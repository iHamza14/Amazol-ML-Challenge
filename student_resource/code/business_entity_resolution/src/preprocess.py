"""
Preprocessing v2 — noise-aware normalisation of business names and addresses.

Every transformation below was motivated by a noise pattern measured on the training
data (see research/DATA_FINDINGS.md):

NAMES
  * Indic-script names (18% of Indian S2/S3)      -> learned transliteration (translit.py)
  * accent injection ('FÁRMS', 'Çentre')            -> unidecode
  * dotted abbreviations ('L.L.C.', 'S.A.R.L.')     -> collapsed ('llc', 'sarl')
  * 'X d/b/a Y', 'X formerly Y'                     -> primary name = Y, alt name = X
  * domain-collapsed names ('cardiologysafecare.com', '@riddhitraders')
                                                    -> TLD/handle stripped + Viterbi word
                                                       segmentation with the S1 vocabulary
  * leetspeak ('Cardi0logy', '5afe', 'A1len')       -> digit->letter inside alphabetic tokens,
                                                       guarded by the S1 vocabulary
  * junk prefixes/suffixes ('*** ', '... ', '#39256', '[INC]', '((LIMITED))')
  * legal suffixes in any position, any variant     -> canonical form, then removed for core
  * filler words inserted by the generator          -> removed for a 'strict' variant

ADDRESSES
  * comma components parsed; state / region / department components canonicalised
    ('Texas', 'TX' -> 'us_tx'; 'Telangana', 'TG', 'తెలంగాణ' -> 'in_tg'; 'Nord' -> 'fr_hauts_de_france')
  * junk tokens ('N/A', '<NULL>', '#', '##', 'N°')  -> removed
  * ordinal suffixes/words ('71st', 'ninth', '1er') -> digits
  * street-type / unit / directional abbreviations  -> single canonical short form
  * hyphenated numbers ('1-02', '628-632')          -> parts kept + joined variant + range
  * leading zeros ('02007', '00354')                -> stripped
  * numbers, first (house) number, postal code, street string, tokens extracted
"""
import re
import math
import logging
import multiprocessing as mp
import numpy as np
import pandas as pd
from unidecode import unidecode

import text_tables as T
from translit import Transliterator, has_indic

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Compiled patterns
# ------------------------------------------------------------------
_WS_RE = re.compile(r'\s+')
_DOTTED_RE = re.compile(r'(?<![a-z0-9])((?:[a-z]\.\s?){1,}[a-z])\b\.?')
_DBA_RE = re.compile(
    r'^(.*?)\b(?:d\s?/\s?b\s?/\s?a|dba|doing business as|formerly known as|formerly|'
    r'f\s?/\s?k\s?/\s?a|fka|t\s?/\s?a|trading as|now known as|nka|a\s?/\s?k\s?/\s?a)\b\s*:?\s*(.+)$')
# a domain needs a literal '.tld'; the dot-less 'com' form ('sreetradecom') is accepted only for long tokens
# that are not vocabulary words (otherwise 'Sirius' -> 'siri', 'Lotus' -> 'lot')
_DOMAIN_RE = re.compile(r'^[\W_]*([a-z0-9][a-z0-9\-\.]*?)\.(?:com|in|fr|net|org|co|io|biz|info|us|uk|eu)$')
_DOMAIN_NODOT_RE = re.compile(r'^[\W_]*([a-z0-9][a-z0-9\-]{5,}?)com$')
_HANDLE_RE = re.compile(r'^[\s]*[@#]\s*([a-z0-9][a-z0-9_\-\.]*)$')
_NONALNUM_RE = re.compile(r'[^a-z0-9\s]')
_LEAD_JUNK_RE = re.compile(r'^[^a-z0-9]+')
_TRAIL_NUM_RE = re.compile(r'(\s\d{4,})+$')
# leet candidates: >=3 letters and exactly 1-2 digits ('cardi0logy', 'a1len'), never long digit runs
_LEET_TOKEN_RE = re.compile(r'^(?=(?:[a-z]*\d){1,2}[a-z]*$)(?=(?:.*[a-z]){3})[a-z0-9]+$')
_CP_CITY_RE = re.compile(r'^(\d{5})\s+([a-z].*)$')
# short legal sigles that are also ordinary name tokens ('Sel Water', 'PC Club', 'Earl Grey', 'SA Apparels'):
# removed only when they are NOT the first token, or when no other legal token exists in the name
_AMBIGUOUS_LEGAL = frozenset({'sa', 'sas', 'sci', 'scs', 'sca', 'sel', 'sem', 'gie', 'earl', 'pc', 'pa', 'ei',
                              'lp', 'snc', 'scm', 'scp', 'sccv', 'prop', 'co'})
_ORD_SUFFIX_RE = re.compile(r'\b(\d+)(?:st|nd|rd|th|er|ere|eme|e)\b')
_NUM_RE = re.compile(r'\d+')
_HYPHEN_NUM_RE = re.compile(r'\b(\d+)\s*[-/]\s*(\d+)\b')
_POSTAL_RE = re.compile(r'^\d{5,6}$')
_JUNK_ADDR_RE = re.compile(r'\b(?:n/a|<null>|null|none|nil|na)\b')
_ELISION_RE = re.compile(r"\b(l|d|qu|j|n|m|t|s|c)'(?=[a-z])")

_STOP = T.STOPWORDS_BLOCKING
_FILLER = T.NAME_FILLER_WORDS
# single-token legal forms only (multi-word forms are canonicalised to these first); never single letters.
# Words that carry identity ('Fer & Fils', 'Liz & Associates', 'Chiffres & Freres') are kept in the core
# name (they are removed only in the 'strict' variant via NAME_FILLER_WORDS).
_IDENTITY_WORDS = frozenset({'associes', 'assoc', 'associates', 'brothers', 'bros', 'sons', 'partners', 'fils',
                             'freres', 'filles', 'firm', 'cie', 'compagnie', 'ets', 'etablissements',
                             'etablissement', 'etab', 'proprietor', 'proprietorship', 'prop', 'chartered',
                             'chtd', 'huf', 'sc', 'company'})
_LEGAL_TOKENS = ((frozenset(s for s in T.LEGAL_SUFFIXES if ' ' not in s and len(s) >= 2)
                  | frozenset(v for v in T.LEGAL_SUFFIX_ABBREV.values() if ' ' not in v and len(v) >= 2)
                  | frozenset({'the'})) - frozenset({'and'})) - _IDENTITY_WORDS
# French big cities must stay address tokens (the tables map them to their region for admin purposes)
for _k in ('lille', 'nantes', 'bordeaux', 'toulouse', 'marseille', 'nice', 'strasbourg', 'montpellier', 'rennes',
           'lyon', 'grenoble', 'rouen', 'reims', 'toulon', 'dijon', 'angers', 'nimes', 'brest', 'tours', 'amiens',
           'limoges', 'metz', 'nancy', 'caen', 'orleans', 'mulhouse', 'besancon', 'perpignan', 'le havre', 'le mans'):
    T.FR_REGIONS.pop(_k, None)
_NUM_MARKER_RE = re.compile(r'\b(?:n|no|nos|num|number|nr)\s*[\.:]?\s*(?=\d)')
_ADMIN_PREFIX = 'zz'
# legal-form families (distractors swap families: Inc->Ltd, EURL->SCI, Private->Public)
_LEGAL_FAMILY = {}
for _fam, _toks in enumerate((
        ('inc', 'incorporated', 'corp', 'corporation'),                 # 1
        ('llc', 'pllc', 'lllp'),                                        # 2
        ('ltd', 'limited', 'plc', 'pvt', 'private', 'pvtltd', 'opc'),   # 3
        ('llp', 'lp'),                                                  # 4
        ('co',),                                                        # 5
        ('pc', 'pa', 'chtd'),                                           # 6
        ('sarl', 'sarlu', 'eurl'),                                      # 7
        ('sas', 'sasu'),                                                # 8
        ('sa', 'sca', 'scs', 'sem'),                                    # 9
        ('sci', 'sccv', 'scm', 'scp', 'snc', 'scop', 'gie', 'sel', 'selarl', 'selas', 'selafa', 'earl', 'gaec'),  # 10
        ('ei', 'eirl', 'prop', 'proprietor', 'proprietorship', 'huf'),  # 11
), start=1):
    for _t in _toks:
        _LEGAL_FAMILY[_t] = _fam


def _phrase_regex(keys):
    """Alternation regex for whole-phrase replacement, longest first."""
    keys = sorted({k for k in keys if k}, key=len, reverse=True)
    if not keys:
        return None
    return re.compile(r'\b(?:' + '|'.join(re.escape(k) for k in keys) + r')\b')


_LEGAL_ABBREV_RE = _phrase_regex(T.LEGAL_SUFFIX_ABBREV.keys())
_ADDR_MULTI_KEYS = [k for k in T.ADDR_ABBREV if ' ' in k]
_ADDR_MULTI_RE = _phrase_regex(_ADDR_MULTI_KEYS)
_CITY_MULTI_KEYS = [k for k in T.CITY_ALIASES if ' ' in k]
_CITY_MULTI_RE = _phrase_regex(_CITY_MULTI_KEYS)
_ORD_MULTI_KEYS = [k for k in T.ORDINALS if ' ' in k]
_ORD_MULTI_RE = _phrase_regex(_ORD_MULTI_KEYS)


# ------------------------------------------------------------------
# Admin (state / region) canonicalisation per country
# ------------------------------------------------------------------
def _admin_lookup(component_norm, country):
    """Return canonical admin token for a normalised comma component, or None."""
    if country == 'us':
        code = T.US_STATES.get(component_norm)
        return f'{_ADMIN_PREFIX}us{code}' if code else None
    if country == 'india':
        code = T.IN_STATES.get(component_norm)
        return f'{_ADMIN_PREFIX}in{code}' if code else None
    if country == 'france':
        reg = T.FR_REGIONS.get(component_norm)
        return _ADMIN_PREFIX + 'fr' + reg.replace(' ', '') if reg else None
    # unknown country: try all
    for tab, pre in ((T.US_STATES, 'us'), (T.IN_STATES, 'in'), (T.FR_REGIONS, 'fr')):
        v = tab.get(component_norm)
        if v:
            return _ADMIN_PREFIX + pre + v.replace(' ', '')
    return None


# ------------------------------------------------------------------
# Word segmentation (for domain-collapsed names) and leet guard
# ------------------------------------------------------------------
class SegVocab:
    """Unigram vocabulary built from S1 business names (provided data only)."""

    def __init__(self, logp, maxlen=18, unk_per_char=-7.0):
        self.logp = logp
        self.maxlen = maxlen
        self.unk = unk_per_char
        # single letters are never vocabulary words for the OOV / leet guards
        self.words = frozenset(w for w in logp.keys() if len(w) >= 2)

    @classmethod
    def build(cls, names, min_count=2):
        cnt = {}
        for nm in names:
            s = _NONALNUM_RE.sub(' ', unidecode(str(nm)).lower())
            for t in s.split():
                if len(t) >= 2:
                    cnt[t] = cnt.get(t, 0) + 1
        tot = float(sum(cnt.values())) or 1.0
        logp = {w: math.log(c / tot) for w, c in cnt.items() if c >= min_count}
        # single letters as words only at a cost clearly above the unknown-run cost, so an unknown chunk
        # stays one token instead of dissolving into letters ('iriecto' must not become 'i r i e c t o')
        for w in ('a', 'i', 'k', 'b', 'j', 'm', 'r', 's', 'n', 'u', 'q', 'h'):
            logp.setdefault(w, math.log(1.0 / tot) - 8.0)
        log.info(f"Segmentation vocabulary: {len(logp):,} words")
        return cls(logp)

    def score(self, s):
        """Best Viterbi log-prob of s (same scoring as segment(); 0.0 for empty / over-long input)."""
        n = len(s)
        if n == 0 or n > 60:
            return 0.0
        best = [0.0] + [-1e18] * n
        lp, unk = self.logp, self.unk
        for i in range(1, n + 1):
            bi = -1e18
            for j in range(max(0, i - self.maxlen), i):
                w = s[j:i]
                sc = lp.get(w)
                if sc is None:
                    sc = -6.0 if w.isdigit() else unk * (i - j)
                if best[j] + sc > bi:
                    bi = best[j] + sc
            best[i] = bi
        return best[n]

    def segment(self, s):
        """Viterbi segmentation maximising sum of unigram log-probs."""
        n = len(s)
        if n == 0:
            return s
        if n > 60:
            return s
        best = [0.0] + [-1e18] * n
        back = [0] * (n + 1)
        lp = self.logp
        unk = self.unk
        for i in range(1, n + 1):
            lo = max(0, i - self.maxlen)
            bi = -1e18
            bj = i - 1
            for j in range(lo, i):
                w = s[j:i]
                sc = lp.get(w)
                if sc is None:
                    if w.isdigit():
                        sc = -6.0
                    else:
                        sc = unk * (i - j)
                cand = best[j] + sc
                if cand > bi:
                    bi = cand
                    bj = j
            best[i] = bi
            back[i] = bj
        out = []
        i = n
        while i > 0:
            j = back[i]
            out.append(s[j:i])
            i = j
        return ' '.join(reversed(out))


_SEG = None          # SegVocab, set per process
_TRANS = None        # Transliterator, set per process


def _deleet_token(tok):
    if not _LEET_TOKEN_RE.match(tok):
        return tok
    if _SEG is not None and tok in _SEG.words:
        return tok
    fixed = ''.join(T.LEET_MAP.get(c, c) if c.isdigit() else c for c in tok)
    if _SEG is None or fixed in _SEG.words or tok not in _SEG.words:
        return fixed
    return tok


# ------------------------------------------------------------------
# NAME normalisation
# ------------------------------------------------------------------
def normalize_name(raw):
    """
    Returns dict with: name_norm, name_core, name_strict, name_collapsed, name_alt,
    is_domain, has_dba, script (0 latin / 1 accented / 2 indic).
    """
    s = str(raw) if raw is not None else ''
    script = 0
    if not s.isascii():
        if has_indic(s):
            script = 2
            s = _TRANS.translit(s) if _TRANS is not None else unidecode(s)
        else:
            script = 1
            s = unidecode(s.replace('�', ''))
    s = s.lower().strip()
    s = _DOTTED_RE.sub(lambda m: m.group(1).replace('.', '').replace(' ', ''), s)

    # d/b/a, formerly ...
    has_dba = 0
    alt = ''
    m = _DBA_RE.search(s)
    if m and len(m.group(2).strip()) >= 3:
        has_dba = 1
        alt = m.group(1)
        s = m.group(2)

    # domain / handle collapsed names
    is_domain = 0
    collapsed_seed = None
    s_nospace = s.replace(' ', '')
    m = _DOMAIN_RE.match(s_nospace)
    if m and (' ' not in s.strip() or s_nospace.endswith(('.com', '.in', '.fr', '.net', '.org'))):
        is_domain = 1
        collapsed_seed = m.group(1).replace('-', '').replace('.', '')
    else:
        m_nd = _DOMAIN_NODOT_RE.match(s_nospace) if ' ' not in s.strip() else None
        if m_nd and (_SEG is None or s_nospace.strip('*.-#@! ') not in _SEG.words):
            is_domain = 1
            collapsed_seed = m_nd.group(1).replace('-', '')
        else:
            m2 = _HANDLE_RE.match(s)
            if m2:
                is_domain = 1
                collapsed_seed = m2.group(1).replace('-', '').replace('.', '').replace('_', '')
    if collapsed_seed is not None:
        if any(ch.isdigit() for ch in collapsed_seed):
            # leet digits inside a collapsed seed break segmentation before any per-token repair could run
            # ('cardi0logysafecare'): compare the raw seed with its de-leeted form by segmentation score —
            # the one the S1 vocabulary explains better wins; a genuine number ('4528mountainview') stays,
            # because LEET_MAP turns it into an unknown letter run
            fixed = ''.join(T.LEET_MAP.get(ch, ch) if ch.isdigit() else ch for ch in collapsed_seed)
            if _SEG is None:
                if _LEET_TOKEN_RE.match(collapsed_seed):
                    collapsed_seed = fixed
            elif _SEG.score(fixed) > _SEG.score(collapsed_seed):
                collapsed_seed = fixed
        s = _SEG.segment(collapsed_seed) if _SEG is not None else collapsed_seed

    # symbols and punctuation; French elisions (l'atelier -> l atelier) before apostrophes are dropped,
    # English possessives (celestyna's) stay glued
    s = s.replace('&', ' and ').replace('+', ' and ').replace('@', ' at ')
    s = s.replace('’', "'").replace('`', "'")
    s = _ELISION_RE.sub(r'\1 ', s)
    s = s.replace("'", '')
    s = _NONALNUM_RE.sub(' ', s)
    s = _WS_RE.sub(' ', s).strip()
    s = _LEAD_JUNK_RE.sub('', s)
    if ' ' in s:
        s = _TRAIL_NUM_RE.sub('', s).strip()

    # leetspeak
    toks = s.split()
    if any(ch.isdigit() for ch in s):
        toks = [_deleet_token(t) for t in toks]
    s = ' '.join(toks)

    # legal suffix canonicalisation and removal
    if _LEGAL_ABBREV_RE is not None:
        s = _LEGAL_ABBREV_RE.sub(lambda m: T.LEGAL_SUFFIX_ABBREV.get(m.group(0), m.group(0)), s)
        s = _WS_RE.sub(' ', s).strip()
    toks = s.split()
    legal_hits = [t for t in toks if t in _LEGAL_TOKENS]
    fams = [_LEGAL_FAMILY[t] for t in toks if t in _LEGAL_FAMILY]
    legal_code = fams[0] if fams else (12 if legal_hits else 0)
    # remove legal forms anywhere in the name ('Private Team Nirman Ltd'), but keep an ambiguous sigle in
    # first position when another legal token exists ('PC Club SAS' -> 'pc club', 'Sel Water Pvt Ltd' -> 'sel water')
    core_toks = []
    for i, t in enumerate(toks):
        if t in _LEGAL_TOKENS:
            if i == 0 and t in _AMBIGUOUS_LEGAL and len(legal_hits) >= 2:
                core_toks.append(t)
            continue
        core_toks.append(t)
    while core_toks and core_toks[-1] == 'and':
        core_toks.pop()
    while core_toks and core_toks[0] == 'and':
        core_toks.pop(0)
    if not core_toks:
        core_toks = toks
    core = ' '.join(core_toks)
    strict_toks = [t for t in core_toks if t not in _FILLER]
    strict = ' '.join(strict_toks) if strict_toks else core
    collapsed = collapsed_seed if collapsed_seed is not None else core.replace(' ', '')
    if alt:
        alt = _WS_RE.sub(' ', _NONALNUM_RE.sub(' ', alt.replace('&', ' and '))).strip()
    # fraction of core tokens (len>=4) unknown to the S1 vocabulary: random generated names ('Iriecto',
    # 'Belonovivio') are 100% out-of-vocabulary, real names ~0%, typos in between
    long_toks = [t for t in core_toks if len(t) >= 4]
    if _SEG is not None and long_toks:
        oov = sum(1 for t in long_toks if t not in _SEG.words) / len(long_toks)
    else:
        oov = 0.0
    return {
        'name_norm': s, 'name_core': core, 'name_strict': strict, 'name_collapsed': collapsed,
        'name_alt': alt, 'is_domain': is_domain, 'has_dba': has_dba, 'name_script': script,
        'legal_code': legal_code, 'name_oov_frac': oov,
    }


def name_tokens(core):
    toks = [t for t in core.split() if len(t) >= 2 and t not in _STOP]
    if not toks:  # names made only of generic words ('global solutions group') must stay retrievable
        toks = [t for t in core.split() if len(t) >= 2]
    return toks


# ------------------------------------------------------------------
# ADDRESS normalisation
# ------------------------------------------------------------------
def _norm_component(c):
    c = c.replace('�', '').replace('°', ' ').replace('º', ' ')
    if not c.isascii():
        # Indic state names etc. are looked up on the raw form first
        key = c.strip().lower()
        for tab in (T.IN_STATES,):
            if key in tab:
                return key
        c = unidecode(c)
    c = c.lower()
    c = _JUNK_ADDR_RE.sub(' ', c)
    c = c.replace('#', ' ').replace('&', ' and ').replace("'", ' ')
    c = _NUM_MARKER_RE.sub(' ', c)
    return _WS_RE.sub(' ', c).strip()


def normalize_address(raw, country):
    """
    Returns dict with: addr_norm, street, admin, postal, nums (list), first_num, ranges (list),
    n_components, addr_tokens (list), addr_words (list).
    """
    s = str(raw) if raw is not None else ''
    if not s.strip():
        return {'addr_norm': '', 'street': '', 'admin': '', 'postal': '', 'nums': [], 'first_num': '',
                'ranges': [], 'n_components': 0, 'addr_tokens': [], 'addr_words': []}
    comps = [c for c in (x.strip() for x in s.split(',')) if c]
    admin = ''
    postal = ''
    kept = []
    for c in comps:
        cn = _norm_component(c)
        if not cn:
            continue
        # admin lookup on a hyphen-free key ('hauts-de-france' -> 'hauts de france'); the component itself
        # keeps its hyphens so hyphenated numbers are still parsed below
        key = _WS_RE.sub(' ', cn.replace('-', ' ')).strip()
        adm = _admin_lookup(key, country)
        if adm is None and not key.isascii():
            adm = _admin_lookup(unidecode(key).lower().strip(), country)
        if adm is not None:
            admin = adm
            kept.append(adm)
            continue
        if _POSTAL_RE.match(cn):
            postal = cn
            kept.append(cn)
            continue
        # French 'CP CITY' inside one component ('59000 LILLE'): the code is a postal code, not a house number
        m_cp = _CP_CITY_RE.match(cn)
        if m_cp and country == 'france':
            postal = m_cp.group(1)
            kept.append(m_cp.group(2))
            continue
        kept.append(cn)
    s = ' , '.join(kept)
    if not s.isascii():
        s = unidecode(s).lower()
    # hyphen / slash numbers: keep parts, remember joined variant and ranges
    ranges = []
    joined = []
    for m in _HYPHEN_NUM_RE.finditer(s):
        a, b = m.group(1), m.group(2)
        sep = s[m.start(1) + len(a):m.start(2)]
        if '-' in sep and len(a) + len(b) <= 4:
            joined.append((a + b).lstrip('0') or '0')
        try:
            ia, ib = int(a), int(b)
            if '-' in sep and ib > ia and ib - ia <= 40:
                ranges.append((ia, ib))
        except ValueError:
            pass
    s = _ORD_SUFFIX_RE.sub(r'\1', s)
    s = _NONALNUM_RE.sub(' ', s)
    s = _WS_RE.sub(' ', s).strip()
    if _ORD_MULTI_RE is not None:
        s = _ORD_MULTI_RE.sub(lambda m: T.ORDINALS[m.group(0)], s)
    if _ADDR_MULTI_RE is not None:
        s = _ADDR_MULTI_RE.sub(lambda m: T.ADDR_ABBREV[m.group(0)], s)
    if _CITY_MULTI_RE is not None:
        s = _CITY_MULTI_RE.sub(lambda m: T.CITY_ALIASES[m.group(0)], s)
    toks = []
    for t in s.split():
        if t in T.ORDINALS:
            t = T.ORDINALS[t]
        elif t.isdigit():
            t = t.lstrip('0') or '0'
        t = T.ADDR_ABBREV.get(t, t)
        # single-token city aliases only for keys of >=4 chars ('jp' must stay 'jp' in 'JP Nagar');
        # multi-word canonical values are split back into tokens
        if len(t) >= 4:
            t = T.CITY_ALIASES.get(t, t)
        if ' ' in t:
            toks.extend(t.split())
        else:
            toks.append(t)
    nums = []
    words = []
    street_toks = []
    for t in toks:
        if t.isdigit():
            nums.append(t)
        else:
            if t.startswith(_ADMIN_PREFIX):
                continue
            digits = ''.join(ch for ch in t if ch.isdigit())
            if digits and any(ch.isalpha() for ch in t) and len(digits) >= 1 and len(t) - len(digits) <= 2:
                nums.append(digits.lstrip('0') or '0')   # '1549a', '16a', 'b3'
            if t not in _STOP:
                words.append(t)
            street_toks.append(t)
    nums_all = nums + [j for j in joined if j not in nums]
    first_num = ''
    for n in nums:
        if n != postal.lstrip('0'):
            first_num = n
            break
    addr_norm = ' '.join(toks)
    addr_tokens = [t for t in toks if (t.isdigit() and len(t) >= 2 and t != postal) or (not t.isdigit() and len(t) >= 2 and t not in _STOP)]
    return {'addr_norm': addr_norm, 'street': ' '.join(street_toks), 'admin': admin, 'postal': postal,
            'nums': nums_all, 'first_num': first_num, 'ranges': ranges, 'n_components': len(comps),
            'addr_tokens': addr_tokens, 'addr_words': words}


# ------------------------------------------------------------------
# Batch processing
# ------------------------------------------------------------------
NAME_COLS = ['name_norm', 'name_core', 'name_strict', 'name_collapsed', 'name_alt',
             'is_domain', 'has_dba', 'name_script', 'legal_code', 'name_oov_frac']
ADDR_COLS = ['addr_norm', 'street', 'admin', 'postal', 'nums', 'first_num', 'ranges',
             'n_components', 'addr_tokens', 'addr_words']


def _init_worker(trans_table, seg_logp):
    global _TRANS, _SEG
    _TRANS = Transliterator(trans_table) if trans_table is not None else None
    _SEG = SegVocab(seg_logp) if seg_logp is not None else None


def _process_chunk(args):
    names, addrs, countries = args
    out_n = {c: [] for c in NAME_COLS}
    out_a = {c: [] for c in ADDR_COLS}
    out_tok = []
    for nm, ad, ct in zip(names, addrs, countries):
        n = normalize_name(nm)
        for c in NAME_COLS:
            out_n[c].append(n[c])
        out_tok.append(name_tokens(n['name_core']))
        a = normalize_address(ad, ct)
        for c in ADDR_COLS:
            out_a[c].append(a[c])
    out_n.update(out_a)
    out_n['name_tokens'] = out_tok
    return out_n


def load_data(file_path, usecols=None):
    """Load a TSV file with explicit tab separator (all columns as str, no NaN)."""
    log.info(f"Loading {file_path}...")
    df = pd.read_csv(file_path, sep='\t', dtype=str, keep_default_na=False, usecols=usecols)
    log.info(f"  -> {len(df):,} rows, columns: {list(df.columns)}")
    return df


def preprocess_dataframe(df, translit=None, seg_vocab=None, n_jobs=1, chunk_size=50000):
    """
    Add normalised columns to a source DataFrame (entity_id, business_name, business_address, country).
    translit : Transliterator (learned on train) or None
    seg_vocab: SegVocab built from S1 names, or None
    """
    n = len(df)
    log.info(f"Preprocessing {n:,} records (jobs={n_jobs})...")
    df = df.reset_index(drop=True)
    df['country_norm'] = df['country'].map(lambda x: unidecode(str(x)).lower().strip())
    names = df['business_name'].values
    addrs = df['business_address'].values
    ctry = df['country_norm'].values
    chunks = [(names[i:i + chunk_size], addrs[i:i + chunk_size], ctry[i:i + chunk_size])
              for i in range(0, n, chunk_size)]
    trans_table = translit.table if translit is not None else None
    seg_logp = seg_vocab.logp if seg_vocab is not None else None
    results = []
    if n_jobs > 1 and len(chunks) > 1:
        with mp.Pool(min(n_jobs, len(chunks)), initializer=_init_worker, initargs=(trans_table, seg_logp)) as pool:
            for i, r in enumerate(pool.imap(_process_chunk, chunks)):
                results.append(r)
                if (i + 1) % 20 == 0:
                    log.info(f"  preprocessed {min((i + 1) * chunk_size, n):,}/{n:,}")
    else:
        _init_worker(trans_table, seg_logp)
        for i, ch in enumerate(chunks):
            results.append(_process_chunk(ch))
            if (i + 1) % 20 == 0:
                log.info(f"  preprocessed {min((i + 1) * chunk_size, n):,}/{n:,}")
    import sys
    intern = sys.intern
    for c in NAME_COLS + ADDR_COLS + ['name_tokens']:
        vals = []
        for r in results:
            vals.extend(r[c])
        if c in ('name_tokens', 'addr_tokens', 'addr_words', 'nums'):
            # one str object per distinct token instead of one per occurrence (~1M distinct vs ~75M
            # occurrences on 6M rows): saves several GB and makes forked workers dirty fewer pages
            for lst in vals:
                if lst:
                    lst[:] = [intern(t) for t in lst]
        df[c] = vals
    for c in ('is_domain', 'has_dba', 'name_script', 'n_components', 'legal_code'):
        df[c] = df[c].astype(np.int8)
    df['addr_empty'] = (df['addr_norm'] == '').astype(np.int8)
    log.info(f"  -> done. domain-like names: {df['is_domain'].mean():.4f}, indic: {(df['name_script'] == 2).mean():.4f}, "
             f"empty addr: {df['addr_empty'].mean():.4f}, admin found: {(df['admin'] != '').mean():.4f}")
    return df


# ------------------------------------------------------------------
# Manual test
# ------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    _init_worker(None, None)
    tests = [
        ("Kimble, Olva S., L.C.S.W., PC", "607 Virginia Street, Terrell, TX", "US"),
        ("cardiologysafecare.com", "617 FIREHOUSE RD, WILLIS, VA", "US"),
        ("Cardi0logy Specialists LLC", "3634 PARK VISTA DR, PMB 8044, MISSOURI CITY, TX", "US"),
        ("*** THE BROWN & WARFIELD LLC", "N/A, CT, GRISWOLD, PLAINFIELD RD", "US"),
        ("Tavotavogild d/b/a Liz & Associates", "Plot No. 16/A, Khasra No. 21/6, Gurugram, Gurgaon, Haryana", "India"),
        ("Anand Ventures Ltd Pvt", "#867 HOUSE NO 121, SECOND FLOOR, ROHINI BLOCK-E, PKT-19, SEC-3, NEW DELHI, Delhi", "India"),
        ("FAB FÁRMS PVT LTD", "GAT NO. ##1174 & 1175/1/1, OFFICE NO. 105, 1ST FLOOR, WAGHOLI (CT) HAVELI PUNE, HAVELI, महाराष्ट्र", "India"),
        ("Lille Amicale Participations E.U.R.L.", "19 R DE LA LIBERTE, LILLE, Nord", "France"),
        ("Article Club (France) S.A.S.U.", "N°17 Av Des Algues, Nantes", "France"),
        ("Atilano, Wheeland and Thompson Ltd", "628-632 NINTH ST, ROCKFORD, IL", "US"),
        ("@riddhitraders", "Plot 428 5, N/A, MH", "India"),
        ("Springdale, City Of Co", "02007 ROBYN RD, SPRINGDALE, AR", "US"),
        ("sportsorion.com", "1-02, ANDHERI EAST, महाराष्ट्र", "India"),
    ]
    for nm, ad, ct in tests:
        n = normalize_name(nm)
        a = normalize_address(ad, ct.lower())
        print(f"\nNAME {nm!r}\n  -> norm={n['name_norm']!r} core={n['name_core']!r} strict={n['name_strict']!r} collapsed={n['name_collapsed']!r} dom={n['is_domain']} dba={n['has_dba']} alt={n['name_alt']!r}")
        print(f"ADDR {ad!r}\n  -> norm={a['addr_norm']!r}\n     street={a['street']!r} admin={a['admin']!r} postal={a['postal']!r} nums={a['nums']} first={a['first_num']!r} ranges={a['ranges']} toks={a['addr_tokens']}")
