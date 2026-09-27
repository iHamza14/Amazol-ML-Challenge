"""
Feature engineering v2 — vectorised pair features for candidate chunks.

Groups
  NAME   fuzzy similarities on several normalised variants (core / strict / with-suffix /
         collapsed / alt-name), TF-IDF cosines (word + char 3-gram), token overlap,
         rare-shared-token IDF, extra-token statistics learned from training pairs
         (words the distractor generator inserts: 'holdings', 'midtown', ... have ~0%
         true-match rate; words the noise generator inserts: 'center', 'services', 'dr'
         have ~95%).
  ADDR   fuzzy similarities on full address and street string, TF-IDF cosine, word overlap,
         admin (state/region) agreement, postal code agreement, component counts.
  NUMBERS the decisive signal for near-miss distractors: first-number relation
         (equal / truncation prefix / truncation suffix / digit substitution / disjoint),
         absolute difference, digit hamming, shared-number counts, range containment,
         digit-string containment ('1-02' vs '102').
  FLAGS  script of the S2/S3 name, domain-collapsed, dba, empty address, source (S2/S3).
  BLOCK  per-channel retrieval scores and ranks, number of channels, candidate count.
  GROUP  relative features within the S1 entity's candidate set: gap to the best
         candidate on name / address / street / retrieval score, rank by combined
         similarity, number of candidates whose house number equals S1's.

All functions operate on a candidate chunk DataFrame (s1_pos, s23_pos, block meta) plus
the two preprocessed frames; strings are pulled by position from numpy arrays.
"""
import json
import logging
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

import config as cfg
from blocking import BLOCK_META_COLS

log = logging.getLogger(__name__)

# rapidfuzz threads in the main process (the cgroup quota, not the host core count: -1 would start 96 threads
# on a 9-CPU pod); forked workers set this to 1
WORKERS = cfg.N_JOBS
# admin units the noise generator swaps in TRUE pairs (Telangana was carved out of Andhra Pradesh in 2014)
ADMIN_ALIASES = {frozenset(('zzintg', 'zzinap'))}


# ------------------------------------------------------------------
# Extra-token statistics (learned on a held-out split of TRAIN, applied everywhere)
# ------------------------------------------------------------------
class ExtraTokenStats:
    """P(true match | token appears in S2/S3 name but not in S1 name), smoothed."""

    def __init__(self, table=None, prior=0.5, alpha=20.0):
        self.table = table or {}
        self.prior = prior
        self.alpha = alpha

    @classmethod
    def learn(cls, toks1_list, toks2_list, labels, min_total=20, alpha=20.0):
        n_true = {}
        n_tot = {}
        pos_with_extra = tot_with_extra = 0
        for t1, t2, y in zip(toks1_list, toks2_list, labels):
            extra = set(t2) - set(t1)
            if not extra:
                continue
            tot_with_extra += 1
            pos_with_extra += int(y)
            for t in extra:
                n_tot[t] = n_tot.get(t, 0) + 1
                if y:
                    n_true[t] = n_true.get(t, 0) + 1
        prior = pos_with_extra / tot_with_extra if tot_with_extra else 0.5
        table = {}
        for t, n in n_tot.items():
            if n >= min_total:
                table[t] = (n_true.get(t, 0) + alpha * prior) / (n + alpha)
        log.info(f"ExtraTokenStats: {len(table):,} tokens (prior={prior:.3f}) from {tot_with_extra:,} pairs with extras")
        return cls(table, prior=prior, alpha=alpha)

    def features(self, toks1_list, toks2_list):
        n = len(toks1_list)
        f_min = np.ones(n, dtype=np.float32)
        f_mean = np.ones(n, dtype=np.float32)
        f_known = np.ones(n, dtype=np.float32)
        f_cnt = np.zeros(n, dtype=np.float32)
        f_miss = np.zeros(n, dtype=np.float32)
        tab = self.table
        prior = self.prior
        for i, (t1, t2) in enumerate(zip(toks1_list, toks2_list)):
            s1 = set(t1)
            s2 = set(t2)
            extra = s2 - s1
            f_miss[i] = len(s1 - s2)
            if not extra:
                continue
            vals = []
            known = 0
            for t in extra:
                v = tab.get(t)
                if v is None:
                    vals.append(prior)
                else:
                    vals.append(v)
                    known += 1
            f_cnt[i] = len(extra)
            f_min[i] = min(vals)
            f_mean[i] = sum(vals) / len(vals)
            f_known[i] = known / len(extra)
        return f_min, f_mean, f_known, f_cnt, f_miss

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'prior': self.prior, 'alpha': self.alpha, 'table': self.table}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        return cls(d['table'], prior=d['prior'], alpha=d['alpha'])


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _cp(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=WORKERS, dtype=np.float32, **kw)


def _cos(vec, docs1, docs2):
    if vec is None or not vec.vocab:
        return np.zeros(len(docs1), dtype=np.float32)
    A = vec.transform(docs1)
    B = vec.transform(docs2)
    return np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(np.float32)


def _first_num_relation(a, b, nums_a, nums_b):
    """
    0 both missing | 1 one missing | 2 equal | 3 prefix-truncation | 4 suffix-truncation |
    5 S1 first appears elsewhere in S2/S3 numbers | 6 S2/S3 first appears elsewhere in S1 numbers |
    7 same length, one digit differs | 8 same length, >=2 digits differ | 9 different length, unrelated
    """
    if not a and not b:
        return 0
    if not a or not b:
        return 1
    if a == b:
        return 2
    if len(a) != len(b):
        s, l = (a, b) if len(a) < len(b) else (b, a)
        if l.startswith(s):
            return 3
        if l.endswith(s):
            return 4
        if a in nums_b:
            return 5
        if b in nums_a:
            return 6
        return 9
    if a in nums_b:
        return 5
    if b in nums_a:
        return 6
    ham = sum(x != y for x, y in zip(a, b))
    return 7 if ham == 1 else 8


def _num_features(first1, first2, nums1, nums2, ranges1, ranges2, addr1, addr2, postal1, postal2):
    n = len(first1)
    rel = np.zeros(n, dtype=np.int8)
    absdiff = np.full(n, -1.0, dtype=np.float32)
    hamm = np.full(n, -1.0, dtype=np.float32)
    len_eq = np.zeros(n, dtype=np.int8)
    shared = np.zeros(n, dtype=np.float32)
    jacc = np.zeros(n, dtype=np.float32)
    s1_in_s23 = np.zeros(n, dtype=np.int8)
    extra23 = np.zeros(n, dtype=np.float32)
    miss1 = np.zeros(n, dtype=np.float32)
    in_range = np.zeros(n, dtype=np.int8)
    digit_contain = np.zeros(n, dtype=np.int8)
    postal_rel = np.zeros(n, dtype=np.int8)
    nn1 = np.zeros(n, dtype=np.float32)
    nn2 = np.zeros(n, dtype=np.float32)
    flen1 = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a = first1[i]
        b = first2[i]
        na = nums1[i]
        nb = nums2[i]
        sa = set(na)
        sb = set(nb)
        nn1[i] = len(sa)
        nn2[i] = len(sb)
        flen1[i] = len(a)
        rel[i] = _first_num_relation(a, b, sa, sb)
        if a and b:
            len_eq[i] = int(len(a) == len(b))
            try:
                d = abs(int(a) - int(b))
                absdiff[i] = min(d, 99999)
            except ValueError:
                pass
            if len(a) == len(b):
                hamm[i] = sum(x != y for x, y in zip(a, b))
        if sa or sb:
            inter = len(sa & sb)
            shared[i] = inter
            jacc[i] = inter / len(sa | sb)
            s1_in_s23[i] = int(bool(sa) and sa <= sb)
            extra23[i] = len(sb - sa)
            miss1[i] = len(sa - sb)
        if a and ranges2[i]:
            try:
                ia = int(a)
                in_range[i] = int(any(lo <= ia <= hi for lo, hi in ranges2[i]))
            except ValueError:
                pass
        if not in_range[i] and b and ranges1[i]:
            try:
                ib = int(b)
                in_range[i] = int(any(lo <= ib <= hi for lo, hi in ranges1[i]))
            except ValueError:
                pass
        if a and b and a != b:
            d1 = ''.join(ch for ch in addr1[i] if ch.isdigit())
            d2 = ''.join(ch for ch in addr2[i] if ch.isdigit())
            if (len(a) >= 2 and a in d2) or (len(b) >= 2 and b in d1):
                digit_contain[i] = 1
        pa = postal1[i]
        pb = postal2[i]
        if pa and pb:
            postal_rel[i] = 2 if pa == pb else 3
        elif pa or pb:
            postal_rel[i] = 1
    return dict(num_first_rel=rel, num_first_absdiff=absdiff, num_first_hamming=hamm, num_first_len_eq=len_eq,
                num_shared=shared, num_jacc=jacc, num_s1_in_s23=s1_in_s23, num_s23_extra=extra23,
                num_s1_missing=miss1, num_in_range=in_range, num_digit_contain=digit_contain,
                postal_rel=postal_rel, num_count_s1=nn1, num_count_s23=nn2, num_first_len_s1=flen1)


def _token_overlap(toks1, toks2, vocab, idf):
    n = len(toks1)
    jacc = np.zeros(n, dtype=np.float32)
    contain = np.zeros(n, dtype=np.float32)
    n_shared = np.zeros(n, dtype=np.float32)
    idf_max = np.zeros(n, dtype=np.float32)
    idf_sum = np.zeros(n, dtype=np.float32)
    first_eq = np.zeros(n, dtype=np.int8)
    last_eq = np.zeros(n, dtype=np.int8)
    nt1 = np.zeros(n, dtype=np.float32)
    nt2 = np.zeros(n, dtype=np.float32)
    has_vocab = bool(vocab)
    for i in range(n):
        t1 = toks1[i]
        t2 = toks2[i]
        nt1[i] = len(t1)
        nt2[i] = len(t2)
        if not t1 or not t2:
            continue
        s1 = set(t1)
        s2 = set(t2)
        inter = s1 & s2
        if inter:
            n_shared[i] = len(inter)
            jacc[i] = len(inter) / len(s1 | s2)
            contain[i] = len(inter) / min(len(s1), len(s2))
            if has_vocab:
                mx = 0.0
                sm = 0.0
                for t in inter:
                    j = vocab.get(t)
                    if j is not None:
                        v = float(idf[j])
                        sm += v
                        if v > mx:
                            mx = v
                idf_max[i] = mx
                idf_sum[i] = sm
        first_eq[i] = int(t1[0] == t2[0])
        last_eq[i] = int(t1[-1] == t2[-1])
    return jacc, contain, n_shared, idf_max, idf_sum, first_eq, last_eq, nt1, nt2


_VOWELS = str.maketrans('', '', 'aeiou')


def _skeleton(s):
    """Consonant skeleton: drop vowels, collapse repeated letters ('limittedd' -> 'lmtd')."""
    if not s:
        return ''
    t = s.translate(_VOWELS)
    out = []
    prev = ''
    for ch in t:
        if ch != prev:
            out.append(ch)
        prev = ch
    return ''.join(out)


def _soft_token_match(toks1_list, toks2_list, jw_threshold=0.9):
    """
    SoftTFIDF-style token coverage: fraction of tokens (len>=3) on each side that have an exact or
    near-identical (Jaro-Winkler >= threshold) token on the other side, plus an acronym flag
    (initials of one side's tokens equal a token of the other side, e.g. 'cpam' vs 'caisse primaire ...').
    """
    n = len(toks1_list)
    cov1 = np.zeros(n, dtype=np.float32)
    cov2 = np.zeros(n, dtype=np.float32)
    acro = np.zeros(n, dtype=np.int8)
    short_edit = np.zeros(n, dtype=np.float32)   # S1 tokens len<=4 (acronyms) with no exact match but a 1-edit neighbour
    long_edit = np.zeros(n, dtype=np.float32)    # S1 tokens len>=5 with no exact match but a <=2-edit neighbour (typos)
    sim = JaroWinkler.similarity
    lev = Levenshtein.distance
    for i in range(n):
        t1 = [t for t in toks1_list[i] if len(t) >= 3]
        t2 = [t for t in toks2_list[i] if len(t) >= 3]
        if not t1 or not t2:
            continue
        s2 = set(t2)
        s1 = set(t1)
        m1 = 0
        for a in t1:
            if a in s2:
                m1 += 1
                continue
            matched = False
            for b in t2:
                if abs(len(a) - len(b)) <= 3 and a[0] == b[0] and sim(a, b) >= jw_threshold:
                    m1 += 1
                    matched = True
                    break
            # distractor fingerprint: a different acronym ('ye'->'ym', 'nrm'->'rm') vs a typo in a long word
            if len(a) <= 4:
                if any(lev(a, b, score_cutoff=1) <= 1 for b in t2 if abs(len(a) - len(b)) <= 1):
                    short_edit[i] += 1
            elif not matched:
                if any(lev(a, b, score_cutoff=2) <= 2 for b in t2 if abs(len(a) - len(b)) <= 2):
                    long_edit[i] += 1
        m2 = 0
        for b in t2:
            if b in s1:
                m2 += 1
                continue
            for a in t1:
                if abs(len(a) - len(b)) <= 3 and a[0] == b[0] and sim(a, b) >= jw_threshold:
                    m2 += 1
                    break
        cov1[i] = m1 / len(t1)
        cov2[i] = m2 / len(t2)
        if len(toks1_list[i]) >= 2:
            ini = ''.join(t[0] for t in toks1_list[i])
            if len(ini) >= 2 and ini in s2:
                acro[i] = 1
        if not acro[i] and len(toks2_list[i]) >= 2:
            ini = ''.join(t[0] for t in toks2_list[i])
            if len(ini) >= 2 and ini in s1:
                acro[i] = 1
    return cov1, cov2, acro, short_edit, long_edit


def _common_prefix_ratio(a_list, b_list):
    out = np.zeros(len(a_list), dtype=np.float32)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        if not a or not b:
            continue
        m = min(len(a), len(b))
        k = 0
        while k < m and a[k] == b[k]:
            k += 1
        out[i] = k / max(len(a), len(b))
    return out


# ------------------------------------------------------------------
# main entry
# ------------------------------------------------------------------
def compute_features(cand, df_s1, df_s23, vecs=None, extra_stats=None):
    """
    cand   : candidate chunk (s1_pos, s23_pos, block meta columns)
    df_s1  : preprocessed S1 frame (RangeIndex)
    df_s23 : preprocessed S2/S3 frame (RangeIndex)
    vecs   : dict channel -> SparseVectorizer fitted on this country's S2/S3 (optional)
    extra_stats : ExtraTokenStats (optional)
    Returns DataFrame with feature columns (float32/int8) aligned with cand rows.
    """
    p1 = cand['s1_pos'].values
    p2 = cand['s23_pos'].values
    n = len(cand)
    vecs = vecs or {}

    def g1(c):
        return df_s1[c].values[p1]

    def g2(c):
        return df_s23[c].values[p2]

    F = {}
    # ---------------- NAME ----------------
    core1, core2 = g1('name_core').tolist(), g2('name_core').tolist()
    strict1, strict2 = g1('name_strict').tolist(), g2('name_strict').tolist()
    norm1, norm2 = g1('name_norm').tolist(), g2('name_norm').tolist()
    col1, col2 = g1('name_collapsed').tolist(), g2('name_collapsed').tolist()
    alt2 = g2('name_alt').tolist()
    F['n_ratio_core'] = _cp(core1, core2, fuzz.ratio)
    F['n_tsort_core'] = _cp(core1, core2, fuzz.token_sort_ratio)
    F['n_tset_core'] = _cp(core1, core2, fuzz.token_set_ratio)
    F['n_partial_core'] = _cp(core1, core2, fuzz.partial_ratio)
    F['n_jw_core'] = _cp(core1, core2, JaroWinkler.normalized_similarity) * 100.0
    F['n_ratio_strict'] = _cp(strict1, strict2, fuzz.ratio)
    F['n_tset_strict'] = _cp(strict1, strict2, fuzz.token_set_ratio)
    F['n_ratio_norm'] = _cp(norm1, norm2, fuzz.ratio)
    F['n_collapsed_ratio'] = _cp(col1, col2, fuzz.ratio)
    F['n_collapsed_partial'] = _cp(col1, col2, fuzz.partial_ratio)
    F['n_prefix_ratio'] = _common_prefix_ratio(col1, col2)
    # --- comparators from the ER literature (Cohen 2003 SoftTFIDF idea, Foursquare LCS features, Splink ladders) ---
    from rapidfuzz.distance import LCSseq, Prefix, Postfix, Indel
    F['n_lcs_core'] = _cp(core1, core2, LCSseq.normalized_similarity) * 100.0
    F['n_indel_collapsed'] = _cp(col1, col2, Indel.normalized_similarity) * 100.0
    F['n_postfix_collapsed'] = _cp(col1, col2, Postfix.normalized_similarity) * 100.0
    sk1 = [_skeleton(s) for s in core1]
    sk2 = [_skeleton(s) for s in core2]
    F['n_skel_ratio'] = _cp(sk1, sk2, fuzz.ratio)            # consonant skeleton: robust to vowel/schwa noise
    F['n_skel_tset'] = _cp(sk1, sk2, fuzz.token_set_ratio)
    soft_cov1, soft_cov2, acro, short_edit, long_edit = _soft_token_match([s.split() for s in core1], [s.split() for s in core2])
    F['n_soft_cov_s1'] = soft_cov1        # fraction of S1 name tokens with a near-identical (JW>=0.9) S2/S3 token
    F['n_soft_cov_s23'] = soft_cov2
    F['n_acronym'] = acro                # initials of one side's tokens == a token of the other side
    F['n_short_edit'] = short_edit       # acronym-letter edits ('ye'->'ym'): distractor fingerprint
    F['n_long_edit'] = long_edit         # typo edits in long words: ordinary noise
    # random generated names ('Iriecto', 'Belonovivio') are 100% out of the S1 vocabulary; the generator's
    # distractors never use random strings, so OOV + matching address is a POSITIVE signal
    F['x_oov_s1'] = g1('name_oov_frac').astype(np.float32) if 'name_oov_frac' in df_s1.columns else np.zeros(n, dtype=np.float32)
    F['x_oov_s23'] = g2('name_oov_frac').astype(np.float32) if 'name_oov_frac' in df_s23.columns else np.zeros(n, dtype=np.float32)
    # how many S1 entities of the country carry exactly the candidate's core name (chains vs unique names)
    F['s23_core_s1_count'] = np.minimum(g2('s1_core_count').astype(np.float32), 5.0) if 's1_core_count' in df_s23.columns else np.zeros(n, dtype=np.float32)
    has_alt = np.array([bool(a) for a in alt2])
    alt_sim = np.zeros(n, dtype=np.float32)
    if has_alt.any():
        idx = np.flatnonzero(has_alt)
        alt_sim[idx] = _cp([core1[i] for i in idx], [alt2[i] for i in idx], fuzz.token_set_ratio)
    F['n_alt_tset'] = alt_sim
    F['n_exact'] = np.array([int(a == b and a != '') for a, b in zip(core1, core2)], dtype=np.int8)
    F['n_sorted_exact'] = np.array([int(a != '' and sorted(a.split()) == sorted(b.split())) for a, b in zip(core1, core2)], dtype=np.int8)
    F['n_len_s1'] = np.array([len(a) for a in core1], dtype=np.float32)
    F['n_len_s23'] = np.array([len(b) for b in core2], dtype=np.float32)
    F['n_len_ratio'] = np.minimum(F['n_len_s1'], F['n_len_s23']) / np.maximum(np.maximum(F['n_len_s1'], F['n_len_s23']), 1)
    toks1, toks2 = g1('name_tokens').tolist(), g2('name_tokens').tolist()
    v = vecs.get('name_tok')
    jacc, contain, n_shared, idf_max, idf_sum, first_eq, last_eq, nt1, nt2 = _token_overlap(
        toks1, toks2, v.vocab if v is not None else {}, v.idf if v is not None else None)
    F['n_jacc_tok'] = jacc
    F['n_contain_tok'] = contain
    F['n_shared_tok'] = n_shared
    F['n_shared_idf_max'] = idf_max
    F['n_shared_idf_sum'] = idf_sum
    F['n_first_tok_eq'] = first_eq
    F['n_last_tok_eq'] = last_eq
    F['n_ntok_s1'] = nt1
    F['n_ntok_s23'] = nt2
    F['n_cos_tok'] = _cos(vecs.get('name_tok'), toks1, toks2)
    from blocking import _char_ngrams
    F['n_cos_chr'] = _cos(vecs.get('name_chr'), [_char_ngrams(s) for s in col1], [_char_ngrams(s) for s in col2])
    # all tokens (incl. fillers) for extra-token stats
    all1 = [s.split() for s in core1]
    all2 = [s.split() for s in core2]
    if extra_stats is not None:
        e_min, e_mean, e_known, e_cnt, e_miss = extra_stats.features(all1, all2)
    else:
        e_min = np.ones(n, dtype=np.float32)
        e_mean = np.ones(n, dtype=np.float32)
        e_known = np.zeros(n, dtype=np.float32)
        e_cnt = np.array([len(set(b) - set(a)) for a, b in zip(all1, all2)], dtype=np.float32)
        e_miss = np.array([len(set(a) - set(b)) for a, b in zip(all1, all2)], dtype=np.float32)
    F['x_extra_min'] = e_min
    F['x_extra_mean'] = e_mean
    F['x_extra_known'] = e_known
    F['x_extra_cnt'] = e_cnt
    F['x_missing_cnt'] = e_miss

    # ---------------- LEGAL FORM / NAME AMBIGUITY ----------------
    lc1 = g1('legal_code').astype(np.int8) if 'legal_code' in df_s1.columns else np.zeros(n, dtype=np.int8)
    lc2 = g2('legal_code').astype(np.int8) if 'legal_code' in df_s23.columns else np.zeros(n, dtype=np.int8)
    legal_rel = np.where((lc1 > 0) & (lc2 > 0), np.where(lc1 == lc2, 2, 3), np.where((lc1 > 0) | (lc2 > 0), 1, 0))
    F['legal_rel'] = legal_rel.astype(np.int8)
    F['legal_code_s23'] = lc2
    # how many S1 entities of the same country share this S1's core name (51% of S1 have a same-name twin).
    # Capped at 5: raw counts scale with the S1 pool (US test pool is half the train pool), the capped
    # value ('unique / twin / small chain / big chain') does not.
    F['s1_name_dup'] = np.minimum(g1('name_dup').astype(np.float32), 5.0) if 'name_dup' in df_s1.columns else np.zeros(n, dtype=np.float32)

    # ---------------- FLAGS ----------------
    F['f_script_s23'] = g2('name_script').astype(np.int8)
    F['f_domain_s23'] = g2('is_domain').astype(np.int8)
    F['f_domain_s1'] = g1('is_domain').astype(np.int8)
    F['f_dba_s23'] = g2('has_dba').astype(np.int8)
    F['f_addr_empty_s23'] = g2('addr_empty').astype(np.int8)
    F['f_addr_empty_s1'] = g1('addr_empty').astype(np.int8)
    F['f_source_s3'] = np.array([int(e.startswith('S3')) for e in g2('entity_id')], dtype=np.int8)

    # ---------------- ADDRESS ----------------
    a1, a2 = g1('addr_norm').tolist(), g2('addr_norm').tolist()
    st1, st2 = g1('street').tolist(), g2('street').tolist()
    F['a_ratio'] = _cp(a1, a2, fuzz.ratio)
    F['a_tset'] = _cp(a1, a2, fuzz.token_set_ratio)
    F['a_tsort'] = _cp(a1, a2, fuzz.token_sort_ratio)
    F['a_partial'] = _cp(a1, a2, fuzz.partial_ratio)
    F['st_ratio'] = _cp(st1, st2, fuzz.ratio)
    F['st_tset'] = _cp(st1, st2, fuzz.token_set_ratio)
    F['st_jw'] = _cp(st1, st2, JaroWinkler.normalized_similarity) * 100.0
    F['st_partial'] = _cp(st1, st2, fuzz.partial_ratio)
    at1, at2 = g1('addr_tokens').tolist(), g2('addr_tokens').tolist()
    F['a_cos_tok'] = _cos(vecs.get('addr_tok'), at1, at2)
    aw1, aw2 = g1('addr_words').tolist(), g2('addr_words').tolist()
    va = vecs.get('addr_tok')
    jacc, contain, n_shared, idf_max, idf_sum, first_eq, last_eq, nt1, nt2 = _token_overlap(
        aw1, aw2, va.vocab if va is not None else {}, va.idf if va is not None else None)
    F['a_jacc_words'] = jacc
    F['a_contain_words'] = contain
    F['a_shared_words'] = n_shared
    F['a_shared_idf_max'] = idf_max
    F['a_shared_idf_sum'] = idf_sum
    F['a_nwords_s1'] = nt1
    F['a_nwords_s23'] = nt2
    F['a_ncomp_s1'] = g1('n_components').astype(np.float32)
    F['a_ncomp_s23'] = g2('n_components').astype(np.float32)
    F['a_len_s1'] = np.array([len(x) for x in a1], dtype=np.float32)
    F['a_len_s23'] = np.array([len(x) for x in a2], dtype=np.float32)
    adm1, adm2 = g1('admin'), g2('admin')
    adm_rel = np.zeros(n, dtype=np.int8)
    for i in range(n):
        x, y = adm1[i], adm2[i]
        if x and y:
            adm_rel[i] = 2 if (x == y or frozenset((x, y)) in ADMIN_ALIASES) else 3
        elif x or y:
            adm_rel[i] = 1
    F['a_admin_rel'] = adm_rel
    F['joint_cos'] = _cos(vecs.get('joint'), [list(a) + list(b) for a, b in zip(toks1, at1)],
                          [list(a) + list(b) for a, b in zip(toks2, at2)])

    # ---------------- NUMBERS ----------------
    numf = _num_features(g1('first_num').tolist(), g2('first_num').tolist(), g1('nums').tolist(), g2('nums').tolist(),
                         g1('ranges').tolist(), g2('ranges').tolist(), a1, a2, g1('postal').tolist(), g2('postal').tolist())
    F.update(numf)

    # ---------------- CROSS ----------------
    F['name_in_addr'] = np.array([int((len(c1) >= 4 and c1 in x2) or (len(c2) >= 4 and c2 in x1))
                                  for c1, c2, x1, x2 in zip(core1, core2, a1, a2)], dtype=np.int8)
    F['combo_sim'] = (0.5 * F['n_tset_core'] + 0.5 * F['a_ratio']).astype(np.float32)

    feat = pd.DataFrame(F)
    # ---------------- BLOCK META ----------------
    for c in BLOCK_META_COLS:
        feat[c] = cand[c].values
    # ---------------- GROUP-RELATIVE ----------------
    feat = add_group_features(feat, p1)
    # Absolute pool-density counts shift between train (4.67 S2/S3 per S1) and test (5.75): teams that used
    # them gained on validation and lost on the leaderboard. Keep ranks/gaps, drop raw counts.
    # legal_code_s23: French legal-form codes (SARL/SAS/SA/SCI families) never occur in training, so the raw
    # code would route unseen values arbitrarily; only the relation (same/different family) is kept.
    feat = feat.drop(columns=[c for c in DROP_COLS if c in feat.columns])
    return feat


DROP_COLS = ['cand_count', 'g_n_first_eq', 'g_n_high_name', 'legal_code_s23']
# ER_DROP_FEATURES=x_oov_s23,s23_core_s1_count  -> ablations without code changes (train and inference
# both read it; the trained feature list is stored in model_config.json so inference stays consistent)
import os as _os
DROP_COLS += [c.strip() for c in _os.environ.get('ER_DROP_FEATURES', '').split(',') if c.strip()]
DENSITY_SENSITIVE_COLS = DROP_COLS


def add_group_features(feat, s1_pos):
    g = pd.Series(s1_pos)
    grp = feat.groupby(g.values, sort=False)
    for col, name in (('n_tset_core', 'g_gap_ntset'), ('a_ratio', 'g_gap_aratio'), ('st_ratio', 'g_gap_st'),
                      ('sum_score', 'g_gap_score'), ('combo_sim', 'g_gap_combo'), ('joint_cos', 'g_gap_joint')):
        mx = grp[col].transform('max')
        feat[name] = (mx - feat[col]).astype(np.float32)
        feat['g_max_' + col] = mx.astype(np.float32)
    feat['g_rank_combo'] = grp['combo_sim'].rank(ascending=False, method='min').astype(np.float32)
    feat['g_rank_joint'] = grp['joint_cos'].rank(ascending=False, method='min').astype(np.float32)
    # vectorised group counts (no Python lambdas)
    first_eq = (feat['num_first_rel'].values == 2).astype(np.float32)
    high_name = (feat['n_tset_core'].values >= 85).astype(np.float32)
    feat['g_n_first_eq'] = pd.Series(first_eq).groupby(g.values, sort=False).transform('sum').values.astype(np.float32)
    feat['g_n_high_name'] = pd.Series(high_name).groupby(g.values, sort=False).transform('sum').values.astype(np.float32)
    return feat


def feature_columns(feat):
    return [c for c in feat.columns if c not in ('s1_pos', 's23_pos', 'label', 'country', 's1_id', 's23_id')]


# ------------------------------------------------------------------
# Parallel feature computation (Linux/fork only; falls back to the main process elsewhere)
#
# Memory model (the reason for the design): the parent holds ~27 GB of Python-object frames on the full
# data. A forked worker that reads those frames dirties every page it touches (refcount writes), which cost
# 4-5 GB per worker and OOM-killed two full runs on a 47 GB pod. Workers therefore receive a pickled copy of
# ONLY their slice (the rows of df_s1 / df_s23 their pairs reference, positions remapped) and never touch the
# inherited frames; the vectorizers and extra-token statistics are inherited read-only and are small.
# Per-worker cost = slice (~150 MB per 100k pairs) + working set (~300 MB); the pool is sized against the RAM
# budget with FEATURE_WORKER_GB = 3 GB per worker, i.e. a 3x margin on the measured need.
# ------------------------------------------------------------------
_PAR = {}


def _par_init(vecs, extra_stats):
    global _PAR, WORKERS
    import gc
    gc.disable()         # workers allocate only short-lived per-task objects; no collections over the inherited heap
    _PAR = {'vecs': vecs, 'stats': extra_stats}
    WORKERS = 1          # rapidfuzz threads: one per process, the pool provides the parallelism


def _par_work(task):
    cand_sub, s1_slice, s23_slice = task
    return compute_features(cand_sub, s1_slice, s23_slice, vecs=_PAR['vecs'], extra_stats=_PAR['stats'])


def _slice_task(part, df_s1, df_s23):
    """
    Self-contained task for one part of the (s1_pos-sorted) candidate chunk: the part with s1_pos / s23_pos
    remapped to positions in compact copies of the referenced rows. compute_features only ever indexes the
    frames by these positions (and groups by s1_pos), so the output is identical to the unsliced call.
    """
    u1, inv1 = np.unique(part['s1_pos'].values, return_inverse=True)
    u2, inv2 = np.unique(part['s23_pos'].values, return_inverse=True)
    sub = part.copy()
    sub['s1_pos'] = inv1.astype(np.int64)
    sub['s23_pos'] = inv2.astype(np.int64)
    s1_slice = df_s1.iloc[u1].reset_index(drop=True)
    s23_slice = df_s23.iloc[u2].reset_index(drop=True)
    return sub, s1_slice, s23_slice


def split_by_s1_groups(cand_sorted, target_pairs):
    """Boundaries [a, b) of consecutive parts of ~target_pairs rows that never split an S1 entity's group."""
    n = len(cand_sorted)
    s1s = cand_sorted['s1_pos'].values
    group_starts = np.r_[0, np.flatnonzero(np.diff(s1s)) + 1]
    n_parts = max(1, min(int(np.ceil(n / max(1, target_pairs))), len(group_starts)))
    targets = (np.arange(1, n_parts) * n) // n_parts
    cuts = [0]
    for t in targets:
        j = np.searchsorted(group_starts, t)
        cuts.append(int(group_starts[j]) if j < len(group_starts) else n)
    cuts.append(n)
    cuts = sorted(set(cuts))
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def _pool_size(n_jobs):
    """Workers allowed by the RAM budget: parent RSS + workers * FEATURE_WORKER_GB + headroom <= TOTAL_RAM_GB."""
    try:
        if cfg.TOTAL_RAM_GB > 0:
            rss = cfg.rss_gb()
            free = cfg.TOTAL_RAM_GB - rss - cfg.FEATURE_RAM_HEADROOM_GB
            allowed = max(1, int(free / cfg.FEATURE_WORKER_GB))
            if allowed < n_jobs:
                log.info(f"  feature workers capped {n_jobs} -> {allowed} (parent rss {rss:.1f} GB of {cfg.TOTAL_RAM_GB:.0f} GB, "
                         f"budget {cfg.FEATURE_WORKER_GB:.0f} GB/worker + {cfg.FEATURE_RAM_HEADROOM_GB:.0f} GB headroom)")
                return allowed
    except Exception:  # noqa
        pass
    return n_jobs


def compute_features_parallel(cand, df_s1, df_s23, vecs=None, extra_stats=None, n_jobs=1, min_pairs=60000):
    """
    Same output as compute_features (rows aligned with `cand`), computed by up to n_jobs forked worker
    processes on disjoint S1 groups (group-relative features need every candidate of an S1 entity in the
    same part). Workers get pickled slices (see the memory model above); on Windows / macOS it runs in the
    main process with rapidfuzz threads.
    """
    import platform
    import multiprocessing as mp
    n = len(cand)
    if n_jobs > 1 and n >= min_pairs and platform.system() == 'Linux':
        n_jobs = _pool_size(n_jobs)
    if n_jobs <= 1 or n < min_pairs or platform.system() != 'Linux':
        return compute_features(cand, df_s1, df_s23, vecs=vecs, extra_stats=extra_stats)
    s1p = cand['s1_pos'].values
    order = np.argsort(s1p, kind='stable')
    cand_sorted = cand.iloc[order].reset_index(drop=True)
    bounds = split_by_s1_groups(cand_sorted, cfg.FEATURE_TASK_PAIRS)
    n_workers = min(n_jobs, len(bounds))
    tasks = (_slice_task(cand_sorted.iloc[a:b], df_s1, df_s23) for a, b in bounds)   # lazy: one slice in flight at a time
    ctx = mp.get_context('fork')
    import gc
    import time
    t0 = time.time()
    gc.collect()
    gc.freeze()          # inherited objects go to the permanent generation: nothing in the parent's heap is traversed later
    try:
        with ctx.Pool(n_workers, initializer=_par_init, initargs=(vecs, extra_stats)) as pool:
            feats = list(pool.imap(_par_work, tasks))          # ordered; results stream back as parts finish
    finally:
        gc.unfreeze()
    feat = pd.concat(feats, ignore_index=True)
    del feats
    dt = max(1e-6, time.time() - t0)
    log.info(f"  features: {n:,} pairs in {dt:.0f}s ({n / dt:,.0f} pairs/s; {n_workers} workers, {len(bounds)} tasks; "
             f"parent rss {cfg.rss_gb():.1f} GB)")
    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    return feat.iloc[inv].reset_index(drop=True)
