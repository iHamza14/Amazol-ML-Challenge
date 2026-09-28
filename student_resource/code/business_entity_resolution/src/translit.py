"""
Indic-script -> Latin transliteration learned from the TRAINING data only.

~18% of Indian S2/S3 business names are written in an Indic script (Devanagari,
Bengali, Gujarati, Gurmukhi, Odia, Tamil, Telugu, Kannada, Malayalam) as a word-by-word
phonetic transliteration of the Latin name ("Fortune Finance Private Limited" ->
"फॉर्च्यून फाइनेंस प्राइवेट लिमिटेड").  `unidecode` gives a crude romanisation
("phoNrcyuun phaaineNs praaivett limittedd") that shares almost no tokens with the
English name, so name similarity collapses to ~0 for these records.

We instead LEARN a token-level dictionary from aligned ground-truth pairs in the
training set (S1 Latin name  <->  S2/S3 Indic name).  When both names have the same
number of tokens they are zipped position-wise; otherwise only the first and last
tokens are aligned.  The most frequent Latin token for each Indic token is kept if it
is seen >= MIN_COUNT times with purity >= MIN_PURITY.

This uses ONLY the provided training data (no external resources), and falls back to
`unidecode` for unseen tokens.
"""
import re
import json
import logging
import collections
from unidecode import unidecode

log = logging.getLogger(__name__)

MIN_COUNT = 2
MIN_PURITY = 0.5

_INDIC_RANGES = ((0x0900, 0x0DFF), (0x0E00, 0x0E7F))
_SPLIT_RE = re.compile(r'[\s\.\,\(\)\[\]\-&/\|:;!\?"\'`~*#@+]+')
_LATIN_CLEAN_RE = re.compile(r'[^a-z0-9\s]')


def has_indic(s):
    """True if any character is in an Indic Unicode block."""
    for c in s:
        o = ord(c)
        if 0x0900 <= o <= 0x0DFF:
            return True
    return False


def indic_tokens(s):
    """Split an Indic-script name into tokens (keeps ASCII tokens too, lower-cased)."""
    return [t.lower() for t in _SPLIT_RE.split(s) if t]


def latin_tokens(s):
    s = _LATIN_CLEAN_RE.sub(' ', unidecode(s).lower())
    return s.split()


class Transliterator:
    """Token dictionary with unidecode fallback."""

    def __init__(self, table=None):
        self.table = dict(table) if table else {}

    # ---------------- learning ----------------
    @classmethod
    def learn(cls, s1_names_by_id, s23_names_by_id, gt_pairs):
        """
        s1_names_by_id : dict entity_id -> raw business_name (S1)
        s23_names_by_id: dict entity_id -> raw business_name (S2/S3)
        gt_pairs       : iterable of (s1_id, s23_id) TRUE matches
        """
        counts = collections.defaultdict(collections.Counter)
        n_pairs = 0
        for s1_id, s23_id in gt_pairs:
            b = s23_names_by_id.get(s23_id)
            if b is None or b.isascii() or not has_indic(b):
                continue
            a = s1_names_by_id.get(s1_id)
            if a is None:
                continue
            at = latin_tokens(a)
            bt = [t for t in indic_tokens(b) if not t.isascii()]
            if not at or not bt:
                continue
            n_pairs += 1
            if len(bt) == len(at):
                for x, y in zip(bt, at):
                    counts[x][y] += 1
            else:
                counts[bt[0]][at[0]] += 1
                if len(bt) > 1 and len(at) > 1:
                    counts[bt[-1]][at[-1]] += 1
        table = {}
        for k, c in counts.items():
            (y, n), = c.most_common(1)
            tot = sum(c.values())
            if n >= MIN_COUNT and n / tot >= MIN_PURITY:
                table[k] = y
        log.info(f"Transliterator learned from {n_pairs:,} Indic pairs -> {len(table):,} tokens")
        return cls(table)

    # ---------------- applying ----------------
    def translit_token(self, t):
        if t.isascii():
            return t.lower()
        y = self.table.get(t)
        if y is not None:
            return y
        return unidecode(t).lower()

    def translit(self, s):
        """Return a Latin string for an Indic-script name (tokens joined by spaces)."""
        out = [self.translit_token(t) for t in indic_tokens(s)]
        return ' '.join(o for o in out if o)

    def coverage(self, names):
        tot = hit = 0
        for n in names:
            for t in indic_tokens(n):
                if t.isascii():
                    continue
                tot += 1
                hit += t in self.table
        return hit / tot if tot else 0.0

    # ---------------- persistence ----------------
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.table, f, ensure_ascii=False)
        log.info(f"Transliteration table saved -> {path} ({len(self.table):,} tokens)")

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            table = json.load(f)
        log.info(f"Transliteration table loaded <- {path} ({len(table):,} tokens)")
        return cls(table)

    @classmethod
    def load_or_empty(cls, path):
        import os
        if os.path.exists(path):
            return cls.load(path)
        log.warning(f"No transliteration table at {path}; falling back to unidecode only")
        return cls()
