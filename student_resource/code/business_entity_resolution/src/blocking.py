"""
Blocking v2 — multi-channel sparse TF-IDF top-k retrieval (per country, chunked) + reverse channel.

Why this design
  * Python dict-of-sets inverted indexes (v1) need >100 GB for 12M records and hours of
    Python loops.  Sparse matrix products with `sparse_dot_topn` retrieve the top-k for
    20k S1 rows against 5M S2/S3 rows in seconds, multi-threaded, in a few hundred MB.
  * Recall analysis on the training data: 14.6% of true pairs share NO name token and
    4.5% share NO address token, but only 0.02% share neither.  Hence several channels
    are unioned; every candidate keeps the score and rank it obtained in every channel
    (these become model features).
  * Reverse channel: each S2/S3 record retrieves ITS top-k S1 entities (joint vector). The
    rank of the S1 in the record's list ("is this S1 the best owner of this record?") is a
    competition feature that the best public pipelines report as their strongest signal,
    and the record's top-1 S1 is added as an extra candidate (recall pass).

Forward channels
  name_tok : IDF-weighted word unigrams of the core business name
  name_chr : char 3-grams of the space-less core name (typos, leetspeak, collapsed domains)
  addr_tok : address words + numbers
  addr_num : (number, street-word) combinations — very precise address signal
  joint    : name + address tokens in one vector — best overall ranking

Output per chunk: DataFrame with s1_pos, s23_pos (row positions in the full preprocessed
frames), per-channel score/rank, n_channels, min_rank, sum_score, fused_rank, cand_count,
rev_rank, rev_score.
"""
import math
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

import config as cfg

log = logging.getLogger(__name__)

CHANNELS = ['name_tok', 'name_chr', 'addr_tok', 'addr_num', 'joint']
NO_RANK = 999


# ------------------------------------------------------------------
# Lightweight TF-IDF vectoriser over pre-tokenised documents
# ------------------------------------------------------------------
class SparseVectorizer:
    """Fit on S2/S3 token lists; transform S1 token lists with the same vocabulary."""

    def __init__(self, max_df_frac=0.03, min_df=1, sublinear=True):
        self.max_df_frac = max_df_frac
        self.min_df = min_df
        self.sublinear = sublinear
        self.vocab = None
        self.idf = None

    @staticmethod
    def _flatten(docs):
        lengths = np.fromiter((len(d) for d in docs), dtype=np.int64, count=len(docs))
        rows = np.repeat(np.arange(len(docs), dtype=np.int64), lengths)
        flat = np.empty(int(lengths.sum()), dtype=object)
        i = 0
        for d in docs:
            n = len(d)
            if n:
                flat[i:i + n] = d
                i += n
        return rows, flat

    def fit_transform(self, docs):
        n_docs = len(docs)
        rows, flat = self._flatten(docs)
        if len(flat) == 0:
            self.vocab = {}
            self.idf = np.zeros(0, dtype=np.float32)
            return sp.csr_matrix((n_docs, 0), dtype=np.float32)
        codes, uniques = pd.factorize(flat, sort=False)
        pair = rows * np.int64(len(uniques)) + codes
        pair_u, tf = np.unique(pair, return_counts=True)
        r_u = pair_u // len(uniques)
        c_u = pair_u % len(uniques)
        df = np.bincount(c_u, minlength=len(uniques))
        # floor the DF cap in documents: for tiny country groups int(0.03*n) would be 1 and every token
        # shared by two records (i.e. the matching ones) would be pruned
        max_df = max(self.min_df, int(self.max_df_frac * n_docs), 50)
        keep = (df >= self.min_df) & (df <= max_df)
        new_index = -np.ones(len(uniques), dtype=np.int64)
        new_index[keep] = np.arange(int(keep.sum()))
        self.vocab = {tok: int(new_index[i]) for i, tok in enumerate(uniques) if keep[i]}
        idf_full = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        self.idf = idf_full[keep].astype(np.float32)
        m = keep[c_u]
        r_u, c_u, tf = r_u[m], new_index[c_u[m]], tf[m]
        vals = (1.0 + np.log(tf)) if self.sublinear else tf.astype(np.float32)
        vals = (vals * self.idf[c_u]).astype(np.float32)
        X = sp.csr_matrix((vals, (r_u, c_u)), shape=(n_docs, len(self.idf)), dtype=np.float32)
        return self._l2(X)

    def transform(self, docs):
        n_docs = len(docs)
        if not self.vocab:
            return sp.csr_matrix((n_docs, 0), dtype=np.float32)
        rows, flat = self._flatten(docs)
        if len(flat) == 0:
            return sp.csr_matrix((n_docs, len(self.idf)), dtype=np.float32)
        cols = np.fromiter((self.vocab.get(t, -1) for t in flat), dtype=np.int64, count=len(flat))
        m = cols >= 0
        rows, cols = rows[m], cols[m]
        pair = rows * np.int64(len(self.idf)) + cols
        pair_u, tf = np.unique(pair, return_counts=True)
        r_u = pair_u // len(self.idf)
        c_u = pair_u % len(self.idf)
        vals = (1.0 + np.log(tf)) if self.sublinear else tf.astype(np.float32)
        vals = (vals * self.idf[c_u]).astype(np.float32)
        X = sp.csr_matrix((vals, (r_u, c_u)), shape=(n_docs, len(self.idf)), dtype=np.float32)
        return self._l2(X)

    @staticmethod
    def _l2(X):
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        inv = sp.diags((1.0 / norms).astype(np.float32))
        return (inv @ X).tocsr()


# ------------------------------------------------------------------
# Document builders for each channel
# ------------------------------------------------------------------
_GRAM_CACHE = {}


def _char_ngrams(s, n=3):
    """Char n-grams with interned strings: 6M names x 17 grams would otherwise be 100M Python str objects;
    only ~30k distinct trigrams exist, so the cache keeps a single object per gram (pointers only)."""
    if len(s) < n:
        return [s] if s else []
    c = _GRAM_CACHE
    out = []
    for i in range(len(s) - n + 1):
        g = s[i:i + n]
        out.append(c.setdefault(g, g))
    return out


def channel_docs(df, channel):
    """Return list of token lists for the given channel."""
    if channel == 'name_tok':
        return list(df['name_tokens'].values)
    if channel == 'name_chr':
        return [_char_ngrams(s) for s in df['name_collapsed'].values]
    if channel == 'addr_tok':
        return list(df['addr_tokens'].values)
    if channel == 'addr_num':
        out = []
        for nums, words in zip(df['nums'].values, df['addr_words'].values):
            if nums and words:
                ws = words[:8]
                out.append([f'{n}|{w}' for n in nums[:4] for w in ws])
            else:
                out.append([])
        return out
    if channel == 'joint':
        return [list(a) + list(b) for a, b in zip(df['name_tokens'].values, df['addr_tokens'].values)]
    raise ValueError(channel)


def _row_ranks(r, d):
    """rank within row for arrays sorted by (row asc, score desc)."""
    starts = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
    return (np.arange(len(r)) - np.repeat(starts, np.diff(np.r_[starts, len(r)]))).astype(np.int16)


# ------------------------------------------------------------------
# Blocker: one per country
# ------------------------------------------------------------------
class Blocker:
    def __init__(self, s23_df, s23_pos, channels=None, n_threads=None,
                 s1_df_country=None, s1_pos_country=None, reverse_k=0):
        """
        s23_df        : preprocessed S2/S3 rows of ONE country (any index)
        s23_pos       : np.array of row positions of those rows in the full S2/S3 frame
        s1_df_country : ALL preprocessed S1 rows of the country (for the reverse channel), optional
        s1_pos_country: their positions in the full S1 frame (ascending)
        reverse_k     : top-k S1 per S2/S3 record for the reverse channel (0 = disabled)
        """
        self.channels = channels or CHANNELS
        self.n_threads = n_threads or cfg.N_JOBS
        self.s23_pos = np.asarray(s23_pos)
        self.n_s23 = len(s23_df)
        self.vec = {}
        self.BT = {}
        X_joint = None
        for ch in self.channels:
            docs = channel_docs(s23_df, ch)
            v = SparseVectorizer(max_df_frac=cfg.BLOCK_MAX_DF_FRAC)
            X = v.fit_transform(docs)
            self.vec[ch] = v
            self.BT[ch] = X.T.tocsr()
            log.info(f"    channel {ch:9s}: vocab={len(v.idf):,} nnz={X.nnz:,}")
            if ch == 'joint' and reverse_k > 0:
                X_joint = X
            else:
                del X
            del docs
        # ---------------- reverse channel ----------------
        self.reverse_k = 0
        if reverse_k > 0 and s1_df_country is not None and len(s1_df_country) > 0 and X_joint is not None and X_joint.nnz > 0:
            self._build_reverse(X_joint, s1_df_country, np.asarray(s1_pos_country), reverse_k)
        del X_joint

    def _build_reverse(self, X_joint, s1_df_country, s1_pos_country, k):
        import time
        t = time.time()
        A = self.vec['joint'].transform(channel_docs(s1_df_country, 'joint'))
        self.s1_pos_country = s1_pos_country
        self.n_s1c = len(s1_df_country)
        if A.nnz == 0:
            return
        # prune common tokens from both sides: they never decide a record's best S1 but their long posting
        # lists dominate the S2/S3 x S1 product (19 min -> minutes per country)
        df_cols = np.diff(X_joint.tocsc().indptr)
        keep_col = df_cols <= max(50, int(cfg.REVERSE_MAX_DF_FRAC * X_joint.shape[0]))
        if not keep_col.all():
            D = sp.diags(keep_col.astype(np.float32))
            Xr = (X_joint @ D).tocsr()
            Ar = (A @ D).tocsr()
            Xr.eliminate_zeros()
            Ar.eliminate_zeros()
            log.info(f"    reverse channel: {int((~keep_col).sum()):,} common tokens pruned "
                     f"({X_joint.nnz:,} -> {Xr.nnz:,} nnz on the S2/S3 side)")
        else:
            Xr, Ar = X_joint, A
        C = sp_matmul_topn(Xr, Ar.T.tocsr(), top_n=k, threshold=cfg.BLOCK_MIN_SCORE, sort=True,
                           n_threads=self.n_threads)
        del Xr, Ar
        C = C.tocoo()
        if C.nnz == 0:
            return
        r = C.row.astype(np.int64)      # s23 local
        c = C.col.astype(np.int64)      # s1 local (country)
        d = C.data.astype(np.float32)
        order = np.lexsort((-d, r))
        r, c, d = r[order], c[order], d[order]
        rank = _row_ranks(r, d)
        key = r * np.int64(self.n_s1c) + c
        o2 = np.argsort(key, kind='stable')
        self.rev_key = key[o2]
        self.rev_rank = rank[o2]
        self.rev_score = d[o2]
        # inverted top-1: s1 local -> its "best-owner" s23 rows
        top1 = rank == 0
        c1, r1, d1 = c[top1], r[top1], d[top1]
        o3 = np.argsort(c1, kind='stable')
        self.rev1_s1 = c1[o3]
        self.rev1_s23 = r1[o3]
        self.rev1_score = d1[o3]
        self.reverse_k = k
        log.info(f"    reverse channel: {len(self.rev_key):,} (record, S1) pairs, top-1 pairs {int(top1.sum()):,} "
                 f"({time.time() - t:.0f}s)")

    def _reverse_lookup(self, s1_local, s23_local):
        """rev_rank / rev_score for pair arrays (NO_RANK / 0 when absent)."""
        n = len(s1_local)
        rank = np.full(n, NO_RANK, dtype=np.int16)
        score = np.zeros(n, dtype=np.float32)
        if self.reverse_k == 0 or n == 0:
            return rank, score
        key = s23_local.astype(np.int64) * np.int64(self.n_s1c) + s1_local.astype(np.int64)
        idx = np.searchsorted(self.rev_key, key)
        idx = np.minimum(idx, len(self.rev_key) - 1)
        hit = self.rev_key[idx] == key
        rank[hit] = self.rev_rank[idx[hit]]
        score[hit] = self.rev_score[idx[hit]]
        return rank, score

    def _reverse_top1_pairs(self, s1_local_chunk):
        """(chunk_row, s23_local, score) for records whose best S1 is in this chunk."""
        if self.reverse_k == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
        lo = np.searchsorted(self.rev1_s1, s1_local_chunk, side='left')
        hi = np.searchsorted(self.rev1_s1, s1_local_chunk, side='right')
        cnt = hi - lo
        if cnt.sum() == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
        rows = np.repeat(np.arange(len(s1_local_chunk), dtype=np.int64), cnt)
        idx = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi) if b > a])
        return rows, self.rev1_s23[idx].astype(np.int64), self.rev1_score[idx]

    def query(self, s1_df, s1_pos, max_candidates=None):
        """Return candidate DataFrame for a chunk of S1 rows (one country)."""
        max_candidates = max_candidates or cfg.BLOCK_MAX_CANDIDATES
        s1_pos = np.asarray(s1_pos)
        keys_all = []
        per_ch = {}
        for ch in self.channels:
            A = self.vec[ch].transform(channel_docs(s1_df, ch))
            if A.nnz == 0 or self.BT[ch].shape[0] == 0:
                per_ch[ch] = None
                continue
            C = sp_matmul_topn(A, self.BT[ch], top_n=cfg.BLOCK_TOPK[ch], threshold=cfg.BLOCK_MIN_SCORE,
                               sort=False, n_threads=self.n_threads)
            C = C.tocoo()
            if C.nnz == 0:
                per_ch[ch] = None
                continue
            r = C.row.astype(np.int64)
            c = C.col.astype(np.int64)
            d = C.data.astype(np.float32)
            order = np.lexsort((-d, r))
            r, c, d = r[order], c[order], d[order]
            rank = _row_ranks(r, d)
            key = r * np.int64(self.n_s23) + c
            per_ch[ch] = (key, d, rank)
            keys_all.append(key)
        # reverse top-1 pairs as extra candidates
        s1_local_chunk = None
        rev_rows = rev_cols = rev_sc = None
        if self.reverse_k > 0:
            s1_local_chunk = np.searchsorted(self.s1_pos_country, s1_pos)
            rev_rows, rev_cols, rev_sc = self._reverse_top1_pairs(s1_local_chunk)
            if len(rev_rows):
                keys_all.append(rev_rows * np.int64(self.n_s23) + rev_cols)
        if not keys_all:
            return self._empty()
        keys_u, inv_all = np.unique(np.concatenate(keys_all), return_inverse=True)
        n_pairs = len(keys_u)
        out = {}
        offset = 0
        n_ch_hit = np.zeros(n_pairs, dtype=np.int8)
        min_rank = np.full(n_pairs, NO_RANK, dtype=np.int16)
        sum_score = np.zeros(n_pairs, dtype=np.float32)
        for ch in self.channels:
            sc = np.zeros(n_pairs, dtype=np.float32)
            rk = np.full(n_pairs, NO_RANK, dtype=np.int16)
            if per_ch.get(ch) is not None:
                key, d, rank = per_ch[ch]
                inv = inv_all[offset:offset + len(key)]
                offset += len(key)
                sc[inv] = d
                rk[inv] = rank
                n_ch_hit[inv] += 1
                np.minimum(min_rank, rk, out=min_rank)
                sum_score += sc
            out[f'bs_{ch}'] = sc
            out[f'br_{ch}'] = rk
        row = (keys_u // self.n_s23).astype(np.int64)
        col = (keys_u % self.n_s23).astype(np.int64)
        if self.reverse_k > 0:
            rev_rank, rev_score = self._reverse_lookup(s1_local_chunk[row], col)
        else:
            rev_rank = np.full(n_pairs, NO_RANK, dtype=np.int16)
            rev_score = np.zeros(n_pairs, dtype=np.float32)
        # cap candidates per S1 by (fused rank asc, total score desc); reverse rank r counts like forward rank 2r
        fused_key = np.minimum(min_rank, np.where(rev_rank < NO_RANK, rev_rank * 2, NO_RANK)).astype(np.int16)
        total = sum_score + rev_score
        order = np.lexsort((-total, fused_key, row))
        row_o = row[order]
        starts = np.r_[0, np.flatnonzero(np.diff(row_o)) + 1]
        counts = np.diff(np.r_[starts, len(row_o)])
        within = np.arange(len(row_o)) - np.repeat(starts, counts)
        keep_o = within < max_candidates
        keep = np.zeros(n_pairs, dtype=bool)
        keep[order[keep_o]] = True
        cand_count_row = np.minimum(counts, max_candidates)
        cc = np.zeros(n_pairs, dtype=np.int16)
        cc[order] = np.repeat(cand_count_row, counts)
        fused = np.zeros(n_pairs, dtype=np.int16)
        fused[order] = np.minimum(within, 32000).astype(np.int16)
        df = pd.DataFrame({
            's1_pos': s1_pos[row[keep]],
            's23_pos': self.s23_pos[col[keep]],
            'n_channels': n_ch_hit[keep],
            'min_rank': min_rank[keep],
            'sum_score': sum_score[keep],
            'cand_count': cc[keep],
            'fused_rank': fused[keep],
            'rev_rank': rev_rank[keep],
            'rev_score': rev_score[keep],
        })
        for k, v in out.items():
            df[k] = v[keep]
        return df

    @staticmethod
    def _empty():
        cols = {'s1_pos': np.array([], dtype=np.int64), 's23_pos': np.array([], dtype=np.int64),
                'n_channels': np.array([], dtype=np.int8), 'min_rank': np.array([], dtype=np.int16),
                'sum_score': np.array([], dtype=np.float32), 'cand_count': np.array([], dtype=np.int16),
                'fused_rank': np.array([], dtype=np.int16), 'rev_rank': np.array([], dtype=np.int16),
                'rev_score': np.array([], dtype=np.float32)}
        for ch in CHANNELS:
            cols[f'bs_{ch}'] = np.array([], dtype=np.float32)
            cols[f'br_{ch}'] = np.array([], dtype=np.int16)
        return pd.DataFrame(cols)


BLOCK_META_COLS = ['n_channels', 'min_rank', 'sum_score', 'cand_count', 'fused_rank', 'rev_rank', 'rev_score'] + \
                  [f'bs_{c}' for c in CHANNELS] + [f'br_{c}' for c in CHANNELS]


# ------------------------------------------------------------------
# Generator over countries and S1 chunks
# ------------------------------------------------------------------
def iter_candidate_chunks(df_s1, df_s23, chunk_size=None, max_candidates=None, s1_mask=None, blocker_cache=None):
    """
    Yield (country, candidate_chunk_df, blocker) for every country present in df_s1.
    df_s1 / df_s23 : full preprocessed frames (default RangeIndex). s1_mask optionally restricts the S1 rows
    that are QUERIED; the reverse channel always sees ALL S1 rows of the country (as at test time).
    The blocker exposes .vec (channel -> SparseVectorizer) for exact cosine features.
    blocker_cache  : optional dict; a Blocker built for a country is stored there and reused by later calls with
                     the same df_s23 (train.py: Pass A and Pass B+C) — the build (five vocabulary fits + the
                     reverse S2/S3 x S1 product) does not depend on s1_mask.
    """
    chunk_size = chunk_size or cfg.BLOCK_S1_CHUNK
    s1_country = df_s1['country_norm'].values
    s23_country = df_s23['country_norm'].values
    if s1_mask is None:
        s1_mask = np.ones(len(df_s1), dtype=bool)
    for country in pd.unique(s1_country[s1_mask]):
        s1_pos_all = np.flatnonzero((s1_country == country) & s1_mask)
        s1_pos_country = np.flatnonzero(s1_country == country)
        s23_pos_all = np.flatnonzero(s23_country == country)
        log.info(f"=== BLOCKING country={country}: S1={len(s1_pos_all):,} (of {len(s1_pos_country):,})  S23={len(s23_pos_all):,} ===")
        if len(s23_pos_all) == 0:
            log.warning(f"  no S2/S3 rows for country {country}; all its S1 entities become singletons")
            continue
        blocker = blocker_cache.get(country) if blocker_cache is not None else None
        if blocker is not None and blocker.n_s23 == len(s23_pos_all):
            log.info("  reusing the cached Blocker for this country")
        else:
            rev_k = cfg.REVERSE_TOPK if cfg.USE_REVERSE_BLOCKING else 0
            blocker = Blocker(df_s23.iloc[s23_pos_all], s23_pos_all,
                              s1_df_country=df_s1.iloc[s1_pos_country] if rev_k else None,
                              s1_pos_country=s1_pos_country, reverse_k=rev_k)
            if blocker_cache is not None:
                blocker_cache[country] = blocker
        n_chunks = math.ceil(len(s1_pos_all) / chunk_size)
        for ci in range(n_chunks):
            pos = s1_pos_all[ci * chunk_size:(ci + 1) * chunk_size]
            cand = blocker.query(df_s1.iloc[pos], pos, max_candidates=max_candidates)
            log.info(f"  chunk {ci + 1}/{n_chunks}: {len(pos):,} S1 -> {len(cand):,} candidates "
                     f"({len(cand) / max(1, len(pos)):.1f}/S1)")
            yield country, cand, blocker
        if blocker_cache is None:
            del blocker
