"""
Decision layer — turn per-pair probabilities into per-entity match sets that maximise
entity-level macro F0.5.

Three ingredients, each validated on the held-out split in train.py:

1. Per-country probability thresholds (France, unseen in training, gets the mean of the
   seen-country thresholds plus a precision-first shift).

2. Expected-F0.5 set selection: for one S1 entity with candidate probabilities
   p1 >= p2 >= ... >= pk (assumed independent and calibrated), predicting the top-j set has
   plug-in expected F0.5
        E[F_j] ~= 1.25*TP_j / (1.25*TP_j + 0.25*FN_j + FP_j)
        TP_j = sum_{i<=j} p_i,  FP_j = j - TP_j,  FN_j = sum_{i>j} p_i
   and predicting the empty set has E[F_0] = prod_i (1 - p_i)  (probability the entity is a
   singleton).  We pick j maximising E[F_j].  This adapts the effective threshold to each
   entity: a lone 0.55 candidate is kept (E[F_1]=0.55 > E[F_0]=0.45) while a 0.55 candidate
   next to a 0.98 one is dropped (it would dilute precision).

3. Conflict resolution: every S2/S3 record belongs to at most ONE S1 entity (verified on the
   training ground truth: 0 violations).  If two S1 entities claim the same S2/S3 id, only
   the higher-probability claim survives.
"""
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def expected_f05_select(s1_code, prob, min_prob=0.05):
    """
    Vectorised expected-F0.5 set selection.
    Returns boolean mask over pairs (True = predict as match).
    """
    n = len(prob)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((-prob, s1_code))
    s = s1_code[order]
    p = prob[order].astype(np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    counts = np.diff(np.r_[starts, n])
    grp_id = np.repeat(np.arange(len(starts)), counts)
    within = np.arange(n) - np.repeat(starts, counts)
    # cumulative sums per group
    csum = np.cumsum(p)
    grp_offset = np.repeat(csum[starts] - p[starts], counts)
    tp = csum - grp_offset                       # TP_j for j = within+1
    total = np.repeat(np.add.reduceat(p, starts), counts)
    fn = total - tp
    j = within + 1.0
    fp = j - tp
    ef = 1.25 * tp / (1.25 * tp + 0.25 * fn + fp)
    # empty-set expectation per group
    log1m = np.log(np.clip(1.0 - p, 1e-12, 1.0))
    ef0 = np.exp(np.add.reduceat(log1m, starts))            # per group
    # best j per group
    best_ef = np.maximum.reduceat(ef, starts)
    keep_group = best_ef > ef0                              # per group: predict non-empty?
    # position of best j per group: first index achieving the max
    is_best = ef == np.repeat(best_ef, counts)
    # cumulative first-best index within group
    first_best = np.full(len(starts), -1, dtype=np.int64)
    idx_best = np.flatnonzero(is_best)
    g_best = grp_id[idx_best]
    # keep the first occurrence per group
    uniq, first_pos = np.unique(g_best, return_index=True)
    first_best[uniq] = within[idx_best[first_pos]]
    keep = (within <= np.repeat(first_best, counts)) & np.repeat(keep_group, counts) & (p >= min_prob)
    mask = np.zeros(n, dtype=bool)
    mask[order] = keep
    return mask


def threshold_select(prob, country, thresholds, default):
    """Boolean mask: prob >= threshold of its country."""
    t = np.full(len(prob), default, dtype=np.float64)
    for c, th in thresholds.items():
        t[country == c] = th
    return prob >= t


def resolve_conflicts(s1_code, s23_code, prob, mask):
    """Among selected pairs, keep for each S2/S3 record only the highest-probability S1."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return mask
    order = idx[np.lexsort((-prob[idx], s23_code[idx]))]
    s23_o = s23_code[order]
    first = np.r_[True, s23_o[1:] != s23_o[:-1]]
    new_mask = np.zeros_like(mask)
    new_mask[order[first]] = True
    n_removed = int(mask.sum() - new_mask.sum())
    if n_removed:
        log.info(f"  conflict resolution removed {n_removed:,} duplicate S2/S3 claims")
    return new_mask


def decide(s1_code, s23_code, prob, country, cfg_dec):
    """
    cfg_dec: dict with keys
        mode: 'threshold' | 'expected_f'
        thresholds: {country: t}, default_threshold: float
        resolve_conflicts: bool
        ef_min_prob: float (expected_f mode floor)
    Returns boolean mask over pairs.
    """
    if cfg_dec.get('mode', 'threshold') == 'expected_f':
        mask = expected_f05_select(s1_code, prob, min_prob=cfg_dec.get('ef_min_prob', 0.05))
        # optional per-country hard floor on top of expected-F
        th = cfg_dec.get('thresholds') or {}
        if th:
            floor = np.full(len(prob), cfg_dec.get('ef_floor_default', 0.0))
            for c, t in th.items():
                floor[country == c] = t
            mask &= prob >= floor
    else:
        mask = threshold_select(prob, country, cfg_dec.get('thresholds') or {}, cfg_dec.get('default_threshold', 0.5))
    delta = float(cfg_dec.get('extra_link_delta', 0.0) or 0.0)
    if delta != 0:
        t_pair = pair_thresholds(country, cfg_dec.get('thresholds') or {}, cfg_dec.get('default_threshold', 0.5))
        mask = size_adaptive(s1_code, prob, mask, t_pair + delta, relax=delta < 0)
    if cfg_dec.get('resolve_conflicts', True):
        mask = resolve_conflicts(s1_code, s23_code, prob, mask)
    return mask


def pair_thresholds(group, thresholds, default):
    """Per-pair threshold array from a {group: threshold} dict."""
    t = np.full(len(group), float(default), dtype=np.float64)
    group = np.asarray(group)
    for g, th in thresholds.items():
        t[group == g] = float(th)
    return t


def size_adaptive(s1_code, prob, mask, floor, relax=False):
    """
    Size-adaptive acceptance (Foursquare 4th place post-processing, adapted to F0.5): within an entity the
    best selected candidate is always kept; every further selected candidate must satisfy
    prob >= floor (= its bin threshold + delta). Extra false links cost 4x a missed link in F0.5.
    `floor` is a per-pair array.
    """
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return mask
    if relax:
        # negative delta: an entity whose best candidate passed its threshold is a confirmed non-singleton, so
        # its further candidates are accepted at the LOWER floor (threshold + delta, delta < 0). Missed links
        # were 66% of the validation loss; only entities with an accepted anchor are relaxed.
        anchored = np.zeros(int(s1_code.max()) + 1, dtype=bool)
        anchored[s1_code[idx]] = True
        return mask | (anchored[s1_code] & (prob >= floor))
    order = idx[np.lexsort((-prob[idx], s1_code[idx]))]
    s = s1_code[order]
    p = prob[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    counts = np.diff(np.r_[starts, len(s)])
    within = np.arange(len(s)) - np.repeat(starts, counts)
    keep = (within == 0) | (p >= floor[order])
    new_mask = np.zeros_like(mask)
    new_mask[order[keep]] = True
    return new_mask


def calibrate_unseen_threshold(prob, s1_code, n_s1, start_t, target_empty_rate, max_t=0.97, step=0.005):
    """
    Label-free threshold for an unseen country (France): the generator produces the same singleton share
    (5.59%) in every country, so raise the threshold (never lower it) until the predicted-empty rate of the
    country's S1 entities reaches the target. Returns (threshold, empty_rate_at_threshold).
    """
    t = float(start_t)
    rate = float((np.bincount(s1_code[prob >= t], minlength=n_s1) == 0).mean()) if n_s1 else 1.0
    while rate < target_empty_rate and t + step <= max_t:
        t = round(t + step, 4)
        rate = float((np.bincount(s1_code[prob >= t], minlength=n_s1) == 0).mean())
    return t, rate


def pair_groups(country, addr_empty_s23, per_bin=True):
    """Group key per pair: '<country>' or '<country>|noaddr' (candidate has no address)."""
    country = np.asarray(country).astype(str)
    if not per_bin:
        return country
    return np.where(np.asarray(addr_empty_s23).astype(bool), np.char.add(country, '|noaddr'), country)


def mask_to_dict(mask, s1_ids, s23_ids):
    out = {}
    for a, b in zip(s1_ids[mask], s23_ids[mask]):
        out.setdefault(a, set()).add(b)
    return out
