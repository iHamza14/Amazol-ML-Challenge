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
        thresholds = cfg_dec.get('thresholds') or {}
        default = cfg_dec.get('default_threshold', 0.5)
        mask = threshold_select(prob, country, thresholds, default)
        # rank ladder (per-rank deltas + link cap); a uniform extra_link_delta is the special case d2 == d3 == d4
        deltas = cfg_dec.get('rank_deltas')
        delta = float(cfg_dec.get('extra_link_delta', 0.0) or 0.0)
        max_links = int(cfg_dec.get('max_links', 0) or 0)
        if deltas is None and delta != 0:
            deltas = [delta, delta, delta]
        if (deltas is not None and any(float(x) != 0 for x in deltas)) or max_links > 0:
            t_pair = pair_thresholds(country, thresholds, default)
            mask = ladder_select(s1_code, prob, t_pair, deltas or [0.0, 0.0, 0.0], max_links)
    if cfg_dec.get('mode', 'threshold') == 'expected_f':
        delta = float(cfg_dec.get('extra_link_delta', 0.0) or 0.0)
        if delta != 0:
            t_pair = pair_thresholds(country, cfg_dec.get('thresholds') or {}, cfg_dec.get('default_threshold', 0.5))
            mask = size_adaptive(s1_code, prob, mask, t_pair + delta, relax=delta < 0)
    if cfg_dec.get('noaddr_rescue'):
        mask = noaddr_rescue(s1_code, s23_code, prob, country, mask, cfg_dec['noaddr_rescue'])
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


def rank_within_entity(s1_code, prob):
    """0-based rank of every pair among its entity's pairs by probability (descending)."""
    order = np.lexsort((-prob, s1_code))
    s = s1_code[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    counts = np.diff(np.r_[starts, len(s)])
    within = np.arange(len(s)) - np.repeat(starts, counts)
    rank = np.empty(len(prob), dtype=np.int64)
    rank[order] = within
    return rank


def ladder_select(s1_code, prob, t_pair, deltas, max_links=0):
    """
    Rank ladder: generalised size-adaptive acceptance. An entity is anchored when some pair passes its own
    threshold (t_pair, per pair: country / address bin). Its top-probability pair is accepted iff it passes its
    threshold; the pair at rank r >= 1 is accepted iff prob >= threshold + deltas[min(r, 3) - 1]. Negative deltas
    relax (an anchored entity is a confirmed non-singleton; under F0.5 the break-even for the k+1-th link rises
    with k, so the deltas are meant to be non-decreasing: d2 <= d3 <= d4), positive ones tighten (Foursquare
    4th place). With non-decreasing deltas the accepted set is a prefix of the entity's ranking. max_links > 0
    additionally keeps only the top max_links pairs of an entity unless a pair's prob >= 0.99 (impossible-count
    guard: same-name chains with city-only addresses collect dozens of claims). deltas (0, 0, 0) and
    max_links 0 reproduce the plain threshold rule exactly.
    """
    m0 = prob >= t_pair
    if len(s1_code) == 0:
        return m0
    anchored = np.zeros(int(s1_code.max()) + 1, dtype=bool)
    anchored[s1_code[m0]] = True
    rank = rank_within_entity(s1_code, prob)
    d = [0.0] + [float(x) for x in list(deltas)[:3]]
    while len(d) < 4:
        d.append(d[-1])
    d = np.asarray(d, dtype=np.float64)
    floor = t_pair + d[np.minimum(rank, 3)]
    keep = np.where(rank == 0, m0, anchored[s1_code] & (prob >= floor))
    if max_links and int(max_links) > 0:
        keep &= (rank < int(max_links)) | (prob >= 0.99)
    return keep


def noaddr_rescue(s1_code, s23_code, prob, group, mask, params):
    """
    Address-empty unique-claimant rescue. Address-less S2/S3 records are noisy copies of SOME S1 almost always
    (the distractor generator keeps addresses; several teams measured ~0.3% empty addresses among unmatched
    records vs ~4-5% among true copies), so for them the question is ownership, not existence. A record in the
    '|noaddr' bin that is not yet selected is accepted for its TOP claimant when that claimant's prob >=
    t[country] and either no other S1 claims the record (claim = prob >= claim_prob) or the top claim leads
    the runner-up by >= margin, and the claimant has fewer than max_links accepted pairs. Applied before
    conflict resolution; selected on the density-adjusted validation metric with a precision report.
    params: {'t': {country: t}, 'claim_prob': 0.10, 'margin': 0.30, 'max_links': 11}
    """
    if not params or not params.get('t') or len(prob) == 0:
        return mask
    group = np.asarray(group).astype(str)
    is_na = np.char.endswith(group, '|noaddr')
    if not is_na.any():
        return mask
    country = np.array([g.split('|')[0] for g in group])
    t = np.full(len(prob), np.inf)
    for c, v in params['t'].items():
        t[country == c] = float(v)
    claim_p = float(params.get('claim_prob', 0.10))
    margin = float(params.get('margin', 0.30))
    max_links = int(params.get('max_links', 11))
    claims = is_na & (prob >= claim_p)
    idx = np.flatnonzero(claims)
    if len(idx) == 0:
        return mask
    order = idx[np.lexsort((-prob[idx], s23_code[idx]))]
    s = s23_code[order]
    p_o = prob[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    counts = np.diff(np.r_[starts, len(s)])
    within = np.arange(len(s)) - np.repeat(starts, counts)
    p_top = np.repeat(p_o[starts], counts)
    second = np.where(counts > 1, p_o[np.minimum(starts + 1, len(s) - 1)], 0.0)
    p_2nd = np.repeat(second, counts)
    n_claim = np.repeat(counts, counts)
    ok = (within == 0) & ((n_claim == 1) | (p_top - p_2nd >= margin))
    cand = np.zeros(len(prob), dtype=bool)
    cand[order[ok]] = True
    n_acc = np.bincount(s1_code[mask], minlength=int(s1_code.max()) + 1)
    rescue = cand & ~mask & (prob >= t) & (n_acc[s1_code] < max_links)
    return mask | rescue


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
