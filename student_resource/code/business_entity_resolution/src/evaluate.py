"""
Evaluation v2 — vectorised entity-level macro F0.5 (the exact competition metric).

  F0.5 per S1 entity = 1.25*TP / (1.25*TP + 0.25*FN + FP)
  singleton (no true matches): 1.0 if predicted empty else 0.0
  macro average over ALL S1 entities of the evaluation set.

Everything works on integer S1 codes (0..n_s1-1) so sweeps over 60+ thresholds on
millions of candidate pairs take seconds.
"""
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def f05_from_counts(n_pred, n_true, tp):
    """Vectorised per-entity F0.5 from count arrays."""
    n_pred = np.asarray(n_pred, dtype=np.float64)
    n_true = np.asarray(n_true, dtype=np.float64)
    tp = np.asarray(tp, dtype=np.float64)
    fp = n_pred - tp
    fn = n_true - tp
    denom = 1.25 * tp + 0.25 * fn + fp
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.where(denom > 0, 1.25 * tp / np.where(denom > 0, denom, 1.0), 0.0)
    singleton = n_true == 0
    f = np.where(singleton, np.where(n_pred == 0, 1.0, 0.0), f)
    return f


def macro_f05(pred_s1_code, pred_is_true, n_true_per_s1):
    """
    pred_s1_code : int array (one per predicted pair) of S1 codes
    pred_is_true : bool/int array (pair is a true match)
    n_true_per_s1: int array of length n_s1 (true match counts, 0 for singletons)
    """
    n_s1 = len(n_true_per_s1)
    n_pred = np.bincount(pred_s1_code, minlength=n_s1)
    tp = np.bincount(pred_s1_code, weights=pred_is_true.astype(np.float64), minlength=n_s1)
    return float(f05_from_counts(n_pred, n_true_per_s1, tp).mean())


def threshold_sweep(s1_code, prob, label, n_true_per_s1, grid, mask=None, fp_weight=None, pair_mask=None):
    """
    Return list of (threshold, macro_f05) for the given entity subset.
    mask      : boolean over S1 codes selecting the entities to average over (default all).
    fp_weight : optional per-pair weight applied to FALSE POSITIVES only (e.g. 1.9 for pairs whose S2/S3
                record is an unmatched distractor, to mimic the test pool's higher distractor density).
    pair_mask : optional boolean over pairs; pairs outside it are never predicted (used for per-bin sweeps).
    """
    if mask is None:
        mask = np.ones(len(n_true_per_s1), dtype=bool)
    if pair_mask is not None:
        s1_code, prob, label = s1_code[pair_mask], prob[pair_mask], label[pair_mask]
        if fp_weight is not None:
            fp_weight = fp_weight[pair_mask]
    order = np.argsort(-prob, kind='stable')
    s1_o = s1_code[order]
    lab_o = label[order].astype(np.float64)
    prob_o = prob[order]
    w_o = None if fp_weight is None else np.where(lab_o > 0, 1.0, fp_weight[order].astype(np.float64))
    results = []
    n_s1 = len(n_true_per_s1)
    for t in grid:
        k = int(np.searchsorted(-prob_o, -t, side='right'))  # pairs with prob >= t
        tp = np.bincount(s1_o[:k], weights=lab_o[:k], minlength=n_s1)
        if w_o is None:
            n_pred = np.bincount(s1_o[:k], minlength=n_s1)
        else:
            n_pred = np.bincount(s1_o[:k], weights=w_o[:k], minlength=n_s1)
        f = f05_from_counts(n_pred, n_true_per_s1, tp)
        results.append((float(t), float(f[mask].mean())))
    return results


def score_selection(mask_sel, s1_code, label, n_true_per_s1, fp_weight=None):
    """Macro F0.5 (optionally with weighted false positives) of a boolean pair selection."""
    n_s1 = len(n_true_per_s1)
    lab = label[mask_sel].astype(np.float64)
    tp = np.bincount(s1_code[mask_sel], weights=lab, minlength=n_s1)
    if fp_weight is None:
        n_pred = np.bincount(s1_code[mask_sel], minlength=n_s1).astype(np.float64)
    else:
        w = np.where(lab > 0, 1.0, fp_weight[mask_sel].astype(np.float64))
        n_pred = np.bincount(s1_code[mask_sel], weights=w, minlength=n_s1)
    return f05_from_counts(n_pred, n_true_per_s1, tp)


def best_threshold(results):
    t, f = max(results, key=lambda r: r[1])
    return t, f


def evaluate_prediction_sets(pred_dict, gt_dict, s1_ids):
    """Dict-based (slower) evaluation for final sanity checks: {s1_id: set(s23_id)}."""
    scores = []
    for s in s1_ids:
        yt = gt_dict.get(s, set())
        yp = pred_dict.get(s, set())
        if not yt:
            scores.append(1.0 if not yp else 0.0)
            continue
        if not yp:
            scores.append(0.0)
            continue
        tp = len(yt & yp)
        fp = len(yp - yt)
        fn = len(yt - yp)
        d = 1.25 * tp + 0.25 * fn + fp
        scores.append(1.25 * tp / d if d > 0 else 0.0)
    return float(np.mean(scores))


def build_ground_truth_dict(gt_df):
    out = {}
    for s1_id, matched in zip(gt_df['source1_entity_id'].values, gt_df['matched_entity_ids'].values):
        matched = str(matched)
        out[s1_id] = set(m.strip() for m in matched.split(',') if m.strip()) if matched else set()
    return out


def per_group_report(s1_code, prob, label, n_true_per_s1, group_of_s1, thresholds_by_group, grid):
    """Log the best per-group threshold and F0.5 for diagnostics."""
    rep = {}
    for g in pd.unique(group_of_s1):
        mask = group_of_s1 == g
        res = threshold_sweep(s1_code, prob, label, n_true_per_s1, grid, mask=mask)
        t, f = best_threshold(res)
        rep[g] = (t, f, int(mask.sum()))
        log.info(f"  group={g}: best threshold={t:.3f} macro-F0.5={f:.5f} (n_s1={int(mask.sum()):,})")
    return rep
