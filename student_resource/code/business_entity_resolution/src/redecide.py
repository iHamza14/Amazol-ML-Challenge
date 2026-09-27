"""
Re-run ONLY the decision layer on the scored pairs saved by inference.py (output/scored_pairs.parquet), and
write a new matching_results file. Minutes instead of hours: retrieval, pruning, features and models are not
repeated. Use it to produce alternative submissions from one inference run (France threshold variants,
consensus mode, a global threshold shift) and to pick among them with leaderboard feedback.

    python redecide.py                                   # identical decision to inference.py
    python redecide.py --france-threshold 0.92           # fixed France threshold (skips the calibration)
    python redecide.py --france-shift 0.03               # calibrated France threshold + 0.03
    python redecide.py --threshold-shift 0.02            # every seen-country threshold + 0.02
    python redecide.py --consensus                       # min over the models instead of the mean

The output file defaults to output/matching_results_<tag>.tsv (the inference result is never overwritten);
candidate_pairs.tsv is unchanged because the scored pairs are a subset of the candidates by construction.
The France calibration below is a verbatim copy of inference.py's (kept in sync by hand; inference.py is
not refactored so that the running job is never touched).
"""
import os
import sys
import json
import time
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd

import config as cfg
from inference import combine_probs, write_id_lists, KEEP_PROB
from decision import decide, pair_groups

log = logging.getLogger('redecide')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default=None, help='where inference.py wrote scored_pairs.parquet (default: OUTPUT_DIR)')
    ap.add_argument('--model-dir', default=None)
    ap.add_argument('--test-dir', default=None, help='for the validator')
    ap.add_argument('--tag', default='redecided', help='output file suffix: matching_results_<tag>.tsv')
    ap.add_argument('--threshold-shift', type=float, default=0.0, help='added to every SEEN-country threshold')
    ap.add_argument('--consensus', action='store_true', help='min over models instead of the mean')
    ap.add_argument('--no-conflicts', action='store_true')
    ap.add_argument('--france-threshold', type=float, default=None, help='fixed main threshold for unseen countries (skips calibration)')
    ap.add_argument('--france-shift', type=float, default=0.0, help='added to the calibrated unseen-country threshold')
    ap.add_argument('--france-max-raise', type=float, default=None, help='override the calibration max raise')
    ap.add_argument('--no-validate', action='store_true')
    a = ap.parse_args()
    if a.france_threshold is not None and a.france_shift:
        ap.error('--france-shift applies to the calibrated threshold; give either --france-threshold or --france-shift')
    t0 = time.time()
    output_dir = a.output_dir or cfg.OUTPUT_DIR
    model_dir = a.model_dir or cfg.MODEL_DIR
    test_dir = a.test_dir or cfg.TEST_DIR

    with open(os.path.join(model_dir, 'model_config.json'), encoding='utf-8') as f:
        mcfg = json.load(f)
    dec = dict(mcfg['decision'])
    weights = mcfg.get('ensemble_weights', {})
    if a.no_conflicts:
        dec['resolve_conflicts'] = False
    if a.consensus:
        dec['prob_mode'] = 'min'
    if a.threshold_shift:
        dec['thresholds'] = {k: v + a.threshold_shift for k, v in dec.get('thresholds', {}).items()}
        dec['default_threshold'] = dec.get('default_threshold', 0.5) + a.threshold_shift
    unseen_t = mcfg.get('unseen_country_threshold', dec.get('default_threshold', 0.5)) + a.threshold_shift
    seen_countries = set(mcfg.get('seen_countries') or [k.split('|')[0] for k in dec.get('thresholds', {})])

    log.info(f"loading scored pairs from {output_dir}")
    sp = pd.read_parquet(os.path.join(output_dir, 'scored_pairs.parquet'))
    s1 = pd.read_parquet(os.path.join(output_dir, 'scored_s1.parquet'))
    s1_ids = s1['entity_id'].values.astype(str)
    s1_country = s1['country'].values.astype(str)
    n_s1 = len(s1_ids)
    P_s1 = sp['s1_pos'].values.astype(np.int64)
    P_s23 = sp['s23_id'].values.astype(str)
    P_st = sp['st_tset'].values.astype(np.float32)
    P_rel = sp['num_first_rel'].values.astype(np.int8)
    P_noaddr = sp['addr_empty_s23'].values.astype(bool)
    P_models = {c[5:]: sp[c].values.astype(np.float32) for c in sp.columns if c.startswith('prob_')}
    if not weights:
        weights = {k: 1.0 for k in P_models}
    del sp
    log.info(f"  {len(P_s1):,} scored pairs for {n_s1:,} S1 entities; models {list(P_models)}; decision {dec}")

    P_p = combine_probs(P_models, weights, dec.get('prob_mode', 'mean'))
    P_country = s1_country[P_s1]
    if dec.get('calibrated') and dec.get('calibrators'):
        cals = dec['calibrators']
        P_c = P_p.copy()
        for c in pd.unique(P_country):
            cm = P_country == c
            if c in cals:
                P_c[cm] = np.interp(P_p[cm], cals[c]['x'], cals[c]['y']).astype(np.float32)
            else:
                P_c[cm] = np.mean([np.interp(P_p[cm], cal['x'], cal['y']) for cal in cals.values()], axis=0).astype(np.float32)
        P_p = P_c
        log.info("  applied isotonic calibration before the expected-F0.5 decision")
    P_group = pair_groups(P_country, P_noaddr, per_bin=mcfg.get('per_bin_thresholds', True))
    s23_code = pd.factorize(P_s23)[0]

    # ---------------- unseen countries (verbatim from inference.py, plus the override options) ----------------
    unseen = mcfg.get('unseen_country', {})
    thresholds = dict(dec.get('thresholds', {}))
    if dec.get('noaddr_rescue') and dec['noaddr_rescue'].get('t'):
        for c in pd.unique(s1_country):
            if c not in seen_countries and str(c) not in dec['noaddr_rescue']['t']:
                dec['noaddr_rescue']['t'][str(c)] = float(max(dec['noaddr_rescue']['t'].values()))
                log.info(f"  unseen country '{c}': noaddr rescue threshold {dec['noaddr_rescue']['t'][str(c)]:.3f} (max of seen)")
    for c in pd.unique(s1_country):
        if c in seen_countries:
            continue
        start_t = float(unseen.get('start_threshold', unseen_t)) + a.threshold_shift
        cm = P_country == c
        n_c = int((s1_country == c).sum())
        codes_c = pd.factorize(P_s1[cm])[0] if cm.any() else np.zeros(0, dtype=np.int64)
        n_with_pairs = len(np.unique(P_s1[cm]))
        target = float(unseen.get('target_singleton_rate', cfg.UNSEEN_TARGET_SINGLETON_RATE))
        delta_na = float(unseen.get('noaddr_delta', 0.0)) if mcfg.get('per_bin_thresholds', True) else 0.0
        if a.france_threshold is not None:
            t_c = float(a.france_threshold)
            log.info(f"  unseen country '{c}': FIXED threshold {t_c:.3f} (calibration skipped)")
        elif n_with_pairs:
            no_cand_rate = (n_c - n_with_pairs) / n_c
            target_adj = max(0.0, (target - no_cand_rate) / max(1e-9, n_with_pairs / n_c))
            max_t = float(unseen.get('max_threshold', cfg.UNSEEN_MAX_THRESHOLD))
            sub_s1, sub_s23, sub_p, sub_g = P_s1[cm], s23_code[cm], P_p[cm], P_group[cm]

            def applied_stats(t):
                cfg_c = dict(dec, thresholds={c: t, c + '|noaddr': min(0.99, t + delta_na)}, default_threshold=t)
                m_c = decide(sub_s1, sub_s23, sub_p, sub_g, cfg_c)
                counts = np.bincount(codes_c[m_c], minlength=n_with_pairs)
                return float((counts == 0).mean()), float(counts.sum() / n_c)

            seen_links = float(unseen.get('seen_mean_links', 3.4))
            max_raise = float(a.france_max_raise if a.france_max_raise is not None else unseen.get('max_raise', 0.20))
            t_c = float(start_t)
            rate_c, links_c = applied_stats(t_c)
            stop_reason = 'target reached' if rate_c >= target_adj else 'max threshold'
            while rate_c < target_adj and t_c + 0.005 <= max_t:
                if t_c + 0.005 > start_t + max_raise:
                    stop_reason = f'max raise {max_raise} above start'
                    break
                r2, l2 = applied_stats(round(t_c + 0.005, 4))
                if l2 < seen_links - 0.2:
                    stop_reason = f'mean links would fall below seen level {seen_links:.2f} - 0.2'
                    break
                t_c = round(t_c + 0.005, 4)
                rate_c, links_c = r2, l2
                if rate_c >= target_adj:
                    stop_reason = 'target reached'
            overall_empty = no_cand_rate + rate_c * n_with_pairs / n_c
            log.info(f"  unseen country '{c}': calibration stopped ({stop_reason}); mean links {links_c:.3f} vs seen {seen_links:.3f}; "
                     f"start {start_t:.3f} -> calibrated {t_c:.3f} (predicted-empty rate {overall_empty:.4f} vs target {target:.4f})")
        else:
            t_c = start_t
        if a.france_shift and a.france_threshold is None:
            t_c = min(0.99, t_c + a.france_shift)
            log.info(f"  unseen country '{c}': threshold shifted to {t_c:.3f}")
        thresholds[c] = t_c
        if mcfg.get('per_bin_thresholds', True):
            thresholds[c + '|noaddr'] = min(0.99, t_c + float(unseen.get('noaddr_delta', 0.0)))
    dec['thresholds'] = thresholds
    mask = decide(P_s1, s23_code, P_p, P_group, dec)
    log.info(f"  decision: {dec}")
    log.info(f"  selected {int(mask.sum()):,} pairs from {len(P_p):,} scored (prob>={KEEP_PROB})")

    # ---------------- outputs ----------------
    match_lists = [[] for _ in range(n_s1)]
    for i, b in zip(P_s1[mask], P_s23[mask]):
        match_lists[i].append(b)
    match_lists = [sorted(set(l)) for l in match_lists]
    out_path = os.path.join(output_dir, f'matching_results_{a.tag}.tsv')
    write_id_lists(out_path, 'matched_entity_ids', s1_ids, match_lists)
    n_links_arr = np.array([len(l) for l in match_lists])
    log.info(f"  S1 entities: {n_s1:,} | with matches: {int((n_links_arr > 0).sum()):,} | links: {int(n_links_arr.sum()):,} ({n_links_arr.mean():.3f}/S1)")
    log.info("  GUARDRAILS (training ground truth: 94.4% of S1 have matches -> empty rate 5.6%; 3.46 links per S1):")
    for c in pd.unique(s1_country):
        m = s1_country == c
        empty_rate = float((n_links_arr[m] == 0).mean())
        mean_links = float(n_links_arr[m].mean())
        flag = ''
        if not (0.040 <= empty_rate <= 0.085):
            flag += '  <-- EMPTY RATE OFF (expected ~0.056)'
        if not (3.0 <= mean_links <= 3.8):
            flag += '  <-- MEAN LINKS OFF (expected ~3.46)'
        log.info(f"    {c:8s} n_s1={int(m.sum()):9,d} empty_rate={empty_rate:.4f} mean_links={mean_links:.3f} max_links={int(n_links_arr[m].max())}{flag}")
    log.info(f"  -> {out_path}")
    with open(os.path.join(output_dir, f'decision_{a.tag}.json'), 'w', encoding='utf-8') as f:
        json.dump({'args': vars(a), 'decision': dec}, f, indent=1, default=str)

    if not a.no_validate:
        validator = os.path.join(cfg.PROJECT_ROOT, 'utils', 'validate_submission.py')
        if not os.path.exists(validator):
            validator = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'utils', 'validate_submission.py'))
        if os.path.exists(validator):
            out = subprocess.run([sys.executable, validator, '--matching', out_path, '--test-dir', test_dir],
                                 capture_output=True, text=True, timeout=1800)
            log.info(out.stdout[-3000:])
            if out.returncode != 0:
                log.error(out.stderr[-2000:])
    log.info(f"REDECIDE COMPLETE in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
