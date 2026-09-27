"""
Inference pipeline v2 (single command: `python inference.py`).

  1. Load models + decision config + transliteration table + segmentation vocab + extra-token stats.
  2. Load test sources; preprocess S1 (all) and S2/S3 per country.
  3. Per country: stream blocking chunks -> features -> ensemble probabilities.
     Keep every candidate (for candidate_pairs.tsv) and every pair with prob >= 0.02 (for the
     decision layer) in compact arrays.
  4. Decision layer: thresholds / expected-F0.5 selection, S2/S3 conflict resolution,
     France (unseen-country) filter.
  5. Write output/matching_results.tsv and output/candidate_pairs.tsv (every S1 entity, in file
     order, deduplicated sorted id lists) and run the official validator if present.

CLI: --test-dir --output-dir --model-dir --threshold-shift (added to every threshold)
     --no-conflicts --no-france-filter --max-candidates
"""
import os
import gc
import json
import time
import logging
import argparse
import subprocess
import sys
import numpy as np
import pandas as pd

import config as cfg
from preprocess import load_data, preprocess_dataframe, SegVocab
from translit import Transliterator
from blocking import iter_candidate_chunks
from features import compute_features_parallel, ExtraTokenStats
from decision import decide, pair_groups, calibrate_unseen_threshold
from france_filter import apply_france_filter

log = logging.getLogger('inference')

KEEP_PROB = cfg.DECISION_KEEP_PROB


def load_models(model_dir):
    with open(os.path.join(model_dir, 'model_config.json')) as f:
        mcfg = json.load(f)
    models = {}
    import lightgbm as lgb
    if 'lgbm' in mcfg['models']:
        models['lgbm'] = lgb.Booster(model_file=os.path.join(model_dir, 'lgbm.txt'))
    if 'catboost' in mcfg['models'] and os.path.exists(os.path.join(model_dir, 'catboost.cbm')):
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier()
        cb.load_model(os.path.join(model_dir, 'catboost.cbm'))
        models['catboost'] = cb
    return mcfg, models


def predict_models(models, X):
    """Per-model probabilities (dict name -> float32 array)."""
    out = {}
    for k, m in models.items():
        if k == 'lgbm':
            out[k] = m.predict(X).astype(np.float32)
        else:
            out[k] = m.predict_proba(X)[:, 1].astype(np.float32)
    return out


def combine_probs(P_models, weights, mode='mean'):
    """'mean': weighted average of the models; 'min': minimum over models (consensus, more precision)."""
    keys = [k for k in P_models if weights.get(k, 1.0) > 0] or list(P_models)
    if not keys:
        return np.zeros(0, dtype=np.float32)
    if mode == 'min' and len(keys) > 1:
        return np.min(np.stack([P_models[k] for k in keys]), axis=0).astype(np.float32)
    wsum = sum(weights.get(k, 1.0) for k in keys)
    tot = sum(P_models[k] * weights.get(k, 1.0) for k in keys)
    return (tot / wsum).astype(np.float32)


def write_id_lists(path, header_col, s1_ids, lists):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f"source1_entity_id\t{header_col}\n")
        for sid, lst in zip(s1_ids, lists):
            f.write(f"{sid}\t{','.join(lst)}\n")


def run_inference(test_dir=None, output_dir=None, model_dir=None, threshold_shift=0.0,
                  resolve_conflicts=None, france_filter=None, max_candidates=None, consensus=False):
    t0 = time.time()
    test_dir = test_dir or cfg.TEST_DIR
    output_dir = output_dir or cfg.OUTPUT_DIR
    model_dir = model_dir or cfg.MODEL_DIR
    os.makedirs(output_dir, exist_ok=True)

    # ---------------- 1. models & resources ----------------
    log.info("STEP 1: loading models and resources")
    mcfg, models = load_models(model_dir)
    feat_cols = mcfg['feature_cols']
    weights = mcfg.get('ensemble_weights', {k: 1.0 for k in models})
    dec = dict(mcfg['decision'])
    if resolve_conflicts is not None:
        dec['resolve_conflicts'] = resolve_conflicts
    if consensus:
        dec['prob_mode'] = 'min'
    if threshold_shift:
        dec['thresholds'] = {k: v + threshold_shift for k, v in dec.get('thresholds', {}).items()}
        dec['default_threshold'] = dec.get('default_threshold', 0.5) + threshold_shift
    unseen_t = mcfg.get('unseen_country_threshold', dec.get('default_threshold', 0.5)) + threshold_shift
    trans = Transliterator.load_or_empty(os.path.join(model_dir, 'translit.json'))
    with open(os.path.join(model_dir, 'seg_vocab.json'), encoding='utf-8') as f:
        seg = SegVocab(json.load(f))
    extra_stats = ExtraTokenStats.load(os.path.join(model_dir, 'extra_token_stats.json'))
    log.info(f"  decision config: {dec} | unseen-country threshold: {unseen_t:.3f}")

    # ---------------- 2. data ----------------
    log.info("STEP 2: loading and preprocessing test data")
    df_s1 = load_data(os.path.join(test_dir, 'test_source1.tsv'))
    df_s2 = load_data(os.path.join(test_dir, 'test_source2.tsv'))
    df_s3 = load_data(os.path.join(test_dir, 'test_source3.tsv'))
    df_s23_raw = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    gc.collect()
    df_s1 = preprocess_dataframe(df_s1, translit=trans, seg_vocab=seg, n_jobs=cfg.N_JOBS)
    df_s1['name_dup'] = df_s1.groupby(['country_norm', 'name_core'])['entity_id'].transform('size').astype(np.int32)
    df_s1 = df_s1.drop(columns=['business_name', 'business_address', 'country'])   # raw text no longer needed
    gc.collect()
    df_s23_raw['country_norm'] = df_s23_raw['country'].map(lambda x: str(x).lower().strip())
    s1_ids = df_s1['entity_id'].values
    s1_country = df_s1['country_norm'].values
    n_s1 = len(df_s1)
    # countries the models were trained/validated on (persisted by train.py; the threshold keys can be empty
    # when the expected-F0.5 mode with floor 0 wins, so they must not be used for this)
    seen_countries = set(mcfg.get('seen_countries') or [k.split('|')[0] for k in dec.get('thresholds', {})]
                         or list(mcfg.get('val_per_country', {}).keys()))
    # blocking settings must match training: candidate cap, reverse channel and top-k define fused_rank,
    # rev_* and the group-relative features the models were trained on
    blk = mcfg.get('blocking')
    if blk:
        eff_cap = max_candidates or cfg.BLOCK_MAX_CANDIDATES
        trained_by_country = dict(blk.get('by_country') or {})
        if eff_cap != blk['max_candidates'] or bool(cfg.USE_REVERSE_BLOCKING) != bool(blk['use_reverse']) \
                or cfg.REVERSE_TOPK != blk.get('reverse_topk', cfg.REVERSE_TOPK) or dict(cfg.BLOCK_TOPK) != dict(blk.get('topk', cfg.BLOCK_TOPK)) \
                or dict(cfg.BLOCK_BY_COUNTRY) != trained_by_country or cfg.BLOCK_FUSION != blk.get('fusion', 'minrank'):
            log.warning(f"  BLOCKING SETTINGS DIFFER FROM TRAINING: now cap={eff_cap} reverse={cfg.USE_REVERSE_BLOCKING} "
                        f"topk={dict(cfg.BLOCK_TOPK)} by_country={dict(cfg.BLOCK_BY_COUNTRY)} vs trained {blk}. Using the TRAINED settings.")
            cfg.BLOCK_MAX_CANDIDATES = int(blk['max_candidates'])
            cfg.USE_REVERSE_BLOCKING = bool(blk['use_reverse'])
            cfg.REVERSE_TOPK = int(blk.get('reverse_topk', cfg.REVERSE_TOPK))
            cfg.BLOCK_TOPK = dict(blk.get('topk', cfg.BLOCK_TOPK))
            cfg.BLOCK_BY_COUNTRY = trained_by_country
            cfg.BLOCK_FUSION = blk.get('fusion', 'minrank')          # models before this option used min-rank fusion
            cfg.BLOCK_RRF_C = float(blk.get('rrf_c', cfg.BLOCK_RRF_C))
            # per-country caps come from cfg.max_candidates_for(); a --max-candidates override is dropped
            max_candidates = None
    keep_prob_trained = float(mcfg.get('keep_prob', KEEP_PROB))
    if abs(keep_prob_trained - KEEP_PROB) > 1e-9:
        log.warning(f"  keep_prob differs from training ({keep_prob_trained} vs {KEEP_PROB}); using the trained value")
    keep_prob = keep_prob_trained
    for c in pd.unique(s1_country):
        if c not in seen_countries:
            dec.setdefault('thresholds', {})[c] = float(unseen_t)
            log.info(f"  unseen country '{c}': threshold {unseen_t:.3f}")

    # ---------------- 3. blocking -> stage-2 pruner -> scoring ----------------
    # cascade: retrieval returns up to 150-200 pairs per entity; the pruner (LightGBM on the retrieval meta features
    # only) keeps the pairs worth scoring; the survivors are the candidate set (candidate_pairs.tsv) and the only
    # pairs the matching model runs on
    pruner = None
    pr_cfg = mcfg.get('pruner')
    pruner_path = os.path.join(model_dir, 'pruner.txt')
    if pr_cfg and os.path.exists(pruner_path):
        import lightgbm as lgb
        pruner = lgb.Booster(model_file=pruner_path)
        pr_cols = list(pr_cfg['features'])
        pr_thr = {str(k): float(v) for k, v in pr_cfg['thresholds'].items()}
        pr_default = float(pr_cfg.get('default_threshold', min(pr_thr.values()) if pr_thr else 0.0))
        log.info(f"  candidate pruner loaded: thresholds {pr_thr} (unseen countries use {pr_default:.5f}); "
                 f"validation: {pr_cfg.get('val_stats')}")
    elif pr_cfg:
        log.warning("  model_config has a pruner but pruner.txt is missing: scoring every retrieved pair")
    cand_lists = [None] * n_s1            # per S1 position: np.array of S2/S3 ids (str)
    pred_s1, pred_s23, pred_p, pred_st, pred_rel, pred_noaddr = [], [], [], [], [], []
    pred_models = {k: [] for k in models}
    for country in pd.unique(s1_country):
        log.info("=" * 70)
        log.info(f"COUNTRY {country}")
        df_s23_c = df_s23_raw[df_s23_raw['country_norm'] == country].copy()
        if len(df_s23_c) == 0:
            log.warning(f"  no S2/S3 rows for {country}: all S1 entities of this country are singletons")
            continue
        df_s23_c = preprocess_dataframe(df_s23_c, translit=trans, seg_vocab=seg, n_jobs=cfg.N_JOBS)
        core_counts = df_s1.loc[s1_country == country, 'name_core'].value_counts()
        df_s23_c['s1_core_count'] = df_s23_c['name_core'].map(core_counts).fillna(0).astype(np.int32)
        df_s23_c = df_s23_c.drop(columns=['business_name', 'business_address', 'country'])   # raw text no longer needed
        gc.collect()
        log.info(f"  parent RSS after preprocessing {country}: {cfg.rss_gb():.1f} GB (limit {cfg.TOTAL_RAM_GB:.0f} GB)")
        s23_ids_c = df_s23_c['entity_id'].values
        mask_c = s1_country == country
        n_pairs = 0
        n_retrieved = 0
        for _, cand, blocker in iter_candidate_chunks(df_s1, df_s23_c, s1_mask=mask_c, max_candidates=max_candidates):
            if len(cand) == 0:
                continue
            n_retrieved += len(cand)
            if pruner is not None:
                s_pr = pruner.predict(cand[pr_cols].values.astype(np.float32))
                keep_pr = s_pr >= pr_thr.get(str(country), pr_default)
                cand = cand.loc[keep_pr].reset_index(drop=True)
                log.info(f"  pruner kept {int(keep_pr.sum()):,} of {len(keep_pr):,} retrieved pairs ({100 * keep_pr.mean():.1f}%)")
                if len(cand) == 0:
                    continue
            s1p = cand['s1_pos'].values
            s23p = cand['s23_pos'].values
            # candidate lists (complete per S1 within a chunk)
            order = np.argsort(s1p, kind='stable')
            s1p_o = s1p[order]
            starts = np.r_[0, np.flatnonzero(np.diff(s1p_o)) + 1]
            ends = np.r_[starts[1:], len(s1p_o)]
            ids_o = s23_ids_c[s23p[order]]
            for a, b in zip(starts, ends):
                cand_lists[s1p_o[a]] = ids_o[a:b]
            feat = compute_features_parallel(cand, df_s1, df_s23_c, vecs=blocker.vec, extra_stats=extra_stats, n_jobs=cfg.FEATURE_WORKERS)
            X = feat[feat_cols].values.astype(np.float32)
            pm = predict_models(models, X)
            p_max = np.max(np.stack(list(pm.values())), axis=0) if pm else np.zeros(len(X), dtype=np.float32)
            keep = p_max >= keep_prob
            pred_s1.append(s1p[keep])
            pred_s23.append(s23_ids_c[s23p[keep]])
            pred_p.append(p_max[keep])
            for k in pm:
                pred_models[k].append(pm[k][keep])
            pred_st.append(feat['st_tset'].values[keep].astype(np.float32))
            pred_rel.append(feat['num_first_rel'].values[keep].astype(np.int8))
            pred_noaddr.append(feat['f_addr_empty_s23'].values[keep].astype(bool))
            n_pairs += len(cand)
            del feat, X
        n_ent_c = max(1, int(mask_c.sum()))
        log.info(f"  {country}: retrieved {n_retrieved:,} pairs ({n_retrieved / n_ent_c:.1f}/S1) -> candidates scored "
                 f"{n_pairs:,} ({n_pairs / n_ent_c:.1f}/S1)")
        del df_s23_c
        gc.collect()
    del df_s23_raw
    gc.collect()

    # ---------------- 4. decision ----------------
    log.info("=" * 70)
    log.info("STEP 4: decision layer")
    if pred_p:
        P_s1 = np.concatenate(pred_s1)
        P_s23 = np.concatenate(pred_s23)
        P_models = {k: np.concatenate(v) for k, v in pred_models.items()}
        P_st = np.concatenate(pred_st)
        P_rel = np.concatenate(pred_rel)
        P_noaddr = np.concatenate(pred_noaddr)
    else:
        P_s1 = np.zeros(0, dtype=np.int64)
        P_s23 = np.zeros(0, dtype=object)
        P_models = {k: np.zeros(0, dtype=np.float32) for k in models}
        P_st = np.zeros(0, dtype=np.float32)
        P_rel = np.zeros(0, dtype=np.int8)
        P_noaddr = np.zeros(0, dtype=bool)
    P_p = combine_probs(P_models, weights, dec.get('prob_mode', 'mean'))
    P_country = s1_country[P_s1]
    # persist the scored pairs (everything the decision layer consumes) so that redecide.py can re-run the
    # decision layer with other settings (France threshold, consensus, shift) in minutes instead of hours.
    # Best effort: a failure here must never stop the run.
    try:
        t_dump = time.time()
        scored = pd.DataFrame({'s1_pos': P_s1.astype(np.int64), 's23_id': P_s23.astype(str),
                               'st_tset': P_st.astype(np.float32), 'num_first_rel': P_rel.astype(np.int8),
                               'addr_empty_s23': P_noaddr.astype(bool)})
        for k, v in P_models.items():
            scored[f'prob_{k}'] = v.astype(np.float32)
        scored.to_parquet(os.path.join(output_dir, 'scored_pairs.parquet'), index=False)
        pd.DataFrame({'entity_id': s1_ids.astype(str), 'country': s1_country.astype(str)}).to_parquet(
            os.path.join(output_dir, 'scored_s1.parquet'), index=False)
        del scored
        log.info(f"  scored pairs saved -> {os.path.join(output_dir, 'scored_pairs.parquet')} ({len(P_s1):,} rows, {time.time() - t_dump:.0f}s)")
    except Exception as e:  # noqa
        log.warning(f"  could not save the scored pairs (redecide.py will not be available): {e}")
    if dec.get('calibrated') and dec.get('calibrators'):
        # per-country isotonic calibration (fitted on validation); unseen countries get the mean of the seen curves
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

    # unseen countries: start at the seen-country mean (+shift), then raise until the predicted-empty
    # rate reaches the generator's singleton share (label-free calibration; never lowered)
    unseen = mcfg.get('unseen_country', {})
    thresholds = dict(dec.get('thresholds', {}))
    # the address-empty rescue threshold of an unseen country is the strictest seen one (raise-only), set before
    # the calibration below so that the calibrated rule is the rule that is applied
    if dec.get('noaddr_rescue') and dec['noaddr_rescue'].get('t'):
        for c in pd.unique(s1_country):
            if c not in seen_countries and str(c) not in dec['noaddr_rescue']['t']:
                dec['noaddr_rescue']['t'][str(c)] = float(max(dec['noaddr_rescue']['t'].values()))
                log.info(f"  unseen country '{c}': noaddr rescue threshold {dec['noaddr_rescue']['t'][str(c)]:.3f} (max of seen)")
    for c in pd.unique(s1_country):
        if c in seen_countries:
            continue
        start_t = float(unseen.get('start_threshold', unseen_t)) + threshold_shift
        cm = P_country == c
        n_c = int((s1_country == c).sum())
        codes_c = pd.factorize(P_s1[cm])[0] if cm.any() else np.zeros(0, dtype=np.int64)
        # entities with no candidate at all are empty regardless -> count them in the rate
        n_with_pairs = len(np.unique(P_s1[cm]))
        target = float(unseen.get('target_singleton_rate', cfg.UNSEEN_TARGET_SINGLETON_RATE))
        if n_with_pairs:
            no_cand_rate = (n_c - n_with_pairs) / n_c
            target_adj = max(0.0, (target - no_cand_rate) / max(1e-9, n_with_pairs / n_c))
            max_t = float(unseen.get('max_threshold', cfg.UNSEEN_MAX_THRESHOLD))
            delta_na = float(unseen.get('noaddr_delta', 0.0)) if mcfg.get('per_bin_thresholds', True) else 0.0
            sub_s1, sub_s23, sub_p, sub_g = P_s1[cm], s23_code[cm], P_p[cm], P_group[cm]

            def applied_stats(t):
                # the rule that will actually be applied (per-bin thresholds, size-adaptive, conflicts, mode)
                cfg_c = dict(dec, thresholds={c: t, c + '|noaddr': min(0.99, t + delta_na)}, default_threshold=t)
                m_c = decide(sub_s1, sub_s23, sub_p, sub_g, cfg_c)
                counts = np.bincount(codes_c[m_c], minlength=n_with_pairs)
                return float((counts == 0).mean()), float(counts.sum() / n_c)

            # raise-only, but never past the point where it starts destroying recall: stop when the country's
            # mean links per entity fall well below the seen countries' level, or after max_raise
            seen_links = float(unseen.get('seen_mean_links', 3.4))
            max_raise = float(unseen.get('max_raise', 0.20))
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
            log.info(f"  unseen country '{c}': calibration stopped ({stop_reason}); mean links {links_c:.3f} vs seen {seen_links:.3f}")
        else:
            t_c, rate_c, overall_empty = start_t, 1.0, 1.0
        thresholds[c] = t_c
        if mcfg.get('per_bin_thresholds', True):
            thresholds[c + '|noaddr'] = min(0.99, t_c + float(unseen.get('noaddr_delta', 0.0)))
        log.info(f"  unseen country '{c}': start threshold {start_t:.3f} -> calibrated {t_c:.3f} "
                 f"(predicted-empty rate {overall_empty:.4f} vs target {target:.4f}; "
                 f"{n_c - n_with_pairs:,} of {n_c:,} entities had no candidate at all)")
    dec['thresholds'] = thresholds
    mask = decide(P_s1, s23_code, P_p, P_group, dec)
    log.info(f"  decision: {dec}")
    log.info(f"  selected {int(mask.sum()):,} pairs from {len(P_p):,} scored (prob>={KEEP_PROB})")
    use_ff = cfg.FRANCE_ENABLED if france_filter is None else france_filter
    if use_ff:
        for c in pd.unique(s1_country):
            if c not in seen_countries:
                mask = apply_france_filter(mask, P_s1, P_country, P_st, P_rel, target_country=c)

    # ---------------- 5. outputs ----------------
    log.info("STEP 5: writing outputs")
    match_lists = [[] for _ in range(n_s1)]
    for a, b in zip(P_s1[mask], P_s23[mask]):
        match_lists[a].append(b)
    match_lists = [sorted(set(l)) for l in match_lists]
    cand_out = [sorted(set(l.tolist())) if l is not None else [] for l in cand_lists]
    for i in range(n_s1):   # guarantee the subset property
        if match_lists[i]:
            cs = set(cand_out[i])
            match_lists[i] = [m for m in match_lists[i] if m in cs]
    match_path = os.path.join(output_dir, 'matching_results.tsv')
    cand_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    write_id_lists(match_path, 'matched_entity_ids', s1_ids, match_lists)
    write_id_lists(cand_path, 'candidate_entity_ids', s1_ids, cand_out)
    n_matched = sum(1 for l in match_lists if l)
    n_links = sum(len(l) for l in match_lists)
    log.info(f"  S1 entities: {n_s1:,} | with matches: {n_matched:,} ({n_matched / n_s1:.4f}) | links: {n_links:,} ({n_links / n_s1:.3f}/S1)")
    log.info("  GUARDRAILS (training ground truth: 94.4% of S1 have matches -> empty rate 5.6%; 3.46 links per S1):")
    n_links_arr = np.array([len(l) for l in match_lists])
    for c in pd.unique(s1_country):
        m = s1_country == c
        empty_rate = float((n_links_arr[m] == 0).mean())
        mean_links = float(n_links_arr[m].mean())
        flag = ''
        if not (0.040 <= empty_rate <= 0.085):
            flag += '  <-- EMPTY RATE OFF (expected ~0.056)'
        if not (3.0 <= mean_links <= 3.8):
            flag += '  <-- MEAN LINKS OFF (expected ~3.46)'
        log.info(f"    {c:8s} n_s1={int(m.sum()):9,d} empty_rate={empty_rate:.4f} mean_links={mean_links:.3f} "
                 f"max_links={int(n_links_arr[m].max()) if m.any() else 0}{flag}")
    log.info(f"  -> {match_path}\n  -> {cand_path}")

    # free the big in-memory structures before the validator subprocess (its candidate cross-check alone
    # needs ~12 KB per S1 entity, ~20 GB on the full test set); the subset property was enforced above, so the
    # validator is run on the matching file only. Re-run it with --candidate from a fresh shell if desired.
    del df_s1, cand_lists, cand_out, match_lists, P_s1, P_s23, P_models, P_p, P_group, P_st, P_rel, P_noaddr, s23_code, mask
    gc.collect()
    validator = os.path.join(cfg.PROJECT_ROOT, 'utils', 'validate_submission.py')
    if not os.path.exists(validator):
        validator = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'utils', 'validate_submission.py'))
    if os.path.exists(validator):
        log.info("STEP 6: running official validator (matching file; candidate file checked separately to save memory)")
        try:
            out = subprocess.run([sys.executable, validator, '--matching', match_path,
                                  '--test-dir', test_dir], capture_output=True, text=True, timeout=1800)
            log.info(out.stdout[-3000:])
            if out.returncode != 0:
                log.error(out.stderr[-2000:])
        except Exception as e:  # noqa
            log.warning(f"validator could not run: {e}")
    log.info(f"INFERENCE COMPLETE in {(time.time() - t0) / 60:.1f} min")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Entity Resolution Inference v2")
    ap.add_argument('--test-dir', default=None)
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--model-dir', default=None)
    ap.add_argument('--threshold-shift', type=float, default=0.0, help='added to every decision threshold')
    ap.add_argument('--no-conflicts', action='store_true', help='disable S2/S3 conflict resolution')
    ap.add_argument('--no-france-filter', action='store_true')
    ap.add_argument('--consensus', action='store_true', help='use min over models instead of the mean (more precision)')
    ap.add_argument('--max-candidates', type=int, default=None)
    a = ap.parse_args()
    run_inference(test_dir=a.test_dir, output_dir=a.output_dir, model_dir=a.model_dir,
                  threshold_shift=a.threshold_shift,
                  resolve_conflicts=(False if a.no_conflicts else None),
                  france_filter=(False if a.no_france_filter else None),
                  max_candidates=a.max_candidates, consensus=a.consensus)
