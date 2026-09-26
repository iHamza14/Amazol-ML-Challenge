"""
Training pipeline v2 (single command: `python train.py`).

  1. Load training sources + ground truth.
  2. Learn the Indic->Latin transliteration table from ground-truth pairs (translit.py).
  3. Build the word-segmentation vocabulary from S1 names (train + test S1 names if present).
  4. Preprocess S1 (all) and, per country, S2/S3.
  5. Split S1 entities into   stats | train | val   (disjoint, seeded).
  6. Pass A (stats split): blocking + labels -> extra-token statistics (features.ExtraTokenStats),
     blocking recall diagnostics.
  7. Pass B (train split): blocking -> features -> labels -> negative subsampling -> matrix.
  8. Pass C (val split):   blocking -> features -> labels (ALL candidates kept).
  9. Train LightGBM (+ CatBoost if enabled), early stopping on val, ensemble.
 10. Decision-layer selection on val: global / per-country thresholds, expected-F0.5 set
     selection, S2/S3 conflict resolution, France-filter sanity check.
 11. Save models/, model_config.json, translit.json, seg_vocab.json, extra_token_stats.json.

Memory: the S2/S3 frame is preprocessed one country at a time; candidate chunks stream through
features; training rows are stored as float32.  Fits in 32 GB for the full data with defaults.
"""
import os
import gc
import json
import time
import logging
import numpy as np
import pandas as pd

import config as cfg
from preprocess import load_data, preprocess_dataframe, SegVocab
from translit import Transliterator
from blocking import iter_candidate_chunks
from features import compute_features, compute_features_parallel, ExtraTokenStats, feature_columns
from evaluate import threshold_sweep, best_threshold, f05_from_counts, score_selection
from decision import decide, resolve_conflicts, expected_f05_select, threshold_select, pair_groups
from france_filter import apply_france_filter

log = logging.getLogger('train')

FEAT_DTYPE = np.float32


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def gt_pairs(gt_df):
    for s1_id, m in zip(gt_df['source1_entity_id'].values, gt_df['matched_entity_ids'].values):
        if m:
            for mid in m.split(','):
                mid = mid.strip()
                if mid:
                    yield s1_id, mid


def build_owner_array(df_s23_c, s23_owner_s1pos):
    """int64 array: for each row of the country S2/S3 frame, position of its true S1 (or -1)."""
    return df_s23_c['entity_id'].map(s23_owner_s1pos).fillna(-1).astype(np.int64).values


class ExtraTokenAccumulator:
    def __init__(self):
        self.n_true = {}
        self.n_tot = {}
        self.pos_extra = 0
        self.tot_extra = 0

    def add(self, core1_list, core2_list, labels):
        nt, no = self.n_true, self.n_tot
        for a, b, y in zip(core1_list, core2_list, labels):
            extra = set(b.split()) - set(a.split())
            if not extra:
                continue
            self.tot_extra += 1
            self.pos_extra += int(y)
            for t in extra:
                no[t] = no.get(t, 0) + 1
                if y:
                    nt[t] = nt.get(t, 0) + 1

    def finish(self, min_total=20, alpha=20.0):
        prior = self.pos_extra / self.tot_extra if self.tot_extra else 0.5
        table = {t: (self.n_true.get(t, 0) + alpha * prior) / (n + alpha)
                 for t, n in self.n_tot.items() if n >= min_total}
        log.info(f"ExtraTokenStats: {len(table):,} tokens, prior={prior:.3f}, pairs-with-extras={self.tot_extra:,}")
        return ExtraTokenStats(table, prior=prior, alpha=alpha)


def subsample_negatives(label, min_rank, rng):
    keep = label.astype(bool) | (min_rank < cfg.NEG_KEEP_ALL_TOP)
    rnd = rng.random(len(label)) < cfg.NEG_RANDOM_FRAC
    return keep | rnd


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    t0 = time.time()
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    # ---------------- 1. load ----------------
    log.info("=" * 70)
    log.info("STEP 1: loading training data")
    df_s1 = load_data(os.path.join(cfg.TRAIN_DIR, 'train_source1.tsv'))
    df_s2 = load_data(os.path.join(cfg.TRAIN_DIR, 'train_source2.tsv'))
    df_s3 = load_data(os.path.join(cfg.TRAIN_DIR, 'train_source3.tsv'))
    gt_df = load_data(os.path.join(cfg.TRAIN_DIR, 'train_ground_truth.tsv'))
    df_s23_raw = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    gc.collect()

    # ---------------- 2. transliteration ----------------
    log.info("STEP 2: learning transliteration table")
    name1 = dict(zip(df_s1['entity_id'].values, df_s1['business_name'].values))
    name23 = dict(zip(df_s23_raw['entity_id'].values, df_s23_raw['business_name'].values))
    trans = Transliterator.learn(name1, name23, gt_pairs(gt_df))
    trans.save(os.path.join(cfg.MODEL_DIR, 'translit.json'))
    del name23
    gc.collect()

    # ---------------- 3. segmentation vocab ----------------
    log.info("STEP 3: building segmentation vocabulary from S1 names")
    names_for_vocab = list(df_s1['business_name'].values)
    test_s1_path = os.path.join(cfg.TEST_DIR, 'test_source1.tsv')
    if os.path.exists(test_s1_path):
        names_for_vocab += list(load_data(test_s1_path, usecols=['business_name'])['business_name'].values)
    seg = SegVocab.build(names_for_vocab)
    with open(os.path.join(cfg.MODEL_DIR, 'seg_vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(seg.logp, f)
    del names_for_vocab
    gc.collect()

    # ---------------- 4. preprocess S1 ----------------
    log.info("STEP 4: preprocessing S1")
    df_s1 = preprocess_dataframe(df_s1, translit=trans, seg_vocab=seg, n_jobs=cfg.N_JOBS)
    # how many S1 entities of the same country share the core name (name ambiguity signal)
    df_s1['name_dup'] = df_s1.groupby(['country_norm', 'name_core'])['entity_id'].transform('size').astype(np.int32)
    s1_pos_of_id = pd.Series(np.arange(len(df_s1)), index=df_s1['entity_id'].values)
    # owner of each S2/S3 id -> S1 position ; true-match counts per S1 position
    owner = {}
    for s1_id, mid in gt_pairs(gt_df):
        owner[mid] = s1_id
    s23_owner_s1pos = pd.Series(owner).map(s1_pos_of_id)   # index: s23 id -> s1 pos
    n_true_all = np.bincount(s23_owner_s1pos.values.astype(np.int64), minlength=len(df_s1)).astype(np.int64)
    log.info(f"  S1 entities: {len(df_s1):,} | singletons: {(n_true_all == 0).mean():.4f} | mean matches: {n_true_all.mean():.3f}")
    del owner, gt_df
    gc.collect()

    # ---------------- 5. splits ----------------
    n_s1 = len(df_s1)
    perm = rng.permutation(n_s1)
    n_train = min(cfg.TRAIN_S1_ENTITIES, int(0.6 * n_s1))
    n_stats = min(300000, int(0.75 * n_train), n_s1 - n_train)
    n_val = min(cfg.VAL_S1_ENTITIES, int(0.25 * n_train), n_s1 - n_train - n_stats)
    train_pos = np.sort(perm[:n_train])
    stats_pos = np.sort(perm[n_train:n_train + n_stats])
    val_pos = np.sort(perm[n_train + n_stats:n_train + n_stats + n_val])
    split = np.zeros(n_s1, dtype=np.int8)      # 0 unused, 1 train, 2 stats, 3 val
    split[train_pos] = 1
    split[stats_pos] = 2
    split[val_pos] = 3
    log.info(f"STEP 5: splits  train={n_train:,}  stats={n_stats:,}  val={n_val:,}")

    s1_country = df_s1['country_norm'].values
    df_s23_raw['country_norm'] = df_s23_raw['country'].map(lambda x: str(x).lower().strip())
    countries = pd.unique(s1_country)

    acc = ExtraTokenAccumulator()
    train_X, train_y, train_meta = [], [], []
    val_X, val_y, val_meta = [], [], []
    feat_cols = None
    recall_stats = {}

    preprocessed_s23 = {}

    def get_s23(country):
        """Preprocess (or fetch cached) S2/S3 rows of one country."""
        if country not in preprocessed_s23:
            df_c = df_s23_raw[df_s23_raw['country_norm'] == country].copy()
            preprocessed_s23[country] = preprocess_dataframe(df_c, translit=trans, seg_vocab=seg, n_jobs=cfg.N_JOBS)
        return preprocessed_s23[country]

    keep_s23_cached = os.environ.get('ER_CACHE_S23', '1' if len(df_s23_raw) < 4_000_000 else '0') == '1'

    # ---------- Pass A (all countries first): extra-token statistics + blocking recall ----------
    for country in countries:
        log.info("=" * 70)
        log.info(f"--- Pass A (stats split) country={country} ---")
        df_s23_c = get_s23(country)
        owner_c = build_owner_array(df_s23_c, s23_owner_s1pos)
        found = 0
        mask_stats = (split == 2) & (s1_country == country)
        truth_total = int(n_true_all[mask_stats].sum())
        true_fused_ranks = []
        for _, cand, blocker in iter_candidate_chunks(df_s1, df_s23_c, s1_mask=mask_stats):
            lab = (owner_c[cand['s23_pos'].values] == cand['s1_pos'].values)
            found += int(lab.sum())
            true_fused_ranks.append(cand['fused_rank'].values[lab])
            acc.add(df_s1['name_core'].values[cand['s1_pos'].values].tolist(),
                    df_s23_c['name_core'].values[cand['s23_pos'].values].tolist(), lab)
        rec = found / max(1, truth_total)
        recall_stats[country] = rec
        log.info(f"  BLOCKING RECALL (stats split, {country}): {rec:.5f}  ({found:,}/{truth_total:,})")
        if true_fused_ranks:
            tfr = np.concatenate(true_fused_ranks)
            curve = {k: round(float((tfr < k).sum()) / max(1, truth_total), 5) for k in (10, 20, 30, 40, 50, 60, 80, 100, 150)}
            log.info(f"  recall@K by fused rank ({country}): {curve}")
            recall_stats[f'{country}_recall_at_k'] = curve
        if not keep_s23_cached:
            del preprocessed_s23[country]
            gc.collect()
    extra_stats = acc.finish()
    extra_stats.save(os.path.join(cfg.MODEL_DIR, 'extra_token_stats.json'))

    for country in countries:
        log.info("=" * 70)
        log.info(f"COUNTRY {country}")
        df_s23_c = get_s23(country)
        owner_c = build_owner_array(df_s23_c, s23_owner_s1pos)
        s23_ids_arr = df_s23_c['entity_id'].values

        # ---------- Pass B: train split ----------
        log.info(f"--- Pass B (train split) country={country} ---")
        mask_train = (split == 1) & (s1_country == country)
        for _, cand, blocker in iter_candidate_chunks(df_s1, df_s23_c, s1_mask=mask_train):
            if len(cand) == 0:
                continue
            feat = compute_features_parallel(cand, df_s1, df_s23_c, vecs=blocker.vec, extra_stats=extra_stats, n_jobs=cfg.N_JOBS)
            if feat_cols is None:
                feat_cols = feature_columns(feat)
                log.info(f"  {len(feat_cols)} features: {feat_cols}")
            lab = (owner_c[cand['s23_pos'].values] == cand['s1_pos'].values).astype(np.int8)
            keep = subsample_negatives(lab, cand['min_rank'].values, rng)
            train_X.append(feat.loc[keep, feat_cols].values.astype(FEAT_DTYPE))
            train_y.append(lab[keep])
            train_meta.append(pd.DataFrame({'s1_pos': cand['s1_pos'].values[keep], 'country': country}))
            del feat
        # ---------- Pass C: val split ----------
        log.info(f"--- Pass C (val split) country={country} ---")
        mask_val = (split == 3) & (s1_country == country)
        for _, cand, blocker in iter_candidate_chunks(df_s1, df_s23_c, s1_mask=mask_val):
            if len(cand) == 0:
                continue
            feat = compute_features_parallel(cand, df_s1, df_s23_c, vecs=blocker.vec, extra_stats=extra_stats, n_jobs=cfg.N_JOBS)
            lab = (owner_c[cand['s23_pos'].values] == cand['s1_pos'].values).astype(np.int8)
            val_X.append(feat[feat_cols].values.astype(FEAT_DTYPE))
            val_y.append(lab)
            val_meta.append(pd.DataFrame({
                's1_pos': cand['s1_pos'].values, 's23_id': s23_ids_arr[cand['s23_pos'].values],
                'country': country, 'st_tset': feat['st_tset'].values, 'num_first_rel': feat['num_first_rel'].values,
                's23_distractor': (owner_c[cand['s23_pos'].values] < 0),
                'addr_empty_s23': feat['f_addr_empty_s23'].values.astype(bool),
            }))
            del feat
        preprocessed_s23.pop(country, None)
        del df_s23_c, owner_c
        gc.collect()

    del df_s23_raw
    gc.collect()

    X_tr = np.concatenate(train_X)
    y_tr = np.concatenate(train_y)
    meta_tr = pd.concat(train_meta, ignore_index=True)
    X_va = np.concatenate(val_X)
    y_va = np.concatenate(val_y)
    meta_va = pd.concat(val_meta, ignore_index=True)
    del train_X, val_X
    gc.collect()
    log.info(f"TRAIN matrix: {X_tr.shape} positives={int(y_tr.sum()):,} ({y_tr.mean():.4f}) | "
             f"VAL matrix: {X_va.shape} positives={int(y_va.sum()):,}")

    # ---------- France-robustness masking: hide extra-token vocabulary for 15% of rows ----------
    col_idx = {c: i for i, c in enumerate(feat_cols)}
    m = rng.random(len(X_tr)) < 0.15
    X_tr[np.ix_(m, [col_idx['x_extra_min'], col_idx['x_extra_mean']])] = extra_stats.prior
    X_tr[m, col_idx['x_extra_known']] = 0.0

    # ---------------- 9. models ----------------
    log.info("=" * 70)
    log.info("STEP 9: training models")
    import lightgbm as lgb
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain, free_raw_data=False)
    params = dict(cfg.LGBM_PARAMS)
    booster = lgb.train(params, dtrain, num_boost_round=cfg.LGBM_ROUNDS, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(cfg.LGBM_EARLY_STOP), lgb.log_evaluation(100)])
    booster.save_model(os.path.join(cfg.MODEL_DIR, 'lgbm.txt'))
    imp = pd.DataFrame({'feature': feat_cols, 'gain': booster.feature_importance('gain'),
                        'split': booster.feature_importance('split')}).sort_values('gain', ascending=False)
    imp.to_csv(os.path.join(cfg.MODEL_DIR, 'feature_importance.csv'), index=False)
    log.info("  top-25 features by gain:\n" + imp.head(25).to_string(index=False))
    p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
    probs = {'lgbm': p_va}

    if cfg.USE_CATBOOST:
        try:
            from catboost import CatBoostClassifier
            cb = CatBoostClassifier(**cfg.CATBOOST_PARAMS)
            cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
            cb.save_model(os.path.join(cfg.MODEL_DIR, 'catboost.cbm'))
            probs['catboost'] = cb.predict_proba(X_va)[:, 1]
        except Exception as e:  # noqa
            log.warning(f"CatBoost failed ({e}); continuing with LightGBM only")
    del X_tr, y_tr
    gc.collect()

    # ensemble
    w = {k: cfg.ENSEMBLE_WEIGHTS.get(k, 0.0) for k in probs}
    wsum = sum(w.values()) or 1.0
    prob = sum(probs[k] * w[k] for k in probs) / wsum
    from sklearn.metrics import log_loss, roc_auc_score
    for k, p in probs.items():
        log.info(f"  {k}: val logloss={log_loss(y_va, p):.5f} auc={roc_auc_score(y_va, p):.6f}")
    log.info(f"  ensemble: val logloss={log_loss(y_va, prob):.5f} auc={roc_auc_score(y_va, prob):.6f}")

    # ---------------- 10. decision layer ----------------
    log.info("=" * 70)
    log.info("STEP 10: decision layer selection on validation (density-adjusted)")
    val_s1_pos = meta_va['s1_pos'].values
    codes_of_pos = -np.ones(n_s1, dtype=np.int64)
    codes_of_pos[val_pos] = np.arange(len(val_pos))
    s1_code = codes_of_pos[val_s1_pos]
    n_true_val = n_true_all[val_pos]
    country_of_code = s1_country[val_pos]
    country_pair = meta_va['country'].values.astype(str)
    group_pair = pair_groups(country_pair, meta_va['addr_empty_s23'].values, per_bin=cfg.PER_BIN_THRESHOLDS)
    s23_code = pd.factorize(meta_va['s23_id'].values)[0]
    lab = y_va.astype(np.int8)
    # false positives on unmatched (distractor) records are ~1.9x more frequent in the test pool
    fp_w = np.where(meta_va['s23_distractor'].values, cfg.TEST_DISTRACTOR_RATIO, 1.0).astype(np.float64)
    grid = cfg.THRESHOLD_GRID

    def score_mask(mask, adjusted=True):
        f = score_selection(mask, s1_code, lab, n_true_val, fp_weight=fp_w if adjusted else None)
        per_c = {c: round(float(f[country_of_code == c].mean()), 5) for c in pd.unique(country_of_code)}
        return float(f.mean()), per_c

    def select_group_thresholds(P):
        """Coordinate ascent over per-group thresholds (country, country|noaddr) on the adjusted metric."""
        res_global = threshold_sweep(s1_code, P, lab, n_true_val, grid, fp_weight=fp_w)
        t_glob, f_glob = best_threshold(res_global)
        thresholds = {}
        for c in pd.unique(country_of_code):
            ent_mask = country_of_code == c
            main_pairs = group_pair == c
            res_c = threshold_sweep(s1_code, P, lab, n_true_val, grid, mask=ent_mask, fp_weight=fp_w, pair_mask=main_pairs)
            thresholds[c] = best_threshold(res_c)[0]
            if cfg.PER_BIN_THRESHOLDS:
                g = c + '|noaddr'
                noaddr_pairs = group_pair == g
                if noaddr_pairs.any():
                    base = main_pairs & (P >= thresholds[c])
                    best_t, best_f = thresholds[c], -1.0
                    for t in grid:
                        m = base | (noaddr_pairs & (P >= t))
                        f = float(score_selection(m, s1_code, lab, n_true_val, fp_weight=fp_w)[ent_mask].mean())
                        if f > best_f:
                            best_f, best_t = f, float(t)
                    thresholds[g] = best_t
        return t_glob, f_glob, thresholds

    prob_modes = {'mean': prob}
    if cfg.USE_CONSENSUS and len(probs) > 1:
        prob_modes['min'] = np.min(np.stack([probs[k] for k in probs]), axis=0)

    best = None
    for pm, P in prob_modes.items():
        t_glob, f_glob, thresholds = select_group_thresholds(P)
        log.info(f"  [{pm}] global threshold t={t_glob:.3f} adj-F0.5={f_glob:.5f} | group thresholds {thresholds}")
        m_thr = threshold_select(P, group_pair, thresholds, t_glob)
        f_adj, pc_adj = score_mask(m_thr)
        f_plain, pc_plain = score_mask(m_thr, adjusted=False)
        log.info(f"  [{pm}] per-group thresholds: adj-F0.5={f_adj:.5f} {pc_adj} | plain-F0.5={f_plain:.5f} {pc_plain}")
        m_rc = resolve_conflicts(s1_code, s23_code, P, m_thr)
        f_adj_rc, pc = score_mask(m_rc)
        f_plain_rc, _ = score_mask(m_rc, adjusted=False)
        log.info(f"  [{pm}] + conflict resolution: adj-F0.5={f_adj_rc:.5f} {pc} | plain={f_plain_rc:.5f}")
        use_rc = f_adj_rc >= f_adj
        cand_best = {'mode': 'threshold', 'prob_mode': pm, 'thresholds': {k: float(v) for k, v in thresholds.items()},
                     'default_threshold': float(t_glob), 'resolve_conflicts': bool(use_rc),
                     'score_adj': max(f_adj, f_adj_rc), 'score_plain': f_plain_rc if use_rc else f_plain}
        if best is None or cand_best['score_adj'] > best['score_adj']:
            best = cand_best
        if cfg.USE_EXPECTED_F05:
            for floor in (0.0, 0.3, 0.4):
                m_ef = expected_f05_select(s1_code, P, min_prob=0.05)
                if floor > 0:
                    m_ef &= P >= floor
                m_ef_rc = resolve_conflicts(s1_code, s23_code, P, m_ef)
                f_ef, _ = score_mask(m_ef)
                f_ef_rc, _ = score_mask(m_ef_rc)
                f_ef_plain, _ = score_mask(m_ef_rc if f_ef_rc >= f_ef else m_ef, adjusted=False)
                log.info(f"  [{pm}] expected-F0.5 (floor={floor}): adj {f_ef:.5f} | +conflicts {f_ef_rc:.5f} | plain {f_ef_plain:.5f}")
                if max(f_ef, f_ef_rc) > best['score_adj']:
                    best = {'mode': 'expected_f', 'prob_mode': pm, 'thresholds': ({c: float(floor) for c in thresholds} if floor > 0 else {}),
                            'ef_floor_default': float(floor), 'ef_min_prob': 0.05, 'default_threshold': float(t_glob),
                            'resolve_conflicts': bool(f_ef_rc >= f_ef), 'score_adj': max(f_ef, f_ef_rc), 'score_plain': f_ef_plain}

    P_best = prob_modes[best['prob_mode']]
    m_best = decide(s1_code, s23_code, P_best, group_pair, best)
    f_best_adj, pc_best_adj = score_mask(m_best)
    f_best, pc_best = score_mask(m_best, adjusted=False)
    for c in pd.unique(country_of_code):
        m_ff = apply_france_filter(m_best, s1_code, country_pair, meta_va['st_tset'].values,
                                   meta_va['num_first_rel'].values, target_country=c)
        f_ff, _ = score_mask(m_ff, adjusted=False)
        log.info(f"  France-style filter applied to {c}: plain {f_best:.5f} -> {f_ff:.5f}  (enabled={cfg.FRANCE_ENABLED})")
    log.info(f"  SELECTED decision config: {best}")
    log.info(f"  VAL macro-F0.5 plain = {f_best:.5f} {pc_best} | density-adjusted (test-like) = {f_best_adj:.5f} {pc_best_adj}")

    # ---------- error analysis of the selected decision (plain) ----------
    n_pred = np.bincount(s1_code[m_best], minlength=len(val_pos))
    tp = np.bincount(s1_code[m_best], weights=lab[m_best].astype(np.float64), minlength=len(val_pos))
    fp = n_pred - tp
    fn = n_true_val - tp
    f_ent = f05_from_counts(n_pred, n_true_val, tp)
    is_single = n_true_val == 0
    cats = {
        'singleton_correct_empty': int((is_single & (n_pred == 0)).sum()),
        'singleton_with_FP (score 0)': int((is_single & (n_pred > 0)).sum()),
        'nonsingleton_perfect': int((~is_single & (fp == 0) & (fn == 0)).sum()),
        'nonsingleton_FN_only': int((~is_single & (fp == 0) & (fn > 0)).sum()),
        'nonsingleton_FP_only': int((~is_single & (fp > 0) & (fn == 0)).sum()),
        'nonsingleton_FP_and_FN': int((~is_single & (fp > 0) & (fn > 0)).sum()),
        'nonsingleton_predicted_empty': int((~is_single & (n_pred == 0)).sum()),
    }
    loss = {k: 0.0 for k in cats}
    loss['singleton_with_FP (score 0)'] = float((is_single & (n_pred > 0)).sum())
    loss['nonsingleton_FN_only'] = float((1 - f_ent)[~is_single & (fp == 0) & (fn > 0)].sum())
    loss['nonsingleton_FP_only'] = float((1 - f_ent)[~is_single & (fp > 0) & (fn == 0)].sum())
    loss['nonsingleton_FP_and_FN'] = float((1 - f_ent)[~is_single & (fp > 0) & (fn > 0)].sum())
    total_loss = float((1 - f_ent).sum())
    log.info("  ERROR ANALYSIS (entities / share of total F0.5 loss):")
    for k in cats:
        log.info(f"    {k:32s} n={cats[k]:7,d}  loss={loss[k]:9.1f}  ({100 * loss[k] / max(total_loss, 1e-9):5.1f}%)")
    log.info(f"    total lost entity-equivalents: {total_loss:.1f} of {len(val_pos):,}")
    fp_distractor = int((m_best & (lab == 0) & meta_va['s23_distractor'].values).sum())
    fp_other = int((m_best & (lab == 0) & ~meta_va['s23_distractor'].values).sum())
    log.info(f"    false-positive pairs: on unmatched distractor rows={fp_distractor:,}  on rows owned by another S1={fp_other:,}")
    val_out = meta_va[['s1_pos', 's23_id', 'country', 'st_tset', 'num_first_rel', 's23_distractor', 'addr_empty_s23']].copy()
    val_out['s1_id'] = df_s1['entity_id'].values[val_s1_pos]
    val_out['prob'] = prob.astype(np.float32)
    for k, p in probs.items():
        val_out[f'prob_{k}'] = p.astype(np.float32)
    val_out['label'] = lab
    val_out['selected'] = m_best
    val_path = os.path.join(cfg.MODEL_DIR, 'val_predictions.parquet')
    try:
        val_out.to_parquet(val_path, index=False)
    except Exception:  # pyarrow missing
        val_path = val_path.replace('.parquet', '.csv')
        val_out.to_csv(val_path, index=False)
    log.info(f"  validation predictions saved -> {val_path}")

    # unseen-country (France) settings: start at mean seen main threshold + shift, then label-free calibration
    main_ts = [v for k, v in best['thresholds'].items() if '|' not in k] or [best['default_threshold']]
    noaddr_deltas = [best['thresholds'][k] - best['thresholds'][k.split('|')[0]] for k in best['thresholds'] if '|' in k and k.split('|')[0] in best['thresholds']]
    unseen = {
        'start_threshold': float(np.mean(main_ts) + cfg.UNSEEN_COUNTRY_THRESHOLD_SHIFT),
        'target_singleton_rate': float(cfg.UNSEEN_TARGET_SINGLETON_RATE),
        'max_threshold': float(cfg.UNSEEN_MAX_THRESHOLD),
        'noaddr_delta': float(np.mean(noaddr_deltas)) if noaddr_deltas else 0.0,
    }
    # observed empty rate on validation per country at the selected decision (sanity reference for inference)
    empty_rate_val = {c: float((n_pred[country_of_code == c] == 0).mean()) for c in pd.unique(country_of_code)}
    mean_links_val = {c: float(n_pred[country_of_code == c].mean()) for c in pd.unique(country_of_code)}
    log.info(f"  validation predicted-empty rate {empty_rate_val} | mean links {mean_links_val} | GT singleton rate {float(is_single.mean()):.4f}")

    config_out = {
        'feature_cols': feat_cols,
        'models': list(probs.keys()),
        'ensemble_weights': w,
        'decision': best,
        'per_bin_thresholds': bool(cfg.PER_BIN_THRESHOLDS),
        'unseen_country': unseen,
        'unseen_country_threshold': unseen['start_threshold'],
        'val_macro_f05': f_best,
        'val_macro_f05_density_adjusted': f_best_adj,
        'val_per_country': pc_best,
        'val_per_country_density_adjusted': pc_best_adj,
        'val_empty_rate': empty_rate_val,
        'val_mean_links': mean_links_val,
        'blocking_recall': recall_stats,
        'lgbm_best_iteration': int(booster.best_iteration),
        'n_train_rows': int(len(meta_tr)),
        'n_val_rows': int(len(meta_va)),
        'test_distractor_ratio': float(cfg.TEST_DISTRACTOR_RATIO),
    }
    with open(os.path.join(cfg.MODEL_DIR, 'model_config.json'), 'w') as f:
        json.dump(config_out, f, indent=2)
    log.info("=" * 70)
    log.info(f"TRAINING COMPLETE in {(time.time() - t0) / 60:.1f} min | VAL macro-F0.5 plain={f_best:.5f} "
             f"adjusted={f_best_adj:.5f} | config -> {cfg.MODEL_DIR}")
    return config_out


if __name__ == '__main__':
    main()
