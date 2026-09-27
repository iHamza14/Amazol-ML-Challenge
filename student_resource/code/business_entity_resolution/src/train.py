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
from decision import (decide, resolve_conflicts, expected_f05_select, threshold_select, pair_groups,
                      size_adaptive, calibrate_unseen_threshold, pair_thresholds)
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


def subsample_negatives(label, fused_rank, rng):
    """
    Keep all positives, all candidates within the top-N of the S1's fused ranking (the hard negatives),
    and a random NEG_RANDOM_FRAC of the deeper negatives. Returns (keep_mask, sample_weight) where the
    randomly kept deep negatives get weight 1/NEG_RANDOM_FRAC so the models still estimate the full-pool
    probability (otherwise deep-rank pairs would look 1/frac more likely to match than they are).
    """
    pos = label.astype(bool)
    top = fused_rank < cfg.NEG_KEEP_ALL_TOP
    rnd = rng.random(len(label)) < cfg.NEG_RANDOM_FRAC
    keep = pos | top | rnd
    w = np.ones(len(label), dtype=np.float32)
    w[~pos & ~top & rnd] = 1.0 / cfg.NEG_RANDOM_FRAC
    return keep, w


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
STATE_DIR = os.path.join(cfg.CACHE_DIR, 'train_state')


def save_state(st):
    os.makedirs(STATE_DIR, exist_ok=True)
    t = time.time()
    for k in ('X_tr', 'y_tr', 'w_sub_all', 'X_va', 'y_va', 'val_pos', 'n_true_all', 's1_country', 's1_ids'):
        np.save(os.path.join(STATE_DIR, f'{k}.npy'), np.asarray(st[k]))
    st['meta_tr'].to_parquet(os.path.join(STATE_DIR, 'meta_tr.parquet'), index=False)
    st['meta_va'].to_parquet(os.path.join(STATE_DIR, 'meta_va.parquet'), index=False)
    with open(os.path.join(STATE_DIR, 'small.json'), 'w') as f:
        json.dump({'feat_cols': st['feat_cols'], 'n_s1': int(st['n_s1']), 'recall_stats': st['recall_stats']}, f)
    log.info(f"  checkpoint saved -> {STATE_DIR} ({time.time() - t:.0f}s; ER_RESUME=1 restarts from the model stage)")


def load_state():
    st = {}
    for k in ('X_tr', 'y_tr', 'w_sub_all', 'X_va', 'y_va', 'val_pos', 'n_true_all', 's1_country', 's1_ids'):
        st[k] = np.load(os.path.join(STATE_DIR, f'{k}.npy'), allow_pickle=True)
    st['meta_tr'] = pd.read_parquet(os.path.join(STATE_DIR, 'meta_tr.parquet'))
    st['meta_va'] = pd.read_parquet(os.path.join(STATE_DIR, 'meta_va.parquet'))
    with open(os.path.join(STATE_DIR, 'small.json')) as f:
        small = json.load(f)
    st.update(feat_cols=small['feat_cols'], n_s1=small['n_s1'], recall_stats=small['recall_stats'], t0=time.time())
    log.info(f"  checkpoint loaded <- {STATE_DIR}: train {st['X_tr'].shape}, val {st['X_va'].shape}")
    return st


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
    df_s1 = df_s1.drop(columns=['business_name', 'business_address', 'country'])   # raw text no longer needed
    gc.collect()
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
    s1_country = df_s1['country_norm'].values
    if cfg.LOCO_COUNTRY:
        # leave-one-country-out: learn everything on one seen country, validate on the other as if unseen
        pool_train = np.flatnonzero(s1_country == cfg.LOCO_COUNTRY)
        pool_val = np.flatnonzero(s1_country != cfg.LOCO_COUNTRY)
        if len(pool_train) == 0 or len(pool_val) == 0:
            raise SystemExit(f"ER_LOCO={cfg.LOCO_COUNTRY}: need S1 rows both in and outside that country")
        perm_t = rng.permutation(pool_train)
        perm_v = rng.permutation(pool_val)
        n_train = min(cfg.TRAIN_S1_ENTITIES, int(0.5 * len(pool_train)))
        n_stats = min(cfg.STATS_S1_MAX, int(0.75 * n_train), max(0, int(0.8 * len(pool_train)) - n_train))
        n_val_unseen = min(cfg.VAL_S1_ENTITIES, len(pool_val))
        n_val_in = min(cfg.VAL_S1_ENTITIES // 2, len(pool_train) - n_train - n_stats)   # in-country holdout
        train_pos = np.sort(perm_t[:n_train])
        stats_pos = np.sort(perm_t[n_train:n_train + n_stats])
        val_pos = np.sort(np.concatenate([perm_v[:n_val_unseen], perm_t[n_train + n_stats:n_train + n_stats + n_val_in]]))
        n_val = len(val_pos)
        log.info(f"STEP 5 (LOCO {cfg.LOCO_COUNTRY}): train={n_train:,} stats={n_stats:,} from {cfg.LOCO_COUNTRY}; "
                 f"val = {n_val_unseen:,} UNSEEN ({sorted(set(s1_country[perm_v[:n_val_unseen]]))}) + {n_val_in:,} in-country holdout")
    else:
        perm = rng.permutation(n_s1)
        n_train = min(cfg.TRAIN_S1_ENTITIES, int(0.6 * n_s1))
        n_stats = min(cfg.STATS_S1_MAX, int(0.75 * n_train), n_s1 - n_train)
        n_val = min(cfg.VAL_S1_ENTITIES, int(0.25 * n_train), n_s1 - n_train - n_stats)
        train_pos = np.sort(perm[:n_train])
        stats_pos = np.sort(perm[n_train:n_train + n_stats])
        val_pos = np.sort(perm[n_train + n_stats:n_train + n_stats + n_val])
        log.info(f"STEP 5: splits  train={n_train:,}  stats={n_stats:,}  val={n_val:,}")
    split = np.zeros(n_s1, dtype=np.int8)      # 0 unused, 1 train, 2 stats, 3 val
    split[train_pos] = 1
    split[stats_pos] = 2
    split[val_pos] = 3
    df_s23_raw['country_norm'] = df_s23_raw['country'].map(lambda x: str(x).lower().strip())
    countries = pd.unique(s1_country)

    acc = ExtraTokenAccumulator()
    train_X, train_y, train_w, train_meta = [], [], [], []
    val_X, val_y, val_meta = [], [], []
    feat_cols = None
    recall_stats = {}

    preprocessed_s23 = {}

    def get_s23(country):
        """Preprocess (or fetch cached) S2/S3 rows of one country."""
        if country not in preprocessed_s23:
            df_c = df_s23_raw[df_s23_raw['country_norm'] == country].copy()
            df_c = preprocess_dataframe(df_c, translit=trans, seg_vocab=seg, n_jobs=cfg.N_JOBS)
            # how many S1 entities of the country carry exactly this record's core name (chain / ambiguity signal)
            core_counts = df_s1.loc[s1_country == country, 'name_core'].value_counts()
            df_c['s1_core_count'] = df_c['name_core'].map(core_counts).fillna(0).astype(np.int32)
            df_c = df_c.drop(columns=['business_name', 'business_address', 'country'])   # raw text no longer needed
            gc.collect()
            log.info(f"  parent RSS after preprocessing {country}: {cfg.rss_gb():.1f} GB (limit {cfg.TOTAL_RAM_GB:.0f} GB)")
            preprocessed_s23[country] = df_c
        return preprocessed_s23[country]

    # keep the preprocessed S2/S3 of every country in RAM between Pass A and Pass B+C when memory allows
    # (~600 B/row -> ~6 GB for the full data): saves one full preprocessing per country
    auto_cache = '1' if (len(df_s23_raw) < 4_000_000 or cfg.TOTAL_RAM_GB >= 45) else '0'
    keep_s23_cached = os.environ.get('ER_CACHE_S23', auto_cache) == '1'
    # the per-country Blocker (5 vocabulary fits + the ~minutes reverse product) is always built once and reused
    # between Pass A and Pass B+C (~3 GB per country, dropped after its Pass B+C); re-preprocessing the S2/S3
    # frame instead is cheap (~3 min per country), so the frame itself is cached only when RAM allows
    blocker_cache = {}
    log.info(f"  S2/S3 frame cache between passes: {keep_s23_cached} (rows={len(df_s23_raw):,}, ram={cfg.TOTAL_RAM_GB:.0f} GB); Blocker cache: on")

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
        cap = cfg.max_candidates_for(country)
        # labels only (no feature cost): retrieve deeper than the cap so the recall curve is informative above it
        for _, cand, blocker in iter_candidate_chunks(df_s1, df_s23_c, s1_mask=mask_stats, max_candidates=max(200, cap),
                                                      blocker_cache=blocker_cache):
            lab = (owner_c[cand['s23_pos'].values] == cand['s1_pos'].values)
            in_cap = cand['fused_rank'].values < cap
            found += int((lab & in_cap).sum())
            true_fused_ranks.append(cand['fused_rank'].values[lab])
            acc.add(df_s1['name_core'].values[cand['s1_pos'].values[in_cap]].tolist(),
                    df_s23_c['name_core'].values[cand['s23_pos'].values[in_cap]].tolist(), lab[in_cap])
        rec = found / max(1, truth_total)
        recall_stats[country] = rec
        log.info(f"  BLOCKING RECALL at the shipped cap {cap} (stats split, {country}): {rec:.5f}  ({found:,}/{truth_total:,})")
        if true_fused_ranks:
            tfr = np.concatenate(true_fused_ranks)
            curve = {k: round(float((tfr < k).sum()) / max(1, truth_total), 5) for k in (10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200)}
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

        # ---------- Pass B+C: train and val splits in ONE blocking pass (one Blocker build per country) ----------
        # Features are per pair and group features are per S1 entity, so routing a chunk's rows by the split of
        # their S1 is exact.
        log.info(f"--- Pass B+C (train + val splits) country={country} ---")
        mask_bc = ((split == 1) | (split == 3)) & (s1_country == country)
        for _, cand, blocker in iter_candidate_chunks(df_s1, df_s23_c, s1_mask=mask_bc, blocker_cache=blocker_cache):
            if len(cand) == 0:
                continue
            feat = compute_features_parallel(cand, df_s1, df_s23_c, vecs=blocker.vec, extra_stats=extra_stats, n_jobs=cfg.FEATURE_WORKERS)
            if feat_cols is None:
                feat_cols = feature_columns(feat)
                log.info(f"  {len(feat_cols)} features: {feat_cols}")
            lab = (owner_c[cand['s23_pos'].values] == cand['s1_pos'].values).astype(np.int8)
            row_split = split[cand['s1_pos'].values]
            is_tr = row_split == 1
            is_va = row_split == 3
            if is_tr.any():
                keep, w_sub = subsample_negatives(lab, cand['fused_rank'].values, rng)
                keep &= is_tr
                train_X.append(feat.loc[keep, feat_cols].values.astype(FEAT_DTYPE))
                train_y.append(lab[keep])
                train_w.append(w_sub[keep])
                train_meta.append(pd.DataFrame({'s1_pos': cand['s1_pos'].values[keep], 'country': country}))
            if is_va.any():
                val_X.append(feat.loc[is_va, feat_cols].values.astype(FEAT_DTYPE))
                val_y.append(lab[is_va])
                val_meta.append(pd.DataFrame({
                    's1_pos': cand['s1_pos'].values[is_va], 's23_id': s23_ids_arr[cand['s23_pos'].values[is_va]],
                    'country': country, 'st_tset': feat['st_tset'].values[is_va],
                    'num_first_rel': feat['num_first_rel'].values[is_va],
                    's23_distractor': (owner_c[cand['s23_pos'].values[is_va]] < 0),
                    'addr_empty_s23': feat['f_addr_empty_s23'].values[is_va].astype(bool),
                }))
            del feat
        preprocessed_s23.pop(country, None)
        if blocker_cache is not None:
            blocker_cache.pop(country, None)
        del df_s23_c, owner_c
        gc.collect()

    del df_s23_raw
    gc.collect()

    def _stack(parts):
        """Concatenate float32 chunks into a preallocated matrix, freeing each chunk as it is copied
        (np.concatenate would hold both the parts and the result: 2x the peak on the full data)."""
        n_rows = sum(p.shape[0] for p in parts)
        if n_rows == 0:
            return np.zeros((0, len(feat_cols)), dtype=FEAT_DTYPE)
        out = np.empty((n_rows, parts[0].shape[1]), dtype=FEAT_DTYPE)
        i = 0
        while parts:
            p = parts.pop(0)
            out[i:i + len(p)] = p
            i += len(p)
            del p
        return out

    X_tr = _stack(train_X)
    y_tr = np.concatenate(train_y)
    w_sub_all = np.concatenate(train_w).astype(np.float32)
    meta_tr = pd.concat(train_meta, ignore_index=True)
    X_va = _stack(val_X)
    y_va = np.concatenate(val_y)
    meta_va = pd.concat(val_meta, ignore_index=True)
    del train_X, val_X
    gc.collect()
    log.info(f"TRAIN matrix: {X_tr.shape} positives={int(y_tr.sum()):,} ({y_tr.mean():.4f}) | "
             f"VAL matrix: {X_va.shape} positives={int(y_va.sum()):,}")

    # ---------- France-robustness masking: hide extra-token vocabulary for 15% of rows ----------
    col_idx = {c: i for i, c in enumerate(feat_cols)}
    # only rows that HAVE extra tokens can be in the 'unknown vocabulary' regime at inference (France)
    has_extra = X_tr[:, col_idx['x_extra_cnt']] > 0
    m = has_extra & (rng.random(len(X_tr)) < 0.25)
    X_tr[np.ix_(m, [col_idx['x_extra_min'], col_idx['x_extra_mean']])] = extra_stats.prior
    X_tr[m, col_idx['x_extra_known']] = 0.0
    log.info(f"  France-robustness masking applied to {int(m.sum()):,} of {int(has_extra.sum()):,} rows with extra tokens")

    # ---------- checkpoint: everything the model + decision stages need (pod restarts, CatBoost crashes,
    # decision-layer re-runs). ER_RESUME=1 restarts from here in seconds instead of ~1 h of features. ----------
    state = dict(X_tr=X_tr, y_tr=y_tr, w_sub_all=w_sub_all, meta_tr=meta_tr, X_va=X_va, y_va=y_va, meta_va=meta_va,
                 feat_cols=feat_cols, val_pos=val_pos, n_s1=n_s1, n_true_all=n_true_all, s1_country=s1_country,
                 s1_ids=df_s1['entity_id'].values, recall_stats=recall_stats, t0=t0)
    if cfg.CHECKPOINT_MATRICES:
        save_state(state)
    return model_and_decision(state)


def model_and_decision(st):
    """Model training + decision-layer selection from the checkpointed matrices (steps 9-11)."""
    X_tr, y_tr, w_sub_all, meta_tr = st['X_tr'], st['y_tr'], st['w_sub_all'], st['meta_tr']
    X_va, y_va, meta_va, feat_cols = st['X_va'], st['y_va'], st['meta_va'], st['feat_cols']
    val_pos, n_s1, n_true_all, s1_country, s1_ids = st['val_pos'], st['n_s1'], st['n_true_all'], st['s1_country'], st['s1_ids']
    recall_stats, t0 = st['recall_stats'], st['t0']
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)

    # ---------------- 9. models ----------------
    log.info("=" * 70)
    log.info("STEP 9: training models")
    import lightgbm as lgb
    # sample weights: subsampled deep negatives are up-weighted (1/NEG_RANDOM_FRAC) so probabilities stay
    # calibrated to the full candidate pool; macro weights (optional) multiply on top
    w_tr = w_sub_all.copy()
    if cfg.MACRO_WEIGHTS:
        # macro metric: every entity counts once -> positive pairs weighted 1/(true matches of the entity)
        nt = np.maximum(1, n_true_all[meta_tr['s1_pos'].values]).astype(np.float32)
        w_tr *= np.where(y_tr > 0, 1.0 / nt, 1.0).astype(np.float32)
        log.info(f"  macro sample weights on (mean positive weight {w_tr[y_tr > 0].mean():.3f})")
    log.info(f"  sample weights: {int((w_tr > 1).sum()):,} deep negatives x{1.0 / cfg.NEG_RANDOM_FRAC:.1f}")
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feat_cols, free_raw_data=False)
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
        from catboost import CatBoostClassifier
        cb_params = dict(cfg.CATBOOST_PARAMS)
        attempts = [cb_params]
        if cb_params.get('task_type') == 'GPU':
            # GPU failures (driver / CUDA runtime mismatch, e.g. very new CUDA on the pod) fall back to CPU
            # with a bounded number of iterations so the ensemble survives without eating the time budget
            attempts.append(dict(cb_params, task_type='CPU', iterations=cfg.CATBOOST_CPU_FALLBACK_ITERATIONS))
        for params in attempts:
            try:
                log.info(f"  CatBoost task_type={params['task_type']} iterations={params['iterations']} threads={params.get('thread_count')}")
                cb = CatBoostClassifier(**params)
                cb.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_va, y_va), use_best_model=True)
                cb.save_model(os.path.join(cfg.MODEL_DIR, 'catboost.cbm'))
                probs['catboost'] = cb.predict_proba(X_va)[:, 1]
                break
            except Exception as e:  # noqa
                log.warning(f"CatBoost ({params['task_type']}) failed: {e}")
        if 'catboost' not in probs:
            log.warning("CatBoost unavailable; continuing with LightGBM only")
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
    # mimic inference exactly: only pairs whose max-model probability >= DECISION_KEEP_PROB reach the decision layer
    p_max_va = np.max(np.stack([probs[k] for k in probs]), axis=0)
    keep_va = p_max_va >= cfg.DECISION_KEEP_PROB
    log.info(f"  decision layer sees {int(keep_va.sum()):,} of {len(keep_va):,} validation pairs (max-model prob >= {cfg.DECISION_KEEP_PROB})")
    meta_va = meta_va.loc[keep_va].reset_index(drop=True)
    y_va = y_va[keep_va]
    prob = prob[keep_va]
    probs = {k: p[keep_va] for k, p in probs.items()}
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
                     'default_threshold': float(t_glob), 'resolve_conflicts': bool(use_rc), 'extra_link_delta': 0.0,
                     'score_adj': max(f_adj, f_adj_rc), 'score_plain': f_plain_rc if use_rc else f_plain}
        if best is None or cand_best['score_adj'] > best['score_adj']:
            best = cand_best
        # size-adaptive acceptance: 2nd+ links of an entity need prob >= entity floor + delta
        t_pair = pair_thresholds(group_pair, thresholds, t_glob)
        for delta in [d for d in cfg.EXTRA_LINK_DELTA_GRID if d > 0]:
            m_sa = size_adaptive(s1_code, P, m_thr, t_pair + delta)
            if use_rc:
                m_sa = resolve_conflicts(s1_code, s23_code, P, m_sa)
            f_sa, _ = score_mask(m_sa)
            f_sa_plain, _ = score_mask(m_sa, adjusted=False)
            log.info(f"  [{pm}] size-adaptive delta={delta:.2f}: adj-F0.5={f_sa:.5f} plain={f_sa_plain:.5f}")
            if f_sa > best['score_adj']:
                best = dict(cand_best, extra_link_delta=float(delta), score_adj=f_sa, score_plain=f_sa_plain)
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

    # ---------- isotonic calibration (2-fold within validation, honest) + expected-F0.5 ----------
    # Boosted trees are systematically mis-calibrated (Niculescu-Mizil & Caruana 2005); the expected-F rule
    # needs calibrated probabilities. Calibrators are fitted per country on one half of the validation
    # entities and applied to the other half for scoring; the final calibrators (all validation pairs)
    # are saved for inference only if this variant wins.
    from sklearn.isotonic import IsotonicRegression

    def fit_iso(p, y):
        ir = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(p.astype(np.float64), y.astype(np.float64))
        return {'x': ir.X_thresholds_.astype(float).tolist(), 'y': ir.y_thresholds_.astype(float).tolist()}

    def apply_iso(cal, p):
        return np.interp(p, cal['x'], cal['y']).astype(np.float32)

    fold = (s1_code % 2).astype(bool)
    P_cal_by_pm = {}
    for pm, P in prob_modes.items():
        P_cal = P.copy()
        for c in pd.unique(country_pair):
            cm = country_pair == c
            for fa in (False, True):
                fit_m = cm & (fold == fa)
                app_m = cm & (fold != fa)
                if fit_m.sum() >= 1000 and app_m.any():
                    P_cal[app_m] = apply_iso(fit_iso(P[fit_m], lab[fit_m]), P[app_m])
        P_cal_by_pm[pm] = P_cal
        for floor in (0.0, 0.2, 0.3):
            m_ef = expected_f05_select(s1_code, P_cal, min_prob=0.05)
            if floor > 0:
                m_ef &= P_cal >= floor
            m_ef_rc = resolve_conflicts(s1_code, s23_code, P_cal, m_ef)
            f_ef, _ = score_mask(m_ef)
            f_ef_rc, _ = score_mask(m_ef_rc)
            use_rc = f_ef_rc >= f_ef
            f_ef_plain, pc_ef = score_mask(m_ef_rc if use_rc else m_ef, adjusted=False)
            log.info(f"  [{pm}] CALIBRATED expected-F0.5 (floor={floor}): adj {f_ef:.5f} | +conflicts {f_ef_rc:.5f} | plain {f_ef_plain:.5f} {pc_ef}")
            if max(f_ef, f_ef_rc) > best['score_adj']:
                best = {'mode': 'expected_f', 'prob_mode': pm, 'calibrated': True,
                        'thresholds': ({c: float(floor) for c in pd.unique(country_pair)} if floor > 0 else {}),
                        'ef_floor_default': float(floor), 'ef_min_prob': 0.05,
                        'default_threshold': float(best['default_threshold']), 'resolve_conflicts': bool(use_rc),
                        'extra_link_delta': 0.0, 'score_adj': max(f_ef, f_ef_rc), 'score_plain': f_ef_plain}
    if best.get('calibrated'):
        P_raw = prob_modes[best['prob_mode']]
        best['calibrators'] = {c: fit_iso(P_raw[country_pair == c], lab[country_pair == c]) for c in pd.unique(country_pair)}
        P_best = P_cal_by_pm[best['prob_mode']]          # honest (2-fold) calibrated probabilities for scoring
        log.info("  calibrated expected-F0.5 selected; per-country isotonic calibrators saved in model_config.json")
    else:
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

    # ---------- LOCO: how does the France strategy behave on a truly unseen country? ----------
    if cfg.LOCO_COUNTRY:
        seen = cfg.LOCO_COUNTRY
        seen_thr = {k: v for k, v in best['thresholds'].items() if k.split('|')[0] == seen}
        t_seen = float(seen_thr.get(seen, best['default_threshold']))
        noaddr_d = float(seen_thr.get(seen + '|noaddr', t_seen) - t_seen)
        for uc in [c for c in pd.unique(country_of_code) if c != seen]:
            ent = country_of_code == uc
            pairs = country_pair == uc
            n_u = int(ent.sum())
            remap = -np.ones(len(val_pos), dtype=np.int64)
            remap[np.flatnonzero(ent)] = np.arange(n_u)
            codes_u = remap[s1_code[pairs]]

            def unseen_score(thr_main):
                cfg_u = dict(best, thresholds={uc: thr_main, uc + '|noaddr': min(0.99, thr_main + noaddr_d)}, default_threshold=thr_main)
                m_u = decide(s1_code, s23_code, P_best, group_pair, cfg_u) & pairs
                f_u = score_selection(m_u, s1_code, lab, n_true_val)
                n_pred_u = np.bincount(s1_code[m_u], minlength=len(val_pos))
                return float(f_u[ent].mean()), float((n_pred_u[ent] == 0).mean()), float(n_pred_u[ent].mean())
            f_a, e_a, l_a = unseen_score(t_seen)
            t_cal, _ = calibrate_unseen_threshold(P_best[pairs], codes_u, n_u, t_seen + cfg.UNSEEN_COUNTRY_THRESHOLD_SHIFT,
                                                  cfg.UNSEEN_TARGET_SINGLETON_RATE, cfg.UNSEEN_MAX_THRESHOLD)
            f_b, e_b, l_b = unseen_score(t_cal)
            t_opt = float(best['thresholds'].get(uc, t_seen))
            f_c, e_c, l_c = unseen_score(t_opt)
            gt_empty = float((n_true_val[ent] == 0).mean()) if n_u else 0.0
            log.info(f"  LOCO {seen}->{uc}: seen-threshold {t_seen:.3f}: F0.5={f_a:.5f} empty={e_a:.4f} links={l_a:.3f} | "
                     f"calibrated {t_cal:.3f}: F0.5={f_b:.5f} empty={e_b:.4f} links={l_b:.3f} | "
                     f"oracle {t_opt:.3f}: F0.5={f_c:.5f} empty={e_c:.4f} links={l_c:.3f}  (GT empty rate {gt_empty:.4f})")

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
    val_out['s1_id'] = s1_ids[val_s1_pos]
    val_out['prob'] = P_best.astype(np.float32)          # the probability the selected decision actually used
    val_out['prob_mean'] = prob.astype(np.float32)
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

    # observed empty rate on validation per country at the selected decision (sanity reference for inference)
    empty_rate_val = {c: float((n_pred[country_of_code == c] == 0).mean()) for c in pd.unique(country_of_code)}
    mean_links_val = {c: float(n_pred[country_of_code == c].mean()) for c in pd.unique(country_of_code)}
    log.info(f"  validation predicted-empty rate {empty_rate_val} | mean links {mean_links_val} | GT singleton rate {float(is_single.mean()):.4f}")
    # unseen-country (France) settings: start at mean seen main threshold + shift, then label-free calibration.
    # The F0.5-optimal decision predicts empty for MORE entities than the true singleton share (precision-weighted
    # metric), so the calibration target is the seen countries' predicted-empty rate, floored by the GT share.
    main_ts = [v for k, v in best['thresholds'].items() if '|' not in k] or [best['default_threshold']]
    noaddr_deltas = [best['thresholds'][k] - best['thresholds'][k.split('|')[0]] for k in best['thresholds'] if '|' in k and k.split('|')[0] in best['thresholds']]
    unseen = {
        'start_threshold': float(np.mean(main_ts) + cfg.UNSEEN_COUNTRY_THRESHOLD_SHIFT),
        'target_singleton_rate': float(max(cfg.UNSEEN_TARGET_SINGLETON_RATE, np.mean(list(empty_rate_val.values())))),
        'max_threshold': float(cfg.UNSEEN_MAX_THRESHOLD),
        'max_raise': 0.20,                                       # never raise more than this above the start
        'seen_mean_links': float(np.mean(list(mean_links_val.values()))),   # stop raising when links fall well below
        'noaddr_delta': float(np.mean(noaddr_deltas)) if noaddr_deltas else 0.0,
    }
    log.info(f"  unseen-country settings: {unseen}")

    config_out = {
        'feature_cols': feat_cols,
        'models': list(probs.keys()),
        'ensemble_weights': w,
        'decision': best,
        'seen_countries': sorted(str(c) for c in pd.unique(country_of_code)),
        'keep_prob': float(cfg.DECISION_KEEP_PROB),
        'blocking': {'max_candidates': int(cfg.BLOCK_MAX_CANDIDATES), 'use_reverse': bool(cfg.USE_REVERSE_BLOCKING),
                     'reverse_topk': int(cfg.REVERSE_TOPK), 'topk': dict(cfg.BLOCK_TOPK),
                     'by_country': dict(cfg.BLOCK_BY_COUNTRY), 'fusion': cfg.BLOCK_FUSION, 'rrf_c': float(cfg.BLOCK_RRF_C),
                     'min_score': float(cfg.BLOCK_MIN_SCORE), 'max_df_frac': float(cfg.BLOCK_MAX_DF_FRAC)},
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
    if cfg.RESUME_FROM_CHECKPOINT:
        model_and_decision(load_state())
    else:
        main()
