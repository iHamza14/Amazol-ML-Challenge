"""
Choose ER_PRUNE_TOPK (per-entity cap on the pruned candidate set) for inference, from the saved checkpoint.

Measures, on the 75k validation entities of the finished training run:
  - how many candidates per S1 survive the pruner threshold alone and with each cap K,
  - the macro-F0.5 (plain and density-adjusted) of the SELECTED decision when pairs outside the top-K by
    pruner score are removed (exact for the selected pairs; ignores second-order effects on group features),
  - the real prediction speed of the trained models on this machine (rows/s with N_JOBS threads),
and estimates the full-test inference duration for each K. Prints a recommendation: the smallest K whose
adjusted F0.5 is within 0.00005 of the best K that finishes by --finish-by-utc.

    python -B prune_cap_eval.py --finish-by-utc 16:45
"""
import os
import sys
import json
import time
import math
import argparse
import datetime as dt
import numpy as np
import pandas as pd

import config as cfg
from evaluate import score_selection

STATE_DIR = os.path.join(cfg.CACHE_DIR, 'train_state')
KS = [10, 15, 20, 25, 30, 40, 50, 60, 80, 0]      # 0 = no cap (threshold only)
# test pool sizes (S2/S3 rows per country, from the test files) and measured retrieval seconds per 10k-S1 chunk
# on the 16-vCPU pod, scaled from the training pools (US 6.19M rows: 85 s; India 4.13M rows: 42 s)
S23_TEST = {'us': 3_817_031, 'india': 4_717_565, 'france': 1_434_993}
SEC_PER_CHUNK = {'us': 85.0 * 3.82 / 6.19, 'india': 42.0 * 4.72 / 4.13, 'france': 42.0 * 1.43 / 4.13}
FEATURE_PAIRS_PER_S = 24000.0          # measured in the training run (21k US, 28k India, 15 workers)
OVERHEAD_MIN = 25.0                    # preprocessing + Blocker builds of 3 countries + output writing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--finish-by-utc', default='16:45', help='HH:MM UTC by which inference must be done')
    a = ap.parse_args()
    t0 = time.time()
    import lightgbm as lgb
    with open(os.path.join(cfg.MODEL_DIR, 'model_config.json'), encoding='utf-8') as f:
        mcfg = json.load(f)
    pr = mcfg.get('pruner')
    if not pr:
        sys.exit('model_config.json has no pruner section')
    with open(os.path.join(STATE_DIR, 'small.json')) as f:
        small = json.load(f)
    feat_cols = small['feat_cols']
    mi = [feat_cols.index(c) for c in pr['features']]
    print(f'loading checkpoint from {STATE_DIR} ...', flush=True)
    X_va = np.load(os.path.join(STATE_DIR, 'X_va.npy'), mmap_mode='r')
    y_va = np.load(os.path.join(STATE_DIR, 'y_va.npy'))
    val_pos = np.load(os.path.join(STATE_DIR, 'val_pos.npy'))
    n_true_all = np.load(os.path.join(STATE_DIR, 'n_true_all.npy'))
    meta = pd.read_parquet(os.path.join(STATE_DIR, 'meta_va.parquet'), columns=['s1_pos', 's23_id', 'country'])
    Xp = np.ascontiguousarray(X_va[:, mi], dtype=np.float32)
    print(f'  {len(y_va):,} validation pairs, {len(val_pos):,} entities ({time.time() - t0:.0f}s)', flush=True)

    pruner = lgb.Booster(model_file=os.path.join(cfg.MODEL_DIR, 'pruner.txt'))
    t = time.time()
    s = pruner.predict(Xp, num_threads=cfg.N_JOBS).astype(np.float32)
    pruner_rows_per_s = len(s) / max(1e-6, time.time() - t)
    country = meta['country'].values.astype(str)
    thr = np.array([pr['thresholds'].get(c, pr['default_threshold']) for c in country])
    passed = s >= thr
    s1p = meta['s1_pos'].values.astype(np.int64)
    o = np.lexsort((-s, s1p))
    so = s1p[o]
    st = np.r_[0, np.flatnonzero(np.diff(so)) + 1]
    rank = np.empty(len(so), dtype=np.int64)
    rank[o] = np.arange(len(so)) - np.repeat(st, np.diff(np.r_[st, len(so)]))
    print(f'  pruner scored in {time.time() - t:.0f}s ({pruner_rows_per_s:,.0f} rows/s)', flush=True)

    # --- selected decision on validation, joined to the pruner rank of each pair
    vp = pd.read_parquet(os.path.join(cfg.MODEL_DIR, 'val_predictions.parquet'),
                         columns=['s1_pos', 's23_id', 'label', 'selected', 's23_distractor'])
    key_all = pd.MultiIndex.from_arrays([s1p, meta['s23_id'].values.astype(str)])
    key_vp = pd.MultiIndex.from_arrays([vp['s1_pos'].values.astype(np.int64), vp['s23_id'].values.astype(str)])
    idx = key_all.get_indexer(key_vp)
    if (idx < 0).any():
        print(f'  WARNING: {int((idx < 0).sum())} decision-layer pairs not found in the checkpoint; treated as rank 0')
    vp_rank = np.where(idx >= 0, rank[np.maximum(idx, 0)], 0)
    code_of = -np.ones(int(max(val_pos.max(), s1p.max())) + 1, dtype=np.int64)
    code_of[val_pos] = np.arange(len(val_pos))
    vcode = code_of[vp['s1_pos'].values.astype(np.int64)]
    lab = vp['label'].values.astype(np.int8)
    sel = vp['selected'].values.astype(bool)
    fp_w = np.where(vp['s23_distractor'].values, cfg.TEST_DISTRACTOR_RATIO, 1.0)
    n_true = n_true_all[val_pos]
    ent_country = pd.Series(country).groupby(s1p).first()
    ent_c = ent_country.reindex(val_pos).values.astype(str)

    def f05(mask, w=None):
        f = score_selection(mask, vcode, lab, n_true, fp_weight=w)
        return float(f.mean()), {c: round(float(f[ent_c == c].mean()), 5) for c in pd.unique(ent_c) if c != 'nan'}

    # --- measured prediction speed of the trained models (this machine, N_JOBS threads)
    models, _ = None, None
    rows = min(200_000, len(y_va))
    Xs = np.ascontiguousarray(X_va[:rows], dtype=np.float32)
    per_pair_predict = 0.0
    b = lgb.Booster(model_file=os.path.join(cfg.MODEL_DIR, 'lgbm.txt'))
    t = time.time(); b.predict(Xs, num_threads=cfg.N_JOBS); dt_l = time.time() - t
    per_pair_predict += dt_l / rows
    msg = f'lightgbm {rows / dt_l:,.0f} rows/s'
    if os.path.exists(os.path.join(cfg.MODEL_DIR, 'catboost.cbm')) and 'catboost' in mcfg.get('models', []):
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier(); cb.load_model(os.path.join(cfg.MODEL_DIR, 'catboost.cbm'))
        t = time.time(); cb.predict_proba(Xs, thread_count=cfg.N_JOBS); dt_c = time.time() - t
        per_pair_predict += dt_c / rows
        msg += f', catboost {rows / dt_c:,.0f} rows/s'
    print(f'  prediction speed with {cfg.N_JOBS} threads: {msg}', flush=True)
    per_pair = per_pair_predict + 1.0 / FEATURE_PAIRS_PER_S

    # --- test-set sizes
    test_c = pd.read_csv(os.path.join(cfg.TEST_DIR, 'test_source1.tsv'), sep='\t', dtype=str,
                         keep_default_na=False, usecols=['country'])['country'].str.lower().str.strip()
    n_test = test_c.value_counts().to_dict()
    retrieval_s = sum(math.ceil(n / 10000) * SEC_PER_CHUNK.get(c, 30.0) for c, n in n_test.items())
    retrieved_pairs = sum(n * (200 if c == 'india' else 150) for c, n in n_test.items())
    base_s = OVERHEAD_MIN * 60 + retrieval_s + retrieved_pairs / max(1.0, pruner_rows_per_s)

    f_base, pc_base = f05(sel)
    fa_base, pca_base = f05(sel, fp_w)
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    hh, mm = map(int, a.finish_by_utc.split(':'))
    deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    print(f'\nvalidation macro-F0.5 of the selected decision: plain {f_base:.5f} {pc_base} | adjusted {fa_base:.5f} {pca_base}')
    print(f'test S1 per country: {n_test} | retrieval estimate {retrieval_s / 3600:.2f} h + overhead {OVERHEAD_MIN:.0f} min')
    print(f'\n{"K":>5} {"cand/S1 us":>10} {"india":>7} {"adjF0.5":>9} {"delta":>8} {"plainF0.5":>9} {"test pairs":>11} {"est hours":>9} {"finish UTC":>10}')
    rows_out = []
    for K in KS:
        keep = passed & ((rank < K) if K else True)
        cps = {c: float(keep[country == c].sum()) / max(1, len(np.unique(s1p[country == c]))) for c in ('us', 'india')}
        m = sel & ((vp_rank < K) if K else True)
        fa, pca = f05(m, fp_w)
        fp_, _ = f05(m)
        # test pairs: US and India at their measured rates; France uses India's (its threshold is the minimum)
        pairs = sum(n * cps.get(c, cps['india']) for c, n in n_test.items())
        est_s = base_s + pairs * per_pair
        finish = now + dt.timedelta(seconds=est_s)
        rows_out.append((K, fa, finish))
        print(f'{("none" if K == 0 else K):>5} {cps["us"]:>10.1f} {cps["india"]:>7.1f} {fa:>9.5f} {fa - fa_base:>+8.5f} {fp_:>9.5f} '
              f'{pairs / 1e6:>10.1f}M {est_s / 3600:>9.2f} {finish.strftime("%H:%M"):>10}')
    fitting = [r for r in rows_out if r[2] <= deadline]
    pool = fitting or [min(rows_out, key=lambda r: r[2])]
    best_f = max(r[1] for r in pool)
    rec = min((r for r in pool if r[1] >= best_f - 0.00005), key=lambda r: (r[0] == 0, r[0]))
    print(f'\nRECOMMENDED: ER_PRUNE_TOPK={rec[0]}  (adjusted F0.5 {rec[1]:.5f}; estimated finish {rec[2].strftime("%H:%M")} UTC; '
          f'{"fits" if fitting else "NOTHING FITS the deadline - fastest option shown"} before {a.finish_by_utc} UTC)')
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
