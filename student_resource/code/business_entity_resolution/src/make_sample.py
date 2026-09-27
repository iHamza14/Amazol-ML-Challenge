"""
Build a small, realistic sample dataset for smoke tests (same TSV layout as the real data).

    python make_sample.py --n-s1 40000 --out /workspace/dataset_sample

train/: the sampled S1 entities, ALL their true S2/S3 records, S2/S3 records that share a rare name
token with a sampled S1 (near-miss look-alikes, incl. records owned by S1 entities outside the sample,
which act like the test pool's extra distractors), and random unmatched records.
test/: the same S1/S2/S3 files with a share of one country's S1 entities relabelled as 'France' so the
unseen-country path (calibration, guardrails) is exercised; test/ground_truth.tsv allows self-scoring:

    python - <<'EOF'
    import pandas as pd, sys; sys.path.insert(0, '.')
    from evaluate import evaluate_prediction_sets, build_ground_truth_dict
    gt = pd.read_csv('/workspace/dataset_sample/test/ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False)
    pr = pd.read_csv('/workspace/sample_project/output/matching_results.tsv', sep='\t', dtype=str, keep_default_na=False)
    print(evaluate_prediction_sets(build_ground_truth_dict(pr), build_ground_truth_dict(gt), gt.source1_entity_id.tolist()))
    EOF
"""
import os
import re
import argparse
import collections
import numpy as np
import pandas as pd
from unidecode import unidecode

import config as cfg

STOP = set('the of and in at to for on by with from or an a as de la le les du des et en un une au aux road rd street st '
           'avenue ave lane ln drive dr court ct circle cir way floor fl suite ste building bldg unit apartment apt block blk '
           'near opposite opp behind north south east west po box nagar colony phase plot flat house sector village district '
           'dist tehsil taluk mandal post via main cross layout rue boulevard blvd bd place allee chemin impasse passage bis '
           'ter cedex bp inc corp llc ltd pvt co sarl sas sa eurl sasu private limited company corporation incorporated'.split())


def toks(x):
    x = unidecode(x).lower()
    x = re.sub(r'[^\w\s]', ' ', x)
    return [t for t in x.split() if t not in STOP and len(t) >= 2]


def load(p):
    return pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-s1', type=int, default=40000)
    ap.add_argument('--n-random', type=int, default=150000, help='random unmatched S2/S3 rows added as distractors')
    ap.add_argument('--fake-unseen-frac', type=float, default=0.15, help="share of one country's test S1 relabelled 'France'")
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)

    s1 = load(os.path.join(cfg.TRAIN_DIR, 'train_source1.tsv'))
    gt = load(os.path.join(cfg.TRAIN_DIR, 'train_ground_truth.tsv'))
    s2 = load(os.path.join(cfg.TRAIN_DIR, 'train_source2.tsv'))
    s3 = load(os.path.join(cfg.TRAIN_DIR, 'train_source3.tsv'))
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    matched_ids = set(t for m in gt.matched_entity_ids.values for t in m.split(',') if t)

    s1_samp = s1.sample(min(a.n_s1, len(s1)), random_state=a.seed)
    ids = set(s1_samp.entity_id)
    gt_samp = gt[gt.source1_entity_id.isin(ids)]
    keep = set(t for m in gt_samp.matched_entity_ids.values for t in m.split(',') if t)

    print("indexing S2/S3 names for look-alikes ...")
    idx = collections.defaultdict(list)
    for i, nm in enumerate(s23.business_name.values):
        for t in set(toks(nm)):
            idx[t].append(i)
    s23_ids = s23.entity_id.values
    for nm in s1_samp.business_name.values:
        for t in set(toks(nm)):
            lst = idx.get(t, [])
            if 0 < len(lst) < 2000:
                for i in lst[:50]:
                    keep.add(s23_ids[i])
    unmatched = s23.entity_id.values[~s23.entity_id.isin(matched_ids).values]
    keep |= set(rng.choice(unmatched, min(a.n_random, len(unmatched)), replace=False))
    s23_samp = s23[s23.entity_id.isin(keep)]

    out_tr = os.path.join(a.out, 'train')
    out_te = os.path.join(a.out, 'test')
    os.makedirs(out_tr, exist_ok=True)
    os.makedirs(out_te, exist_ok=True)
    cols = ['entity_id', 'business_name', 'business_address', 'country']
    s1_samp[cols].to_csv(os.path.join(out_tr, 'train_source1.tsv'), sep='\t', index=False)
    s23_samp[s23_samp.entity_id.str.startswith('S2-')][cols].to_csv(os.path.join(out_tr, 'train_source2.tsv'), sep='\t', index=False)
    s23_samp[s23_samp.entity_id.str.startswith('S3-')][cols].to_csv(os.path.join(out_tr, 'train_source3.tsv'), sep='\t', index=False)
    gt_samp[['source1_entity_id', 'matched_entity_ids']].to_csv(os.path.join(out_tr, 'train_ground_truth.tsv'), sep='\t', index=False)

    # test split: same records, with a share of one country's S1 (and their true records) relabelled as an unseen country
    t1 = s1_samp[cols].copy()
    t23 = s23_samp[cols].copy()
    countries = t1.country.unique()
    donor = countries[0]
    fake_ids = set(t1[t1.country == donor].entity_id.sample(frac=a.fake_unseen_frac, random_state=a.seed))
    fake_s23 = set(t for sid, m in zip(gt_samp.source1_entity_id, gt_samp.matched_entity_ids) if sid in fake_ids
                   for t in m.split(',') if t)
    t1.loc[t1.entity_id.isin(fake_ids), 'country'] = 'France'
    t23.loc[t23.entity_id.isin(fake_s23), 'country'] = 'France'
    t1.to_csv(os.path.join(out_te, 'test_source1.tsv'), sep='\t', index=False)
    t23[t23.entity_id.str.startswith('S2-')].to_csv(os.path.join(out_te, 'test_source2.tsv'), sep='\t', index=False)
    t23[t23.entity_id.str.startswith('S3-')].to_csv(os.path.join(out_te, 'test_source3.tsv'), sep='\t', index=False)
    gt_samp[['source1_entity_id', 'matched_entity_ids']].to_csv(os.path.join(out_te, 'ground_truth.tsv'), sep='\t', index=False)
    print(f"sample written to {a.out}: S1 {len(s1_samp):,}, S2/S3 {len(s23_samp):,} "
          f"(true matches {len(set(t for m in gt_samp.matched_entity_ids.values for t in m.split(',') if t)):,}); "
          f"test: {len(fake_ids):,} S1 relabelled as France")


if __name__ == '__main__':
    main()
