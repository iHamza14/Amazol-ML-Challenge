# Business Entity Resolution — pipeline v2

Candidate generation as a cascade (five-channel sparse TF-IDF retrieval at top-k 250, reciprocal-rank fusion,
cap 150/200 per S1 → stage-2 pruner on the retrieval scores) → 124 pair features → LightGBM + CatBoost →
entity-level decision layer (per-country / per-bin thresholds, rank ladder, one-S1-per-S2/S3 conflict rule,
address-empty unique-claimant rule, expected-F0.5 alternative) → label-free France calibration →
`matching_results.tsv` + `candidate_pairs.tsv`.

Everything is learned from the provided training data only. No external data, APIs, registries or
geocoding are used anywhere. Static normalisation tables (state abbreviations, street-type
abbreviations, legal-form abbreviations, ordinal words) live in `text_tables.py`.

## Reproduce end-to-end (two commands)

```bash
pip install -r ../requirements.txt          # pandas, numpy, scipy, scikit-learn, lightgbm, catboost,
                                            # rapidfuzz, unidecode, sparse_dot_topn, pyarrow
export DATA_ROOT=/path/to/dataset           # contains train/ and test/ (case-insensitive)
export PROJECT_ROOT=/path/for/models_cache_output   # models/, cache/, output/ are created here
export ER_N_JOBS=16                         # CPU count; set it explicitly inside containers that report the host's cores
cd src
ER_SAMPLE_S1=300000 ER_STATS_S1=60000 python train.py     # ~3 h on 16 vCPU + GPU: writes $PROJECT_ROOT/models/*
python prune_cap_eval.py --finish-by-utc HH:MM             # optional: measures the candidate-cap trade-off (2 min)
ER_PRUNE_TOPK=20 python inference.py                       # ~3.2 h on the full test set: writes $PROJECT_ROOT/output/
                                                           # matching_results.tsv, candidate_pairs.tsv, scored_pairs.parquet
                                                           # and runs utils/validate_submission.py on the matching file
```

The submitted run used exactly these settings (300 000 training entities, 60 000 statistics entities, 75 000
validation entities, each searched against the full S2/S3 pool of its country; inference with a cap of 20
candidates per S1 entity, `ER_PRUNE_TOPK=20`).

After the feature stage `train.py` writes a checkpoint (`cache/train_state`); `ER_RESUME=1 python train.py`
restarts at the model stage, and `ER_RESUME=1 ER_REUSE_MODELS=1 python train.py` re-runs only the pruner and
the decision layer with the saved models (minutes). `python redecide.py` re-runs only the decision layer on
the scored pairs saved by inference (`--france-threshold`, `--france-shift`, `--threshold-shift`,
`--consensus`; output `matching_results_<tag>.tsv`, the inference result is never overwritten).
`../../../../package_submission.sh` assembles and validates the submission zip.

Environment variables (all in `config.py`): `ER_N_JOBS`, `ER_SAMPLE_S1` (training entities, default 400000),
`ER_STATS_S1` (cap on the statistics split), `ER_MAX_CANDIDATES` (default cap 150) and `ER_BLOCK_BY_COUNTRY`
(per-country cap / depth overrides, default India cap 200), `ER_BLOCK_TOPK` (per-channel depths, default
250 each), `ER_BLOCK_FUSION` (`rrf` default, `minrank`), `ER_PRUNE=0` (score every retrieved pair),
`ER_PRUNE_RECALL` (default 0.999), `ER_FEATURE_WORKERS`, `ER_NO_CATBOOST=1`, `ER_GPU=0/1`, `ER_REVERSE=0`,
`ER_CACHE_S23=0/1`, `ER_CHECKPOINT=0`, `ER_LOCO=<country>` (leave-one-country-out diagnostics),
`ER_FRANCE_FILTER=1` (street-support filter for unseen countries; off by default because it hurt seen
countries on validation). Every blocking, pruner and decision setting used in training is persisted in
`models/model_config.json`, and inference applies the trained values regardless of the environment.

`inference.py` flags: `--threshold-shift 0.02` (add to every decision threshold; positive = more
precision), `--consensus` (minimum over LightGBM/CatBoost instead of the mean), `--no-conflicts`,
`--no-france-filter`, `--test-dir`, `--output-dir`, `--model-dir`.

## How the pieces fit

1. **Preprocessing** (`preprocess.py`, `translit.py`, `text_tables.py`): transliteration dictionary learned from
   ground-truth pairs, unidecode, dotted acronyms, d/b/a and "formerly" splits, domain-name detection with
   Viterbi word segmentation over the S1 vocabulary, guarded leetspeak repair, legal-form families; addresses
   parsed into components with admin units as single tokens and truncation-aware house numbers.
2. **Retrieval** (`blocking.py`): per country, five TF-IDF channels with `sparse_dot_topn` (top-k 250 each),
   a reverse channel (each S2/S3 record's top-5 S1; its top-1 becomes a candidate), reciprocal-rank fusion,
   cap 150 (US, France) / 200 (India). Measured on a 1.24 M-record India pool: the previous min-rank fusion at
   depths 60/40/60/40/80 saturated at 0.9928 recall even uncapped; depth 250 + RRF gives 0.9939 at cap 200.
3. **Stage-2 pruner** (`train.py` step 9b, applied in `inference.py`): LightGBM on the 16 retrieval meta
   features only; per-country threshold = keep 99.9 % of the retrieved true matches on validation. Survivors
   are the candidate set (`candidate_pairs.tsv`) and the only pairs that get features and model scores.
4. **Features** (`features.py`, 124) computed in forked workers that receive pickled slices of their rows.
5. **Models**: LightGBM + CatBoost (GPU), averaged; negatives subsampled by retrieval rank and re-weighted.
6. **Decision layer** (`decision.py`), selected on validation with a density-adjusted macro-F0.5 (false
   positives on distractor rows weighted 1.9x for the denser test pool): thresholds per country and address
   bin, rank ladder, conflict resolution, address-empty rescue, expected-F0.5 alternatives. France (unseen):
   raise-only calibration of the applied rule to the seen countries' predicted-empty rate.

## Files

| File | Role |
|---|---|
| `config.py` | paths (auto-detect RunPod / SageMaker / Windows), CPU and RAM detection inside containers, all hyper-parameters |
| `text_tables.py` | static normalisation tables (US/India/France admin names, address & legal abbreviations, ordinals, leetspeak map, blocking stopwords) |
| `translit.py` | Indic→Latin transliteration dictionary learned from ground-truth pairs (99.7% token coverage on validation) |
| `preprocess.py` | name/address normalisation (see above); copy-on-write-safe multiprocessing |
| `blocking.py` | per-country retrieval, reverse channel, reciprocal-rank fusion, per-country caps and depths |
| `features.py` | 124 vectorised pair features (rapidfuzz `cpdist`, sparse cosines, house-number relation codes, learned extra-token statistics, group-relative features); RAM-budgeted worker pool |
| `evaluate.py` | vectorised entity-level macro F0.5 and threshold sweeps |
| `decision.py` | thresholds, rank ladder, address-empty rescue, expected-F0.5 set selection, conflict resolution, unseen-country calibration |
| `france_filter.py` | optional unseen-country safety filter (off by default) |
| `train.py` | full training pipeline (splits, statistics, retrieval recall, features, models, pruner, decision selection, checkpoints) |
| `inference.py` | full inference pipeline: retrieval → pruner → features → models → decision → outputs + validator |
| `redecide.py` | decision layer only, on the saved scored pairs |
| `make_sample.py` | stratified sample dataset for smoke tests |
| `cross_encoder.py` | legacy experiment, not used |

## Outputs of `train.py` (in `models/`)
`lgbm.txt`, `catboost.cbm`, `pruner.txt`, `model_config.json` (feature list, ensemble weights, blocking
settings, pruner thresholds, decision config, validation scores, blocking recall), `translit.json`,
`seg_vocab.json`, `extra_token_stats.json`, `feature_importance.csv`, `val_predictions.parquet`.

## Outputs of `inference.py` (in `output/`)
`matching_results.tsv`, `candidate_pairs.tsv`, `scored_pairs.parquet` + `scored_s1.parquet` (inputs of
`redecide.py`).
