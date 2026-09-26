# Business Entity Resolution — pipeline v2

Blocking (multi-channel sparse TF-IDF top-k) → 109 pair features → LightGBM + CatBoost →
entity-level decision layer (per-country thresholds / expected-F0.5 set selection, one-S1-per-S2/S3
conflict resolution) → unseen-country (France) filter → `matching_results.tsv` + `candidate_pairs.tsv`.

Everything is learned from the provided training data only. No external data, APIs, registries or
geocoding are used anywhere. Static normalisation tables (state abbreviations, street-type
abbreviations, legal-form abbreviations, ordinal words) live in `text_tables.py`.

## Reproduce end-to-end

```bash
pip install -r ../requirements.txt          # pandas, numpy, scipy, scikit-learn, lightgbm, catboost,
                                            # rapidfuzz, unidecode, sparse_dot_topn, pyarrow
export DATA_ROOT=/path/to/dataset           # contains train/ and test/  (auto-detected on SageMaker:
                                            # /home/ec2-user/SageMaker/dataset ; on the dev laptop: repo/Dataset ML Amazon)
cd src
python train.py                             # ~1-2 h on 8 vCPU: writes ../../../models/*
python inference.py                         # ~1 h: writes ../../../output/matching_results.tsv and candidate_pairs.tsv
                                            # and runs utils/validate_submission.py automatically
```

Useful environment variables: `ER_N_JOBS` (worker processes, default all cores), `ER_SAMPLE_S1`
(number of S1 entities used for training pairs; default 400000; e.g. 100000 for a 30-40 min run),
`ER_MAX_CANDIDATES` (candidate cap per S1, default 80, identical for train and inference),
`ER_NO_CATBOOST=1` (LightGBM only), `ER_FRANCE_FILTER=1` (enable the street-support filter for unseen
countries; off by default because it hurt seen countries on validation), `ER_CACHE_S23=1` (keep the
preprocessed S2/S3 frames of all countries in memory between passes — needs ~64 GB on the full data).

Verified end-to-end on a 40k-S1 local sample: `train.py` 11 min (val macro-F0.5 0.985 with only 20k
training entities), `inference.py` + official validator PASS.

`inference.py` flags: `--threshold-shift 0.02` (add to every decision threshold; positive = more
precision), `--consensus` (use the minimum over LightGBM/CatBoost instead of the mean — a precision
filter), `--no-conflicts`, `--no-france-filter`, `--max-candidates N`, `--test-dir`, `--output-dir`,
`--model-dir`.

Decision-layer selection in `train.py` is **density-adjusted**: false positives on unmatched (distractor)
S2/S3 rows are weighted 1.9x (the test pool has 5.75 S2/S3 rows per S1 vs 4.67 in train with the same
3.46 true matches per S1), because plain validation was found to overstate the leaderboard by ~0.015 in
every public log. Thresholds are per country and per bin (candidate address present / empty). France,
unseen in training, starts at the mean seen threshold + 0.02 and is then raised (never lowered) until
its predicted-empty rate reaches the generator's 5.6% singleton share.

## Files

| File | Role |
|---|---|
| `config.py` | paths (auto-detect SageMaker/Windows), all hyper-parameters |
| `text_tables.py` | static normalisation tables (US/India/France admin names, address & legal abbreviations, ordinals, leetspeak map, blocking stopwords) |
| `translit.py` | Indic→Latin transliteration dictionary learned from ground-truth pairs (99.7% token coverage on validation) |
| `preprocess.py` | name/address normalisation: dotted abbreviations, d/b/a, domain-collapsed names (Viterbi word segmentation with the S1 vocabulary), leetspeak, legal forms, comma-component address parsing, admin canonicalisation, number extraction |
| `blocking.py` | per-country multi-channel sparse TF-IDF retrieval with `sparse_dot_topn` (name tokens, name char-3grams, address tokens, number+street combos, joint); union with per-channel score/rank meta-features; reverse channel (each S2/S3 record's top-5 S1 → rank/score feature and its top-1 S1 as an extra candidate), `ER_REVERSE=0` disables |
| `features.py` | 109 vectorised pair features (rapidfuzz `cpdist`, sparse cosines, house-number relation codes, learned extra-token statistics, group-relative features) |
| `evaluate.py` | vectorised entity-level macro F0.5 and threshold sweeps |
| `decision.py` | thresholds, expected-F0.5 set selection, S2/S3 conflict resolution |
| `france_filter.py` | unseen-country safety filter (street similarity + number-relation support) |
| `train.py` | full training pipeline (splits, statistics, features, models, decision selection) |
| `inference.py` | full inference pipeline + output writing + validator |
| `cross_encoder.py` | optional Phase-2 multilingual cross-encoder (not used in the submitted run unless stated) |

## Outputs of `train.py` (in `models/`)
`lgbm.txt`, `catboost.cbm`, `model_config.json` (feature list, ensemble weights, decision config,
validation scores, blocking recall), `translit.json`, `seg_vocab.json`, `extra_token_stats.json`,
`feature_importance.csv`.
