# SageMaker runbook — pipeline v2

Instance: **ml.g5.2xlarge** (8 vCPU, 32 GB, A10G). If `train.py` is killed for memory on the full
data, restart on **ml.g5.4xlarge** (16 vCPU, 64 GB) — same code, add `ER_CACHE_S23=1`.

Verified locally (Windows laptop, 40k-S1 sample): `train.py` end-to-end in 11 min, `inference.py`
end-to-end + validator PASS, blocking recall 99.87% (US) / 99.72% (India) at 150 candidates.

## 0. One-time setup (JupyterLab terminal)
```bash
cd /home/ec2-user/SageMaker
git clone https://github.com/iHamza14/Amazol-ML-Challenge.git && cd Amazol-ML-Challenge && git checkout deep-learning-sota
# dataset so that these exist:
#   /home/ec2-user/SageMaker/dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
#   /home/ec2-user/SageMaker/dataset/test/test_source{1,2,3}.tsv
# fastest: aws s3 sync s3://<bucket>/dataset /home/ec2-user/SageMaker/dataset
source activate python3
pip install -r student_resource/code/business_entity_resolution/requirements.txt
nvidia-smi        # CatBoost trains on the GPU automatically on SageMaker
cd student_resource/code/business_entity_resolution/src
```

## 1. Quick run with 1 lakh S1 entities (~30-40 min) — recommended first
```bash
ER_SAMPLE_S1=100000 nohup python train.py > ../../../train_100k.log 2>&1 &
tail -f ../../../train_100k.log
```
`ER_SAMPLE_S1` = number of S1 entities whose candidate pairs form the training matrix (all S2/S3
rows are still indexed, so blocking is realistic). The stats split (extra-token statistics) and the
validation split scale with it (75% / 25% of it, capped at 300k / 120k).

Read these lines in the log:
- `BLOCKING RECALL (stats split, us|india)` and `recall@K by fused rank` — recall at K=80 (the
  default cap) vs 150. If recall@80 is more than ~0.001 below recall@150, rerun with
  `ER_MAX_CANDIDATES=120` (costs 1.5x feature time).
- `TRAIN matrix ... VAL matrix ...`, `lgbm/catboost/ensemble: val logloss / auc`.
- `[mean] ... group thresholds {...}`, `+ conflict resolution`, `expected-F0.5 (...)`, `[min] ...`
  (consensus of the two models), `SELECTED decision config`, then
  `VAL macro-F0.5 plain = ... | density-adjusted (test-like) = ...`. The **adjusted** number is the one
  that should track the leaderboard (public logs show plain validation overstates it by ~0.015).
- `ERROR ANALYSIS` — where the remaining loss sits (singletons with a false match vs FN vs FP) and how
  many false-positive pairs land on unmatched distractor rows vs rows owned by another S1.
- `validation predicted-empty rate {...} | mean links {...}` — the reference for the inference guardrails.
- `feature_importance.csv` and `val_predictions.parquet` land in `student_resource/models/`.

## 2. Inference on the full test set (expect 45-90 min with 8 vCPU)
```bash
nohup python inference.py > ../../../infer.log 2>&1 &
tail -f ../../../infer.log
```
Ends with the official validator (`PASS`). Read:
- `unseen country 'france': start threshold X -> calibrated Y (predicted-empty rate ..., target 0.0559)`:
  France's threshold is raised from the seen-country mean until France's empty rate hits the known
  singleton share. If Y hit the 0.97 cap, France is over-matching badly — inspect a sample of France
  predictions before uploading.
- `GUARDRAILS`: per country `empty_rate` (expect ~0.056, flagged outside 0.040-0.085) and `mean_links`
  (expect ~3.46, flagged outside 3.0-3.8). A flagged country means a broken threshold; do not upload.
- Variants worth a submission slot: default; `--consensus` (min of the two models, more precision);
  `--threshold-shift 0.02`.

## 3. Full training (optional, 1.5-3 h) once the 1-lakh run looks right
```bash
nohup python train.py > ../../../train_full.log 2>&1 &     # default ER_SAMPLE_S1=400000
```

## 4. Environment switches
| Variable | Default | Effect |
|---|---|---|
| `ER_SAMPLE_S1` | 400000 | S1 entities used for training pairs |
| `ER_MAX_CANDIDATES` | 80 | candidate cap per S1 (train and inference; keep identical) |
| `ER_N_JOBS` | all cores | preprocessing processes / feature-computation processes / LightGBM threads |
| `ER_NO_CATBOOST=1` | off | LightGBM only (faster; CatBoost GPU adds ~5-10 min and a small ensemble gain) |
| `ER_REVERSE=0` | on | disable the reverse channel (each S2/S3 record's top-5 S1 as rank feature + extra candidate); saves one S2/S3 x S1 sparse product per country (~5-15 min each at full scale) |
| `ER_DISTRACTOR_RATIO` | 1.9 | false-positive weight on unmatched rows during threshold selection (test pool density) |
| `ER_LOCO=us` | off | leave-one-country-out experiment: train on US only, validate on India as an UNSEEN country (France proxy). The log line `LOCO us->india: seen-threshold ... | calibrated ... | oracle ...` tells you how much the France calibration recovers. Run once with `ER_SAMPLE_S1=100000`; do not use its models for the submission. |
| `ER_MACRO_WEIGHTS=1` | off | weight positive pairs by 1/(true matches of the entity) so the loss follows the macro metric; compare `VAL macro-F0.5 density-adjusted` with and without |

The decision selection also evaluates (and logs) size-adaptive acceptance (`size-adaptive delta=...`: 2nd+ links
of an entity need a higher probability) and isotonic-calibrated expected-F0.5 (`CALIBRATED expected-F0.5`);
whichever variant wins on the density-adjusted metric is stored in `model_config.json` and applied by
`inference.py` automatically.
| `ER_FRANCE_FILTER=1` | off | enable the street-support post filter for unseen countries (hurt seen countries on validation) |
| `ER_CACHE_S23=1` | auto | keep preprocessed S2/S3 of all countries in RAM (needs 64 GB on full data) |
| `DATA_ROOT`, `PROJECT_ROOT` | auto | dataset root / where models/ and output/ are written |

`inference.py` flags: `--threshold-shift x`, `--no-conflicts`, `--no-france-filter`, `--max-candidates N`,
`--test-dir`, `--output-dir`, `--model-dir`.

## 5. Submit
Upload `student_resource/output/matching_results.tsv` (max 5 submissions per day). Keep the
`models/model_config.json` of every submission (version history is required).

## 6. Final package
```bash
cd /home/ec2-user/SageMaker/Amazol-ML-Challenge/student_resource
zip -r <team>_submission.zip output/matching_results.tsv output/candidate_pairs.tsv \
    code/business_entity_resolution/src code/business_entity_resolution/README.md \
    code/business_entity_resolution/requirements.txt Documentation_template.md
```
Fill the bracketed numbers in `Documentation_template.md` from the train/infer logs.
