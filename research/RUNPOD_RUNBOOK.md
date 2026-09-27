# RunPod runbook — pipeline v2

Pod: 1x A40 (48 GB), 50 GB RAM, 9 vCPU, CUDA 13.0, PyTorch image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`,
50 GB **ephemeral** container disk, persistent `/workspace`. Everything below lives under `/workspace`.

## 0. Setup (one command, idempotent — re-run after any pod restart)
```bash
# dataset first (7 TSVs, ~2.5 GB) so that these exist:
#   /workspace/dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
#   /workspace/dataset/test/test_source{1,2,3}.tsv
cd /workspace && git clone --branch deep-learning-sota https://github.com/iHamza14/Amazol-ML-Challenge.git   # first time only
bash /workspace/Amazol-ML-Challenge/setup_runpod.sh          # ~15 min incl. a 40k-entity smoke test; --no-smoke to skip
source /workspace/env.sh && cd /workspace/Amazol-ML-Challenge/student_resource/code/business_entity_resolution/src
```
The setup prints the config line `Environment: RunPod | jobs=9 (host cpus=N) | ram=50 GB | gpu=True`.
If `jobs` shows the host core count (e.g. 64/128), `export ER_N_JOBS=9` in `/workspace/env.sh` — otherwise LightGBM,
CatBoost and the forked feature workers oversubscribe the 9 vCPU and the run crawls or OOMs.
`CatBoost GPU: OK` means the CUDA path works; if the probe fails, training falls back to CPU automatically
(1500 iterations, bounded) and the ensemble is kept.

## 1. Quick run — 1 lakh entities (~35-50 min train, ~50-70 min inference)
```bash
ER_SAMPLE_S1=100000 nohup python -B train.py > /workspace/train_100k.log 2>&1 &
tail -f /workspace/train_100k.log
nohup python -B inference.py > /workspace/infer.log 2>&1 &
```
Lines to read (details in SAGEMAKER_RUNBOOK.md, identical pipeline):
- `BLOCKING RECALL at the shipped cap 100` and `recall@K by fused rank`
- `checkpoint saved -> .../cache/train_state` — from here `ER_RESUME=1 python -B train.py` restarts at the
  model stage in seconds (pod restart, CatBoost crash, or to re-run only the decision layer)
- `[mean] ... group thresholds`, `size-adaptive`, `CALIBRATED expected-F0.5`, `SELECTED decision config`
- `VAL macro-F0.5 plain = ... | density-adjusted (test-like) = ...` — the adjusted number tracks the leaderboard
- `ERROR ANALYSIS`, `unseen-country settings`
- inference: `unseen country 'france': ... calibrated ...`, `GUARDRAILS` (empty rate ~0.056-0.08, mean links 3.0-3.8), `PASS`

## 2. Full run (~2-3 h train with 400k entities, same inference)
```bash
nohup python -B train.py > /workspace/train_full.log 2>&1 &
```

## 3. Submit
`/workspace/Amazol-ML-Challenge/student_resource/output/matching_results.tsv` (5 uploads/day; ties go to the earlier upload).
Keep each submission's `models/model_config.json`. Candidate-file validator (memory-heavy, run separately):
```bash
cd /workspace/Amazol-ML-Challenge/student_resource
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir /workspace/dataset/test
```

## 4. Resources on this pod
| | full data | notes |
|---|---|---|
| RAM peak (train, 400k entities) | ~30 GB | S2/S3 of both countries cached between passes (auto when RAM ≥ 45 GB); matrices ~15 GB; LightGBM binning ~3 GB |
| RAM peak (inference) | ~20 GB | per-country streaming; validator run on the matching file only |
| VRAM (CatBoost) | < 10 GB | |
| Disk under /workspace | dataset 2.5 GB + checkpoint ~15 GB + models < 1 GB + outputs ~0.3 GB | container disk unused except pip |
| Time budget | quick run 1.5 h, full run 3.5 h incl. inference | deadline 23:59 IST |

## 5. Switches
Same as SAGEMAKER_RUNBOOK.md section 4, plus: `ER_RESUME=1` (restart at the model stage from the checkpoint),
`ER_CHECKPOINT=0` (do not write the ~15 GB checkpoint), `ER_GPU=0/1` (force CatBoost device),
`ER_CATBOOST_CPU_ITERS` (fallback iterations), `ER_N_JOBS` (pin CPU count).

## 5b. Incident: BrokenPipeError in ForkPoolWorker-* after "chunk 1/4: 20,000 S1 -> 2,000,000 candidates"
Workers print `BrokenPipeError` when the parent that owns their result pipe has died — i.e. the parent was
OOM-killed during the feature stage (`dmesg | tail -20` shows `Out of memory: Killed process ... python`).
Root cause: the parent held ~25 GB of Python-object frames (6.2M-row US S2/S3 frame ~11 GB + its raw copy,
2.2M-row S1 frame, raw 10.3M-row S2/S3 frame, Blocker matrices) and every forked worker adds its own working
set plus copy-on-write pages. Fixed in commit after `54d7e0f`: token strings interned (one object per distinct
token), raw text columns dropped after preprocessing, feature chunks halved (10k S1), the worker pool capped
by `TOTAL_RAM_GB - parent RSS - 6 GB` / 2.5 GB, RAM limit read from the cgroup (not the host), the reverse
product pruned of common tokens (19 min -> minutes per country) and the Blocker reused between passes.
Watch the new log lines `parent RSS after preprocessing ...` and `feature workers capped N -> M`.

Before re-running, check what the container really gives you:
```bash
cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max 2>/dev/null; nproc; free -g | head -2; dmesg | tail -5
```
If `cpu.max` shows `400000 100000` the pod really has 4 CPUs (the log's `jobs=4`); otherwise set
`ER_N_JOBS=9` in `/workspace/env.sh`. If `memory.max` is not 50 GB, the config line's `ram=` must match it.

## 6. Warnings
- **Pod restart wipes pip packages**: re-run `setup_runpod.sh --no-smoke` (2 min), then `source /workspace/env.sh`.
- **Do not run two trainings at once** on 50 GB.
- **CUDA 13 + CatBoost**: the wheel ships its own CUDA runtime; if the GPU probe fails the CPU fallback keeps the
  ensemble but adds ~20-40 min. `ER_NO_CATBOOST=1` is the escape hatch (LightGBM alone was within 0.0002 on validation).
- **Thread env**: `env.sh` sets `OMP_NUM_THREADS=1`; LightGBM/CatBoost/sparse products get `N_JOBS` explicitly, so this
  only stops BLAS oversubscription inside the forked feature workers. Do not remove it.
- **First run writes a ~15 GB checkpoint** to `/workspace/.../student_resource/cache/train_state`; delete it if disk gets tight.
- **The smoke-test score is optimistic** (sample entities overlap the training split); it only proves the end-to-end path.
