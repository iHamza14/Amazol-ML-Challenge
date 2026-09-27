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

## 1. Quick run — 1 lakh entities (~1 h train; inference on the full test ~4.5-5.5 h, blocking-bound:
## US 10k-S1 chunks take ~2 min each x 66, India ~1 min x 82, France ~0.5 min x 26; features ~1 h at 4-5 workers)
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
`ER_CATBOOST_CPU_ITERS` (fallback iterations), `ER_N_JOBS` (pin CPU count), `ER_FEATURE_WORKERS` (feature-stage
worker processes, default = jobs, capped by the RAM budget; `1` = main process only), `ER_STATS_S1` (cap on the
stats split; default min(300k, 0.75 x train) — 50000 is plenty and saves ~10 min of blocking per 25k entities),
`ER_BLOCK_BY_COUNTRY` (JSON per-country blocking overrides, e.g. `'{"india": {"max_candidates": 200, "topk_scale": 2}}'`;
the trained values are persisted in model_config.json and inference applies them).

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

## 5c. Incident 2 (same symptom, 47 GB cgroup): `feature workers capped 9 -> 5` then BrokenPipeError
Second full run died at Pass B+C US chunk 1/8 with the parent at 26 GB and 5 forked workers. Root cause
measured, not guessed: a forked worker that reads the parent's frames writes a refcount into every string it
touches, so each worker copied 4-5 GB of the parent's pages (200k pairs x ~15 object columns), not the 2.5 GB
budgeted. Fix (commit after `cc27587`): workers no longer touch the parent's frames at all. The parent builds
each task as a self-contained slice (the pairs plus compact copies of the S1 / S2-S3 rows they reference,
positions remapped) and sends it pickled, exactly the mechanism `preprocess_dataframe` has used at 9 workers on
this pod without incident. Measured on 250k pairs: identical output (all 124 columns), 33 MB pickled per 100k
pairs, ~10k pairs/s per worker. Budget: 3 GB per worker + 6 GB headroom against the live parent RSS (a >3x
margin), tasks of 100k pairs, `ER_FEATURE_WORKERS=1` forces the main process (~12k pairs/s) if ever wanted.
Log lines: `feature workers capped 9 -> N (parent rss ...)`. Since commit after `bd2ae3b` the budget is MEASURED: a
thread samples each worker's private memory (`/proc/<pid>/smaps_rollup` Private_Dirty, i.e. pages not shared with the
parent) once a second; the next chunk uses 1.5x the observed peak per worker (the 20k run showed the parent flat at
26.6 GB with 4 workers, so the real cost is far below the 3 GB static guess and the pool grows to all 9 CPUs).
Log line: `features: ... worker peak private X GB -> next budget Y GB/worker`. Memory monitor while a run is going:
```bash
nohup bash -c 'while true; do echo "$(date +%T) $(cat /sys/fs/cgroup/memory.current 2>/dev/null | awk "{printf \"%.1f GB\", \$1/1e9}")"; sleep 30; done' > /workspace/mem.log 2>&1 &
```

## 5d. Blocking depth, fusion and caps (commit after `a9e55ed`)
The second full run measured blocking recall at cap 100 of US 0.99192 and India 0.97669, far below the 40k-sample
numbers (candidate collisions grow with the pool). A probe on a 1.24M-record India pool (30% of full scale,
6,000 queries, 22,090 true matches) showed the base top-k union saturating at ~198 candidates (ceiling 0.9928
even uncapped), while top-k 250 on every channel with reciprocal-rank fusion reached 0.9928 at cap 150 and
0.9939 at cap 200 (vs 0.9890 for the old rule at cap 100). Retrieval depth is nearly free (48 s for 6,000
entities x 5 channels at k=250); the cap sets the feature cost. New defaults: top-k 250 everywhere, RRF fusion,
cap 150 (US, France) and 200 (India). Inference pairs rise from ~173M to ~300M (+50 min of features at 5
workers); training pairs ~22M for 1 lakh entities. The trained values are persisted in model_config.json.
Watch `BLOCKING RECALL at the shipped cap` in Pass A: expect US ≥ 0.994 and India ≥ 0.986 (both were lower before).

## 5e. Second pod (L40S, 16 vCPU, 188 GB RAM, 50 GB container disk, no network volume assumed)
Nothing in the code is tied to the 47 GB pod: CPUs and RAM are read from the cgroup, every pool is sized from the
live budget, GPU from nvidia-smi. On this pod all 16 workers run in every stage and the S2/S3 frames stay cached
between passes (auto when RAM >= 45 GB). Expect: blocking ~2x faster (sparse products use 16 threads), features
~100k pairs/s, LightGBM on 23M rows ~10 min, CatBoost GPU ~10 min. Recommended final run:
`ER_SAMPLE_S1=300000 ER_STATS_S1=60000` (~1.5 h training; matrices ~18 GB; checkpoint ~20 GB on disk).
Fresh setup on a pod (dataset copied pod-to-pod with scp over the exposed TCP port, see section 5e in the chat log):
```bash
cd /workspace && git clone -q --branch runpod https://github.com/iHamza14/Amazol-ML-Challenge.git code
python -m pip install -q -r /workspace/code/student_resource/code/business_entity_resolution/requirements.txt
cat > /workspace/env.sh <<'EOS'
export DATA_ROOT=/workspace/dataset PROJECT_ROOT=/workspace/er_project ER_N_JOBS=16 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
EOS
```

## 6. Warnings
- **Pod restart wipes pip packages**: re-run `setup_runpod.sh --no-smoke` (2 min), then `source /workspace/env.sh`.
- **Do not run two trainings at once** on 50 GB.
- **CUDA 13 + CatBoost**: the wheel ships its own CUDA runtime; if the GPU probe fails the CPU fallback keeps the
  ensemble but adds ~20-40 min. `ER_NO_CATBOOST=1` is the escape hatch (LightGBM alone was within 0.0002 on validation).
- **Thread env**: `env.sh` sets `OMP_NUM_THREADS=1`; LightGBM/CatBoost/sparse products get `N_JOBS` explicitly, so this
  only stops BLAS oversubscription inside the forked feature workers. Do not remove it.
- **First run writes a ~15 GB checkpoint** to `/workspace/.../student_resource/cache/train_state`; delete it if disk gets tight.
- **The smoke-test score is optimistic** (sample entities overlap the training split); it only proves the end-to-end path.
