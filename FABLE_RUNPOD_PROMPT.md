# PROMPT: Adapt Pipeline for RunPod A40

**Paste this into a new Claude chat. Attach the full repo.**

---

## Context

Read `CLAUDE_CONTEXT_TRANSFER.md` in the repo root for the full project history. Then read ALL files in `student_resource/code/business_entity_resolution/src/` — a second Claude session significantly evolved the pipeline beyond what's in the context transfer doc. The current v2 code includes:

- `config.py` — 175 lines, LightGBM + CatBoost ensemble, sparse TF-IDF blocking with `sparse_dot_topn`, reverse channel, per-country thresholds, distractor ratio adjustment, LOCO experiments, size-adaptive acceptance
- `preprocess.py` — 24KB, includes `SegVocab` (word segmentation), much more advanced than v1
- `blocking.py` — 18KB, 5-channel sparse TF-IDF blocking with reverse channel via `sparse_dot_topn` (NOT dict-of-sets inverted indexes anymore)
- `features.py` — 28KB, far more features than the original 40, includes `ExtraTokenStats`
- `train.py` — 34KB, 623-line full pipeline with stats/train/val splits, negative subsampling, LightGBM + CatBoost ensemble, decision layer selection
- `decision.py` — 7.7KB, expected-F0.5 set selection, conflict resolution, consensus, size-adaptive
- `translit.py` — 5KB, learned Indic→Latin transliteration table from ground-truth pairs
- `text_tables.py` — 118KB (likely lookup tables for transliteration/segmentation)
- `inference.py` — 16KB, full inference with France calibration, guardrails, validator
- `france_filter.py` — 2.5KB, street-support post filter (disabled by default — hurts seen countries)
- `evaluate.py` — 5.4KB, threshold sweep, f05_from_counts, score_selection

Also read `research/SAGEMAKER_RUNBOOK.md` — it has the verified performance numbers (blocking recall 99.87% US, 99.72% India at 150 candidates; quick run 11 min on 40k sample locally).

## The Compute Target

We are running on **RunPod**, NOT SageMaker. The pod specs:

```
1× NVIDIA A40
48 GB VRAM
50 GB RAM
9 vCPU
CUDA 13.0
PyTorch 2.8.0 (runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404)
50 GB container disk (ephemeral — wiped on restart)
Persistent storage mounted at /workspace
$0.50/hr
```

## What You Must Do

### 1. Fix config.py for RunPod

The current config.py has these problems on RunPod:

**Problem A — GPU detection:**
```python
# Line 115: CatBoost GPU only on SageMaker
task_type='GPU' if IS_SAGEMAKER else 'CPU'
```
On RunPod, `IS_SAGEMAKER` is False, so CatBoost falls back to CPU. Fix this by detecting CUDA availability:
```python
import torch
HAS_GPU = torch.cuda.is_available()
# Then use task_type='GPU' if HAS_GPU else 'CPU'
```
BUT: `torch` is a heavy import and may not be installed if the user skips the cross-encoder. Consider a lighter detection: check if `/dev/nvidia0` exists, or just check `IS_SAGEMAKER or os.path.exists('/dev/nvidia0')`.

**Problem B — Path detection:**
```python
# Lines 28-35: Only handles Windows and SageMaker explicitly
```
RunPod hits the `else` branch which uses a relative path. The user will set `DATA_ROOT=/workspace/dataset` via env var, so this actually works — but verify.

**Problem C — N_JOBS:**
```python
N_JOBS = int(os.environ.get('ER_N_JOBS', max(1, multiprocessing.cpu_count())))
```
This should auto-detect 9 vCPUs. Fine. But verify `sparse_dot_topn` and LightGBM both respect this.

### 2. Verify dependency compatibility

The `requirements.txt` needs:
```
pandas>=2.0,<3
numpy>=1.24,<3
scipy>=1.10
scikit-learn>=1.3
lightgbm>=4.0
catboost>=1.2
rapidfuzz>=3.8
unidecode>=1.3
sparse_dot_topn>=1.1
pyarrow>=14.0
tqdm>=4.65
```

**Flag:** CatBoost with CUDA 13.0 (very new as of Sep 2026). If `catboost` GPU mode crashes with a CUDA error, the code must gracefully fall back to CPU. Check if `config.py` handles this or if it'll crash the entire train.py run.

**Flag:** `sparse_dot_topn` C++ extension — must compile against the CUDA toolkit in the container. Should work since it's CPU-only, but verify the install succeeds.

### 3. Verify memory fits in 50 GB RAM

The SAGEMAKER_RUNBOOK says "Fits in 32 GB for the full data with defaults." The pipeline processes S2/S3 per-country (peak is ~6M US records). With 50 GB RAM we have generous headroom.

However: `ER_CACHE_S23=1` (keep all preprocessed S23 in RAM) needs ~64 GB and will NOT work. Make sure nothing in the code defaults to caching all S23 in 50 GB. Check if there's a `ER_CACHE_S23` env var check in preprocess.py or train.py.

### 4. Verify disk fits in 50 GB

Dataset: ~2.5 GB. Cache/intermediates (parquet, sparse matrices): ~10-15 GB. Models: ~1 GB. Logs: < 100 MB. Total: ~15-20 GB.

The container disk is 50 GB BUT is ephemeral. Everything must go on `/workspace` (persistent). Check that `PROJECT_ROOT` and `MODEL_DIR` and `OUTPUT_DIR` and `CACHE_DIR` all resolve to paths under `/workspace` when the user sets:
```bash
export DATA_ROOT=/workspace/dataset
export PROJECT_ROOT=/workspace/Amazol-ML-Challenge/student_resource
```

### 5. Create a simple setup script

Write a `setup_runpod.sh` script that:
1. Clones the repo to `/workspace`
2. Installs requirements
3. Sets all env vars
4. Verifies GPU, Python, and CUDA
5. Runs a micro test (10k S1 entities) to confirm everything works end-to-end

### 6. Think critically about these edge cases

- **What if CatBoost GPU fails?** The pipeline should still work with `ER_NO_CATBOOST=1` (LightGBM only). Is the ensemble code in train.py robust to this? Does it skip CatBoost gracefully or crash?
- **What if the pod restarts mid-training?** Are there checkpoints? Can training be resumed? (Probably not — but at least the dataset and code survive on /workspace.)
- **What about the cross-encoder (Phase 2)?** The container has PyTorch 2.8.0 pre-installed. sentence-transformers should install fine. But torch + sentence-transformers may eat 5-10 GB of disk for model downloads. Does 50 GB container disk handle this?
- **What about time?** Deadline is Sep 27 23:59 IST. It's now ~06:20 IST. That's ~17.5 hours. Quick run (40 min) + full training (2.5 hrs) + inference (1 hr) + iteration (2 hrs) = ~6 hrs minimum. Tight but doable. No room for debugging env issues — the setup script must work first try.

## Output Expected

1. **Modified `config.py`** — RunPod-aware GPU detection, no other behavior changes
2. **`setup_runpod.sh`** — One-command setup script for the RunPod pod
3. **Any other files that need RunPod-specific changes** — if any paths, imports, or resource assumptions break on RunPod
4. **A brief list of warnings** — things to watch for during the run

Do NOT rewrite the pipeline. Do NOT change the ML approach. The pipeline is proven and verified. Only adapt it for RunPod's environment. Minimal changes, maximum reliability.
