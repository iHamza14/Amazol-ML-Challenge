#!/usr/bin/env bash
# One-command RunPod setup for the entity-resolution pipeline.
#
#   bash setup_runpod.sh            # clone/update repo, install deps, verify GPU, build a sample, smoke-test
#   bash setup_runpod.sh --no-smoke # skip the sample build + smoke test
#
# Assumes the dataset TSVs are (or will be) at /workspace/dataset/{train,test}/ .
# Everything persistent lives under /workspace (the container disk is wiped on restart; pip packages are
# re-installed by re-running this script, which is idempotent).
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/iHamza14/Amazol-ML-Challenge.git}"
BRANCH="${BRANCH:-deep-learning-sota}"
WS="${WS:-/workspace}"
REPO="$WS/Amazol-ML-Challenge"
SRC="$REPO/student_resource/code/business_entity_resolution/src"
DATASET="${DATA_ROOT:-$WS/dataset}"
SMOKE=1
[[ "${1:-}" == "--no-smoke" ]] && SMOKE=0

echo "== [1/6] repository"
if [[ -d "$REPO/.git" ]]; then
  git -C "$REPO" fetch -q origin && git -C "$REPO" checkout -q "$BRANCH" && git -C "$REPO" pull -q --ff-only origin "$BRANCH" || true
else
  git clone -q --branch "$BRANCH" "$REPO_URL" "$REPO"
fi
git -C "$REPO" log --oneline -1

echo "== [2/6] python packages"
python -m pip install -q --upgrade pip
python -m pip install -q -r "$REPO/student_resource/code/business_entity_resolution/requirements.txt"

echo "== [3/6] environment file -> $WS/env.sh"
cat > "$WS/env.sh" <<EOF
export DATA_ROOT="$DATASET"
export PROJECT_ROOT="$REPO/student_resource"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
# uncomment to pin the CPU count if the auto-detection (cgroup quota) looks wrong in the config log line:
# export ER_N_JOBS=9
EOF
# shellcheck disable=SC1090
source "$WS/env.sh"
mkdir -p "$PROJECT_ROOT/models" "$PROJECT_ROOT/output" "$PROJECT_ROOT/cache"

echo "== [4/6] verification"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "WARNING: nvidia-smi failed - CatBoost will run on CPU"
python - <<'EOF'
import os, sys
sys.path.insert(0, os.path.join(os.environ['PROJECT_ROOT'], 'code', 'business_entity_resolution', 'src'))
import numpy, pandas, scipy, sklearn, lightgbm, catboost, rapidfuzz, unidecode, sparse_dot_topn, pyarrow
print(f"numpy {numpy.__version__} pandas {pandas.__version__} lightgbm {lightgbm.__version__} catboost {catboost.__version__} "
      f"rapidfuzz {rapidfuzz.__version__} sparse_dot_topn {sparse_dot_topn.__version__}")
import config as cfg   # prints the environment line: jobs (cgroup-aware), ram, gpu, paths
assert cfg.N_JOBS <= 64, f"N_JOBS={cfg.N_JOBS} looks like the host core count; set ER_N_JOBS"
# CatBoost GPU probe (a few seconds); the pipeline also falls back to CPU automatically
if cfg.HAS_GPU:
    try:
        from catboost import CatBoostClassifier
        import numpy as np
        X = np.random.rand(2000, 5).astype('float32'); y = (X[:, 0] > 0.5).astype(int)
        CatBoostClassifier(iterations=20, task_type='GPU', verbose=0).fit(X, y)
        print("CatBoost GPU: OK")
    except Exception as e:
        print(f"CatBoost GPU probe FAILED ({e}); training will fall back to CPU (bounded iterations)")
for d in (cfg.TRAIN_DIR, cfg.TEST_DIR):
    print(("OK   " if os.path.isdir(d) else "MISSING ") + d)
EOF

echo "== [5/6] dataset"
for f in train/train_source1.tsv train/train_source2.tsv train/train_source3.tsv train/train_ground_truth.tsv \
         test/test_source1.tsv test/test_source2.tsv test/test_source3.tsv; do
  if [[ -f "$DATASET/$f" ]]; then echo "OK      $f ($(du -h "$DATASET/$f" | cut -f1))"; else echo "MISSING $f"; fi
done
df -h "$WS" | tail -1

if [[ "$SMOKE" == "1" ]]; then
  echo "== [6/6] sample dataset + smoke test (~10-15 min)"
  cd "$SRC"
  if [[ ! -f "$WS/dataset_sample/train/train_source1.tsv" ]]; then
    python -B make_sample.py --n-s1 40000 --out "$WS/dataset_sample"
  fi
  DATA_ROOT="$WS/dataset_sample" PROJECT_ROOT="$WS/sample_project" ER_SAMPLE_S1=20000 python -B train.py 2>&1 | tee "$WS/smoke_train.log" | grep -E "BLOCKING RECALL|TRAIN matrix|val logloss|SELECTED|VAL macro|ERROR ANALYSIS|total lost|checkpoint|COMPLETE|Traceback|Error"
  DATA_ROOT="$WS/dataset_sample" PROJECT_ROOT="$WS/sample_project" python -B inference.py 2>&1 | tee "$WS/smoke_infer.log" | grep -E "unseen country|GUARDRAILS|empty_rate|PASS|FAIL|COMPLETE|Traceback|Error"
  python - <<'EOF'
import os, sys, pandas as pd
sys.path.insert(0, os.path.join(os.environ['PROJECT_ROOT'], 'code', 'business_entity_resolution', 'src'))
from evaluate import evaluate_prediction_sets, build_ground_truth_dict
ws = os.environ.get('WS', '/workspace')
gt = pd.read_csv(f'{ws}/dataset_sample/test/ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False)
pr = pd.read_csv(f'{ws}/sample_project/output/matching_results.tsv', sep='\t', dtype=str, keep_default_na=False)
s1 = pd.read_csv(f'{ws}/dataset_sample/test/test_source1.tsv', sep='\t', dtype=str, keep_default_na=False)
g = build_ground_truth_dict(gt); p = build_ground_truth_dict(pr)
print(f"SMOKE TEST macro-F0.5 on the sample test split: {evaluate_prediction_sets(p, g, s1.entity_id.tolist()):.5f} "
      f"(optimistic: sample entities overlap the training split)")
for c in s1.country.unique():
    ids = s1[s1.country == c].entity_id.tolist(); print(f"   {c:8s} {evaluate_prediction_sets(p, g, ids):.5f} (n={len(ids)})")
EOF
  echo "Smoke test done. Logs: $WS/smoke_train.log, $WS/smoke_infer.log"
fi

cat <<EOF

Setup complete. Next (see research/RUNPOD_RUNBOOK.md):
  source $WS/env.sh && cd $SRC
  ER_SAMPLE_S1=100000 nohup python -B train.py > $WS/train_100k.log 2>&1 &
  nohup python -B inference.py > $WS/infer.log 2>&1 &
EOF
