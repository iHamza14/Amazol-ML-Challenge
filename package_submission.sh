#!/usr/bin/env bash
# Build the final submission zip and check it before upload.
#   bash package_submission.sh                       # uses output/matching_results.tsv
#   bash package_submission.sh matching_results_fr92.tsv   # a redecide.py variant as the matching file
# Layout required by the organizers:
#   submission.zip
#     output/matching_results.tsv, output/candidate_pairs.tsv
#     code/business_entity_resolution/...
#     Documentation_template.md
set -euo pipefail
CODE_ROOT="${CODE_ROOT:-/root/code}"                 # git clone of the repo
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/er_project}" # models/ cache/ output/
DATA_ROOT="${DATA_ROOT:-/workspace/dataset}"
OUT="${OUT:-/workspace/submission}"
MATCH_FILE="${1:-matching_results.tsv}"
SRC="$CODE_ROOT/student_resource"
TEST_DIR="$DATA_ROOT/test"; [[ -d "$TEST_DIR" ]] || TEST_DIR="$DATA_ROOT/Test"

echo "== inputs"
ls -la "$PROJECT_ROOT/output/$MATCH_FILE" "$PROJECT_ROOT/output/candidate_pairs.tsv"
n_s1=$(($(wc -l < "$TEST_DIR/test_source1.tsv") - 1))
n_m=$(($(wc -l < "$PROJECT_ROOT/output/$MATCH_FILE") - 1))
n_c=$(($(wc -l < "$PROJECT_ROOT/output/candidate_pairs.tsv") - 1))
echo "test S1 entities: $n_s1 | matching rows: $n_m | candidate rows: $n_c"
[[ "$n_m" == "$n_s1" && "$n_c" == "$n_s1" ]] || { echo "ROW COUNT MISMATCH - do not submit"; exit 1; }
grep -q "^\[Your Team Name\]\|\[List all team members\]\|\[from " "$SRC/Documentation_template.md" && \
  echo "WARNING: Documentation_template.md still has placeholders" || echo "documentation: no placeholders"

echo "== assemble"
rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT/output" "$OUT/code"
cp "$PROJECT_ROOT/output/$MATCH_FILE" "$OUT/output/matching_results.tsv"
cp "$PROJECT_ROOT/output/candidate_pairs.tsv" "$OUT/output/candidate_pairs.tsv"
cp -r "$SRC/code/business_entity_resolution" "$OUT/code/"
rm -rf "$OUT/code/business_entity_resolution/src/__pycache__"
cp "$SRC/Documentation_template.md" "$OUT/Documentation_template.md"

echo "== validator (matching + candidate files)"
python "$SRC/utils/validate_submission.py" --matching "$OUT/output/matching_results.tsv" \
  --candidate "$OUT/output/candidate_pairs.tsv" --test-dir "$TEST_DIR" | tail -15

echo "== candidate-set statistics (what the organizers will look at)"
python - "$OUT/output/candidate_pairs.tsv" "$OUT/output/matching_results.tsv" <<'EOF'
import sys
tot = n = mx = empty = 0
with open(sys.argv[1], encoding='utf-8') as f:
    next(f)
    for line in f:
        ids = line.rstrip('\n').split('\t')[1]
        k = len(ids.split(',')) if ids else 0
        tot += k; n += 1; mx = max(mx, k); empty += (k == 0)
print(f"candidates per S1: mean {tot / max(1, n):.2f}, max {mx}, entities with none {empty}")
tot = n = empty = 0
with open(sys.argv[2], encoding='utf-8') as f:
    next(f)
    for line in f:
        ids = line.rstrip('\n').split('\t')[1]
        k = len(ids.split(',')) if ids else 0
        tot += k; n += 1; empty += (k == 0)
print(f"matches per S1: mean {tot / max(1, n):.3f}, predicted-empty rate {empty / max(1, n):.4f} (training truth: 3.46 and 0.056)")
EOF

echo "== zip"
python -c "import shutil; shutil.make_archive('$OUT', 'zip', '$OUT')"
ls -la "$OUT.zip"
python -c "import zipfile; z=zipfile.ZipFile('$OUT.zip'); n=z.namelist(); print(len(n),'files'); print([x for x in n if not x.startswith('code/')])"
echo "DONE -> $OUT.zip  (download it from the Jupyter file browser; upload output/matching_results.tsv to the leaderboard first)"
