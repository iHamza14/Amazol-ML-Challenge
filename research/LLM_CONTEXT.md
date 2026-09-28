# Handover context: Amazon ML Challenge 2026, business entity resolution

You are taking over or assisting in the **final hours** of a competition. A full training run is in progress on a
cloud GPU pod. Your job is to help the user get a valid, high-scoring submission uploaded before the deadline,
without breaking what already works. Read sections 1 to 3 before advising anything.

---

## 1. The task, the metric, the rules

**Task.** Each Source-1 (S1) business record (name, address, country) must be linked to all matching records in
Source 2 and Source 3 (S2/S3). Countries: US, India, France. **France never appears in training**; it is test-only.

**Metric.** Macro F0.5 per S1 entity (precision weighted 4x recall), averaged over all S1 entities. An S1 with no
true matches (a "singleton", 5.6% of entities) scores 1 only if predicted empty, else 0.

**Leaderboard reference.** Public rank 50 is 0.98756; rank 1 is 0.9908. Ties go to the earlier upload.
**Max 5 leaderboard uploads per day.**

**Deadline.** 27 Sep 2026, 23:59 IST. **The user's own cutoff is 23:00 IST.** Do not propose anything that risks it.

**Hard rules.**
- Models must be MIT/Apache-2.0 licensed and at most 8B parameters. The pipeline uses LightGBM and CatBoost.
- **Strictly no external data lookup**: no APIs, geocoding or business registries.
- `matching_results.tsv`: tab-separated, header `source1_entity_id	matched_entity_ids`, one row per test S1 entity
  (all 1,732,544 of them), comma-separated S2/S3 ids, empty field allowed, no duplicates, only ids that exist.
- `candidate_pairs.tsv`: header `source1_entity_id	candidate_entity_ids`, same row rules. Per the README it must be
  **the exact set the matching model scores**, and matches must be a subset of candidates.
- Must pass `student_resource/utils/validate_submission.py`.
- **Final zip**: `output/` (both TSVs), `code/business_entity_resolution/`, `Documentation_template.md`.

**Late rule from the organizers** (final 10 hours): candidate_pairs.tsv is reviewed for the final ranking, and a
**smaller candidate set per S1 entity ranks higher**, beyond the leaderboard score. The pipeline was adapted: see
the stage-2 pruner in section 4.

---

## 2. Current status (as of about 16:10 IST, 27 Sep)

A single chained job is running on the pod: training, then inference automatically if training succeeds.

```
ER_SAMPLE_S1=300000 ER_STATS_S1=60000 python -B train.py > /workspace/train_300k.log 2>&1 && python -B inference.py > /workspace/infer.log 2>&1
```

Code commit on the pod: **c16f9fb** (branch `runpod`). Started 10:01 UTC (15:31 IST).
Pod log timestamps are **UTC; add 5:30 for IST**.

**Measured so far in this run:**
- Blocking recall on the held-out stats split: **US 0.99240** at cap 150, **India 0.98709** at cap 200
  (the previous settings gave 0.99185 and 0.97593).
- Pass B+C feature stage is healthy: 15 workers, about 21k pairs/s, parent RSS about 30 GB of 175 GB, 1.5 GB per
  worker. One 10k-entity US chunk takes about 2.6 min (about 85 s retrieval plus 70 s features).

**Expected timeline** (estimates from measured chunk times):

| milestone | UTC | IST |
|---|---|---|
| Pass B+C US done (23 chunks) | 11:20 | 16:50 |
| Pass B+C India done (15 chunks) | 11:55 | 17:25 |
| `checkpoint saved`, models, pruner, decision layer, `TRAINING COMPLETE` | about 12:30 | about 18:00 |
| inference auto-starts, then `INFERENCE COMPLETE` | 16:15 to 16:45 | 21:45 to 22:15 |
| package, download, upload | | 22:30 to 22:45 |

The margin to 23:00 IST is thin. Inference time is dominated by retrieval (about 175 chunks of 10k test entities:
US 66, India 82, France 26).

**Previous full-scale validation (20k training entities, old pod):** macro-F0.5 plain 0.97560, density-adjusted
0.97391. Loss shares: missed links 66%, false links 16%, singletons given a link 15%. Ensemble AUC 0.99989.
**Estimate for the current run:** density-adjusted validation about 0.980 to 0.985 (not yet measured).

---

## 3. Environment and operations

**Pod:** RunPod, NVIDIA L40S (46 GB VRAM), 16 vCPU, 188 GB RAM (cgroup limit read as 175 GB). The container reports
the host's **128 CPUs**, so `ER_N_JOBS=16` is mandatory (it is set in the env file). Jupyter terminal, user `root`.

**Paths.**
- Code: `/root/code` (a git clone of branch `runpod`). **Not** on `/workspace`: git clone onto the network volume
  fails with "chmod ... Operation not permitted".
- Source directory: `/root/code/student_resource/code/business_entity_resolution/src`.
- Data: `/workspace/dataset/{train,test}/*.tsv`, from the public HF dataset `akshatbakshi/amazon-ml-challenge-2026`
  (`tsv/` folder, byte-identical to the official files).
- Project root (models, cache, outputs): `/workspace/er_project/{models,cache,output}`.
- Logs: `/workspace/train_300k.log`, `/workspace/infer.log`, `/workspace/run.log`, `/workspace/mem.log`.
- Env file `/workspace/env.sh`:
  `export DATA_ROOT=/workspace/dataset PROJECT_ROOT=/workspace/er_project ER_N_JOBS=16 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1`

**The GitHub repo is private.** Pulling on the pod needs the user's token, which the user holds. Never write the
token into files. Pull form: `git pull -q --ff-only https://<user>:<token>@github.com/iHamza14/Amazol-ML-Challenge.git runpod`.

**Monitoring.**
```bash
tail -F /workspace/train_300k.log          # capital F: waits for the file
tail -F /workspace/infer.log
grep -E "features:|Traceback" /workspace/train_300k.log | tail -3
```

**Healthy training lines, in order:** `BLOCKING RECALL` per country, then `features: ... workers` per chunk, then
`TRAIN matrix`, `checkpoint saved`, LightGBM early stopping, CatBoost (GPU), `STEP 9b ... pruner`,
`pruner <country>: threshold ... keeps 0.999 ... N retrieved -> M candidates per S1`, `rank ladder best of 168`,
`noaddr rescue ADOPTED` or `rejected` (either is fine), `SELECTED decision config`,
`VAL macro-F0.5 plain = ... | density-adjusted = ...`, `unseen-country settings`, `TRAINING COMPLETE`.

**Healthy inference lines:** `candidate pruner loaded`, per chunk `pruner kept X of Y`, per country
`retrieved N pairs (x/S1) -> candidates scored M (y/S1)`, `unseen country 'france': ... calibrated`, `GUARDRAILS`
with empty rates near 0.06 and mean links 3.2 to 3.5, validator `PASS`, `INFERENCE COMPLETE`.

**Recovery commands.**
- Training fails **after** `checkpoint saved`: re-run from the checkpoint (minutes, not an hour):
  `ER_RESUME=1 ER_SAMPLE_S1=300000 ER_STATS_S1=60000 python -B train.py`
- Re-run only the pruner and decision layer with the saved models:
  `ER_RESUME=1 ER_REUSE_MODELS=1 python -B train.py`
- Inference alone: `python -B inference.py` (optionally `--test-dir`, `--output-dir`).
- Faster inference at a small recall cost: `ER_PRUNE_RECALL=0.998` must be set **at training time** (the threshold is
  stored in model_config.json), so apply it via `ER_RESUME=1 ER_REUSE_MODELS=1 ER_PRUNE_RECALL=0.998 python -B train.py`
  before inference. Use only if the timeline demands it.
- If CatBoost's GPU path fails, training falls back to CPU CatBoost automatically. `ER_NO_CATBOOST=1` skips it
  entirely (LightGBM alone was within 0.0002 on validation).

**Packaging** (after `INFERENCE COMPLETE`):
```bash
cd /workspace && rm -rf submission submission.zip && mkdir -p submission/output submission/code
cp /workspace/er_project/output/matching_results.tsv /workspace/er_project/output/candidate_pairs.tsv submission/output/
cp -r /root/code/student_resource/code/business_entity_resolution submission/code/
rm -rf submission/code/business_entity_resolution/src/__pycache__
cp /root/code/student_resource/Documentation_template.md submission/
python -c "import shutil; shutil.make_archive('/workspace/submission', 'zip', '/workspace/submission')"
```
Download via the Jupyter file browser. Upload `matching_results.tsv` to the leaderboard, then the zip.
**`Documentation_template.md` still has bracketed placeholders** (`[from inference log ...]`, team name, members):
fill them from the final logs before zipping.

---

## 4. Pipeline architecture (what the code does)

Entry points: `train.py` and `inference.py`. Supporting modules: `config.py`, `preprocess.py`, `translit.py`,
`text_tables.py`, `blocking.py`, `features.py`, `decision.py`, `evaluate.py`, `france_filter.py` (disabled).

1. **Preprocessing** (`preprocess.py`). Name normalisation includes an Indic-to-Latin transliteration dictionary
   *learned from training ground-truth pairs* (1,347 tokens, 99.7% validation coverage) plus unidecode. Also handled:
   dotted acronyms; d/b/a and "formerly" splits; domain-name detection and segmentation; Viterbi word segmentation
   with a vocabulary built from S1 names; guarded leetspeak repair; French elisions; legal-form canonicalisation into
   families. Address parsing covers comma components, admin units as single `zz...` tokens, and house numbers
   (truncation-aware, ranges, ordinals). Tokens are interned to save memory. Multiprocessing pool with
   `gc.freeze()` in the parent and `gc.disable()` in the workers, sized by the RAM budget.
2. **Stage 1, retrieval** (`blocking.py`). Per country, five TF-IDF channels are matched with `sparse_dot_topn`:
   name words, name char-3-grams, address words, (number|street word) combos, and a joint channel. **Top-k 250 per
   channel.** A reverse channel gives each S2/S3 record its top-5 S1 entities on the joint vector: the reverse rank
   is a feature, and the top-1 is added as a candidate. **Reciprocal-rank fusion** (1/(60+rank); reverse rank r
   counts as 2r) orders the union, capped at **150 per S1 (US, France)** and **200 (India)**. These settings came from
   a probe on a 1.24M-record India pool: the old min-rank fusion at small k saturated at 0.9928 recall uncapped.
3. **Stage 2, candidate pruner** (train step 9b). A LightGBM (63 leaves, 300 rounds) on the **16 blocking meta
   features only** scores every retrieved pair. The per-country threshold keeps **99.9% of the retrieved true
   matches** on validation. Only survivors get full features and model scores, and **candidate_pairs.tsv =
   survivors**. On a local sample this kept 35 to 80 candidates per S1 from 150 to 200 retrieved.
4. **Features** (`features.py`, 124). rapidfuzz comparisons (`cpdist`) on several name variants; TF-IDF cosines;
   soft token coverage; acronyms; short vs long edit counts; out-of-vocabulary fraction; learned "extra token"
   statistics (words the distractor generator inserts, e.g. "Holdings", have near-zero true-match rate); legal-form
   relation; address fuzzy scores; admin agreement; a 10-class house-number relation (equal, truncation, digit
   substitution, disjoint...); postal codes; all blocking meta; group-relative gaps and ranks within each S1's
   candidate set. Density-sensitive raw counts are dropped. Parallel: forked workers receive **pickled slices** of
   only the rows they need (never touching the parent's frames), and the pool is sized from measured per-worker
   private memory.
5. **Models.** LightGBM (255 leaves, lr 0.05, up to 4000 rounds, early stop 150) and CatBoost (GPU, depth 8, 4000
   iterations), averaged 0.5/0.5. Negatives are subsampled (top 25 by fused rank kept, plus 35% of the rest,
   reweighted). France-robustness masking hides extra-token statistics on 25% of the rows that have extras.
6. **Decision layer** (`decision.py`, selected on validation by a **density-adjusted macro-F0.5**, where false
   positives on distractor rows are weighted 1.9 because the test pool has 1.9x more distractors per S1):
   - per-country and per-address-bin thresholds (grid 0.30 to 0.95);
   - one-owner conflict resolution (each S2/S3 record goes to at most one S1);
   - **rank ladder**: separate deltas for the 2nd / 3rd / 4th+ link of an entity plus an optional link cap;
   - **address-empty unique-claimant rescue**, adopted only if it gains at least 0.0002;
   - expected-F0.5 set selection, raw and isotonic-calibrated, as alternatives;
   - `min` over the two models as an alternative probability mode.
7. **France (unseen)** at inference: start at the mean seen threshold + 0.02, then **raise only** until the
   predicted-empty rate reaches the seen countries' rate. It stops if links per entity fall more than 0.2 below the
   seen level, or after raising by 0.25. The 20k run calibrated France from 0.68 to 0.88. Unseen countries get the
   strictest seen rescue threshold.
8. **Outputs** go through a subset guarantee (matches within candidates), per-country guardrail logs, and the
   official validator on the matching file.

Everything from blocking settings to pruner thresholds to decision parameters is persisted in
`models/model_config.json`, and inference applies the **trained** values.

---

## 5. History you should know (to avoid repeating mistakes)

- **OOM kills** on a 47 GB pod (BrokenPipeError in ForkPoolWorker-*) were caused by forked workers dirtying
  copy-on-write pages of the parent's large Python-object frames. Fixed by pickled slices, gc.freeze/gc.disable, and
  RAM-budgeted pools. On the current 175 GB pod memory is not a concern (about 30 GB used).
- A **thread bug** crashed the first 300k attempt: the memory-probe thread's attribute `_stop` shadowed
  `threading.Thread._stop`. Fixed in c16f9fb, and the probe is now wrapped so it cannot fail the stage.
- Container quirks: the host CPU count leaks into `nproc` (hence `ER_N_JOBS`); `sed -i` and `git clone` fail on the
  `/workspace` network volume, so use `cat >` for files and clone into `/root`.
- Local sample scores (40k entities, about 0.987) **overstate** full-scale performance: collisions grow with pool
  size. Trust only full-scale validation numbers.

---

## 6. Guidance for you

- **Do not change code while the run is in progress** unless it has crashed. A `git pull` on the pod does not affect
  the running process, but a restart costs over an hour.
- The single most important rule tonight: **a valid submission must be uploaded before 23:00 IST.** If inference
  cannot finish in time, anything valid (every S1 row present, format correct) beats nothing.
- When the user pastes logs, compare against the healthy lines in section 3 and the timeline in section 2.
- The final number to watch is `VAL macro-F0.5 ... density-adjusted`. It is the best available estimate of the
  leaderboard score, though France (unlabelled) is the main uncertainty.
- Fuller operational detail: `research/RUNPOD_RUNBOOK.md` in the repo. Data findings: `research/DATA_FINDINGS.md`.
