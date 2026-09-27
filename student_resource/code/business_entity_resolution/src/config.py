"""
Central configuration for the Entity Resolution pipeline (v2).

Auto-detects local Windows vs SageMaker environment. Every hyper-parameter of the
pipeline lives here so experiments are reproducible from a single file.

Environment overrides (all optional):
  DATA_ROOT      root folder containing train/ and test/ sub-folders
  PROJECT_ROOT   folder that will receive output/ and models/
  ER_N_JOBS      number of worker processes / threads (default: all cores)
  ER_SAMPLE_S1   if set (int), train.py only uses this many S1 entities (quick runs)
"""
import os
import json
import platform
import logging
import multiprocessing

# BLAS/OpenMP threads: the feature stage forks N_JOBS worker processes; a multi-threaded BLAS inside each
# would oversubscribe the CPUs. LightGBM / CatBoost / sparse_dot_topn get their thread counts explicitly.
# (Effective only if set before numpy is imported — setup_runpod.sh also exports it.)
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

# ============================================================
# Environment Detection
# ============================================================
IS_SAGEMAKER = os.path.exists('/home/ec2-user/SageMaker')
# RunPod sets RUNPOD_POD_ID; pods without a network volume may still mount /workspace on the container disk
IS_RUNPOD = (bool(os.environ.get('RUNPOD_POD_ID')) or os.path.isdir('/workspace')) and not IS_SAGEMAKER
IS_WINDOWS = platform.system() == 'Windows'
_HERE = os.path.dirname(os.path.abspath(__file__))


def _detect_cpus():
    """
    CPUs actually available to this process. Inside a container `cpu_count()` reports the HOST's cores
    (e.g. 128 on a RunPod node with a 9-vCPU quota); using it would start 128 LightGBM threads and 128
    forked feature workers. Order of trust: ER_N_JOBS > cgroup v2/v1 quota > sched affinity > cpu_count.
    """
    env = os.environ.get('ER_N_JOBS')
    if env:
        return max(1, int(env))
    n = multiprocessing.cpu_count()
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    try:
        with open('/sys/fs/cgroup/cpu.max') as f:          # cgroup v2: "<quota> <period>" or "max <period>"
            q, p = f.read().split()
            if q != 'max':
                n = min(n, max(1, int(int(q) / int(p) + 0.5)))
    except (OSError, ValueError):
        try:
            with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us') as f:   # cgroup v1
                q = int(f.read())
            with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us') as f:
                p = int(f.read())
            if q > 0:
                n = min(n, max(1, int(q / p + 0.5)))
        except (OSError, ValueError):
            pass
    return max(1, n)


def _total_ram_gb():
    """RAM available to this process: min(physical, cgroup limit). Inside a container /proc/meminfo shows the
    HOST's memory (hundreds of GB) while the cgroup limit (e.g. 50 GB on RunPod) is what the OOM killer enforces."""
    ram = 0.0
    try:
        ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024 ** 3
    except (AttributeError, ValueError, OSError):
        pass
    for path in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            with open(path) as f:
                v = f.read().strip()
            if v.isdigit() and int(v) < (1 << 60):
                lim = int(v) / 1024 ** 3
                ram = min(ram, lim) if ram else lim
        except OSError:
            continue
    return ram


def rss_gb():
    """Resident memory of the current process in GB (Linux; 0.0 elsewhere)."""
    try:
        with open('/proc/self/statm') as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf('SC_PAGE_SIZE') / 1024 ** 3
    except (OSError, ValueError, AttributeError, IndexError):
        return 0.0


# ============================================================
# Paths
# ============================================================
if 'DATA_ROOT' in os.environ:
    DATA_ROOT = os.environ['DATA_ROOT']
elif IS_SAGEMAKER:
    DATA_ROOT = '/home/ec2-user/SageMaker/dataset'
elif IS_RUNPOD:
    DATA_ROOT = '/workspace/dataset'
elif IS_WINDOWS:
    DATA_ROOT = r'D:\Downloads\Amazol-ML-Challenge\Dataset ML Amazon'
else:
    DATA_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..', 'dataset'))

if 'PROJECT_ROOT' in os.environ:
    PROJECT_ROOT = os.environ['PROJECT_ROOT']
else:
    # .../student_resource/code/business_entity_resolution/src -> .../student_resource
    PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))


def _find_subdir(root, name):
    """Find subdirectory case-insensitively (handles Train vs train)."""
    if not os.path.isdir(root):
        return os.path.join(root, name)
    for entry in os.listdir(root):
        if entry.lower() == name.lower():
            return os.path.join(root, entry)
    return os.path.join(root, name)


TRAIN_DIR = _find_subdir(DATA_ROOT, 'train')
TEST_DIR = _find_subdir(DATA_ROOT, 'test')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')

# ============================================================
# Compute
# ============================================================
N_JOBS = _detect_cpus()
TOTAL_RAM_GB = _total_ram_gb()
RANDOM_SEED = 42

# ============================================================
# Blocking (sparse TF-IDF top-k retrieval, per country)
# ============================================================
# Each channel retrieves its own top-k per S1 entity; candidates are the union.
# Top-k per channel. Retrieval depth is nearly free (the sparse product dominates, not the top-n selection:
# 6,000 India entities x 5 channels at k=250 against 1.24M records took 48 s on a laptop) while the CAP sets
# the feature cost, so every channel retrieves deep and the fusion rule decides what enters the cap.
# Measured on the 1.24M-record India pool (true-match recall within the cap, RRF fusion):
#   base k (60/40/60/40/80) cap 100: 0.98895 | cap 150: 0.99144      <- the union saturates at ~198 candidates
#   k=250 all channels      cap 100: 0.99099 | cap 150: 0.99280 | cap 200: 0.99389 | cap 300: 0.99538
# ER_BLOCK_TOPK=60,40,60,40,80 restores the old depths (order: name_tok,name_chr,addr_tok,addr_num,joint).
_TOPK_ENV = [int(x) for x in os.environ.get('ER_BLOCK_TOPK', '').split(',') if x.strip()]
BLOCK_TOPK = {
    'name_tok': 250,     # word unigrams of core name (IDF weighted)
    'name_chr': 250,     # char 3-grams of space-less core name (typos, leetspeak, domain-collapse)
    'addr_tok': 250,     # address words + numbers
    'addr_num': 250,     # (house-number, street-word) combos -- very precise
    'joint':    250,     # name + address tokens together (best overall ranking)
}
if len(_TOPK_ENV) == 5:
    BLOCK_TOPK = dict(zip(['name_tok', 'name_chr', 'addr_tok', 'addr_num', 'joint'], _TOPK_ENV))
# Fusion of the channel rankings into one candidate order per S1: 'rrf' (reciprocal-rank fusion, sum over
# channels of 1/(C + rank); the reverse channel's rank r counts as 2r) or 'minrank' (previous rule: best
# single-channel rank, ties by summed score). RRF won every uniform-k configuration measured (see above).
BLOCK_FUSION = os.environ.get('ER_BLOCK_FUSION', 'rrf')
BLOCK_RRF_C = 60.0
BLOCK_MIN_SCORE = 0.08           # cosine below this is never a candidate
BLOCK_MAX_DF_FRAC = 0.03         # drop tokens present in > 3% of S23 docs of that country
# Hard cap per S1 entity after channel union. The union almost always exceeds the cap, so the cap
# directly sets feature cost (pairs = S1 x cap). Measured on a 40k-entity sample (true-pair recall by
# fused rank): US 99.75% @80, 99.80% @100, 99.84% @150; India 99.55% @80, 99.65% @100, 99.75% @150.
# The tail is empty-address records with garbled names and Indic names with truncated addresses.
# Keep train == inference.
# Cap 150 (US, France): US recall at full scale was 0.99192 @100 vs 0.99314 @150 with the old fusion.
BLOCK_MAX_CANDIDATES = int(os.environ.get('ER_MAX_CANDIDATES', 150))
BLOCK_MAX_CANDIDATES_INFER = BLOCK_MAX_CANDIDATES
# Per-country overrides. The full-scale run (Sep 27, 75k stats entities) measured blocking recall at cap 100 of
# US 0.99192 but India 0.97669 (0.98199 @150): the India pool has far more same-name records per S1 than the
# 40k sample, and every unreachable match is a guaranteed miss. Values: 'max_candidates' (cap) and either
# 'topk_scale' (multiplies every channel's top-k) or 'topk' (explicit dict). Keys are country_norm values.
# train.py persists this in model_config.json and inference.py applies the TRAINED values.
# Override: ER_BLOCK_BY_COUNTRY='{"india": {"max_candidates": 200, "topk_scale": 2.0}}'
BLOCK_BY_COUNTRY = json.loads(os.environ.get('ER_BLOCK_BY_COUNTRY') or 'null') or {
    'india': {'max_candidates': 200},      # 0.99389 vs 0.99280 at cap 150 on the India pool (measured above)
}


def topk_for(country):
    """Per-channel top-k for one country (BLOCK_TOPK scaled or replaced by BLOCK_BY_COUNTRY)."""
    o = BLOCK_BY_COUNTRY.get(country) or {}
    if o.get('topk'):
        return {ch: int(o['topk'].get(ch, BLOCK_TOPK[ch])) for ch in BLOCK_TOPK}
    sc = float(o.get('topk_scale', 1.0))
    return {ch: int(round(k * sc)) for ch, k in BLOCK_TOPK.items()}


def max_candidates_for(country):
    o = BLOCK_BY_COUNTRY.get(country) or {}
    return int(o.get('max_candidates', BLOCK_MAX_CANDIDATES))
BLOCK_S1_CHUNK = 10000           # S1 rows per sparse matmul / feature chunk (bounds per-worker memory: ~1M pairs)
# Reverse product (S2/S3 x all S1 of the country): tokens present in more than this fraction of S2/S3 rows are
# dropped from BOTH sides before the product. Common tokens never decide a record's best S1 but dominate the
# cost (posting lists of 100k+ rows); measured 19 min per country without pruning on 4 threads.
REVERSE_MAX_DF_FRAC = float(os.environ.get('ER_REVERSE_MAX_DF', 0.005))
# Feature-stage workers. Two full-data runs on a 47 GB pod were OOM-killed (BrokenPipeError in ForkPoolWorker-*)
# with the previous design, in which forked workers read the parent's 6M-row frames directly: every string a
# worker touches gets a refcount write, so each worker copied 4-5 GB of the parent's pages. Workers now receive
# a pickled copy of ONLY their ~100k-pair slice (the same mechanism preprocess_dataframe has used without
# incident at 9 workers on this pod) and never touch the shared frames, so a worker costs its own slice plus
# working set (< 1 GB measured budget below is 3x that). The pool size is still capped by the RAM budget
# (TOTAL_RAM_GB - parent RSS - headroom) / FEATURE_WORKER_GB.  ER_FEATURE_WORKERS=1 forces the main process.
FEATURE_WORKERS = int(os.environ.get('ER_FEATURE_WORKERS', N_JOBS))
FEATURE_WORKER_GB = 3.0
FEATURE_RAM_HEADROOM_GB = 6.0
FEATURE_TASK_PAIRS = 100_000      # pairs per worker task (bounds a worker's slice + working set)
# Reverse channel: every S2/S3 record retrieves its top-k S1 entities (joint vector); the S1's rank in that
# list is a competition feature and each record's top-1 S1 becomes an extra candidate. Costs one extra
# sparse product per country (S2/S3 x all S1). Disable with ER_REVERSE=0.
USE_REVERSE_BLOCKING = os.environ.get('ER_REVERSE', '1') != '0'
REVERSE_TOPK = 5

# ============================================================
# Training data budget
# ============================================================
TRAIN_S1_ENTITIES = int(os.environ.get('ER_SAMPLE_S1', 400000))  # S1 entities used for training pairs
VAL_S1_ENTITIES = 120000                                         # S1 entities held out for validation
# stats split (extra-token statistics + blocking-recall measurement): min(this, 0.75 x train). 50k entities
# (~5M pairs) already give stable token statistics; each 10k US entities cost ~2 min of blocking on the pod
STATS_S1_MAX = int(os.environ.get('ER_STATS_S1', 300000))
NEG_KEEP_ALL_TOP = 25            # always keep the top-N ranked negatives per S1
NEG_RANDOM_FRAC = 0.35           # keep this fraction of the remaining negatives

# ============================================================
# Model
# ============================================================
LGBM_PARAMS = dict(
    objective='binary', learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
    max_bin=255, verbose=-1, seed=RANDOM_SEED, num_threads=N_JOBS,
)
LGBM_ROUNDS = 4000
LGBM_EARLY_STOP = 150
USE_CATBOOST = os.environ.get('ER_NO_CATBOOST', '0') != '1'   # second model for the ensemble
import shutil as _shutil
# GPU decided by hardware presence (nvidia-smi), overridable with ER_GPU=1/0 — not by the SageMaker path
HAS_GPU = (os.environ.get('ER_GPU') == '1') or (os.environ.get('ER_GPU') != '0' and _shutil.which('nvidia-smi') is not None)
CATBOOST_PARAMS = dict(iterations=4000, learning_rate=0.06, depth=8, l2_leaf_reg=3.0,
                       loss_function='Logloss', eval_metric='Logloss', random_seed=RANDOM_SEED,
                       od_type='Iter', od_wait=150, verbose=200, thread_count=N_JOBS,
                       task_type='GPU' if HAS_GPU else 'CPU')
# if GPU training fails (driver/CUDA mismatch), CatBoost is retried on CPU with this many iterations
# (bounded so the fallback cannot eat the time budget); ER_NO_CATBOOST=1 skips CatBoost entirely
CATBOOST_CPU_FALLBACK_ITERATIONS = int(os.environ.get('ER_CATBOOST_CPU_ITERS', 1500))
# training-matrix checkpoint (train.py): matrices + metadata are saved to CACHE_DIR/train_state after the
# feature stage; ER_RESUME=1 skips straight to model training from that checkpoint (pod restarts, CatBoost
# crashes, decision-layer re-runs). Disable saving with ER_CHECKPOINT=0.
CHECKPOINT_MATRICES = os.environ.get('ER_CHECKPOINT', '1') != '0'
RESUME_FROM_CHECKPOINT = os.environ.get('ER_RESUME', '0') == '1'
ENSEMBLE_WEIGHTS = {'lgbm': 0.5, 'catboost': 0.5}

# ============================================================
# Decision layer (entity-level)
# ============================================================
THRESHOLD_GRID = [round(0.30 + 0.01 * i, 3) for i in range(66)]   # 0.30 .. 0.95 step 0.01
PER_COUNTRY_THRESHOLD = True
PER_BIN_THRESHOLDS = True                # separate threshold for candidates whose S2/S3 address is empty
UNSEEN_COUNTRY_THRESHOLD_SHIFT = 0.02   # France starts at (mean of seen thresholds + shift): precision-first
# Label-free France calibration: singletons are 5.59% of S1 in BOTH seen countries (same generator), so the
# France threshold is raised (never lowered) until France's predicted-empty rate reaches this target.
UNSEEN_TARGET_SINGLETON_RATE = 0.0559
UNSEEN_MAX_THRESHOLD = 0.97
# The test pool has ~1.9x more unmatched near-miss records per S1 than train (5.75 vs 4.67 S2/S3 rows per S1
# with the same 3.46 true matches per S1). False positives on unmatched (distractor) rows are weighted by
# this ratio when selecting thresholds so validation tracks the leaderboard instead of overstating it.
TEST_DISTRACTOR_RATIO = float(os.environ.get('ER_DISTRACTOR_RATIO', 1.9))
SINGLETON_FLOOR = 0.0                    # legacy; decision uses thresholds directly
DECISION_KEEP_PROB = 0.02                # pairs below this max-model probability never reach the decision layer
                                         # (applied identically in train.py validation and inference.py)
RESOLVE_S23_CONFLICTS = True             # each S2/S3 record belongs to at most one S1 entity
USE_EXPECTED_F05 = True                  # compare threshold rule vs expected-F0.5 set selection on val
USE_CONSENSUS = True                     # also evaluate min(model probs) instead of the mean (precision filter)
# size-adaptive acceptance: 2nd+ links of an entity need prob >= threshold + delta. Positive deltas tighten
# (Foursquare-style); NEGATIVE deltas relax: once an entity's best candidate passed its threshold it is a confirmed
# non-singleton and its further candidates are accepted at the lower bar (missed links were 66% of the validation
# loss in the 20k run, and tightening lost 0.0003). Selected on validation like every other decision option.
EXTRA_LINK_DELTA_GRID = [-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]

# ============================================================
# Experiments (all off by default)
# ============================================================
# ER_LOCO=us  -> train/stats on that country only, validate on the OTHER seen country as if unseen
#                (measures the unseen-country threshold shift; France proxy)
LOCO_COUNTRY = os.environ.get('ER_LOCO', '').lower().strip() or None
# ER_MACRO_WEIGHTS=1 -> weight positive pairs by 1/(true matches of the entity) so the loss follows the
#                       macro (per-entity) metric instead of favouring many-match entities
MACRO_WEIGHTS = os.environ.get('ER_MACRO_WEIGHTS', '0') == '1'

# ============================================================
# France post-processing
# ============================================================
# Disabled by default: on the seen countries the street-similarity support rule removes true
# matches whose address components were reordered/dropped (validation: US 0.985 -> 0.970).
# The model + the unseen-country threshold shift handle France; enable only to experiment.
FRANCE_ENABLED = os.environ.get('ER_FRANCE_FILTER', '0') == '1'
FRANCE_STREET_SIM_THRESHOLD = 75

# ============================================================
# Cross-Encoder (optional Phase 2 — GPU only)
# ============================================================
CROSS_ENCODER_MODEL = 'microsoft/mdeberta-v3-base'   # MIT license, 278M params
CROSS_ENCODER_BATCH_SIZE = 256
CROSS_ENCODER_EPOCHS = 2
CROSS_ENCODER_LR = 2e-5
CROSS_ENCODER_WARMUP = 500

# ============================================================
# Logging
# ============================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)
log.info(f"Environment: {'SageMaker' if IS_SAGEMAKER else 'RunPod' if IS_RUNPOD else 'Local ' + platform.system()} | "
         f"jobs={N_JOBS} (host cpus={multiprocessing.cpu_count()}) | ram={TOTAL_RAM_GB:.0f} GB | gpu={HAS_GPU} | "
         f"data={DATA_ROOT} | project={PROJECT_ROOT}")
if not os.path.isdir(TRAIN_DIR) and not os.path.isdir(TEST_DIR):
    log.warning(f"Neither {TRAIN_DIR} nor {TEST_DIR} exists — set DATA_ROOT to the folder containing train/ and test/")
