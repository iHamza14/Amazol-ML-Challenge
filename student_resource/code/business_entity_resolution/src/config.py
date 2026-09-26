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
import platform
import logging
import multiprocessing

# ============================================================
# Environment Detection
# ============================================================
IS_SAGEMAKER = os.path.exists('/home/ec2-user/SageMaker')
IS_WINDOWS = platform.system() == 'Windows'
_HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Paths
# ============================================================
if 'DATA_ROOT' in os.environ:
    DATA_ROOT = os.environ['DATA_ROOT']
elif IS_SAGEMAKER:
    DATA_ROOT = '/home/ec2-user/SageMaker/dataset'
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
N_JOBS = int(os.environ.get('ER_N_JOBS', max(1, multiprocessing.cpu_count())))
RANDOM_SEED = 42

# ============================================================
# Blocking (sparse TF-IDF top-k retrieval, per country)
# ============================================================
# Each channel retrieves its own top-k per S1 entity; candidates are the union.
BLOCK_TOPK = {
    'name_tok': 60,      # word unigrams of core name (IDF weighted)
    'name_chr': 40,      # char 3-grams of space-less core name (typos, leetspeak, domain-collapse)
    'addr_tok': 60,      # address words + numbers
    'addr_num': 40,      # (house-number, street-word) combos -- very precise
    'joint':    80,      # name + address tokens together (best overall ranking)
}
BLOCK_MIN_SCORE = 0.08           # cosine below this is never a candidate
BLOCK_MAX_DF_FRAC = 0.03         # drop tokens present in > 3% of S23 docs of that country
# Hard cap per S1 entity after channel union. The union almost always exceeds the cap, so the cap
# directly sets feature cost (pairs = S1 x cap). train.py logs recall@K by fused rank on the stats
# split; raise the cap only if recall@80 is materially below recall@150. Keep train == inference.
BLOCK_MAX_CANDIDATES = int(os.environ.get('ER_MAX_CANDIDATES', 80))
BLOCK_MAX_CANDIDATES_INFER = BLOCK_MAX_CANDIDATES
BLOCK_S1_CHUNK = 20000           # S1 rows per sparse matmul chunk (memory bound)
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
CATBOOST_PARAMS = dict(iterations=4000, learning_rate=0.06, depth=8, l2_leaf_reg=3.0,
                       loss_function='Logloss', eval_metric='Logloss', random_seed=RANDOM_SEED,
                       od_type='Iter', od_wait=150, verbose=200,
                       task_type='GPU' if IS_SAGEMAKER else 'CPU')
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
RESOLVE_S23_CONFLICTS = True             # each S2/S3 record belongs to at most one S1 entity
USE_EXPECTED_F05 = True                  # compare threshold rule vs expected-F0.5 set selection on val
USE_CONSENSUS = True                     # also evaluate min(model probs) instead of the mean (precision filter)

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
log.info(f"Environment: {'SageMaker' if IS_SAGEMAKER else 'Local ' + platform.system()} | "
         f"jobs={N_JOBS} | data={DATA_ROOT} | project={PROJECT_ROOT}")
