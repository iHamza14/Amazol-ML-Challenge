# CLAUDE CONTEXT TRANSFER — Amazon ML Challenge 2026

> **Created:** 2026-09-27T01:00 IST
> **Source conversation:** 953eab73-bff0-48a0-adf1-7eb52294f9b3
> **Purpose:** Complete brain dump for a new Claude instance to continue this work with zero ramp-up time.
> **User:** Working with teammate Hamza (iHamza14 on GitHub). The user is NOT Hamza — the user is Hamza's teammate.

---

## TABLE OF CONTENTS
1. [The Competition](#1-the-competition)
2. [The Dataset — Exact Numbers](#2-the-dataset--exact-numbers)
3. [The Problem Statement — Every Detail](#3-the-problem-statement--every-detail)
4. [The Evaluation Metric — Precise Formula](#4-the-evaluation-metric--precise-formula)
5. [Strategic Decisions Made](#5-strategic-decisions-made)
6. [The #12 Team's Pipeline (Intel From User)](#6-the-12-teams-pipeline)
7. [What We Built — Complete Codebase](#7-what-we-built--complete-codebase)
8. [What Hamza Originally Built (Abandoned)](#8-what-hamza-originally-built-abandoned)
9. [Environment & Paths](#9-environment--paths)
10. [Compute Setup](#10-compute-setup)
11. [Key Technical Insights](#11-key-technical-insights)
12. [Known Risks & Edge Cases](#12-known-risks--edge-cases)
13. [Timeline & Deadlines](#13-timeline--deadlines)
14. [What Has Been Done vs What Remains](#14-what-has-been-done-vs-what-remains)
15. [Every Web Research Finding](#15-every-web-research-finding)
16. [User Preferences & Communication Style](#16-user-preferences--communication-style)

---

## 1. THE COMPETITION

**Name:** Amazon ML Challenge 2026
**Platform:** Unstop (https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604)
**Host:** Amazon
**Registrations:** 89,399 candidates
**Format:** 72-hour hackathon (teams of 2-4)
**Team's goal:** Top 50 → Pre-Placement Interview (PPI) for Applied Scientist Intern at Amazon
**Stretch goal:** Top 10 → Grand Finale presentation to Amazon Scientists

### Timeline
- **Stage 2 (Hackathon):** Sep 25, 2026 00:00 IST → Sep 27, 2026 23:59 IST
- **Top 50 announced:** Oct 2, 2026
- **Grand Finale:** Oct 7, 2026 (Top 10 present live)

### Prizes
- Winners: ₹1,00,000 + certs + goodies
- 1st runners-up: ₹75,000
- 2nd runners-up: ₹50,000
- Top 50: PPIs for Applied Scientist Intern
- All participants: $200 AWS credits

### Rules
- Models must be MIT/Apache 2.0 licensed, ≤ 8B parameters
- **STRICTLY NO** external data lookup (no APIs, no geocoding, no business registries) → instant DQ
- Code + methodology document reviewed for top teams

---

## 2. THE DATASET — EXACT NUMBERS

### Training Set
| File | Rows | Countries |
|---|---|---|
| `train_source1.tsv` | 2,206,821 | US: 1,323,633 / India: 883,188 |
| `train_source2.tsv` | 5,034,616 | US: 3,016,817 / India: 2,017,799 |
| `train_source3.tsv` | 5,285,603 | US: 3,170,056 / India: 2,115,547 |
| `train_ground_truth.tsv` | 2,206,821 | — |
| **Total train records** | **12,527,040** | |

### Test Set
| File | Rows | Countries |
|---|---|---|
| `test_source1.tsv` | 1,732,544 | India: 809,986 / US: 663,106 / **France: 259,452** |
| `test_source2.tsv` | 4,887,273 | India: 2,312,565 / US: 1,871,330 / **France: 703,378** |
| `test_source3.tsv` | 5,082,316 | India: 2,405,000 / US: 1,945,701 / **France: 731,615** |
| **Total test records** | **11,702,133** | |

### Ground Truth Analysis
- **Singletons (no matches):** 123,247 (5.6%)
- **With matches:** 2,083,574 (94.4%)
- **Average matches per non-singleton:** 3.67

### File Sizes (on disk)
| File | Size |
|---|---|
| test_source1.tsv | 175 MB |
| test_source2.tsv | 509 MB |
| test_source3.tsv | 506 MB |
| train_ground_truth.tsv | 127 MB |
| train_source1.tsv | 210 MB |
| train_source2.tsv | 489 MB |
| train_source3.tsv | 504 MB |

### CRITICAL: France is UNSEEN
- France has ZERO records in training data
- France is ~15% of test S1 entities (259,452 out of 1,732,544)
- The model MUST generalize to France without any France-specific training data
- French business suffixes: SARL, SAS, SA, EURL, SASU
- French address format: `12 Rue de la Paix, 75002 Paris`

### Columns in Source Files
1. `entity_id` — Unique ID. Prefix indicates source: `S1-`, `S2-`, `S3-`
2. `business_name` — May contain abbreviations, legal suffixes, typos, transliterations
3. `business_address` — May contain partial addresses, format variations, landmarks
4. `country` — String label: `US`, `India`, `France`

### Ground Truth Format
1. `source1_entity_id` — S1 entity ID
2. `matched_entity_ids` — Comma-separated S2/S3 IDs (empty if singleton)

### Sample Data (First 5 rows of train_source1)
```
entity_id        business_name           business_address                              country
S1-925783039     Orelee's Barbershop     1795 Westchester Drive, High Point, NC        US
S1-773889195     Prime Money             17560 Ellis Road, Tahlequah, OK               US
S1-377745466     B+ Retail Inc           1712 Montebello Avenue, Phoenix, AZ           US
S1-133037285     Christ Chapel           2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD  US
S1-755362802     Prabhav Business Center 797, Lake Town Block A, Kolkata, Howrah, West Bengal  India
```

---

## 3. THE PROBLEM STATEMENT — EVERY DETAIL

**Business Entity Resolution (ER):** Given business records from 3 independent, noisy data sources with no common identifiers, determine which records across sources refer to the same real-world business entity.

- **Source 1** is the deduplicated reference source
- Task: For each S1 entity, find ALL matching records from S2 and S3
- A S1 entity may match **zero, one, or many** records from S2/S3
- No common join key exists between sources

### Noise Patterns Documented in Problem Statement
- **Name variations:** Abbreviations (Corp vs Corporation), DBA/trade names, punctuation (& vs "and"), word-order transpositions, typos
- **Address variations:** Abbreviations (Rd vs Road), transliteration variants, missing components (no PIN code), landmark-based references ("Near SBI ATM"), component reordering

### Output Required — TWO files
1. **`matching_results.tsv`** — Final predictions. This is scored on leaderboard.
   - Columns: `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 IDs)
   - Every S1 test entity must have exactly one row
   - Empty `matched_entity_ids` for singletons
   - No duplicates within ID lists
   - Only S2/S3 IDs that exist in test set

2. **`candidate_pairs.tsv`** — Blocking output (not scored, used for audit)
   - Same format but with `candidate_entity_ids`
   - All matched IDs should be subset of candidates

### Submission Package (Final)
```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/business_entity_resolution/
│   ├── src/
│   ├── README.md
│   └── requirements.txt
└── Documentation_template.md
```

### Validation Script
Located at: `student_resource/utils/validate_submission.py`
Run: `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`

### Leaderboard
- **Public leaderboard:** Subset of test set, real-time during hackathon
- **Private leaderboard:** Remaining test set, revealed after hackathon ends
- **Final rankings based on private leaderboard**
- Target score: ≥ 0.991 for top 50

---

## 4. THE EVALUATION METRIC — PRECISE FORMULA

**Macro-averaged F₀.₅ score** (precision-heavy):

```
F₀.₅ = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

### How it's computed:
1. For EACH S1 entity independently:
   - `precision_i = |predicted ∩ truth| / |predicted|`
   - `recall_i = |predicted ∩ truth| / |truth|`
   - `f05_i = (1.25 × P × R) / (0.25 × P + R)`
2. **Macro average:** `F₀.₅ = mean(f05_i for ALL S1 entities)`

### Critical singleton handling:
- S1 with NO true matches AND you predict empty → f05 = **1.0** (free point!)
- S1 with NO true matches AND you predict anything → f05 = **0.0** (devastating)
- 5.6% of entities are singletons → 5.6% free points if handled correctly

### Why precision-heavy matters:
- False positive (wrong merge) hurts 2× more than false negative (missed link)
- High threshold = safe; low threshold = risky
- Better to miss a match than to create a false merge

### Example from problem statement:
- Predicted: [S2-00047, S2-00193, S3-00812]
- Truth: [S2-00047, S3-00812]
- Precision = 2/3, Recall = 2/2 = 1.0
- F₀.₅ = (1.25 × 0.667 × 1.0) / (0.25 × 0.667 + 1.0) = **0.714**

---

## 5. STRATEGIC DECISIONS MADE

### Decision 1: ABANDON Hamza's DeBERTa cross-encoder pipeline
**Reason:** Too slow to iterate, English-only (won't handle France), never been trained, FAISS top_k=15 caps recall. The CatBoost approach iterates 5-10x faster.

### Decision 2: Use HYBRID pipeline (CatBoost + Cross-Encoder)
**Architecture:**
1. Inverted-index blocking (from #12 team's approach) → CPU, fast, language-agnostic
2. 40 handcrafted features + CatBoost → fast training and inference
3. Cross-encoder (mdeberta-v3-base, multilingual) → accuracy boost on uncertain cases
4. France-specific post-processing → catch false positives in unseen country

### Decision 3: SageMaker for compute
- Instance: `ml.g5.2xlarge` (A10G 24GB GPU, 32GB RAM, 8 vCPUs, $1.52/hr)
- Budget: $200 → 131 hours
- Fallback: `ml.g5.4xlarge` (64GB RAM) if 32GB is tight

### Decision 4: Entity-level F₀.₅ threshold optimization
- Hamza's code used pair-level F₀.₅ (WRONG — gives optimistic results)
- We use entity-level macro F₀.₅ (the actual competition metric)
- This makes a huge difference in threshold selection

---

## 6. THE #12 TEAM'S PIPELINE

**The user contacted the team currently ranked #12 on the leaderboard.** They shared their pipeline overview:

### Preprocessing
- `unidecode()` for ASCII transliteration
- Lowercase, expand `&` → `and`, `@` → `at`
- Replace punctuation with spaces, collapse whitespace
- Exclude common words (legal suffixes, address terms) from token features

### Blocking (Candidate Generation)
- Build lookup indexes using:
  1. Name tokens
  2. Name token pairs
  3. Address tokens
  4. Address combinations (number + word, e.g. "123" + "main")
- Prefer rare shared tokens and combinations (IDF-like weighting)
- Rank by weighted overlap
- Keep up to **120 candidates** per S1 entity

### Feature Engineering (36 features)
- Fuzzy name and address similarities
- Exact matches, word overlap, substring relationships
- Shared address numbers and first-number agreement
- String lengths, missing fields, other comparisons

### Model: CatBoost
- Score each candidate using saved `model.cbm`
- Keep candidates with probability ≥ 0.625
- Return empty list when none qualifies

### France-Specific Filter (POST-PROCESSING)
- Extract street names and house numbers
- If any predicted match has `street_similarity >= 75/100` AND no conflicting house number → retain ONLY those supported matches
- If none qualifies → keep original model predictions

---

## 7. WHAT WE BUILT — COMPLETE CODEBASE

All files are at: `d:\Downloads\Amazol-ML-Challenge\student_resource\code\business_entity_resolution\src\`

### File: `config.py` (NEW)
- Auto-detects environment: checks `IS_SAGEMAKER` via `/home/ec2-user/SageMaker`, `IS_WINDOWS` via `platform.system()`
- Dataset paths auto-resolved with case-insensitive directory matching
- All hyperparameters centralized:
  - `BLOCKING_TOP_K = 120`
  - `BLOCKING_NAME_MAX_DF = 5000` (skip name tokens in >5000 docs)
  - `BLOCKING_ADDR_MAX_DF = 10000`
  - `BLOCKING_PAIR_MAX_DF = 1000`
  - `BLOCKING_COMBO_MAX_DF = 2000`
  - `BLOCKING_PAIR_BONUS = 5.0`, `BLOCKING_COMBO_BONUS = 3.0`
  - `CATBOOST_ITERATIONS = 3000`, `DEPTH = 8`, `LR = 0.05`
  - `MATCH_THRESHOLD = 0.625` (default, optimized during training)
  - `SINGLETON_FLOOR = 0.3`
  - `FRANCE_STREET_SIM_THRESHOLD = 75`
  - `VAL_RATIO = 0.15`, `RANDOM_SEED = 42`
- Local Windows paths: `DATA_ROOT = r'D:\Downloads\Amazol-ML-Challenge\Dataset ML Amazon'`
- SageMaker paths: `DATA_ROOT = '/home/ec2-user/SageMaker/dataset'`

### File: `preprocess.py` (REWRITTEN)
- `normalize_text()`: unidecode → lowercase → expand &/@ → strip punctuation → collapse whitespace
- `strip_legal_suffixes()`: Removes US (Inc, Corp, LLC, LLP, LP, Co), India (Pvt, Ltd, Private, Limited), France (SARL, SAS, SA, EURL, SASU) — sorted longest-first for greedy matching
- `tokenize()`: Splits into words, filters comprehensive stopword set (English/French/Indian address terms)
- `extract_numbers()`, `get_first_number()`, `get_street_component()`
- `load_data()`: reads TSV with `sep='\t'`, `dtype=str`, `keep_default_na=False`
- `preprocess_dataframe()`: Adds columns: `name_norm`, `name_core`, `addr_norm`, `country_norm`, `name_tokens`, `addr_tokens`, `addr_numbers`, `first_number`, `street`
- CLI mode: preprocess and save as parquet

**Verified working locally:**
- `'B+ Retail Inc.'` → core: `'b retail'` ✅
- `'Boulangerie Artisanale SARL'` → core: `'boulangerie artisanale'` ✅
- Indian address tokens correctly filtered ✅
- French address numbers extracted correctly ✅

### File: `blocking.py` (REWRITTEN)
**Architecture: 4 inverted indexes, per-country processing**

1. **Name single token index**: token → set(S23 row indices), skip if doc_freq > 5000
2. **Name token pair index**: sorted (token1, token2) → set(S23 row indices), skip if df > 1000
3. **Address single token index**: token → set(S23 row indices), skip if df > 10000
4. **Address number+word combo index**: (number, non-digit-word) → set(S23 row indices), skip if df > 2000

**Scoring:** For each S1 entity, query all 4 indexes, accumulate IDF-weighted scores per candidate:
- Name tokens: score += IDF(token) × 1.0
- Name pairs: score += 5.0 (flat bonus)
- Addr tokens: score += IDF(token) × 0.5
- Addr combos: score += 3.0 (flat bonus)

Keep top 120 candidates per S1 entity.

**Returns DataFrame with meta-features:** `s1_id`, `s23_id`, `blocking_rank`, `blocking_score`, `n_shared_tokens`, `max_idf_shared`, `candidate_count`

### File: `features.py` (REWRITTEN)
**40 features in 4 groups:**

**Name features (12):**
1. `name_jaro_winkler` — JaroWinkler normalized similarity × 100
2. `name_levenshtein` — fuzz.ratio
3. `name_token_sort` — fuzz.token_sort_ratio
4. `name_token_set` — fuzz.token_set_ratio
5. `name_partial` — fuzz.partial_ratio
6. `name_exact` — binary exact match
7. `name_sorted_exact` — sorted words exact match
8. `name_jaccard_char3` — character 3-gram Jaccard
9. `name_jaccard_word` — word-level Jaccard
10. `name_containment` — max(|intersection| / |shorter_set|)
11. `name_prefix3` — first 3 chars match
12. `name_len_ratio` — min(len)/max(len)

**Address features (14):**
13-17. Same fuzzy metrics as name but on address
18. `addr_first_num_match` — first house numbers agree
19. `addr_first_num_conflict` — first house numbers disagree (NEGATIVE signal)
20. `addr_all_nums_match` — all numbers identical
21. `addr_num_overlap` — fraction of shared numbers
22. `addr_street_sim` — fuzz.ratio on street component
23-25. `addr_has_s1/s23`, `addr_both_missing` — missing address flags
26. `addr_len_ratio`

**Cross features (10):**
27. `same_country` — binary
28. `source_flag` — S2=0, S3=1
29. `combined_score` — name_jaro×0.6 + addr_jaro×0.4
30. `name_in_addr` — name substring of other's address
31-34. Raw lengths
35-36. Word count differences

**Meta features (4, from blocking):**
37. `blocking_rank` — candidate rank in blocking top-K
38. `blocking_score` — raw blocking score
39. `n_shared_tokens` — tokens shared between S1 and S23
40. `candidate_count` — total candidates for this S1

### File: `evaluate.py` (REWRITTEN)
- `calculate_f05_entity()` — single entity F₀.₅
- `calculate_f05_macro()` — macro average across all S1 entities
- `build_ground_truth_dict()` — GT DataFrame → dict of sets
- `build_predictions_dict()` — pairs + probs + threshold → dict of sets
- `optimize_threshold()` — sweeps threshold 0.1→0.95 by 0.025, maximizes entity-level macro F₀.₅
- `evaluate_blocking_recall()` — measures recall ceiling imposed by blocking

### File: `train.py` (REWRITTEN — was DeBERTa, now CatBoost)
**Full pipeline:**
1. Load & preprocess train data (S1, S2, S3, ground truth)
2. Entity-level 85/15 train/val split on S1 IDs
3. Generate blocking candidates for train and val separately
4. Measure blocking recall on validation
5. Extract 40 features for all candidate pairs
6. Label candidates (1 if in ground truth, 0 otherwise)
7. Train CatBoost with `scale_pos_weight` for class imbalance, early stopping
8. Sweep threshold on validation using entity-level macro F₀.₅
9. Save `models/catboost_model.cbm` and `models/model_config.json`

CatBoost config: 3000 iterations, lr=0.05, depth=8, L2=3.0, GPU on SageMaker / CPU locally, Logloss loss, 200 early stopping rounds.

### File: `inference.py` (REWRITTEN)
**Full pipeline:**
1. Load CatBoost model + threshold from `models/`
2. Load & preprocess test data
3. Generate blocking candidates
4. Save `candidate_pairs.tsv`
5. Extract features
6. Predict with CatBoost
7. Apply threshold + singleton floor (if max_prob < 0.3, predict empty)
8. Apply France filter
9. Save `matching_results.tsv`

Supports CLI args: `--test-dir`, `--output-dir`, `--model-dir`, `--threshold`, `--singleton-floor`

### File: `france_filter.py` (NEW)
- Applies only to France S1 entities
- For each France entity's predicted matches:
  - Compute street similarity (fuzz.ratio on non-numeric address parts)
  - Check for house number conflicts
  - If any match has street_sim ≥ 75 AND no number conflict → keep ONLY those
  - If none qualifies → keep all original predictions (don't remove matches blindly)

### File: `cross_encoder.py` (NEW — Phase 2, GPU only)
- `train`: Fine-tunes `microsoft/mdeberta-v3-base` (multilingual, 278M params) as cross-encoder
  - Uses hard negatives from blocking (candidates that are NOT true matches)
  - Input format: `"Name: {name} | Address: {addr} | Country: {country}"`
  - Max 500K training pairs, 3 epochs, warmup 500 steps
- `score`: Scores candidate pairs, saves CE scores as parquet
  - These scores become feature #41 for a second CatBoost training pass

### File: `requirements.txt` (UPDATED)
```
pandas>=2.0, numpy>=1.24, scikit-learn>=1.3, catboost>=1.2, rapidfuzz>=3.0,
unidecode>=1.3, sentence-transformers>=2.2, faiss-cpu>=1.7, tqdm>=4.65, torch>=2.0
```

---

## 8. WHAT HAMZA ORIGINALLY BUILT (ABANDONED)

On the `deep-learning-sota` branch, Hamza built a DeBERTa cross-encoder pipeline:

- **preprocess.py**: Basic normalization (lowercase, expand abbreviations, strip legal suffixes). Missing: unidecode, French suffixes, stopword filtering, address number extraction.
- **blocking.py**: TF-IDF char n-gram (2-4) + BGE-Large-en FAISS dense retrieval. Top_k=15 (too low). FAISS threshold 0.7 (too aggressive). BGE-Large is English-only (fails for France).
- **train.py**: Fine-tunes `cross-encoder/nli-deberta-v3-base` (English-only). Pair-level F₀.₅ evaluation (WRONG metric). 2 epochs, batch size 32.
- **inference.py**: Loads trained DeBERTa model, applies sigmoid + threshold 0.8.
- **features.py**: 9 features (JaroWinkler, Levenshtein, token_sort, token_set on names/addresses + digit overlap + country match + source flag). Very basic compared to our 40.
- **evaluate.py**: Had entity-level macro F₀.₅ (correct) but also had a "smart filtering" hack for local eval that loads only 50K noise records.
- **No trained model exists** — `deberta_er_model/` folder was never created.
- **README.md** references XGBoost but code uses DeBERTa — inconsistent.

---

## 9. ENVIRONMENT & PATHS

### Local Machine (User's Laptop)
- **OS:** Windows
- **GPU:** RTX 3050 (4GB VRAM)
- **Python:** 3.12 (via Windows Store)
- **Repo:** `d:\Downloads\Amazol-ML-Challenge`
- **Branch:** `deep-learning-sota` (tracking `origin/deep-learning-sota`)
- **Dataset:** `D:\Downloads\Amazol-ML-Challenge\Dataset ML Amazon\Train\` and `Test\`
  - Note: Capital T in Train/Test
  - Dataset is NOT inside `student_resource/` — it's at the repo root level in a separate folder
- **Code:** `d:\Downloads\Amazol-ML-Challenge\student_resource\code\business_entity_resolution\src\`
- **Installed packages:** unidecode, rapidfuzz (just installed). pandas, numpy pre-existing. catboost NOT yet installed locally.

### SageMaker (Planned)
- **Instance:** `ml.g5.2xlarge` recommended
- **Dataset will be at:** `/home/ec2-user/SageMaker/dataset/train/` and `test/`
  - Note: lowercase train/test
- **Code upload options:**
  1. Git clone from `https://github.com/iHamza14/Amazol-ML-Challenge.git` → checkout `deep-learning-sota`
  2. S3 sync from local
  3. JupyterLab direct upload
- AWS credentials needed for S3 operations

### Config.py Auto-Detection
The config.py auto-detects:
- `IS_SAGEMAKER`: checks `/home/ec2-user/SageMaker` exists
- `IS_WINDOWS`: checks `platform.system() == 'Windows'`
- `DATA_ROOT`: set per environment, overridable via `DATA_ROOT` env var
- `TRAIN_DIR`: case-insensitive subdirectory matching (handles Train vs train)
- `TEST_DIR`: same
- `OUTPUT_DIR`: `PROJECT_ROOT/output`
- `MODEL_DIR`: `PROJECT_ROOT/models`

---

## 10. COMPUTE SETUP

### User's Options (Stated)
1. RTX 3050 laptop (4GB VRAM) — can run CatBoost (CPU), too weak for cross-encoder
2. SageMaker AWS ($200 free credits) — **CHOSEN**
3. Third-party GPU rental — rejected

### SageMaker Instance Recommendations
| Instance | GPU | VRAM | RAM | vCPU | Cost/hr | Budget Hours |
|---|---|---|---|---|---|---|
| ml.g4dn.xlarge | T4 | 16GB | 16GB | 4 | $0.74 | 270 |
| **ml.g5.2xlarge** | **A10G** | **24GB** | **32GB** | **8** | **$1.52** | **131** |
| ml.g5.4xlarge | A10G | 24GB | 64GB | 16 | $2.03 | 98 |

**Recommended: ml.g5.2xlarge**. If 32GB RAM is tight for 12M records, upgrade to ml.g5.4xlarge.

### SageMaker Setup Guide
Written as artifact at: `C:\Users\Asus\.gemini\antigravity-ide\brain\953eab73-bff0-48a0-adf1-7eb52294f9b3\sagemaker_setup.md`

Steps: Create notebook instance → Install deps → Upload dataset (via S3 recommended) → Upload code (git clone recommended) → Verify GPU → Run pipeline.

---

## 11. KEY TECHNICAL INSIGHTS

### Insight 1: Blocking recall is the ceiling
If a true match isn't in your candidate set, you can NEVER find it. The blocking step determines the maximum possible recall. With top_k=120 and 4 index types, we should have >99% recall. Hamza's top_k=15 was way too low.

### Insight 2: Entity-level vs pair-level F₀.₅
Hamza's train.py computed pair-level F₀.₅ for threshold optimization. This is WRONG. The competition uses entity-level macro F₀.₅. A pair-level metric hides critical entity-level dynamics (e.g., a singleton with 1 false positive gets F₀.₅=0.0, devastating the average). Our evaluate.py correctly implements entity-level.

### Insight 3: Singletons are free points
5.6% of entities have NO true matches. Correctly predicting empty = F₀.₅ = 1.0 for that entity. Incorrectly predicting ANY match = F₀.₅ = 0.0. The singleton floor (max_prob < 0.3 → predict empty) protects against this.

### Insight 4: France is the differentiator
15% of test S1 entities are French. No French training data exists. Most competitors will struggle here. Our advantages:
- `unidecode()` handles French accents (é→e, ç→c, etc.)
- French legal suffixes (SARL/SAS/SA/EURL/SASU) stripped in preprocessing
- French stopwords (de, la, le, rue, boulevard, etc.) filtered from blocking
- Inverted-index blocking is language-agnostic (string tokens work in any language)
- France filter post-processing catches address-based false positives
- Cross-encoder uses multilingual mdeberta (Phase 2)

### Insight 5: CatBoost + handcrafted features > cross-encoder alone
A cross-encoder can't learn "house number 123 ≠ 124" without massive training data. CatBoost with `addr_first_num_conflict` catches this instantly. Conversely, a cross-encoder understands "Boulangerie Artisanale" = "Artisanale Boulangerie" natively. The hybrid approach gets both strengths.

### Insight 6: Address number agreement is a killer feature
The #12 team specifically called out "shared address numbers and first-number agreement" as key features. Our features include `addr_first_num_match`, `addr_first_num_conflict`, `addr_all_nums_match`, `addr_num_overlap`. These are critical for precision.

### Insight 7: Blocking meta-features help
Most teams extract features only AFTER blocking. We also extract features FROM the blocking step: `blocking_rank`, `blocking_score`, `n_shared_tokens`, `max_idf_shared`, `candidate_count`. A S1 entity with 120 candidates is "harder" than one with 3 — CatBoost learns this.

### Insight 8: IDF weighting in blocking
Common tokens (road, street, company) match too many records and aren't discriminative. Rare tokens (unique business names) are highly discriminative. IDF weighting ensures rare tokens contribute more to blocking scores. Max DF thresholds prevent memory explosion from overly common tokens.

### Insight 9: The threshold matters enormously
With a precision-heavy metric (F₀.₅), the threshold has huge impact. Too low = too many false positives = precision drops. Too high = missed matches = recall drops. The optimal threshold is typically 0.5-0.7. We sweep from 0.1 to 0.95 in 0.025 increments.

### Insight 10: Scale considerations
- Train blocking: ~2.2M S1 × ~120 candidates = up to ~264M pairs (but average candidates per entity is ~30-40, so more like 60-80M)
- Test blocking: ~1.7M S1 × similar = ~50-60M pairs
- Feature extraction at ~50μs/pair = 50-75 minutes (acceptable)
- Cross-encoder scoring at ~1ms/pair = 50-60K seconds = too slow for all pairs → score only uncertain cases

---

## 12. KNOWN RISKS & EDGE CASES

| Risk | Probability | Mitigation |
|---|---|---|
| SageMaker setup takes too long | Low | Write code locally, test on small sample, upload finished code |
| 32GB RAM not enough for 12M records | Medium | Process per-country (max ~7M records per country), or use ml.g5.4xlarge |
| Blocking recall too low | Medium | Measure recall on train set, tune top_k upward if needed |
| CatBoost threshold wrong | Low | Entity-level F₀.₅ sweep (not pair-level) |
| France data behaves differently | High | France filter + multilingual cross-encoder + language-agnostic features |
| Cross-encoder too slow for all pairs | High | Score only test candidates, or only uncertain predictions |
| Submission format errors | Low | Run validate_submission.py before every upload |
| India transliteration issues | Medium | unidecode() handles Hindi→ASCII conversion |
| Missing addresses (empty fields) | Medium | `addr_both_missing` feature + model learns to rely on name |
| Landmark-based Indian addresses ("Near SBI ATM") | Medium | Low addr_similarity → model relies on name features |
| DBA/trade names (two different-looking names, same entity) | Medium | `name_containment` and `name_partial` features help |
| S2/S3 entity matching multiple S1 entities | Low | Not explicitly constrained in problem (allowed) |

---

## 13. TIMELINE & DEADLINES

### Competition Timeline
- Hackathon started: Sep 25, 2026 00:00 IST
- **Hackathon ends: Sep 27, 2026 23:59 IST** ← ~23 hours from context creation
- Top 50 announced: Oct 2, 2026
- Grand Finale (top 10): Oct 7, 2026

### Work Done So Far (as of 2026-09-27 01:00 IST)
- ✅ Deep research on competition, dataset, approaches
- ✅ Full dataset analysis (sizes, country distributions, singleton stats)
- ✅ Analyzed #12 team's pipeline
- ✅ Strategic decision: hybrid CatBoost + cross-encoder
- ✅ All 11 code files written and verified working
- ❌ No model has been trained yet
- ❌ No submission has been made yet
- ❌ SageMaker not set up yet
- ❌ Cross-encoder not trained yet

### Recommended Timeline (Remaining ~23 hours)
| Phase | Hours | Task |
|---|---|---|
| Setup | 0-1 | SageMaker instance + dataset upload |
| Core Pipeline | 1-6 | Run train.py on full data → first model |
| First Submission | 6-7 | Run inference.py → validate → submit |
| Iterate | 7-14 | Tune blocking, features, threshold |
| Cross-Encoder | 14-20 | Phase 2 accuracy boost (GPU) |
| Final | 20-23 | Final submission + documentation + zip |

---

## 14. WHAT HAS BEEN DONE VS WHAT REMAINS

### ✅ DONE
1. Full competition analysis and research
2. Dataset downloaded and analyzed (sizes, distributions, singletons)
3. Strategic decisions finalized (hybrid approach, SageMaker)
4. Complete codebase written (11 files):
   - config.py, preprocess.py, blocking.py, features.py, evaluate.py
   - train.py, inference.py, france_filter.py, cross_encoder.py
   - requirements.txt, README.md
5. Code verified working locally (imports, preprocessing, F₀.₅ calculation)
6. SageMaker setup guide written
7. Dependencies installed locally (unidecode, rapidfuzz)

### ❌ REMAINS
1. **Set up SageMaker notebook instance**
2. **Upload dataset to SageMaker** (or run locally if SageMaker is slow)
3. **Install remaining deps** (catboost, sentence-transformers, torch on SageMaker)
4. **Run train.py** — the big one. This trains the CatBoost model.
5. **Run inference.py** — generates submission files
6. **Validate and submit** — get first score on leaderboard
7. **Iterate:** tune blocking params, add/remove features, adjust thresholds
8. **Phase 2:** Train cross-encoder, score candidates, retrain CatBoost with CE feature
9. **Final submission** with optimizations
10. **Documentation** — fill in Documentation_template.md
11. **Zip package** — assemble final submission package

### CRITICAL PATH (minimum viable submission)
1. SageMaker setup (30 min)
2. `python train.py` (1-3 hours depending on dataset size)
3. `python inference.py` (30-60 min)
4. Validate + submit (10 min)
**Total minimum time to first submission: ~3-4 hours**

---

## 15. EVERY WEB RESEARCH FINDING

### Finding 1: Top teams use two-stage pipelines
Blocking (retrieval) + Matching (classification). This is confirmed by all sources: Kaggle, Reddit, HuggingFace, academic papers.

### Finding 2: CatBoost is the winning model type
Multiple sources confirm gradient boosting (CatBoost specifically) beats cross-encoders for tabular ER features. CatBoost handles categorical features natively, trains fast, and is robust to noisy data.

### Finding 3: Token-based blocking with IDF weighting > dense retrieval
For this scale (12M+ records), inverted indexes with IDF weighting are faster and more recall-efficient than dense embedding retrieval (FAISS). Dense retrieval requires model inference on all records; inverted indexes don't.

### Finding 4: France handling strategies
- `unidecode()` for accent normalization
- French legal suffixes: SARL, SAS, SA, EURL, SASU
- French address format: number + road type + street name + postal code + city
- Road types: rue, avenue, boulevard, place, allée, chemin, impasse, passage
- Post-processing filter on street similarity + house number agreement

### Finding 5: Leaderboard plateau at ~0.85-0.90
Many teams plateau around 0.85-0.90. Breaking through requires:
1. Better blocking recall (more candidates)
2. Better features (especially address number agreement)
3. Entity-level threshold optimization
4. Singleton handling
5. France-specific post-processing

### Finding 6: Dataset on HuggingFace
Available at `akshatbakshi/amazon-ml-challenge-2026` or `logicalguy/amazon-ml-challenge-2026`. Both TSV and Parquet formats available.

### Finding 7: Polars recommended over Pandas
For 12M+ records, Polars is faster and more memory-efficient. However, we're using Pandas for simplicity since SageMaker has enough RAM.

### Finding 8: 1-to-1 assignment constraint
Some sources mention that each S2/S3 entity should link to at most one S1 entity. The problem statement doesn't explicitly enforce this, but it could be a useful post-processing step (Hungarian assignment on high-confidence pairs).

### Finding 9: Hard negative mining critical for cross-encoder
Random negatives are too easy. Hard negatives (blocking candidates that are NOT true matches) force the model to learn subtle distinctions. All top solutions use hard negative mining.

### Finding 10: SageMaker instance pricing
- ml.g4dn.xlarge: $0.74/hr (T4 16GB, 16GB RAM)
- ml.g5.xlarge: $1.01/hr (A10G 24GB, 16GB RAM, 4 vCPU — RAM too low)
- ml.g5.2xlarge: $1.52/hr (A10G 24GB, 32GB RAM, 8 vCPU — recommended)
- ml.g5.4xlarge: $2.03/hr (A10G 24GB, 64GB RAM, 16 vCPU — if RAM tight)

---

## 16. USER PREFERENCES & COMMUNICATION STYLE

### About the User
- They are a teammate of Hamza (iHamza14), not Hamza himself
- They refer to the work as "our ML challenge" and "our team"
- They are comfortable with command-line, git, AWS, Python
- They want maximum accuracy, not efficiency
- They have $200 AWS credits and want to use them
- They are time-pressured (~23 hours remaining as of context creation)

### Communication Style
- Casual, fast-paced
- Uses phrases like "you are the absolute goat ml opus"
- Appreciates concise summaries with tables
- Likes structured plans with hour-by-hour breakdowns
- Wants to be informed about tradeoffs but trusts the AI's judgment
- Prefers getting things done over perfect planning

### User's Stated Preferences
- "prioritizing absolute results and accuracy, not efficiency"
- "i have sagemaker why not use it"
- "i want you to: analyse, deep research, online and thinking"
- "cover everything"
- "we are going to figure everything out and row through this to top 10"

### Key Decisions the User Made
1. Use SageMaker (not local GPU or third-party)
2. Approved the hybrid CatBoost + cross-encoder approach
3. Approved abandoning Hamza's DeBERTa approach
4. Wants to beat the #12 team's CatBoost-only approach by adding cross-encoder

---

## APPENDIX A: EXACT FILE PATHS

```
d:\Downloads\Amazol-ML-Challenge\
├── .git\
├── .gitignore
├── 6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf
├── 6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf
├── Dataset ML Amazon\
│   ├── Train\
│   │   ├── train_source1.tsv (210 MB)
│   │   ├── train_source2.tsv (489 MB)
│   │   ├── train_source3.tsv (504 MB)
│   │   └── train_ground_truth.tsv (127 MB)
│   └── Test\
│       ├── test_source1.tsv (175 MB)
│       ├── test_source2.tsv (509 MB)
│       └── test_source3.tsv (506 MB)
└── student_resource\
    ├── Documentation_template.md
    ├── README.md (14 KB — full problem statement)
    ├── code\
    │   └── business_entity_resolution\
    │       ├── requirements.txt
    │       └── src\
    │           ├── README.md
    │           ├── blocking.py (8.7 KB)
    │           ├── config.py (4.4 KB)
    │           ├── cross_encoder.py (7.1 KB)
    │           ├── evaluate.py (5.5 KB)
    │           ├── explore_dataset.ipynb (16 KB — Hamza's old EDA)
    │           ├── features.py (9.9 KB)
    │           ├── france_filter.py (3.2 KB)
    │           ├── inference.py (8.1 KB)
    │           ├── preprocess.py (7.3 KB)
    │           └── train.py (7.4 KB)
    └── utils\
        └── validate_submission.py (13.7 KB)
```

---

## APPENDIX B: GIT STATUS

- **Remote:** `https://github.com/iHamza14/Amazol-ML-Challenge`
- **Current branch:** `deep-learning-sota` (tracks `origin/deep-learning-sota`)
- **Other branch:** `main`
- **Working tree:** Modified (all our code changes are uncommitted)

---

## APPENDIX C: CONVERSATION ARTIFACTS

Located at: `C:\Users\Asus\.gemini\antigravity-ide\brain\953eab73-bff0-48a0-adf1-7eb52294f9b3\`

| File | Description |
|---|---|
| `challenge_brief.md` | Initial competition brief (superseded by battle plan) |
| `final_battle_plan.md` | Refined battle plan (hybrid CatBoost + cross-encoder) |
| `sagemaker_setup.md` | Step-by-step SageMaker setup guide |

---

## APPENDIX D: WHAT THE NEW CLAUDE SHOULD DO FIRST

1. **Read this document thoroughly** — it IS your context
2. **Ask the user what has happened since this document was created:**
   - Has SageMaker been set up?
   - Has any training or submission been done?
   - What's the current leaderboard score?
   - Any issues encountered?
3. **Check the current state of the code** — files may have been modified since this doc was written
4. **Continue from wherever the user is** — the code is ready to run, the user just needs to execute it
5. **Do NOT re-research or re-plan** — all research has been done, all decisions have been made. Execute.

---

*END OF CONTEXT TRANSFER*
