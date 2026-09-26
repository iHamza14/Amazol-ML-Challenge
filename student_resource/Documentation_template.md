# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Antigravity  
**Team Members:** Antigravity  
**Submission Date:** 2026-09-26

---

## 1. Executive Summary
Our approach employs a robust blocking and classification pipeline. We normalize noise using unidecode and stopword filtering, build tf-idf weighted inverted indexes for precise and scalable candidate generation, and classify pairs using a CatBoost model enriched with 36 distinct similarity features. Additionally, we use a custom France-specific heuristic filter for geographical precision.

---

## 2. Methodology

### 2.1 Problem Analysis
During EDA, we observed significant variations in company names (e.g., abbreviations, suffix inconsistencies) and addresses (e.g., landmarks, number formats). Missing fields and non-ASCII character transliterations were rampant. The France test-set introduced specific house numbering formats requiring tailored heuristics.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier Pipeline  
**Core Innovation:** Implementing rigorous token combinations (number + word) with IDF scoring for blocking, ensuring highly precise and manageable candidate sets. Utilizing exactly 36 structural and fuzzy string features to provide robust signals to the CatBoost classifier.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Name tokens, name-token bigrams, address tokens, and combination tokens of (number + word).
- **Candidate pairs generated:** Top 120 highest-scoring candidates per S1 entity based on IDF-weighted feature overlap.
- **How you ensured true matches were not lost:** By heavily penalizing common stopwords (like 'st', 'inc', 'ltd') and prioritizing rare, distinguishing tokens (like specific house numbers and rare business words).

---

## 4. Matching Model

**Features used:**
- Name features: Fuzz ratio variants (partial, token_sort, token_set), Jaro-Winkler, Exact match, Jaccard overlap, Substring containment, Length differences, missing indicators.
- Address features: Fuzz ratio variants, Jaro-Winkler, Jaccard digit overlap, first-number agreement, length differences, missing indicators.
- Other: Token counts for names and addresses.

**Model type:** CatBoostClassifier (depth=6, iterations=300).
**Threshold selection method:** F_0.5 optimization focusing on a conservative precision threshold (probability >= 0.625) to avoid false merges. France specifically filters candidates if street similarity >= 75 and no conflicting house numbers.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** Optimized via high-precision classification.
- **Common false positives (wrong merges):** Entities sharing same building complex and similar industry keywords.
- **Common false negatives (missed matches):** Drastically different DBA (Doing Business As) names not captured by string similarities.

---

## 6. Conclusion
The combination of robust text normalization, IDF-weighted blocking, and a precision-focused CatBoost model effectively tackles the entity resolution problem. The France-specific logic optimally filters unseen regions while preserving high-confidence matches.

---

## Appendix

### A. Code Artefacts
Our complete code is located in `code/business_entity_resolution/src/`. The primary entry point is `pipeline.py`, which:
- Preprocesses both training and test data.
- Blocks and extracts features for both sets.
- Trains a `model.cbm` CatBoost classifier on the training split.
- Uses `model.cbm` to predict matches on the test set, applying the France filter.
- Outputs `matching_results.tsv` and `candidate_pairs.tsv` to the `output/` directory.

### B. Additional Results
*(Not applicable)*
