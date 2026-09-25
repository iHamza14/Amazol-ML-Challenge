# Business Entity Resolution Pipeline

This folder contains the complete machine learning pipeline for the Business Entity Resolution challenge. The pipeline is optimized for precision (F0.5) and high-recall blocking.

## Prerequisites

Install the required packages in your Python environment:
```bash
pip install -r ../requirements.txt
```

*(Note: We assume a compute instance like SageMaker ml.t3.medium or larger is used for training and inference)*

## Execution Instructions

### 1. Training (Optional / If starting from scratch)
The `train.py` script will load the training data, execute the blocking stage to generate candidates, extract features using `rapidfuzz`, and train an XGBoost baseline model.

```bash
python3 train.py
```
*This will output `xgb_model.json` to be used for inference.*

### 2. Inference
To generate the final `.tsv` files for submission, run `inference.py`. This script reads the test files, generates candidate pairs (Blocking), extracts features, applies the XGBoost model (with a strict threshold), and formats the output perfectly.

Run from the `src` directory:
```bash
python3 inference.py --test-dir ../../../dataset/test --output-dir ../../../output
```

If you don't have a trained `xgb_model.json`, the script will gracefully fallback to a high-precision heuristic (Jaro-Winkler > 95 + Country Match) to demonstrate the end-to-end flow.

### 3. Validation
Run the provided validation script to ensure formatting is correct before uploading:
```bash
python3 ../../../utils/validate_submission.py \
    --matching ../../../output/matching_results.tsv \
    --candidate ../../../output/candidate_pairs.tsv \
    --test-dir ../../../dataset/test
```

## Pipeline Architecture
- **Preprocessing:** Expands abbreviations (`corp`, `pvt`, `ltd`), standardizes addresses, and strips legal suffixes for a 'core name'.
- **Blocking:** Combines TF-IDF, BM25, and Country-Prefix rules to maximize recall.
- **Features:** Computes string similarities (Jaro-Winkler, Levenshtein, Token Sort/Set).
- **Model:** XGBoost trained to optimize F0.5.
