# Business Entity Resolution Pipeline

This directory contains the end-to-end pipeline for the ML Challenge 2026 Business Entity Resolution task.

## Prerequisites

Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

## Running the Pipeline

To run the full pipeline (training, feature extraction, inference, filtering, and output generation), execute:

```bash
cd src
python pipeline.py
```

### What it does:
1. **Preprocessing**: Normalizes text (unidecode, expanding symbols, stopwords).
2. **Blocking**: Generates candidate pairs using IDF-weighted token features.
3. **Feature Extraction**: Computes exactly 36 comparison features for CatBoost.
4. **Training**: Trains a CatBoost classifier on the `dataset/train/` split (if `model.cbm` does not exist).
5. **Inference**: Scores candidates on `dataset/test/` and applies a threshold of `0.625`.
6. **France Filter**: Applies specific house number + street name heuristic rules for French entities.
7. **Output**: Writes `matching_results.tsv` and `candidate_pairs.tsv` to the root `output/` directory in the correct format.
