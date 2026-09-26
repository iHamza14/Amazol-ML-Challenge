import pandas as pd
import numpy as np
import os
import argparse
import logging
import pickle
import gc
from collections import defaultdict
from preprocess import load_data, preprocess_dataframe
from blocking import generate_candidates
from features import extract_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_matches(predictions, s1_ids, output_path):
    logging.info(f"Saving matches to {output_path}")
    grouped = defaultdict(list)
    for s1, s23 in predictions:
        grouped[s1].append(s23)
        
    with open(output_path, 'w') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in s1_ids:
            matches = grouped.get(s1, [])
            matches = list(dict.fromkeys(matches))
            f.write(f"{s1}\t{','.join(matches)}\n")

def run_inference(test_dir, output_dir, model_path="xgb_model.json", limit=None):
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info("Loading and preprocessing test data chunk-by-chunk to save RAM...")
    
    # Load and preprocess S1
    df_s1 = load_data(os.path.join(test_dir, "test_source1.tsv"))
    if limit: df_s1 = df_s1.head(limit)
    df_s1 = preprocess_dataframe(df_s1)
    
    # Load and preprocess S2
    df_s2 = load_data(os.path.join(test_dir, "test_source2.tsv"))
    if limit: df_s2 = df_s2.head(limit)
    df_s2 = preprocess_dataframe(df_s2)
    
    # Load and preprocess S3
    df_s3 = load_data(os.path.join(test_dir, "test_source3.tsv"))
    if limit: df_s3 = df_s3.head(limit)
    df_s3 = preprocess_dataframe(df_s3)
    
    logging.info("Concatenating S2 and S3...")
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    
    # Free up memory immediately
    del df_s2, df_s3
    gc.collect()
    
    logging.info("Generating candidates (Blocking)...")
    candidate_pairs = generate_candidates(df_s1, df_s23)
    
    s1_all_ids = df_s1['entity_id'].tolist()
    if not candidate_pairs:
        save_matches([], s1_all_ids, os.path.join(output_dir, "matching_results.tsv"))
        return
        
    logging.info(f"Generated {len(candidate_pairs)} candidate pairs. Extracting features...")
    feature_df = extract_features(df_s1, df_s23, candidate_pairs)
    feature_cols = [c for c in feature_df.columns if c not in ['s1_id', 's23_id']]
    X = feature_df[feature_cols]
    
    logging.info(f"Loading model from {model_path}...")
    
    predicted_pairs = []
    if model_path.endswith('.json'):
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        preds = model.predict_proba(X)[:, 1]
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        preds = model.predict_proba(X)[:, 1]
        
    for i, prob in enumerate(preds):
        if prob >= 0.7:  # Match threshold
            predicted_pairs.append(candidate_pairs[i])
            
    match_path = os.path.join(output_dir, "matching_results.tsv")
    save_matches(predicted_pairs, s1_all_ids, match_path)
    logging.info("Inference complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="../../../dataset/test")
    parser.add_argument("--output-dir", default="../../../output")
    parser.add_argument("--model-path", default="xgb_model.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    run_inference(args.test_dir, args.output_dir, args.model_path, limit=args.limit)
