import pandas as pd
import numpy as np
import os
import argparse
import logging
import pickle
from collections import defaultdict
from preprocess import load_data, preprocess_dataframe
from blocking import generate_candidates
from features import extract_features

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except (ImportError, Exception) as e:
    XGB_AVAILABLE = False
    logging.warning(f"XGBoost could not be loaded due to Mac architecture. Falling back to Scikit-Learn.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_candidates(candidate_pairs, s1_ids, output_path):
    logging.info(f"Saving candidates to {output_path}")
    grouped = defaultdict(list)
    for s1, s23 in candidate_pairs:
        grouped[s1].append(s23)
        
    with open(output_path, 'w') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in s1_ids:
            cands = grouped.get(s1, [])
            cands = list(dict.fromkeys(cands))
            f.write(f"{s1}\t{','.join(cands)}\n")

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

def run_inference(test_dir, output_dir, model_path="xgb_model.json", threshold=0.7, limit=None):
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info("Loading test data...")
    df_s1 = load_data(os.path.join(test_dir, "test_source1.tsv"))
    if limit:
        df_s1 = df_s1.head(limit)
        logging.info(f"Limited Source 1 to {limit} rows")
    df_s2 = load_data(os.path.join(test_dir, "test_source2.tsv"))
    if limit:
        df_s2 = df_s2.head(limit)
        logging.info(f"Limited Source 2 to {limit} rows")
    df_s3 = load_data(os.path.join(test_dir, "test_source3.tsv"))
    if limit:
        df_s3 = df_s3.head(limit)
        logging.info(f"Limited Source 3 to {limit} rows")
    
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    
    logging.info("Preprocessing...")
    df_s1 = preprocess_dataframe(df_s1)
    df_s23 = preprocess_dataframe(df_s23)
    
    logging.info("Generating candidates (Blocking)...")
    candidate_pairs = generate_candidates(df_s1, df_s23)
    
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    s1_all_ids = df_s1['entity_id'].tolist()
    save_candidates(candidate_pairs, s1_all_ids, cand_path)
    
    logging.info("Extracting features...")
    feature_df = extract_features(df_s1, df_s23, candidate_pairs)
    
    logging.info("Running inference...")
    predicted_pairs = []
    
    feature_cols = [c for c in feature_df.columns if c not in ['s1_id', 's23_id', 'label']]
    X = feature_df[feature_cols] if not feature_df.empty else pd.DataFrame()
    
    if not feature_df.empty:
        if XGB_AVAILABLE and os.path.exists(model_path):
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            probs = model.predict_proba(X)[:, 1]
            for i, prob in enumerate(probs):
                if prob >= threshold:
                    predicted_pairs.append((feature_df.iloc[i]['s1_id'], feature_df.iloc[i]['s23_id']))
        elif os.path.exists("fallback_model.pkl"):
            with open("fallback_model.pkl", "rb") as f:
                model = pickle.load(f)
            probs = model.predict_proba(X)[:, 1]
            for i, prob in enumerate(probs):
                if prob >= threshold:
                    predicted_pairs.append((feature_df.iloc[i]['s1_id'], feature_df.iloc[i]['s23_id']))
        else:
            logging.warning("No model found. Using heuristic threshold on Jaro-Winkler distance for demonstration.")
            for i, row in feature_df.iterrows():
                if row['name_jaro_winkler'] > 95 and row['same_country'] == 1:
                    predicted_pairs.append((row['s1_id'], row['s23_id']))
                
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
