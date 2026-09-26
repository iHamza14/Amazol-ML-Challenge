import pandas as pd
import numpy as np
import pickle
import logging
from preprocess import load_data, preprocess_dataframe
from blocking import generate_candidates
from features import extract_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_f05_macro(predictions_dict, ground_truth_dict, s1_all_ids):
    f05_scores = []
    
    for s1 in s1_all_ids:
        y_true = ground_truth_dict.get(s1, set())
        y_pred = predictions_dict.get(s1, set())
        
        if len(y_true) == 0:
            if len(y_pred) == 0:
                f05_scores.append(1.0)
            else:
                f05_scores.append(0.0)
            continue
            
        if len(y_pred) == 0:
            f05_scores.append(0.0)
            continue
            
        tp = len(y_true.intersection(y_pred))
        fp = len(y_pred - y_true)
        fn = len(y_true - y_pred)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision == 0 and recall == 0:
            f05_scores.append(0.0)
        else:
            f05 = (1.25 * precision * recall) / ((0.25 * precision) + recall)
            f05_scores.append(f05)
            
    return np.mean(f05_scores)

def evaluate(model_path="fallback_model.pkl", limit=2000, offset=10000):
    logging.info(f"Loading S1 validation data (Rows {offset} to {offset+limit})...")
    
    # We only take a subset of S1
    df_s1 = load_data("../../../dataset/train/train_source1.tsv").iloc[offset:offset+limit]
    s1_ids = df_s1['entity_id'].tolist()
    
    # We MUST find the true matches for these specific S1 rows
    gt_full = load_data("../../../dataset/train/train_ground_truth.tsv")
    gt = gt_full[gt_full['source1_entity_id'].isin(s1_ids)]
    
    gt_dict = {}
    true_s23_ids = set()
    for _, row in gt.iterrows():
        s1 = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',')
        valid_matches = set(m for m in matches if m)
        gt_dict[s1] = valid_matches
        true_s23_ids.update(valid_matches)
        
    logging.info(f"Identified {len(true_s23_ids)} true S2/S3 matches for these S1 rows.")
    
    # We load S2 and S3, but to prevent RAM explosion locally, we filter them to include:
    # 1. All true matches for our S1 set
    # 2. 50,000 random rows as noise (haystack)
    logging.info("Loading S2 and S3 (with smart filtering for local eval)...")
    df_s2_full = load_data("../../../dataset/train/train_source2.tsv")
    df_s3_full = load_data("../../../dataset/train/train_source3.tsv")
    df_s23_full = pd.concat([df_s2_full, df_s3_full], ignore_index=True)
    
    # Create the smart subset
    mask_true = df_s23_full['entity_id'].isin(true_s23_ids)
    df_s23_true = df_s23_full[mask_true]
    df_s23_noise = df_s23_full[~mask_true].sample(n=50000, random_state=42)
    df_s23 = pd.concat([df_s23_true, df_s23_noise], ignore_index=True)
    
    logging.info("Preprocessing...")
    df_s1 = preprocess_dataframe(df_s1)
    df_s23 = preprocess_dataframe(df_s23)
    
    logging.info("Generating candidates (Blocking)...")
    candidate_pairs = generate_candidates(df_s1, df_s23)
    
    logging.info("Extracting features...")
    feature_df = extract_features(df_s1, df_s23, candidate_pairs)
    
    logging.info("Loading model and running predictions...")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        logging.error("Model not found! Did you run train.py first?")
        return
        
    feature_cols = [c for c in feature_df.columns if c not in ['s1_id', 's23_id', 'label']]
    X = feature_df[feature_cols] if not feature_df.empty else pd.DataFrame()
    
    predictions_dict = {}
    if not feature_df.empty:
        probs = model.predict_proba(X)[:, 1]
        for i, prob in enumerate(probs):
            if prob >= 0.7:  # Threshold
                s1 = feature_df.iloc[i]['s1_id']
                s23 = feature_df.iloc[i]['s23_id']
                if s1 not in predictions_dict:
                    predictions_dict[s1] = set()
                predictions_dict[s1].add(s23)
                
    logging.info("Calculating Macro F0.5 Score...")
    score = calculate_f05_macro(predictions_dict, gt_dict, s1_ids)
    
    print("\n" + "="*50)
    print(f"📊 LOCAL VALIDATION F0.5 SCORE: {score:.5f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    evaluate()
