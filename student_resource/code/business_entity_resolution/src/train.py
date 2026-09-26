import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocess import load_data, preprocess_dataframe
from blocking import generate_candidates
from features import extract_features
import logging
import pickle

# Attempt to load XGBoost. If architecture mismatch occurs on Mac, fallback to Scikit-Learn.
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except (ImportError, Exception) as e:
    from sklearn.ensemble import HistGradientBoostingClassifier
    XGB_AVAILABLE = False
    logging.warning(f"XGBoost failed to load due to Mac architecture mismatch. Using HistGradientBoostingClassifier fallback.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_training_labels(candidate_pairs, ground_truth_df):
    gt_dict = {}
    for _, row in ground_truth_df.iterrows():
        s1 = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',')
        gt_dict[s1] = set(m for m in matches if m)
        
    labels = []
    for s1, s23 in candidate_pairs:
        if s1 in gt_dict and s23 in gt_dict[s1]:
            labels.append(1)
        else:
            labels.append(0)
    return labels

def train_xgboost(limit=10000):
    logging.info(f"Loading datasets (limit={limit})...")
    df_s1 = load_data("../../../dataset/train/train_source1.tsv").head(limit)
    df_s2 = load_data("../../../dataset/train/train_source2.tsv").head(limit)
    df_s3 = load_data("../../../dataset/train/train_source3.tsv").head(limit)
    gt = load_data("../../../dataset/train/train_ground_truth.tsv")
    
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    
    logging.info("Preprocessing...")
    df_s1 = preprocess_dataframe(df_s1)
    df_s23 = preprocess_dataframe(df_s23)
    
    logging.info("Generating candidates...")
    candidate_pairs = generate_candidates(df_s1, df_s23)
    
    if not candidate_pairs:
        logging.warning("No candidates generated! Please increase the dataset limit.")
        return
        
    logging.info(f"Generated {len(candidate_pairs)} candidate pairs. Extracting features...")
    feature_df = extract_features(df_s1, df_s23, candidate_pairs)
    
    logging.info("Creating training labels...")
    labels = create_training_labels(candidate_pairs, gt)
    feature_df['label'] = labels
    
    feature_cols = [c for c in feature_df.columns if c not in ['s1_id', 's23_id', 'label']]
    X = feature_df[feature_cols]
    y = feature_df['label']
    
    logging.info(f"Training Model on {len(X)} samples...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if XGB_AVAILABLE:
        model = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, 
            use_label_encoder=False, eval_metric='logloss'
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=10)
        model.save_model("xgb_model.json")
        logging.info("Model saved to xgb_model.json")
    else:
        # Mac Fallback Model
        model = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=100, max_depth=5)
        model.fit(X_train, y_train)
        logging.info(f"Validation Accuracy: {model.score(X_val, y_val):.4f}")
        with open("fallback_model.pkl", "wb") as f:
            pickle.dump(model, f)
        logging.info("Model saved to fallback_model.pkl")

if __name__ == "__main__":
    train_xgboost(limit=10000)
