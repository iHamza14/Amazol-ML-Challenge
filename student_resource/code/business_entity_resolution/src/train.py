import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from preprocess import load_data, preprocess_dataframe
from blocking import generate_candidates
from features import extract_features
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_training_labels(candidate_pairs, ground_truth_df):
    """Assigns 1 to true matches and 0 to false matches in the candidate set."""
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

def train_xgboost():
    # Load Data (using a small sample for demonstration if running locally)
    # df_s1 = load_data("../../dataset/train/train_source1.tsv").head(1000)
    # df_s2 = load_data("../../dataset/train/train_source2.tsv").head(1000)
    # df_s3 = load_data("../../dataset/train/train_source3.tsv").head(1000)
    # gt = load_data("../../dataset/train/train_ground_truth.tsv")
    
    # Normally we load everything, but this is a template
    pass

if __name__ == "__main__":
    logging.info("Training script placeholder.")
