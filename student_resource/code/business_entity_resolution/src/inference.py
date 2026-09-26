import pandas as pd
import numpy as np
import os
import argparse
import logging
import torch
import gc
from collections import defaultdict
from sentence_transformers import CrossEncoder
from preprocess import load_data, preprocess_dataframe
from blocking import generate_candidates

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

def run_inference(test_dir, output_dir, model_path="deberta_er_model", threshold=0.8):
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info("Loading and preprocessing test data chunk-by-chunk to save RAM...")
    
    df_s1 = load_data(os.path.join(test_dir, "test_source1.tsv"))
    df_s1 = preprocess_dataframe(df_s1)
    
    df_s2 = load_data(os.path.join(test_dir, "test_source2.tsv"))
    df_s2 = preprocess_dataframe(df_s2)
    
    df_s3 = load_data(os.path.join(test_dir, "test_source3.tsv"))
    df_s3 = preprocess_dataframe(df_s3)
    
    logging.info("Concatenating S2 and S3...")
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    
    del df_s2, df_s3
    gc.collect()
    
    logging.info("Generating candidates (Lexical + Dense)...")
    candidate_pairs = generate_candidates(df_s1, df_s23)
    s1_all_ids = df_s1['entity_id'].tolist()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Loading Fine-Tuned DeBERTa from {model_path} on {device}...")
    try:
        model = CrossEncoder(model_path, device=device)
    except:
        logging.error(f"Could not load {model_path}. Please run train.py first!")
        return

    s1_dict = df_s1.set_index('entity_id').to_dict('index')
    s23_dict = df_s23.set_index('entity_id').to_dict('index')
    
    inference_texts = []
    for s1_id, s23_id in candidate_pairs:
        r1, r2 = s1_dict[s1_id], s23_dict[s23_id]
        text1 = f"{r1['name_norm']} | {r1['address_norm']}"
        text2 = f"{r2['name_norm']} | {r2['address_norm']}"
        inference_texts.append([text1, text2])
        
    logging.info("Running Cross-Encoder Inference...")
    preds = model.predict(inference_texts, batch_size=256, show_progress_bar=True)
    probs = 1 / (1 + np.exp(-preds))
    
    predicted_pairs = []
    for i, prob in enumerate(probs):
        if prob >= threshold:
            predicted_pairs.append(candidate_pairs[i])
            
    match_path = os.path.join(output_dir, "matching_results.tsv")
    save_matches(predicted_pairs, s1_all_ids, match_path)
    logging.info("Inference complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="../../../dataset/test")
    parser.add_argument("--output-dir", default="../../../output")
    parser.add_argument("--model-path", default="deberta_er_model")
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()
    
    run_inference(args.test_dir, args.output_dir, args.model_path, args.threshold)
