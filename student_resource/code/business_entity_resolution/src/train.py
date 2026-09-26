import pandas as pd
import numpy as np
import logging
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from preprocess import load_data, preprocess_dataframe
from blocking import generate_candidates

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

def calculate_f05(y_true, y_pred):
    # This is pair-level F0.5. To be strict, we'd do macro entity-level, 
    # but this is fast for threshold sweeping.
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision == 0 and recall == 0:
        return 0.0
    return (1.25 * precision * recall) / ((0.25 * precision) + recall)

def train_cross_encoder(limit=None):
    logging.info(f"Loading datasets (limit={limit})...")
    df_s1 = load_data("../../../dataset/train/train_source1.tsv")
    df_s2 = load_data("../../../dataset/train/train_source2.tsv")
    df_s3 = load_data("../../../dataset/train/train_source3.tsv")
    if limit:
        df_s1 = df_s1.head(limit)
        df_s2 = df_s2.head(limit)
        df_s3 = df_s3.head(limit)
        
    gt = load_data("../../../dataset/train/train_ground_truth.tsv")
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    
    logging.info("Preprocessing...")
    df_s1 = preprocess_dataframe(df_s1)
    df_s23 = preprocess_dataframe(df_s23)
    
    logging.info("Generating candidates (Lexical + Dense)...")
    candidate_pairs = generate_candidates(df_s1, df_s23)
    
    if not candidate_pairs:
        logging.warning("No candidates generated!")
        return
        
    logging.info(f"Generated {len(candidate_pairs)} candidate pairs.")
    labels = create_training_labels(candidate_pairs, gt)
    
    # Prepare text for CrossEncoder: "[CLS] name1 | addr1 [SEP] name2 | addr2 [SEP]"
    s1_dict = df_s1.set_index('entity_id').to_dict('index')
    s23_dict = df_s23.set_index('entity_id').to_dict('index')
    
    train_examples = []
    val_texts = []
    val_labels = []
    
    # 80/20 train/val split logic
    split_idx = int(len(candidate_pairs) * 0.8)
    
    for i, (s1_id, s23_id) in enumerate(candidate_pairs):
        r1, r2 = s1_dict[s1_id], s23_dict[s23_id]
        text1 = f"{r1['name_norm']} | {r1['address_norm']}"
        text2 = f"{r2['name_norm']} | {r2['address_norm']}"
        label = float(labels[i])
        
        if i < split_idx:
            train_examples.append(InputExample(texts=[text1, text2], label=label))
        else:
            val_texts.append([text1, text2])
            val_labels.append(label)
            
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Initializing DeBERTa-v3 Cross-Encoder on {device}...")
    model = CrossEncoder('cross-encoder/nli-deberta-v3-base', num_labels=1, device=device)
    
    logging.info("Fine-tuning DeBERTa... (This will take a while!)")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=2,
        warmup_steps=100,
        show_progress_bar=True
    )
    
    model_path = "deberta_er_model"
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")
    
    # --- DYNAMIC THRESHOLD SWEEP ---
    logging.info("Predicting on Validation Set to find optimal F0.5 Threshold...")
    val_preds = model.predict(val_texts, batch_size=128, show_progress_bar=True)
    # Apply sigmoid if model outputs logits
    val_probs = 1 / (1 + np.exp(-val_preds))
    
    best_t = 0.5
    best_f05 = 0.0
    val_labels_arr = np.array(val_labels)
    
    for t in np.arange(0.1, 0.95, 0.05):
        y_pred = (val_probs >= t).astype(int)
        score = calculate_f05(val_labels_arr, y_pred)
        if score > best_f05:
            best_f05 = score
            best_t = t
            
    logging.info(f"=====================================")
    logging.info(f"🎯 OPTIMAL THRESHOLD FOUND: {best_t:.3f}")
    logging.info(f"📈 VALIDATION F0.5 SCORE: {best_f05:.4f}")
    logging.info(f"Use this threshold in inference.py!")
    logging.info(f"=====================================")

if __name__ == "__main__":
    train_cross_encoder(limit=None)
