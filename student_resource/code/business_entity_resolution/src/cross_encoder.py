"""
Cross-Encoder module — Phase 2 accuracy boost (GPU only).

Fine-tunes microsoft/mdeberta-v3-base (multilingual) as a cross-encoder
on training data, then scores candidate pairs. The score becomes an
additional feature (#41) for CatBoost.

Usage:
  1. python cross_encoder.py train     # Fine-tune on training data
  2. python cross_encoder.py score     # Score test candidates
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import logging
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

import config as cfg
from preprocess import load_data, preprocess_dataframe
from evaluate import build_ground_truth_dict

log = logging.getLogger(__name__)

CE_MODEL_DIR = os.path.join(cfg.MODEL_DIR, 'cross_encoder')


def _make_text(name, addr, country):
    """Format entity as structured text for the cross-encoder."""
    parts = []
    if name:
        parts.append(f"Name: {name}")
    if addr:
        parts.append(f"Address: {addr}")
    if country:
        parts.append(f"Country: {country}")
    return " | ".join(parts)


def train_cross_encoder(max_train_pairs=500000):
    """
    Fine-tune a multilingual cross-encoder on training data.
    Uses hard negatives from blocking (much better than random negatives).
    """
    log.info("=" * 60)
    log.info("TRAINING CROSS-ENCODER")
    log.info("=" * 60)

    os.makedirs(CE_MODEL_DIR, exist_ok=True)

    # Load preprocessed training data
    log.info("Loading training data...")
    df_s1 = load_data(os.path.join(cfg.TRAIN_DIR, 'train_source1.tsv'))
    df_s2 = load_data(os.path.join(cfg.TRAIN_DIR, 'train_source2.tsv'))
    df_s3 = load_data(os.path.join(cfg.TRAIN_DIR, 'train_source3.tsv'))
    gt_df = load_data(os.path.join(cfg.TRAIN_DIR, 'train_ground_truth.tsv'))

    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3

    df_s1 = preprocess_dataframe(df_s1)
    df_s23 = preprocess_dataframe(df_s23)
    gt_dict = build_ground_truth_dict(gt_df)

    # Load blocking candidates (from training run)
    cand_path = os.path.join(cfg.MODEL_DIR, 'train_candidates.parquet')
    if os.path.exists(cand_path):
        cand_df = pd.read_parquet(cand_path)
        log.info(f"Loaded {len(cand_df):,} training candidates from cache")
    else:
        log.info("No cached candidates found. Run blocking first:")
        log.info("  python blocking.py")
        return

    # Build training examples with hard negatives
    log.info("Building training examples...")
    s1_lookup = df_s1.set_index('entity_id')
    s23_lookup = df_s23.set_index('entity_id')

    examples = []
    for _, row in cand_df.iterrows():
        s1_id = row['s1_id']
        s23_id = row['s23_id']

        try:
            s1_row = s1_lookup.loc[s1_id]
            s23_row = s23_lookup.loc[s23_id]
        except KeyError:
            continue

        text1 = _make_text(
            str(s1_row['name_norm']), str(s1_row['addr_norm']),
            str(s1_row['country_norm'])
        )
        text2 = _make_text(
            str(s23_row['name_norm']), str(s23_row['addr_norm']),
            str(s23_row['country_norm'])
        )

        label = 1.0 if s23_id in gt_dict.get(s1_id, set()) else 0.0
        examples.append(InputExample(texts=[text1, text2], label=label))

        if len(examples) >= max_train_pairs:
            break

    log.info(f"Created {len(examples):,} training examples")
    n_pos = sum(1 for e in examples if e.label > 0.5)
    log.info(f"  Positive: {n_pos:,}, Negative: {len(examples) - n_pos:,}")

    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"Device: {device}")

    model = CrossEncoder(cfg.CROSS_ENCODER_MODEL, num_labels=1, device=device)

    train_dl = DataLoader(examples, shuffle=True, batch_size=32)

    model.fit(
        train_dataloader=train_dl,
        epochs=cfg.CROSS_ENCODER_EPOCHS,
        warmup_steps=cfg.CROSS_ENCODER_WARMUP,
        show_progress_bar=True,
    )

    model.save(CE_MODEL_DIR)
    log.info(f"Cross-encoder saved → {CE_MODEL_DIR}")


def score_candidates(candidates_path, output_path, source='test'):
    """
    Score candidate pairs using the fine-tuned cross-encoder.
    Saves cross-encoder scores to a parquet file.
    """
    log.info("=" * 60)
    log.info(f"SCORING CANDIDATES ({source})")
    log.info("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CrossEncoder(CE_MODEL_DIR, device=device)
    log.info(f"Loaded cross-encoder from {CE_MODEL_DIR} on {device}")

    # Load candidates
    cand_df = pd.read_parquet(candidates_path)
    log.info(f"Loaded {len(cand_df):,} candidates")

    # Load preprocessed data
    if source == 'test':
        src_dir = cfg.TEST_DIR
        prefix = 'test'
    else:
        src_dir = cfg.TRAIN_DIR
        prefix = 'train'

    df_s1 = load_data(os.path.join(src_dir, f'{prefix}_source1.tsv'))
    df_s2 = load_data(os.path.join(src_dir, f'{prefix}_source2.tsv'))
    df_s3 = load_data(os.path.join(src_dir, f'{prefix}_source3.tsv'))
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3

    df_s1 = preprocess_dataframe(df_s1)
    df_s23 = preprocess_dataframe(df_s23)

    s1_lookup = df_s1.set_index('entity_id')
    s23_lookup = df_s23.set_index('entity_id')

    # Build text pairs
    log.info("Building text pairs...")
    texts = []
    valid_indices = []
    for i, row in cand_df.iterrows():
        try:
            s1_row = s1_lookup.loc[row['s1_id']]
            s23_row = s23_lookup.loc[row['s23_id']]
        except KeyError:
            continue

        t1 = _make_text(
            str(s1_row['name_norm']), str(s1_row['addr_norm']),
            str(s1_row['country_norm'])
        )
        t2 = _make_text(
            str(s23_row['name_norm']), str(s23_row['addr_norm']),
            str(s23_row['country_norm'])
        )
        texts.append([t1, t2])
        valid_indices.append(i)

    log.info(f"Scoring {len(texts):,} pairs...")
    scores = model.predict(
        texts,
        batch_size=cfg.CROSS_ENCODER_BATCH_SIZE,
        show_progress_bar=True
    )

    # Apply sigmoid if needed
    scores = 1.0 / (1.0 + np.exp(-np.array(scores)))

    # Save scores
    score_df = cand_df.iloc[valid_indices][['s1_id', 's23_id']].copy()
    score_df['ce_score'] = scores

    score_df.to_parquet(output_path, index=False)
    log.info(f"Scores saved → {output_path} ({len(score_df):,} pairs)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python cross_encoder.py [train|score]")
        sys.exit(1)

    action = sys.argv[1]

    if action == 'train':
        train_cross_encoder()
    elif action == 'score':
        # Score test candidates
        cand_path = os.path.join(cfg.MODEL_DIR, 'test_candidates.parquet')
        out_path = os.path.join(cfg.MODEL_DIR, 'test_ce_scores.parquet')
        score_candidates(cand_path, out_path, source='test')
    else:
        print(f"Unknown action: {action}")
        print("Usage: python cross_encoder.py [train|score]")
