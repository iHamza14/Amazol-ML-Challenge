import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from sentence_transformers import SentenceTransformer
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_sparse_top_k(X_s1, X_s23, top_k=15, threshold=0.3):
    similarity_matrix = X_s1.dot(X_s23.T)
    candidates = []
    for i in range(similarity_matrix.shape[0]):
        row = similarity_matrix.getrow(i)
        if row.nnz == 0: continue
        data, indices = row.data, row.indices
        valid = data >= threshold
        data, indices = data[valid], indices[valid]
        if len(data) == 0: continue
        if len(data) > top_k:
            top_k_idx = np.argpartition(data, -top_k)[-top_k:]
            indices = indices[top_k_idx]
        for j in indices:
            candidates.append((i, j))
    return candidates

def generate_candidates(df_s1, df_s2_s3, top_k=15):
    """Hierarchical Blocking: TF-IDF (Lexical) + BGE-Large (Semantic)"""
    logging.info("Running Deep Hierarchical Blocking...")
    all_candidates = set()
    
    df_s1 = df_s1.reset_index(drop=True)
    df_s2_s3 = df_s2_s3.reset_index(drop=True)
    countries = df_s1['country_norm'].unique()
    
    # Check GPU availability for dense embeddings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Loading BGE-Large Dense Encoder on {device}...")
    embedder = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)
    
    for country in countries:
        logging.info(f"Blocking for country: {country}")
        
        mask1 = df_s1['country_norm'] == country
        mask23 = df_s2_s3['country_norm'] == country
        
        s1_country = df_s1[mask1]
        s23_country = df_s2_s3[mask23]
        
        if s1_country.empty or s23_country.empty:
            continue
            
        s1_ids = s1_country['entity_id'].values
        s23_ids = s23_country['entity_id'].values
        
        # --- 1. TF-IDF Lexical Blocking ---
        logging.info(f"[{country}] TF-IDF Retrieval...")
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2, max_features=100000)
        s1_names = s1_country['name_core'].tolist()
        s23_names = s23_country['name_core'].tolist()
        
        vectorizer.fit(s1_names + s23_names)
        X_s1 = vectorizer.transform(s1_names)
        X_s23 = vectorizer.transform(s23_names)
        
        lexical_pairs = get_sparse_top_k(X_s1, X_s23, top_k=top_k, threshold=0.3)
        for local_i, local_j in lexical_pairs:
            all_candidates.add((s1_ids[local_i], s23_ids[local_j]))
            
        # --- 2. Dense Semantic Blocking (FAISS) ---
        logging.info(f"[{country}] Dense FAISS Retrieval...")
        def create_doc(row):
            return f"{row['name_norm']} {row['address_norm']}"
            
        s1_docs = s1_country.apply(create_doc, axis=1).tolist()
        s23_docs = s23_country.apply(create_doc, axis=1).tolist()
        
        # Encode (batch_size optimized for GPU)
        logging.info("Encoding S23 corpus...")
        s23_embs = embedder.encode(s23_docs, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
        logging.info("Encoding S1 queries...")
        s1_embs = embedder.encode(s1_docs, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
        
        # FAISS Index
        dim = s23_embs.shape[1]
        # Use Inner Product (Cosine Similarity because normalized)
        index = faiss.IndexFlatIP(dim)
        # If GPU is available, move index to GPU for blazing fast search
        if device == 'cuda':
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            
        index.add(s23_embs)
        scores, indices = index.search(s1_embs, top_k)
        
        for i, s1_local in enumerate(s1_ids):
            for j, s23_idx in enumerate(indices[i]):
                if scores[i][j] > 0.7:  # Semantic similarity threshold
                    all_candidates.add((s1_local, s23_ids[s23_idx]))
                    
    return list(all_candidates)
