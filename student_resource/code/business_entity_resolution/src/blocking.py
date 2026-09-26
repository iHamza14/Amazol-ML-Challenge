import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_sparse_top_k(X_s1, X_s23, top_k=15, threshold=0.3, batch_size=1000):
    """
    Computes top-k nearest neighbors using sparse matrix dot product.
    Processed in batches to prevent Terabyte Memory explosions.
    """
    candidates = []
    
    for start_idx in tqdm(range(0, X_s1.shape[0], batch_size), desc="Finding Top Matches"):
        end_idx = min(start_idx + batch_size, X_s1.shape[0])
        
        # Only calculate dot product for a small chunk of S1 at a time!
        batch_sim = X_s1[start_idx:end_idx].dot(X_s23.T)
        
        for i in range(batch_sim.shape[0]):
            row = batch_sim.getrow(i)
            if row.nnz == 0: continue
            data, indices = row.data, row.indices
            valid = data >= threshold
            data, indices = data[valid], indices[valid]
            if len(data) == 0: continue
            if len(data) > top_k:
                top_k_idx = np.argpartition(data, -top_k)[-top_k:]
                indices = indices[top_k_idx]
                
            global_i = start_idx + i
            for j in indices:
                candidates.append((global_i, j))
                
    return candidates

def generate_candidates(df_s1, df_s2_s3, top_k=15):
    logging.info("Running hierarchical sparse blocking...")
    all_candidates = set()
    
    df_s1 = df_s1.reset_index(drop=True)
    df_s2_s3 = df_s2_s3.reset_index(drop=True)
    
    countries = df_s1['country_norm'].unique()
    
    for country in countries:
        logging.info(f"Blocking for country: {country}")
        
        mask1 = df_s1['country_norm'] == country
        mask23 = df_s2_s3['country_norm'] == country
        
        s1_country = df_s1[mask1]
        s23_country = df_s2_s3[mask23]
        
        if s1_country.empty or s23_country.empty:
            continue
            
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2, max_features=100000)
        
        s1_names = s1_country['name_core'].tolist()
        s23_names = s23_country['name_core'].tolist()
        
        vectorizer.fit(s1_names + s23_names)
        X_s1 = vectorizer.transform(s1_names)
        X_s23 = vectorizer.transform(s23_names)
        
        local_candidates = get_sparse_top_k(X_s1, X_s23, top_k=top_k, threshold=0.3, batch_size=1000)
        
        s1_ids = s1_country['entity_id'].values
        s23_ids = s23_country['entity_id'].values
        
        for local_i, local_j in local_candidates:
            all_candidates.add((s1_ids[local_i], s23_ids[local_j]))
            
    return list(all_candidates)
