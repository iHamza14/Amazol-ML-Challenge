import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_sparse_top_k(X_s1, X_s23, top_k=15, threshold=0.4):
    """
    Computes X_s1 * X_s23.T sparsely. 
    Extracts top_k indices per row without ever instantiating a dense matrix.
    """
    # Sparse dot product (keeps memory incredibly low)
    similarity_matrix = X_s1.dot(X_s23.T)
    
    candidates = []
    # Iterate through rows and get top K
    from tqdm import tqdm
    for i in tqdm(range(similarity_matrix.shape[0]), desc="Finding Top Matches"):
        row = similarity_matrix.getrow(i)
        if row.nnz == 0:
            continue
        
        # Get data and indices
        data = row.data
        indices = row.indices
        
        # Filter by threshold
        valid = data >= threshold
        data = data[valid]
        indices = indices[valid]
        
        if len(data) == 0:
            continue
            
        # Sort and get top K
        if len(data) > top_k:
            top_k_idx = np.argpartition(data, -top_k)[-top_k:]
            indices = indices[top_k_idx]
            
        for j in indices:
            candidates.append((i, j))
            
    return candidates

def generate_candidates(df_s1, df_s2_s3, top_k=15):
    """Hierarchical, memory-efficient candidate generation."""
    logging.info("Running hierarchical sparse blocking...")
    all_candidates = []
    
    # Reset indices to map them easily
    df_s1 = df_s1.reset_index(drop=True)
    df_s2_s3 = df_s2_s3.reset_index(drop=True)
    
    # We block STRICTLY by country to prevent useless cross-country matches
    countries = df_s1['country_norm'].unique()
    
    for country in countries:
        logging.info(f"Blocking for country: {country}")
        
        # 1. Filter dataframes
        mask1 = df_s1['country_norm'] == country
        mask23 = df_s2_s3['country_norm'] == country
        
        s1_country = df_s1[mask1]
        s23_country = df_s2_s3[mask23]
        
        if s1_country.empty or s23_country.empty:
            continue
            
        # 2. Vectorize names (sparse char n-grams)
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2, max_features=100000)
        
        s1_names = s1_country['name_core'].tolist()
        s23_names = s23_country['name_core'].tolist()
        
        vectorizer.fit(s1_names + s23_names)
        X_s1 = vectorizer.transform(s1_names)
        X_s23 = vectorizer.transform(s23_names)
        
        # 3. Get Sparse Top K
        local_candidates = get_sparse_top_k(X_s1, X_s23, top_k=top_k, threshold=0.3)
        
        # 4. Map local indices back to global entity_ids
        s1_ids = s1_country['entity_id'].values
        s23_ids = s23_country['entity_id'].values
        
        for local_i, local_j in local_candidates:
            all_candidates.append((s1_ids[local_i], s23_ids[local_j]))
            
    # Use set to drop exact duplicates if any
    return list(set(all_candidates))
