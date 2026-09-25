import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prefix_blocking(df_s1, df_s2_s3):
    """Country + First 3 chars of normalized name."""
    logging.info("Running prefix blocking...")
    s1_grouped = df_s1.groupby('prefix_block')['entity_id'].apply(list).to_dict()
    s23_grouped = df_s2_s3.groupby('prefix_block')['entity_id'].apply(list).to_dict()
    
    candidates = []
    for prefix, s1_ids in s1_grouped.items():
        if prefix in s23_grouped:
            s23_ids = s23_grouped[prefix]
            for s1_id in s1_ids:
                for s23_id in s23_ids:
                    candidates.append((s1_id, s23_id))
    return set(candidates)

def tfidf_blocking(df_s1, df_s2_s3, threshold=0.7):
    """TF-IDF Cosine Similarity on core name."""
    logging.info("Running TF-IDF blocking...")
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    
    s1_names = df_s1['name_core'].tolist()
    s23_names = df_s2_s3['name_core'].tolist()
    
    vectorizer.fit(s1_names + s23_names)
    X_s1 = vectorizer.transform(s1_names)
    X_s23 = vectorizer.transform(s23_names)
    
    # Process in batches to save memory
    batch_size = 10000
    candidates = []
    for i in range(0, X_s1.shape[0], batch_size):
        sim_matrix = cosine_similarity(X_s1[i:i+batch_size], X_s23)
        rows, cols = np.where(sim_matrix >= threshold)
        for r, c in zip(rows, cols):
            candidates.append((df_s1.iloc[i+r]['entity_id'], df_s2_s3.iloc[c]['entity_id']))
    return set(candidates)

def bm25_blocking(df_s1, df_s2_s3, top_k=5):
    """BM25 on addresses."""
    logging.info("Running BM25 blocking...")
    tokenized_s23_addresses = [str(addr).split() for addr in df_s2_s3['address_norm']]
    
    # Filter empty addresses
    tokenized_s23_addresses = [addr if len(addr) > 0 else ['EMPTY_ADDR'] for addr in tokenized_s23_addresses]
    
    bm25 = BM25Okapi(tokenized_s23_addresses)
    
    candidates = []
    for i, s1_row in df_s1.iterrows():
        query = str(s1_row['address_norm']).split()
        if not query:
            continue
        scores = bm25.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        for idx in top_indices:
            if scores[idx] > 0: # Only keep if there's some overlap
                candidates.append((s1_row['entity_id'], df_s2_s3.iloc[idx]['entity_id']))
    return set(candidates)

def dense_blocking(df_s1, df_s2_s3, top_k=5):
    """Dense bi-encoder + FAISS."""
    logging.info("Running dense blocking...")
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    
    def create_doc(row):
        return f"{row['name_norm']} {row['address_norm']} {row['country_norm']}"
    
    s1_docs = df_s1.apply(create_doc, axis=1).tolist()
    s23_docs = df_s2_s3.apply(create_doc, axis=1).tolist()
    
    s1_embeddings = model.encode(s1_docs, normalize_embeddings=True, show_progress_bar=True)
    s23_embeddings = model.encode(s23_docs, normalize_embeddings=True, show_progress_bar=True)
    
    dim = s23_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(s23_embeddings)
    
    scores, indices = index.search(s1_embeddings, top_k)
    
    candidates = []
    for i, s1_id in enumerate(df_s1['entity_id']):
        for j, s23_idx in enumerate(indices[i]):
            if scores[i][j] > 0.7: # Similarity threshold
                candidates.append((s1_id, df_s2_s3.iloc[s23_idx]['entity_id']))
    return set(candidates)

def generate_candidates(df_s1, df_s2_s3):
    candidates = set()
    
    # 1. Prefix blocking
    candidates.update(prefix_blocking(df_s1, df_s2_s3))
    
    # 2. TF-IDF blocking
    candidates.update(tfidf_blocking(df_s1, df_s2_s3))
    
    # 3. BM25 blocking
    candidates.update(bm25_blocking(df_s1, df_s2_s3))
    
    # 4. Dense blocking
    candidates.update(dense_blocking(df_s1, df_s2_s3))
    
    return candidates
