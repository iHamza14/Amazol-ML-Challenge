import pandas as pd
import numpy as np
import re
import math
from unidecode import unidecode
from rapidfuzz import fuzz, distance
from collections import defaultdict
from catboost import CatBoostClassifier, Pool
import os
import tqdm
import pickle

import warnings
warnings.filterwarnings('ignore')

STOPWORDS = {
    'corp', 'corporation', 'inc', 'incorporated', 'llc', 'pvt', 'private', 'ltd', 'limited',
    'street', 'st', 'road', 'rd', 'avenue', 'ave', 'block', 'blk', 'near', 'opp', 'opposite',
    'floor', 'fl', 'room', 'rm', 'building', 'bldg', 'and', 'at', 'company', 'co'
}

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text)
    text = unidecode(text).lower()
    text = text.replace('&', ' and ').replace('@', ' at ')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_tokens(text, remove_stopwords=False):
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens

def preprocess_df(df):
    df['name_clean'] = df['business_name'].apply(clean_text)
    df['addr_clean'] = df['business_address'].apply(clean_text)
    
    df['name_tokens'] = df['name_clean'].apply(lambda x: get_tokens(x, True))
    df['addr_tokens'] = df['addr_clean'].apply(lambda x: get_tokens(x, True))
    
    # Extract numbers from address
    df['addr_numbers'] = df['addr_clean'].apply(lambda x: re.findall(r'\b\d+\b', x))
    return df

def generate_indexes(df):
    token_idf = defaultdict(float)
    corpus_size = len(df)
    
    # Calculate document frequencies
    doc_freqs = defaultdict(int)
    for idx, row in df.iterrows():
        tokens_set = set()
        
        # Name tokens
        for t in row['name_tokens']: tokens_set.add(f"N_{t}")
            
        # Name token pairs
        nt = row['name_tokens']
        for i in range(len(nt)-1):
            tokens_set.add(f"NP_{nt[i]}_{nt[i+1]}")
            
        # Address tokens
        for t in row['addr_tokens']: tokens_set.add(f"A_{t}")
            
        # Number + word combinations
        nums = row['addr_numbers']
        for n in nums:
            for t in row['addr_tokens']:
                if not t.isdigit():
                    tokens_set.add(f"NW_{n}_{t}")
                    
        for t in tokens_set:
            doc_freqs[t] += 1
            
    # Calculate IDF
    for t, freq in doc_freqs.items():
        token_idf[t] = math.log((corpus_size + 1) / (freq + 1)) + 1
        
    return token_idf

def get_entity_features(row):
    features = set()
    for t in row['name_tokens']: features.add(f"N_{t}")
    nt = row['name_tokens']
    for i in range(len(nt)-1): features.add(f"NP_{nt[i]}_{nt[i+1]}")
    for t in row['addr_tokens']: features.add(f"A_{t}")
    nums = row['addr_numbers']
    for n in nums:
        for t in row['addr_tokens']:
            if not t.isdigit():
                features.add(f"NW_{n}_{t}")
    return features

def blocking(df_s1, df_s23):
    candidate_pairs = []
    
    # Group by country
    for country in df_s1['country'].unique():
        s1_country = df_s1[df_s1['country'] == country]
        s23_country = df_s23[df_s23['country'] == country]
        
        if len(s23_country) == 0: continue
            
        token_idf = generate_indexes(s23_country)
        
        # Inverted index: feature -> list of S2/S3 entity indices
        inverted_index = defaultdict(list)
        s23_records = s23_country.to_dict('records')
        s23_features_list = []
        
        for i, row in enumerate(s23_records):
            feats = get_entity_features(row)
            s23_features_list.append(feats)
            for f in feats:
                inverted_index[f].append(i)
                
        # Query for each S1
        s1_records = s1_country.to_dict('records')
        for s1_row in tqdm.tqdm(s1_records, desc=f"Blocking for {country}"):
            s1_feats = get_entity_features(s1_row)
            candidate_scores = defaultdict(float)
            
            for f in s1_feats:
                if f in inverted_index:
                    idf_weight = token_idf.get(f, 1.0)
                    for s23_idx in inverted_index[f]:
                        candidate_scores[s23_idx] += idf_weight
                        
            # Rank and keep top 120
            top_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:120]
            for s23_idx, _ in top_candidates:
                candidate_pairs.append((s1_row['entity_id'], s23_records[s23_idx]['entity_id']))
                
    return candidate_pairs

def get_jaccard(list1, list2):
    s1, s2 = set(list1), set(list2)
    if not s1 and not s2: return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def extract_features(df_s1, df_s23, candidate_pairs):
    s1_df = df_s1.set_index('entity_id')
    s23_df = df_s23.set_index('entity_id')
    
    features = []
    
    for s1_id, s23_id in tqdm.tqdm(candidate_pairs, desc="Extracting Features"):
        r1 = s1_df.loc[s1_id]
        r2 = s23_df.loc[s23_id]
        
        n1, n2 = str(r1['name_clean']), str(r2['name_clean'])
        a1, a2 = str(r1['addr_clean']), str(r2['addr_clean'])
        
        feat = {
            's1_id': s1_id,
            's23_id': s23_id,
            # 1-5 Fuzzy name
            'name_fuzz_ratio': fuzz.ratio(n1, n2),
            'name_fuzz_partial_ratio': fuzz.partial_ratio(n1, n2),
            'name_fuzz_token_sort_ratio': fuzz.token_sort_ratio(n1, n2),
            'name_fuzz_token_set_ratio': fuzz.token_set_ratio(n1, n2),
            'name_jaro_winkler': distance.JaroWinkler.normalized_similarity(n1, n2) * 100,
            
            # 6-10 Fuzzy address
            'addr_fuzz_ratio': fuzz.ratio(a1, a2),
            'addr_fuzz_partial_ratio': fuzz.partial_ratio(a1, a2),
            'addr_fuzz_token_sort_ratio': fuzz.token_sort_ratio(a1, a2),
            'addr_fuzz_token_set_ratio': fuzz.token_set_ratio(a1, a2),
            'addr_jaro_winkler': distance.JaroWinkler.normalized_similarity(a1, a2) * 100,
            
            # 11-12 Exact matches
            'name_exact_match': int(n1 == n2 and len(n1) > 0),
            'addr_exact_match': int(a1 == a2 and len(a1) > 0),
            
            # 13-14 Word overlap
            'name_word_overlap': get_jaccard(r1['name_tokens'], r2['name_tokens']),
            'addr_word_overlap': get_jaccard(r1['addr_tokens'], r2['addr_tokens']),
            
            # 15-18 Substring relationships
            'name_s1_in_s2': int(n1 in n2 and len(n1) > 3),
            'name_s2_in_s1': int(n2 in n1 and len(n2) > 3),
            'addr_s1_in_s2': int(a1 in a2 and len(a1) > 3),
            'addr_s2_in_s1': int(a2 in a1 and len(a2) > 3),
            
            # 19 Shared numbers
            'shared_addr_numbers': get_jaccard(r1['addr_numbers'], r2['addr_numbers']),
            
            # 20 First number agreement
            'first_number_agreement': 1 if len(r1['addr_numbers']) > 0 and len(r2['addr_numbers']) > 0 and r1['addr_numbers'][0] == r2['addr_numbers'][0] else 0,
            
            # 21-26 String lengths
            'name1_len': len(n1),
            'name2_len': len(n2),
            'name_len_diff': abs(len(n1) - len(n2)),
            'addr1_len': len(a1),
            'addr2_len': len(a2),
            'addr_len_diff': abs(len(a1) - len(a2)),
            
            # 27-30 Missing fields
            'is_name1_missing': int(len(n1) == 0),
            'is_name2_missing': int(len(n2) == 0),
            'is_addr1_missing': int(len(a1) == 0),
            'is_addr2_missing': int(len(a2) == 0),
            
            # 31-36 Token counts
            'name1_num_tokens': len(r1['name_tokens']),
            'name2_num_tokens': len(r2['name_tokens']),
            'addr1_num_tokens': len(r1['addr_tokens']),
            'addr2_num_tokens': len(r2['addr_tokens']),
            'name_tokens_diff': abs(len(r1['name_tokens']) - len(r2['name_tokens'])),
            'addr_tokens_diff': abs(len(r1['addr_tokens']) - len(r2['addr_tokens']))
        }
        features.append(feat)
        
    return pd.DataFrame(features)

def extract_france_components(addr_str):
    addr_str = str(addr_str).strip()
    match = re.search(r'^(\d+)\s+(.*)', addr_str)
    if match:
        return match.group(1), match.group(2)
    return "", addr_str

def apply_france_filter(s1_id, predicted_matches, df_s1_dict, df_s23_dict):
    if not predicted_matches:
        return []
        
    s1_row = df_s1_dict[s1_id]
    if s1_row['country'] != 'France':
        return predicted_matches
        
    s1_num, s1_street = extract_france_components(s1_row['addr_clean'])
    
    supported_matches = []
    
    for s23_id in predicted_matches:
        s23_row = df_s23_dict[s23_id]
        s23_num, s23_street = extract_france_components(s23_row['addr_clean'])
        
        street_sim = fuzz.ratio(s1_street, s23_street)
        has_conflict = (s1_num != "" and s23_num != "" and s1_num != s23_num)
        
        if street_sim >= 75 and not has_conflict:
            supported_matches.append(s23_id)
            
    if supported_matches:
        return supported_matches
    return predicted_matches

def train():
    print("Loading train data...")
    df_s1 = pd.read_csv("../../../dataset/train/train_source1.tsv", sep="\t")
    df_s2 = pd.read_csv("../../../dataset/train/train_source2.tsv", sep="\t")
    df_s3 = pd.read_csv("../../../dataset/train/train_source3.tsv", sep="\t")
    gt = pd.read_csv("../../../dataset/train/train_ground_truth.tsv", sep="\t")
    
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    
    print("Preprocessing train data...")
    df_s1 = preprocess_df(df_s1)
    df_s23 = preprocess_df(df_s23)
    
    print("Blocking...")
    # Limit S1 for training to avoid memory explosion, e.g., 5000 records
    df_s1_train = df_s1.sample(n=min(5000, len(df_s1)), random_state=42)
    candidate_pairs = blocking(df_s1_train, df_s23)
    
    print(f"Generated {len(candidate_pairs)} candidate pairs.")
    
    features_df = extract_features(df_s1_train, df_s23, candidate_pairs)
    
    # Create labels
    gt_dict = {}
    for _, row in gt.iterrows():
        s1 = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',')
        gt_dict[s1] = set(m for m in matches if m)
        
    labels = []
    for _, row in features_df.iterrows():
        s1, s23 = row['s1_id'], row['s23_id']
        if s1 in gt_dict and s23 in gt_dict[s1]:
            labels.append(1)
        else:
            labels.append(0)
            
    features_df['label'] = labels
    
    # Train CatBoost
    feature_cols = [c for c in features_df.columns if c not in ['s1_id', 's23_id', 'label']]
    X = features_df[feature_cols]
    y = features_df['label']
    
    print("Training CatBoost model...")
    model = CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6, verbose=50)
    model.fit(X, y)
    model.save_model("model.cbm")
    print("Model saved to model.cbm")

def predict():
    print("Loading test data...")
    df_s1 = pd.read_csv("../../../dataset/test/test_source1.tsv", sep="\t")
    df_s2 = pd.read_csv("../../../dataset/test/test_source2.tsv", sep="\t")
    df_s3 = pd.read_csv("../../../dataset/test/test_source3.tsv", sep="\t")
    
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    
    print("Preprocessing test data...")
    df_s1 = preprocess_df(df_s1)
    df_s23 = preprocess_df(df_s23)
    
    print("Blocking test data...")
    candidate_pairs = blocking(df_s1, df_s23)
    
    print(f"Generated {len(candidate_pairs)} test candidate pairs.")
    
    # Save candidate pairs
    cand_dict = defaultdict(list)
    for s1, s23 in candidate_pairs:
        cand_dict[s1].append(s23)
        
    os.makedirs("../../../output", exist_ok=True)
    with open("../../../output/candidate_pairs.tsv", "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in df_s1['entity_id']:
            cands = list(dict.fromkeys(cand_dict.get(s1, [])))
            f.write(f"{s1}\t{','.join(cands)}\n")
            
    if not candidate_pairs:
        print("No candidates found.")
        with open("../../../output/matching_results.tsv", "w") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1 in df_s1['entity_id']:
                f.write(f"{s1}\t\n")
        return
        
    features_df = extract_features(df_s1, df_s23, candidate_pairs)
    feature_cols = [c for c in features_df.columns if c not in ['s1_id', 's23_id', 'label']]
    X = features_df[feature_cols]
    
    print("Loading CatBoost model...")
    model = CatBoostClassifier()
    model.load_model("model.cbm")
    
    print("Scoring candidates...")
    preds = model.predict_proba(X)[:, 1]
    
    # Keep candidates with prob >= 0.625
    predicted_matches = defaultdict(list)
    for i, prob in enumerate(preds):
        if prob >= 0.625:
            s1, s23 = features_df.iloc[i]['s1_id'], features_df.iloc[i]['s23_id']
            predicted_matches[s1].append(s23)
            
    # Apply France filter
    print("Applying France filter...")
    s1_dict = df_s1.set_index('entity_id').to_dict('index')
    s23_dict = df_s23.set_index('entity_id').to_dict('index')
    
    final_matches = defaultdict(list)
    for s1 in df_s1['entity_id']:
        matches = predicted_matches.get(s1, [])
        if s1_dict[s1]['country'] == 'France':
            matches = apply_france_filter(s1, matches, s1_dict, s23_dict)
        final_matches[s1] = matches
        
    print("Saving matching results...")
    with open("../../../output/matching_results.tsv", "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in df_s1['entity_id']:
            m = list(dict.fromkeys(final_matches.get(s1, [])))
            f.write(f"{s1}\t{','.join(m)}\n")

if __name__ == "__main__":
    if not os.path.exists("model.cbm"):
        train()
    predict()

