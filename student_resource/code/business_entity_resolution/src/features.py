"""
Feature engineering v2 — vectorised pair features for candidate chunks.

Groups
  NAME   fuzzy similarities on several normalised variants (core / strict / with-suffix /
         collapsed / alt-name), TF-IDF cosines (word + char 3-gram), token overlap,
         rare-shared-token IDF, extra-token statistics learned from training pairs
         (words the distractor generator inserts: 'holdings', 'midtown', ... have ~0%
         true-match rate; words the noise generator inserts: 'center', 'services', 'dr'
         have ~95%).
  ADDR   fuzzy similarities on full address and street string, TF-IDF cosine, word overlap,
         admin (state/region) agreement, postal code agreement, component counts.
  NUMBERS the decisive signal for near-miss distractors: first-number relation
         (equal / truncation prefix / truncation suffix / digit substitution / disjoint),
         absolute difference, digit hamming, shared-number counts, range containment,
         digit-string containment ('1-02' vs '102').
  FLAGS  script of the S2/S3 name, domain-collapsed, dba, empty address, source (S2/S3).
  BLOCK  per-channel retrieval scores and ranks, number of channels, candidate count.
  GROUP  relative features within the S1 entity's candidate set: gap to the best
         candidate on name / address / street / retrieval score, rank by combined
         similarity, number of candidates whose house number equals S1's.

All functions operate on a candidate chunk DataFrame (s1_pos, s23_pos, block meta) plus
the two preprocessed frames; strings are pulled by position from numpy arrays.
"""
import json
import logging
import threading
import numpy as np
from rapidfuzz import fuzz, distance

def extract_features(df_s1, df_s2_s3, candidate_pairs):
    """Memory-efficient batch extraction without dict-of-dicts."""
    
    # Create fast lookup arrays
    s1_df = df_s1.set_index('entity_id')
    s23_df = df_s2_s3.set_index('entity_id')
    
    s1_ids = [p[0] for p in candidate_pairs]
    s23_ids = [p[1] for p in candidate_pairs]
    
    # Extract columns as lists for fast iteration
    core1 = s1_df.loc[s1_ids, 'name_core'].values.astype(str)
    core2 = s23_df.loc[s23_ids, 'name_core'].values.astype(str)
    name1 = s1_df.loc[s1_ids, 'name_norm'].values.astype(str)
    name2 = s23_df.loc[s23_ids, 'name_norm'].values.astype(str)
    addr1 = s1_df.loc[s1_ids, 'address_norm'].values.astype(str)
    addr2 = s23_df.loc[s23_ids, 'address_norm'].values.astype(str)
    c1 = s1_df.loc[s1_ids, 'country_norm'].values.astype(str)
    c2 = s23_df.loc[s23_ids, 'country_norm'].values.astype(str)
    
    features = []
    
    from tqdm import tqdm
    for i in tqdm(range(len(candidate_pairs)), desc="Calculating String Features"):
        feat = {
            's1_id': s1_ids[i],
            's23_id': s23_ids[i],
            
            # Using JaroWinkler normalized similarity (returns 0-1, so multiply by 100 to match fuzz scale)
            'name_jaro_winkler': distance.JaroWinkler.normalized_similarity(core1[i], core2[i]) * 100,
            'name_levenshtein_ratio': fuzz.ratio(core1[i], core2[i]),
            'name_token_sort_ratio': fuzz.token_sort_ratio(core1[i], core2[i]),
            'full_name_ratio': fuzz.ratio(name1[i], name2[i]),
            
            'addr_jaro_winkler': distance.JaroWinkler.normalized_similarity(addr1[i], addr2[i]) * 100,
            'addr_token_set_ratio': fuzz.token_set_ratio(addr1[i], addr2[i]),
            
            'same_country': int(c1[i] == c2[i]),
            'is_source_2': int(s23_ids[i].startswith('S2-')),
        }
        
        # Fast digit overlap
        d1 = set(filter(str.isdigit, addr1[i].split()))
        d2 = set(filter(str.isdigit, addr2[i].split()))
        overlap = len(d1.intersection(d2)) / max(len(d1), len(d2)) if d1 and d2 else 0.0
        feat['addr_digit_overlap'] = overlap
        
        features.append(feat)
        
    return pd.DataFrame(features)
