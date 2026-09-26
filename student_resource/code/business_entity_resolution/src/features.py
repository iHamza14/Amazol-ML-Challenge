import pandas as pd
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
