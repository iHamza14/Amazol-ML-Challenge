import pandas as pd
import numpy as np
from rapidfuzz import fuzz, distance

def extract_features(df_s1, df_s2_s3, candidate_pairs):
    """
    Extracts features for all given candidate pairs.
    candidate_pairs is a list of tuples (s1_id, s23_id).
    """
    s1_dict = df_s1.set_index('entity_id').to_dict('index')
    s23_dict = df_s2_s3.set_index('entity_id').to_dict('index')
    
    features = []
    
    for s1_id, s23_id in candidate_pairs:
        if s1_id not in s1_dict or s23_id not in s23_dict:
            continue
            
        r1 = s1_dict[s1_id]
        r2 = s23_dict[s23_id]
        
        name1 = r1['name_norm']
        name2 = r2['name_norm']
        core1 = r1['name_core']
        core2 = r2['name_core']
        addr1 = r1['address_norm']
        addr2 = r2['address_norm']
        
        feat = {
            's1_id': s1_id,
            's23_id': s23_id,
            
            # Name features (Core)
            'name_jaro_winkler': fuzz.jaro_winkler(core1, core2),
            'name_levenshtein_ratio': fuzz.ratio(core1, core2),
            'name_token_sort_ratio': fuzz.token_sort_ratio(core1, core2),
            'name_token_set_ratio': fuzz.token_set_ratio(core1, core2),
            
            # Name features (Full norm)
            'full_name_ratio': fuzz.ratio(name1, name2),
            
            # Address features
            'addr_jaro_winkler': fuzz.jaro_winkler(addr1, addr2),
            'addr_token_set_ratio': fuzz.token_set_ratio(addr1, addr2),
            'addr_token_sort_ratio': fuzz.token_sort_ratio(addr1, addr2),
            
            # Structural features
            'same_country': int(r1['country_norm'] == r2['country_norm']),
            'is_source_2': int(s23_id.startswith('S2-')),
            'is_source_3': int(s23_id.startswith('S3-'))
        }
        
        # Address digit overlap
        digits1 = set(filter(str.isdigit, addr1.split()))
        digits2 = set(filter(str.isdigit, addr2.split()))
        
        if digits1 and digits2:
            overlap = len(digits1.intersection(digits2)) / max(len(digits1), len(digits2))
        else:
            overlap = 0.0
        feat['addr_digit_overlap'] = overlap
        
        features.append(feat)
        
    return pd.DataFrame(features)
