import pandas as pd
import re

def load_data(file_path):
    """Loads a TSV file robustly."""
    return pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)

def preprocess_dataframe(df):
    """Fully vectorized preprocessing to survive million-row datasets on limited RAM."""
    print("Normalizing data (Vectorized)...")
    
    # 1. Lowercase and strip punctuation
    df['name_norm'] = df['business_name'].astype(str).str.lower()
    df['name_norm'] = df['name_norm'].str.replace('&', ' and ', regex=False)
    df['name_norm'] = df['name_norm'].str.replace(r'[^\w\s]', ' ', regex=True)
    df['name_norm'] = df['name_norm'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # 2. Expand abbreviations (vectorized)
    abbrev = {
        r'\bcorp\b': 'corporation', r'\bpvt\b': 'private', r'\bltd\b': 'limited',
        r'\binc\b': 'incorporated', r'\bco\b': 'company', r'\bllc\b': 'limited liability company',
        r'\bllp\b': 'limited liability partnership'
    }
    df['name_core'] = df['name_norm']
    for pat, repl in abbrev.items():
        df['name_core'] = df['name_core'].str.replace(pat, repl, regex=True)
    
    # Strip legal suffixes for core name
    suffixes = r'\b(corporation|private|limited|incorporated|company|limited liability company|sarl|sa|sas)\b'
    df['name_core'] = df['name_core'].str.replace(suffixes, '', regex=True)
    df['name_core'] = df['name_core'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # 3. Address norm
    df['address_norm'] = df['business_address'].astype(str).str.lower()
    df['address_norm'] = df['address_norm'].str.replace(r'[^\w\s]', ' ', regex=True)
    df['address_norm'] = df['address_norm'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # 4. Country norm
    df['country_norm'] = df['country'].astype(str).str.lower().str.strip()
    
    # 5. Cheap blocking prefix (country + first 3 chars)
    df['prefix_block'] = df['country_norm'] + "_" + df['name_norm'].str[:3]
    
    return df
