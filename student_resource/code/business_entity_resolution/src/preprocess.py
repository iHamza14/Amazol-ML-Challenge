import pandas as pd
import re
import string

def load_data(file_path):
    """Loads a TSV file robustly."""
    return pd.read_csv(file_path, sep="\t", dtype=str).fillna("")

def normalize_text(text):
    """Basic lowercasing, punctuation stripping, and whitespace normalization."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Replace '&' with 'and'
    text = text.replace('&', ' and ')
    # Remove punctuation
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def expand_abbreviations(text):
    """Expands common business and address abbreviations."""
    abbreviations = {
        r'\bcorp\b': 'corporation',
        r'\bpvt\b': 'private',
        r'\bltd\b': 'limited',
        r'\binc\b': 'incorporated',
        r'\bco\b': 'company',
        r'\bllc\b': 'limited liability company',
        r'\bllp\b': 'limited liability partnership',
        r'\brd\b': 'road',
        r'\bst\b': 'street',
        r'\bave\b': 'avenue',
        r'\bblvd\b': 'boulevard',
        r'\bdr\b': 'drive',
        r'\bln\b': 'lane',
        r'\bct\b': 'court',
        r'\bpl\b': 'place',
        r'\bsq\b': 'square',
        r'\bapt\b': 'apartment',
        r'\bste\b': 'suite',
    }
    
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text)
    
    # Clean up whitespace again just in case
    return re.sub(r'\s+', ' ', text).strip()

def strip_legal_suffixes(text):
    """Removes common legal suffixes to isolate the 'core name'."""
    suffixes = [
        r'\bcorporation\b', r'\bprivate\b', r'\blimited\b', r'\bincorporated\b',
        r'\bcompany\b', r'\blimited liability company\b', r'\blimited liability partnership\b',
        r'\bsarl\b', r'\bsa\b', r'\bsas\b' # Adding French suffixes just in case
    ]
    core_name = text
    for suffix in suffixes:
        core_name = re.sub(suffix, '', core_name)
    return re.sub(r'\s+', ' ', core_name).strip()

def preprocess_dataframe(df):
    """Applies all preprocessing steps to the dataframe."""
    print("Normalizing names and addresses...")
    
    # Name normalization
    df['name_norm'] = df['business_name'].apply(normalize_text)
    df['name_norm'] = df['name_norm'].apply(expand_abbreviations)
    df['name_core'] = df['name_norm'].apply(strip_legal_suffixes)
    
    # Address normalization
    df['address_norm'] = df['business_address'].apply(normalize_text)
    df['address_norm'] = df['address_norm'].apply(expand_abbreviations)
    
    # Country normalization
    df['country_norm'] = df['country'].apply(lambda x: str(x).lower().strip())
    
    # Create prefix block feature (country + first 3 chars of normalized name)
    df['prefix_block'] = df.apply(
        lambda row: f"{row['country_norm']}_{row['name_norm'][:3]}", axis=1
    )
    
    return df

if __name__ == "__main__":
    # Small test
    test_df = pd.DataFrame({
        'business_name': ['Orelee\'s Barbershop', 'Consulting Nyasa Pvt. Ltd.', 'B+ Retail Inc & Co'],
        'business_address': ['1795 Westchester Dr, NC', 'Oakwood, Mulund Rd', '1712 Montebello Ave'],
        'country': ['US', 'India', 'US']
    })
    processed = preprocess_dataframe(test_df)
    print(processed[['name_norm', 'name_core', 'address_norm', 'prefix_block']])
