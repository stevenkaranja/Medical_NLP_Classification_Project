# src/preprocess.py
# Small helper functions for loading and cleaning the mtsamples.csv file.

import pandas as pd
import re

def load_raw(path: str):
    df = pd.read_csv(path)
    return df

def basic_clean_text(text: str) -> str:
    if pd.isna(text):
        return ''
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    # remove weird characters but keep basic punctuation
    text = re.sub(r'[^a-z0-9 .,;:\-()\n]', '', text)
    return text.strip()

if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else '../data/mtsamples.csv'
    df = load_raw(p)
    print('Rows:', len(df))
    print('Columns:', df.columns.tolist())
