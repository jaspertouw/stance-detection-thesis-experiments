# src/data_preprocessing.py
import pandas as pd
import random
from collections import Counter

def load_and_clean_data(data_path, annotator_cols):
    """Load TSV, drop first 5 rows, remove duplicates and missing, lowercase sentences."""
    df_raw = pd.read_csv(data_path, sep='\t')
    df = df_raw.iloc[5:].reset_index(drop=True)
    df = df.drop_duplicates(subset=['sentence']).dropna(subset=annotator_cols)
    df['sentence'] = df['sentence'].str.lower()
    return df

def add_majority_and_soft_labels(df, annotator_cols, seed=10):
    """Add 'majority_label' with seeded tie-breaker and 'soft_label' distribution."""
    rng = random.Random(seed)
    def majority_vote(row):
        votes = row[annotator_cols].tolist()
        counts = Counter(votes)
        max_count = max(counts.values())
        candidates = [lbl for lbl, cnt in counts.items() if cnt == max_count]
        return rng.choice(candidates)
    df['majority_label'] = df.apply(majority_vote, axis=1)
    label_list = sorted(df[annotator_cols].stack().unique())
    def compute_soft_label(row):
        votes = row[annotator_cols].tolist()
        total = len(votes)
        return [votes.count(lbl) / total for lbl in label_list]
    df['soft_label'] = df.apply(compute_soft_label, axis=1)
    return df, label_list