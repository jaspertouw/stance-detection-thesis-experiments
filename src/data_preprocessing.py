import pandas as pd
import random
from collections import Counter

def load_and_clean_data(data_path, annotator_cols):
    """
    Load the TSV data, remove the first five screening rows, duplicates, and missing values;
    convert sentences to lowercase.
    """
    df_raw = pd.read_csv(data_path, sep='\t')
    df = df_raw.iloc[5:].reset_index(drop=True)
    df = df.drop_duplicates(subset=['sentence'])
    df = df.dropna(subset=annotator_cols)
    df['sentence'] = df['sentence'].str.lower()
    return df

def add_majority_and_soft_labels(df, annotator_cols, seed=10):
    """
    Adds two columns to the DataFrame:
    - 'majority_label': using a seeded tie-breaker
    - 'soft_label': the distribution of annotator votes
    """
    rng = random.Random(seed)
    
    def majority_vote(row):
        votes = row[annotator_cols].tolist()
        vote_counts = Counter(votes)
        max_count = max(vote_counts.values())
        candidates = [label for label, count in vote_counts.items() if count == max_count]
        return rng.choice(candidates)
    
    df['majority_label'] = df.apply(majority_vote, axis=1)
    label_list = sorted(df[annotator_cols].stack().unique())
    
    def compute_soft_label(row):
        votes = row[annotator_cols].tolist()
        total = len(votes)
        return [votes.count(label) / total for label in label_list]
    
    df['soft_label'] = df.apply(compute_soft_label, axis=1)
    return df, label_list
