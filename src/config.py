#src/config.py
import os

# Path to the dataset (relative to repo root)
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'GWSD.tsv')
# Seeds for experiments
SEEDS = [10, 51, 100]
# Annotator columns in the data
ANNOTATOR_COLS = [f'worker_{i}' for i in range(8)]
# Base directory for BERT checkpoints
CHECKPOINT_BASE = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')