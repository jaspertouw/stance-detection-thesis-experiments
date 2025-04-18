# config.py
# Central paths & constants

DATA_PATH = "data/GWSD.tsv"    # path relative to repo root
SEEDS     = [10, 51, 100]
ANNOTATOR_COLS = [f"worker_{i}" for i in range(8)]
CHECKPOINT_BASE = "checkpoints"
