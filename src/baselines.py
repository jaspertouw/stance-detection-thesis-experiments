# src/baselines.py
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from data_preprocessing import load_and_clean_data, add_majority_and_soft_labels
from evaluation import manual_cross_entropy, compute_brier_score, compute_kl_divergence, evaluate_by_disagreement_aggregated
from config import DATA_PATH, ANNOTATOR_COLS

def run_baselines(seed=10):
    """Execute majority and uniform baselines."""
    df = load_and_clean_data(DATA_PATH, ANNOTATOR_COLS)
    df, label_list = add_majority_and_soft_labels(df, ANNOTATOR_COLS, seed=seed)
    X, y = df['sentence'], df['majority_label']
    soft = df['soft_label'].tolist()
    results = {}
    maj = DummyClassifier(strategy='most_frequent', random_state=seed)
    maj.fit(X, y)
    yp, pp = maj.predict(X), maj.predict_proba(X)
    results['majority'] = {
        'accuracy': accuracy_score(y, yp),
        'f1': f1_score(y, yp, average='macro'),
        'ce': manual_cross_entropy(soft, pp),
        'brier': compute_brier_score(soft, pp),
        'kl': compute_kl_divergence(soft, pp),
        'mis': evaluate_by_disagreement_aggregated(yp.tolist(), pp.tolist(), y.tolist(), soft)
    }
    n = len(label_list)
    unif_p = np.full((len(X), n), 1/n)
    unif_y = np.random.choice(label_list, size=len(X))
    results['uniform'] = {
        'accuracy': accuracy_score(y, unif_y),
        'f1': f1_score(y, unif_y, average='macro'),
        'ce': manual_cross_entropy(soft, unif_p),
        'brier': compute_brier_score(soft, unif_p),
        'kl': compute_kl_divergence(soft, unif_p),
        'mis': evaluate_by_disagreement_aggregated(unif_y.tolist(), unif_p.tolist(), y.tolist(), soft)
    }
    return results

if __name__ == '__main__':
    metrics = run_baselines()
    for name, m in metrics.items():
        print(f"\n{name.capitalize()} Baseline")
        print(f"Accuracy:      {m['accuracy']*100:.2f}%")
        print(f"Macro F1:      {m['f1']:.3f}")
        print(f"Cross-Entropy: {m['ce']:.3f}")
        print(f"Brier Score:   {m['brier']:.3f}")
        print(f"KL Divergence: {m['kl']:.3f}")
