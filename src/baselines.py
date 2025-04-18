# baselines.py

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from data_preprocessing import load_and_clean_data, add_majority_and_soft_labels
from evaluation import (
    manual_cross_entropy,
    compute_brier_score,
    compute_kl_divergence,
    evaluate_by_disagreement_aggregated,
)

def run_baselines(data_path, seed=10):
    # 1) Load & preprocess
    annotator_cols = [f'worker_{i}' for i in range(8)]
    df = load_and_clean_data(data_path, annotator_cols)
    df, label_list = add_majority_and_soft_labels(df, annotator_cols, seed=seed)

    X = df['sentence']
    y = df['majority_label']
    soft_true = df['soft_label'].tolist()

    results = {}

    # --- Majority Class Baseline ---
    maj = DummyClassifier(strategy='most_frequent', random_state=seed)
    maj.fit(X, y)
    y_pred_maj = maj.predict(X)
    y_prob_maj = maj.predict_proba(X)

    acc_maj = accuracy_score(y, y_pred_maj)
    f1_maj  = f1_score(y, y_pred_maj, average='macro')
    ce_maj      = manual_cross_entropy(soft_true, y_prob_maj)
    brier_maj   = compute_brier_score(soft_true, y_prob_maj)
    kl_maj      = compute_kl_divergence(soft_true, y_prob_maj)
    mis_maj     = evaluate_by_disagreement_aggregated(
                      y_pred_maj.tolist(), y_prob_maj.tolist(), y.tolist(), soft_true
                  )

    results['majority'] = {
        'accuracy': acc_maj,
        'f1':        f1_maj,
        'ce':        ce_maj,
        'brier':     brier_maj,
        'kl':        kl_maj,
        'mis':       mis_maj
    }

    # --- Uniform Random Baseline ---
    n_classes    = len(label_list)
    uniform_prob = np.full((len(X), n_classes), 1.0/n_classes)
    uniform_pred = np.random.choice(label_list, size=len(X))

    acc_unif  = accuracy_score(y, uniform_pred)
    f1_unif   = f1_score(y, uniform_pred, average='macro')
    ce_unif      = manual_cross_entropy(soft_true, uniform_prob)
    brier_unif   = compute_brier_score(soft_true, uniform_prob)
    kl_unif      = compute_kl_divergence(soft_true, uniform_prob)
    mis_unif     = evaluate_by_disagreement_aggregated(
                       uniform_pred.tolist(), uniform_prob.tolist(), y.tolist(), soft_true
                   )

    results['uniform'] = {
        'accuracy': acc_unif,
        'f1':        f1_unif,
        'ce':        ce_unif,
        'brier':     brier_unif,
        'kl':        kl_unif,
        'mis':       mis_unif
    }

    return results


if __name__ == "__main__":
    DATA_PATH = r'C:\Users\jaspe\OneDrive\Documenten\Bachelor thesis\GWSD.tsv'
    metrics = run_baselines(DATA_PATH)
    print("Baseline metrics:")
    for name, m in metrics.items():
        print(f"\n{name.capitalize()} Baseline:")
        print(f"  Accuracy:       {m['accuracy']*100:.2f}%")
        print(f"  Macro F1:       {m['f1']:.3f}")
        print(f"  Cross-Entropy:  {m['ce']:.3f}")
        print(f"  Brier Score:    {m['brier']:.3f}")
        print(f"  KL Divergence:  {m['kl']:.3f}")
