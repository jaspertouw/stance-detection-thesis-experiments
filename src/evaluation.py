# src/evaluation.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def manual_cross_entropy(y_true, y_prob, epsilon=1e-15):
    """Cross-entropy loss between soft labels and predicted distributions."""
    y_prob = np.clip(y_prob, epsilon, 1)
    return -np.mean(np.sum(np.array(y_true) * np.log(y_prob), axis=1))

def compute_brier_score(y_true, y_prob):
    """Brier score: MSE between probabilities and true soft labels."""
    return np.mean(np.sum((np.array(y_prob) - np.array(y_true))**2, axis=1))

def compute_kl_divergence(y_true, y_prob, epsilon=1e-15):
    """KL divergence between soft labels and predicted probabilities."""
    y_true = np.clip(np.array(y_true), epsilon, 1)
    y_prob = np.clip(np.array(y_prob), epsilon, 1)
    kl = np.sum(y_true * np.log(y_true / y_prob), axis=1)
    return np.mean(kl)

def evaluate_by_disagreement_aggregated(pred_labels, pred_probs, true_labels, soft_labels):
    """Aggregate metrics by Low/Moderate/High annotator disagreement."""
    def level(soft):
        m = max(soft)
        return 'Low' if m >= 0.75 else 'Moderate' if m >= 0.5 else 'High'
    groups = {'Low': [], 'Moderate': [], 'High': []}
    for i, soft in enumerate(soft_labels):
        groups[level(soft)].append(i)
    agg = {}
    for lvl, idxs in groups.items():
        if not idxs:
            continue
        gt = [true_labels[i] for i in idxs]
        gp = [pred_labels[i] for i in idxs]
        acc = accuracy_score(gt, gp)
        _, _, f1, _ = precision_recall_fscore_support(gt, gp, average='macro', zero_division=0)
        st = np.array([soft_labels[i] for i in idxs])
        pp = np.array([pred_probs[i] for i in idxs])
        agg[lvl] = {
            'accuracy': acc,
            'f1': f1,
            'cross_entropy': manual_cross_entropy(st, pp),
            'brier_score': compute_brier_score(st, pp),
            'kl_divergence': compute_kl_divergence(st, pp)
        }
    return agg

def aggregate_misclassification_results(mis_list):
    """Compute mean and std of misclassification metrics across runs."""
    res = {}
    for lvl in ['Low', 'Moderate', 'High']:
        accs, f1s, ces, briers, kls = [], [], [], [], []
        for m in mis_list:
            if lvl in m:
                accs.append(m[lvl]['accuracy'])
                f1s.append(m[lvl]['f1'])
                ces.append(m[lvl]['cross_entropy'])
                briers.append(m[lvl]['brier_score'])
                kls.append(m[lvl]['kl_divergence'])
        if accs:
            res[lvl] = {
                'accuracy_mean': np.mean(accs), 'accuracy_std': np.std(accs),
                'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s),
                'cross_entropy_mean': np.mean(ces), 'cross_entropy_std': np.std(ces),
                'brier_score_mean': np.mean(briers), 'brier_score_std': np.std(briers),
                'kl_divergence_mean': np.mean(kls), 'kl_divergence_std': np.std(kls)
            }
    return res