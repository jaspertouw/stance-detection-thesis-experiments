import numpy as np

def manual_cross_entropy(y_true, y_prob, epsilon=1e-15):
    """
    Compute manual cross-entropy loss between the true soft labels and predicted probability distributions.
    """
    y_prob = np.clip(y_prob, epsilon, 1)
    return -np.mean(np.sum(np.array(y_true) * np.log(y_prob), axis=1))

def compute_brier_score(y_true, y_prob):
    """
    Compute the Brier score: mean squared error between predicted probabilities and true soft labels.
    """
    return np.mean(np.sum((np.array(y_prob) - np.array(y_true)) ** 2, axis=1))

def compute_kl_divergence(y_true, y_prob, epsilon=1e-15):
    """
    Compute the Kullback-Leibler divergence between true soft labels and predicted probabilities.
    """
    y_true = np.clip(np.array(y_true), epsilon, 1)
    y_prob = np.clip(np.array(y_prob), epsilon, 1)
    kl = np.sum(y_true * np.log(y_true / y_prob), axis=1)
    return np.mean(kl)

def evaluate_by_disagreement_aggregated(pred_labels, pred_probs, true_labels, soft_labels):
    """
    Aggregate evaluation metrics for each disagreement group:
      - Low Disagreement: if at least 6 out of 8 annotators agree (i.e. max(soft) >= 0.75)
      - Moderate Disagreement: if exactly 4 or 5 annotators agree (i.e. 0.5 <= max(soft) < 0.75)
      - High Disagreement: if less than 4 annotators agree (i.e. max(soft) < 0.5)
      
    Returns a dictionary with aggregated accuracy, macro F1, cross-entropy, Brier score, and KL divergence for each group.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    groups = {'Low': [], 'Moderate': [], 'High': []}

    def disagreement_level(soft):
        if max(soft) >= 0.75:
            return 'Low'
        elif max(soft) >= 0.5:
            return 'Moderate'
        else:
            return 'High'

    for i, soft in enumerate(soft_labels):
        group = disagreement_level(soft)
        groups[group].append(i)

    aggregated_metrics = {}
    for level, indices in groups.items():
        if len(indices) == 0:
            continue
        group_true = [true_labels[i] for i in indices]
        group_pred = [pred_labels[i] for i in indices]
        acc = accuracy_score(group_true, group_pred)
        _, _, f1, _ = precision_recall_fscore_support(group_true, group_pred, average='macro', zero_division=0)
        group_soft_true = np.array([soft_labels[i] for i in indices])
        group_pred_probs = np.array([pred_probs[i] for i in indices])
        ce = manual_cross_entropy(group_soft_true, group_pred_probs)
        brier = compute_brier_score(group_soft_true, group_pred_probs)
        kl = compute_kl_divergence(group_soft_true, group_pred_probs)
        aggregated_metrics[level] = {
            'accuracy': acc,
            'f1': f1,
            'cross_entropy': ce,
            'brier_score': brier,
            'kl_divergence': kl
        }
    return aggregated_metrics

def aggregate_misclassification_results(mis_results_list):
    """
    Given a list of misclassification results (dictionary for each run), aggregate them by computing
    mean and standard deviation for each metric for each disagreement group.
    """
    aggregated = {}
    groups = ['Low', 'Moderate', 'High']
    for group in groups:
        group_acc = []
        group_f1 = []
        group_ce = []
        group_brier = []
        group_kl = []
        for res in mis_results_list:
            if group in res:
                group_acc.append(res[group]['accuracy'])
                group_f1.append(res[group]['f1'])
                group_ce.append(res[group]['cross_entropy'])
                group_brier.append(res[group]['brier_score'])
                group_kl.append(res[group]['kl_divergence'])
        if group_acc:
            aggregated[group] = {
                'accuracy_mean': np.mean(group_acc),
                'accuracy_std': np.std(group_acc),
                'f1_mean': np.mean(group_f1),
                'f1_std': np.std(group_f1),
                'cross_entropy_mean': np.mean(group_ce),
                'cross_entropy_std': np.std(group_ce),
                'brier_score_mean': np.mean(group_brier),
                'brier_score_std': np.std(group_brier),
                'kl_divergence_mean': np.mean(group_kl),
                'kl_divergence_std': np.std(group_kl)
            }
    return aggregated
