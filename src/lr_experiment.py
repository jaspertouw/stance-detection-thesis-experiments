# lr_experiment.py

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
from data_preprocessing import load_and_clean_data, add_majority_and_soft_labels
from evaluation import (
    manual_cross_entropy,
    compute_brier_score,
    compute_kl_divergence,
    evaluate_by_disagreement_aggregated,
    aggregate_misclassification_results
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def split_data(X, y, test_size=0.15, dev_ratio=15/85, random_state=42):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_val, y_train_val, test_size=dev_ratio,
        random_state=random_state, stratify=y_train_val
    )
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def build_pipeline_mv(random_state=42, C=1):
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=2000, ngram_range=(1,1), min_df=1, max_df=0.7, sublinear_tf=True
        )),
        ('lr', LogisticRegression(
            C=C, solver='lbfgs', max_iter=1000,
            class_weight='balanced', random_state=random_state
        ))
    ])

def build_pipeline_aw(random_state=42, C=10):
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=2000, ngram_range=(1,1), min_df=1, max_df=0.7, sublinear_tf=True
        )),
        ('lr', LogisticRegression(
            C=C, solver='lbfgs', max_iter=1000,
            class_weight='balanced', random_state=random_state
        ))
    ])

def run_lr_experiment_mv(seed, data_path):
    set_seed(seed)
    annotator_cols = [f'worker_{i}' for i in range(8)]
    df, label_list = add_majority_and_soft_labels(
        load_and_clean_data(data_path, annotator_cols),
        annotator_cols,
        seed=seed
    )

    X = df['sentence']
    y = df['majority_label']

    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(X, y, random_state=seed)
    X_train_dev = pd.concat([X_train, X_dev])
    y_train_dev = pd.concat([y_train, y_dev])
    test_fold = [-1]*len(X_train) + [0]*len(X_dev)

    grid = GridSearchCV(
        build_pipeline_mv(random_state=seed),
        {
            'tfidf__max_features': [1000,2000],
            'tfidf__ngram_range': [(1,1),(1,2)],
            'tfidf__min_df': [1,2],
            'tfidf__max_df': [0.7,0.8],
            'lr__C': [0.1,1,10]
        },
        cv=PredefinedSplit(test_fold),
        scoring='f1_macro',
        n_jobs=-1
    )
    grid.fit(X_train_dev, y_train_dev)
    best = grid.best_estimator_

    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro')
    ce    = manual_cross_entropy(df.loc[X_test.index,'soft_label'].tolist(), y_prob)
    brier = compute_brier_score(df.loc[X_test.index,'soft_label'].tolist(), y_prob)
    kl    = compute_kl_divergence(df.loc[X_test.index,'soft_label'].tolist(), y_prob)
    mis   = evaluate_by_disagreement_aggregated(
                y_pred.tolist(), y_prob.tolist(), y_test.tolist(), df.loc[X_test.index,'soft_label'].tolist()
            )

    return {'accuracy': acc, 'f1': f1, 'ce': ce, 'brier': brier, 'kl': kl, 'mis': mis}

def run_lr_experiment_aw(seed, data_path):
    set_seed(seed)
    annotator_cols = [f'worker_{i}' for i in range(8)]
    df, label_list = add_majority_and_soft_labels(
        load_and_clean_data(data_path, annotator_cols),
        annotator_cols,
        seed=seed
    )

    df_long = df.melt(
        id_vars=['sentence','soft_label','majority_label'],
        value_vars=annotator_cols,
        var_name='annotator',
        value_name='label'
    )

    X = df_long['sentence']
    y = df_long['label']
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(X, y, random_state=seed)
    X_train_dev = pd.concat([X_train, X_dev])
    y_train_dev = pd.concat([y_train, y_dev])
    test_fold = [-1]*len(X_train) + [0]*len(X_dev)

    grid = GridSearchCV(
        build_pipeline_aw(random_state=seed),
        {
            'tfidf__max_features': [1000,2000],
            'tfidf__ngram_range': [(1,1),(1,2)],
            'tfidf__min_df': [1,2],
            'tfidf__max_df': [0.7,0.8],
            'lr__C': [0.1,1,10]
        },
        cv=PredefinedSplit(test_fold),
        scoring='f1_macro',
        n_jobs=-1
    )
    grid.fit(X_train_dev, y_train_dev)
    best = grid.best_estimator_

    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)

    # Soft labels aligned to df_long test indices
    soft_test = df_long.loc[X_test.index, 'soft_label'].tolist()

    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average='macro')
    ce     = manual_cross_entropy(soft_test, y_prob)
    brier  = compute_brier_score(soft_test, y_prob)
    kl     = compute_kl_divergence(soft_test, y_prob)
    mis    = evaluate_by_disagreement_aggregated(
                 y_pred.tolist(), y_prob.tolist(), y_test.tolist(), soft_test
             )

    return {'accuracy': acc, 'f1': f1, 'ce': ce, 'brier': brier, 'kl': kl, 'mis': mis}

if __name__ == "__main__":
    DATA_PATH = r'C:\Users\jaspe\OneDrive\Documenten\Bachelor thesis\GWSD.tsv'
    seeds     = [10, 51, 100]

    # Run experiments
    mv_res = [run_lr_experiment_mv(s, DATA_PATH) for s in seeds]
    aw_res = [run_lr_experiment_aw(s, DATA_PATH) for s in seeds]

    # Aggregate & print summary
    import numpy as np
    from evaluation import aggregate_misclassification_results

    def agg(vals):
        return np.mean(vals), np.std(vals)

    # MV summary
    mv_accs = [r['accuracy'] for r in mv_res]
    mv_f1s  = [r['f1'] for r in mv_res]
    mv_ces  = [r['ce'] for r in mv_res]
    mv_briers = [r['brier'] for r in mv_res]
    mv_kls  = [r['kl'] for r in mv_res]
    mv_mis_list = [r['mis'] for r in mv_res]

    print("\n=== Aggregated MV LR ===")
    mean_acc, std_acc = agg(mv_accs)
    mean_f1, std_f1   = agg(mv_f1s)
    mean_ce, std_ce   = agg(mv_ces)
    mean_br, std_br   = agg(mv_briers)
    mean_kl, std_kl   = agg(mv_kls)
    print(f"Accuracy:       {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"Macro F1:       {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"Cross-Entropy:  {mean_ce:.3f} ± {std_ce:.3f}")
    print(f"Brier Score:    {mean_br:.3f} ± {std_br:.3f}")
    print(f"KL Divergence:  {mean_kl:.3f} ± {std_kl:.3f}")

    mv_mis_agg = aggregate_misclassification_results(mv_mis_list)
    for lvl, m in mv_mis_agg.items():
        print(f"\n{lvl} Disagreement (MV):")
        print(f"  Accuracy:      {m['accuracy_mean']*100:.2f}% ± {m['accuracy_std']*100:.2f}%")
        print(f"  Macro F1:      {m['f1_mean']:.3f} ± {m['f1_std']:.3f}")
        print(f"  Cross-Entropy: {m['cross_entropy_mean']:.3f} ± {m['cross_entropy_std']:.3f}")
        print(f"  Brier Score:   {m['brier_score_mean']:.3f} ± {m['brier_score_std']:.3f}")
        print(f"  KL Divergence: {m['kl_divergence_mean']:.3f} ± {m['kl_divergence_std']:.3f}")

    # AW summary
    aw_accs = [r['accuracy'] for r in aw_res]
    aw_f1s  = [r['f1'] for r in aw_res]
    aw_ces  = [r['ce'] for r in aw_res]
    aw_briers = [r['brier'] for r in aw_res]
    aw_kls  = [r['kl'] for r in aw_res]
    aw_mis_list = [r['mis'] for r in aw_res]

    print("\n=== Aggregated AW LR ===")
    mean_acc, std_acc = agg(aw_accs)
    mean_f1, std_f1   = agg(aw_f1s)
    mean_ce, std_ce   = agg(aw_ces)
    mean_br, std_br   = agg(aw_briers)
    mean_kl, std_kl   = agg(aw_kls)
    print(f"Accuracy:       {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"Macro F1:       {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"Cross-Entropy:  {mean_ce:.3f} ± {std_ce:.3f}")
    print(f"Brier Score:    {mean_br:.3f} ± {std_br:.3f}")
    print(f"KL Divergence:  {mean_kl:.3f} ± {std_kl:.3f}")

    aw_mis_agg = aggregate_misclassification_results(aw_mis_list)
    for lvl, m in aw_mis_agg.items():
        print(f"\n{lvl} Disagreement (AW):")
        print(f"  Accuracy:      {m['accuracy_mean']*100:.2f}% ± {m['accuracy_std']*100:.2f}%")
        print(f"  Macro F1:      {m['f1_mean']:.3f} ± {m['f1_std']:.3f}")
        print(f"  Cross-Entropy: {m['cross_entropy_mean']:.3f} ± {m['cross_entropy_std']:.3f}")
        print(f"  Brier Score:   {m['brier_score_mean']:.3f} ± {m['brier_score_std']:.3f}")
        print(f"  KL Divergence: {m['kl_divergence_mean']:.3f} ± {m['kl_divergence_std']:.3f}")