# src/lr_experiment.py
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from data_preprocessing import load_and_clean_data, add_majority_and_soft_labels
from evaluation import manual_cross_entropy, compute_brier_score, compute_kl_divergence, evaluate_by_disagreement_aggregated, aggregate_misclassification_results
from config import DATA_PATH, SEEDS, ANNOTATOR_COLS

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def split_data(X, y, test_size=0.15, dev_ratio=15/85, random_state=42):
    X_trval, X_te, y_trval, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_tr, X_dev, y_tr, y_dev = train_test_split(X_trval, y_trval, test_size=dev_ratio, random_state=random_state, stratify=y_trval)
    return X_tr, X_dev, X_te, y_tr, y_dev, y_te

build_pipeline_mv = lambda rs, C=1: Pipeline([('tfidf', TfidfVectorizer(max_features=2000, ngram_range=(1,1), min_df=1, max_df=0.7, sublinear_tf=True)), ('lr', LogisticRegression(C=C, solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=rs))])
build_pipeline_aw = lambda rs, C=10: Pipeline([('tfidf', TfidfVectorizer(max_features=2000, ngram_range=(1,1), min_df=1, max_df=0.7, sublinear_tf=True)), ('lr', LogisticRegression(C=C, solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=rs))])

def run_lr_experiment_mv(seed):
    set_seed(seed)
    df, labels = add_majority_and_soft_labels(load_and_clean_data(DATA_PATH, ANNOTATOR_COLS), ANNOTATOR_COLS, seed)
    X, y = df['sentence'], df['majority_label']
    X_tr, X_dev, X_te, y_tr, y_dev, y_te = split_data(X, y, random_state=seed)
    X_tv, y_tv = pd.concat([X_tr, X_dev]), pd.concat([y_tr, y_dev])
    fold = [-1]*len(X_tr) + [0]*len(X_dev)
    grid = GridSearchCV(build_pipeline_mv(seed), {'tfidf__max_features': [1000, 2000], 'tfidf__ngram_range': [(1,1), (1,2)], 'tfidf__min_df': [1,2], 'tfidf__max_df': [0.7,0.8], 'lr__C': [0.1,1,10]}, cv=PredefinedSplit(fold), scoring='f1_macro', n_jobs=-1)
    grid.fit(X_tv, y_tv)
    yp, pp = grid.predict(X_te), grid.predict_proba(X_te)
    soft = df.loc[X_te.index, 'soft_label'].tolist()
    return {'accuracy': accuracy_score(y_te, yp), 'f1': f1_score(y_te, yp, average='macro'), 'ce': manual_cross_entropy(soft, pp), 'brier': compute_brier_score(soft, pp), 'kl': compute_kl_divergence(soft, pp), 'mis': evaluate_by_disagreement_aggregated(yp.tolist(), pp.tolist(), y_te.tolist(), soft)}

def run_lr_experiment_aw(seed):
    set_seed(seed)
    df, labels = add_majority_and_soft_labels(load_and_clean_data(DATA_PATH, ANNOTATOR_COLS), ANNOTATOR_COLS, seed)
    df_long = df.melt(id_vars=['sentence','soft_label','majority_label'], value_vars=ANNOTATOR_COLS, var_name='annotator', value_name='label')
    X, y = df_long['sentence'], df_long['label']
    X_tr, X_dev, X_te, y_tr, y_dev, y_te = split_data(X, y, random_state=seed)
    X_tv, y_tv = pd.concat([X_tr, X_dev]), pd.concat([y_tr, y_dev])
    fold = [-1]*len(X_tr) + [0]*len(X_dev)
    grid = GridSearchCV(build_pipeline_aw(seed), {'tfidf__max_features': [1000, 2000], 'tfidf__ngram_range': [(1,1), (1,2)], 'tfidf__min_df': [1,2], 'tfidf__max_df': [0.7,0.8], 'lr__C': [0.1,1,10]}, cv=PredefinedSplit(fold), scoring='f1_macro', n_jobs=-1)
    grid.fit(X_tv, y_tv)
    yp, pp = grid.predict(X_te), grid.predict_proba(X_te)
    soft = df_long.loc[X_te.index, 'soft_label'].tolist()
    return {'accuracy': accuracy_score(y_te, yp), 'f1': f1_score(y_te, yp, average='macro'), 'ce': manual_cross_entropy(soft, pp), 'brier': compute_brier_score(soft, pp), 'kl': compute_kl_divergence(soft, pp), 'mis': evaluate_by_disagreement_aggregated(yp.tolist(), pp.tolist(), y_te.tolist(), soft)}

if __name__ == '__main__':
    mv_results = [run_lr_experiment_mv(s) for s in SEEDS]
    aw_results = [run_lr_experiment_aw(s) for s in SEEDS]
    import numpy as np
    from evaluation import aggregate_misclassification_results
    def agg(vals): return np.mean(vals), np.std(vals)
    # MV summary
    accs = [r['accuracy'] for r in mv_results]
    f1s = [r['f1'] for r in mv_results]
    mis = [r['mis'] for r in mv_results]
    print("\n=== Aggregated MV LR ===")
    ma, sa = agg(accs); mf, sf = agg(f1s)
    print(f"Accuracy: {ma*100:.2f}% ± {sa*100:.2f}%")
    print(f"Macro F1: {mf:.3f} ± {sf:.3f}")
    for lvl, m in aggregate_misclassification_results(mis).items():
        print(f"{lvl} Disagreement (MV): {m['accuracy_mean']*100:.2f}% ± {m['accuracy_std']*100:.2f}%")

    # AW summary
    accs = [r['accuracy'] for r in aw_results]
    f1s = [r['f1'] for r in aw_results]
    mis = [r['mis'] for r in aw_results]
    print("\n=== Aggregated AW LR ===")
    aa, sa = agg(accs); af, sf = agg(f1s)
    print(f"Accuracy: {aa*100:.2f}% ± {sa*100:.2f}%")
    print(f"Macro F1: {af:.3f} ± {sf:.3f}")
    for lvl, m in aggregate_misclassification_results(mis).items():
        print(f"{lvl} Disagreement (AW): {m['accuracy_mean']*100:.2f}% ± {m['accuracy_std']*100:.2f}%")
