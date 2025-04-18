# src/bert_experiment.py
import random
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data_preprocessing import load_and_clean_data, add_majority_and_soft_labels
from evaluation import manual_cross_entropy, compute_brier_score, compute_kl_divergence, evaluate_by_disagreement_aggregated, aggregate_misclassification_results
from config import DATA_PATH, ANNOTATOR_COLS, SEEDS

class AWStanceDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts, self.labels, self.tokenizer, self.max_length = texts, labels, tokenizer, max_length
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {k: v.squeeze() for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    acc    = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

def run_bert_experiment(seed, approach='MV'):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    df = load_and_clean_data(DATA_PATH, ANNOTATOR_COLS)
    df, label_list = add_majority_and_soft_labels(df, ANNOTATOR_COLS, seed)
    if approach=='AW':
        df_long = df.melt(id_vars=['sentence','soft_label'], value_vars=ANNOTATOR_COLS, var_name='annotator', value_name='label')
        X, y, soft = df_long['sentence'], df_long['label'], df_long['soft_label'].tolist()
    else:
        X, y, soft = df['sentence'], df['majority_label'], df['soft_label'].tolist()
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    X_dev, X_te, y_dev, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)
    label_to_id = {lab:i for i,lab in enumerate(label_list)}
    tr_labels = [label_to_id[l] for l in y_tr]
    dev_labels= [label_to_id[l] for l in y_dev]
    te_labels = [label_to_id[l] for l in y_te]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model     = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_list))
    tr_ds = AWStanceDataset(X_tr.tolist(), tr_labels, tokenizer)
    dv_ds = AWStanceDataset(X_dev.tolist(), dev_labels, tokenizer)
    te_ds = AWStanceDataset(X_te.tolist(), te_labels, tokenizer)
    args  = TrainingArguments(output_dir=f'./bert_{approach}_seed_{seed}', num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=16, evaluation_strategy='epoch', save_strategy='epoch', learning_rate=3e-5, load_best_model_at_end=True, metric_for_best_model='f1', greater_is_better=True, seed=seed, logging_dir=f'./logs_{approach}_{seed}', report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=tr_ds, eval_dataset=dv_ds, compute_metrics=compute_metrics)
    trainer.train()
    hard = trainer.evaluate(te_ds)
    preds = trainer.predict(te_ds)
    probs = torch.softmax(torch.from_numpy(preds.predictions), dim=-1).numpy()
    soft_test = soft if approach=='AW' else [df.loc[X_te.index,'soft_label'][i] for i in range(len(X_te))]
    ce    = manual_cross_entropy(soft_test, probs)
    brier = compute_brier_score(soft_test, probs)
    kl    = compute_kl_divergence(soft_test, probs)
    mis   = evaluate_by_disagreement_aggregated(preds.predictions.argmax(-1).tolist(), probs.tolist(), te_labels, soft_test)
    return {'hard':hard, 'ce':ce, 'brier':brier, 'kl':kl, 'mis':mis}

if __name__=='__main__':
    for approach in ['AW','MV']:
        print(f"\n=== BERT Experiments: {approach} Approach ===")
        all_res, all_mis = [], []
        for s in SEEDS:
            print(f"Running seed={s}")
            res = run_bert_experiment(s, approach)
            all_res.append(res); all_mis.append(res['mis'])
        accs = [r['hard']['eval_accuracy'] for r in all_res]
        f1s  = [r['hard']['eval_f1'] for r in all_res]
        ces  = [r['ce'] for r in all_res]
        brs  = [r['brier'] for r in all_res]
        kls  = [r['kl'] for r in all_res]
        import numpy as _np
        print(f"\n>>> Aggregated over seeds ({approach}):")
        print(f"Hard Acc: {_np.mean(accs)*100:.2f}% ± {_np.std(accs)*100:.2f}%")
        print(f"Macro F1:   {_np.mean(f1s):.3f} ± {_np.std(f1s):.3f}")
        print(f"Cross-Ent: {_np.mean(ces):.3f} ± {_np.std(ces):.3f}")
        print(f"Brier:      {_np.mean(brs):.3f} ± {_np.std(brs):.3f}")
        print(f"KL Div:     {_np.mean(kls):.3f} ± {_np.std(kls):.3f}")
        agg = aggregate_misclassification_results(all_mis)
        for lvl,m in agg.items():
            print(f"{lvl} Dis: Acc={m['accuracy_mean']*100:.2f}% ± {m['accuracy_std']*100:.2f}%")
