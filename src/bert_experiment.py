# bert_experiment.py

import random
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AWStanceDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {k: v.squeeze() for k, v in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    acc    = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

def run_bert_experiment(seed, approach="MV"):
    set_seed(seed)
    data_path     = r'C:\Users\jaspe\OneDrive\Documenten\Bachelor thesis\GWSD.tsv'
    annotator_cols = [f'worker_{i}' for i in range(8)]

    # 1) Load & label
    df, label_list = add_majority_and_soft_labels(
        load_and_clean_data(data_path, annotator_cols),
        annotator_cols,
        seed=seed
    )

    # 2) Prepare splits
    if approach == "AW":
        df_long = df.melt(
            id_vars=['sentence', 'soft_label'],
            value_vars=annotator_cols,
            var_name='annotator',
            value_name='label'
        )
        X, y          = df_long['sentence'], df_long['label']
        soft_labels   = df_long['soft_label'].tolist()
    else:
        X, y          = df['sentence'], df['majority_label']
        soft_labels   = df['soft_label'].tolist()

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_dev, X_test, y_dev, y_test  = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
    )

    # 3) Label → ID (consistent ordering from add_majority_and_soft_labels)
    label_to_id = {lab: i for i, lab in enumerate(label_list)}

    train_labels = [label_to_id[l] for l in y_train]
    dev_labels   = [label_to_id[l] for l in y_dev]
    test_labels  = [label_to_id[l] for l in y_test]

    # 4) Model & tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model     = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(label_list)
    )

    # 5) Datasets
    train_ds = AWStanceDataset(X_train.tolist(), train_labels, tokenizer)
    dev_ds   = AWStanceDataset(X_dev.tolist(),   dev_labels,   tokenizer)
    test_ds  = AWStanceDataset(X_test.tolist(),  test_labels,  tokenizer)

    # 6) Trainer
    training_args = TrainingArguments(
        output_dir=f'./bert_{approach}_seed_{seed}',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=seed,
        logging_dir=f'./logs_{approach}_{seed}',
        report_to=[]
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics
    )

    # 7) Train & hard eval
    trainer.train()
    hard_metrics = trainer.evaluate(test_ds)

    # 8) Soft metrics on the test split
    preds = trainer.predict(test_ds)
    probs = torch.softmax(torch.from_numpy(preds.predictions), dim=-1).numpy()

    # Align soft‐labels to the test split
    if approach == "AW":
        soft_test = df_long.loc[X_test.index, 'soft_label'].tolist()
    else:
        soft_test = df.loc[X_test.index, 'soft_label'].tolist()

    ce     = manual_cross_entropy(soft_test, probs)
    brier  = compute_brier_score(soft_test, probs)
    kl     = compute_kl_divergence(soft_test, probs)

    # 9) Misclassification breakdown
    pred_labels = preds.predictions.argmax(-1).tolist()
    mis         = evaluate_by_disagreement_aggregated(
                      pred_labels, probs.tolist(), test_labels, soft_test
                  )

    # 10) Optional: aggregate probs per sentence
    df_out     = pd.DataFrame({
        'sentence': X_test.tolist(),
        **{f'prob_{lab}': probs[:, i] for lab, i in label_to_id.items()}
    })
    df_grouped = df_out.groupby('sentence').mean().reset_index()

    return {
        'hard': hard_metrics,
        'ce':   ce,
        'brier': brier,
        'kl':   kl,
        'mis':  mis,
        'probabilities_per_sentence': df_grouped
    }


if __name__ == "__main__":
    seeds     = [10, 51, 100]
    approaches = ["AW", "MV"]

    for approach in approaches:
        print(f"\n=== BERT Experiments: {approach} Approach ===")
        all_res = []
        all_mis = []
        for seed in seeds:
            print(f"\nRunning seed = {seed}")
            res = run_bert_experiment(seed, approach=approach)
            all_res.append(res)
            all_mis.append(res['mis'])

        # Aggregate & print
        accs   = [r['hard']['eval_accuracy'] for r in all_res]
        f1s    = [r['hard']['eval_f1'] for r in all_res]
        ces    = [r['ce'] for r in all_res]
        briers = [r['brier'] for r in all_res]
        kls    = [r['kl'] for r in all_res]

        print(f"\n>>> Aggregated over seeds ({approach}):")
        print(f"Hard Accuracy: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
        print(f"Macro F1:      {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
        print(f"Cross-Entropy: {np.mean(ces):.3f} ± {np.std(ces):.3f}")
        print(f"Brier Score:   {np.mean(briers):.3f} ± {np.std(briers):.3f}")
        print(f"KL Divergence: {np.mean(kls):.3f} ± {np.std(kls):.3f}")

        agg_mis = aggregate_misclassification_results(all_mis)
        for lvl, m in agg_mis.items():
            print(f"\n  {lvl} Disagreement:")
            print(f"    Accuracy:      {m['accuracy_mean']*100:.2f}% ± {m['accuracy_std']*100:.2f}%")
            print(f"    Macro F1:      {m['f1_mean']:.3f} ± {m['f1_std']:.3f}")
            print(f"    Cross-Entropy: {m['cross_entropy_mean']:.3f} ± {m['cross_entropy_std']:.3f}")
            print(f"    Brier Score:   {m['brier_score_mean']:.3f} ± {m['brier_score_std']:.3f}")
            print(f"    KL Divergence: {m['kl_divergence_mean']:.3f} ± {m['kl_divergence_std']:.3f}")
