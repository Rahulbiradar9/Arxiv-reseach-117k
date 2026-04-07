"""
Quick Report (small subset) – generates all graphs using only the first 200 evaluation samples.
Useful for fast preview without exhausting resources.
"""

import os, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report,
    multilabel_confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# Config – same as full script but with a sample limit
MODEL_PATH = "../bert-multi-label-model/final"
CHECKPOINT_STATE = "../bert-multi-label-model/checkpoint-6616/trainer_state.json"
DATA_FILE = "../data/processed_dataset.json"
OUTPUT_DIR = "../screenshots"
LABEL_NAMES = ["AI", "Networks", "Security", "Systems"]
THRESHOLD = 0.5
MAX_EVAL_SAMPLES = 200  # limit for quick run

# Simple style
plt.rcParams.update({"figure.facecolor": "#fff", "axes.facecolor": "#fafafa", "font.size": 12})

def ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_trainer_state():
    with open(CHECKPOINT_STATE, "r") as f:
        return json.load(f)["log_history"]

def load_model_and_data():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=4, problem_type="multi_label_classification")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    eval_set = dataset["test"]
    # Trim to a small subset for speed
    eval_set = eval_set.select(range(min(MAX_EVAL_SAMPLES, len(eval_set))))
    return model, tokenizer, eval_set, device

def run_inference(model, tokenizer, eval_set, device, batch=32):
    texts = eval_set["text"]
    labels = eval_set["multi_hot_vector"]
    all_preds, all_labels = [], []
    for i in range(0, len(texts), batch):
        batch_txt = texts[i:i+batch]
        batch_lbl = labels[i:i+batch]
        inputs = tokenizer(batch_txt, padding=True, truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = torch.sigmoid(model(**inputs).logits).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(np.array(batch_lbl))
    return np.concatenate(all_preds), np.concatenate(all_labels)

def plot_training_loss(log_history):
    train = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    steps = [e["step"] for e in train]
    loss = [e["loss"] for e in train]
    plt.figure(figsize=(8,4))
    plt.plot(steps, loss, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/train_loss_small.png", dpi=300)
    plt.close()

def plot_confusion(true, pred):
    y_pred = (pred >= THRESHOLD).astype(int)
    mcm = multilabel_confusion_matrix(true, y_pred)
    fig, axes = plt.subplots(1,4,figsize=(16,4))
    for i, (mat, label) in enumerate(zip(mcm, LABEL_NAMES)):
        sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(label)
        axes[i].set_xlabel('Pred')
        axes[i].set_ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_small.png", dpi=300)
    plt.close()

def plot_roc(true, pred):
    plt.figure(figsize=(6,5))
    for i, label in enumerate(LABEL_NAMES):
        fpr, tpr, _ = roc_curve(true[:,i], pred[:,i])
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_small.png", dpi=300)
    plt.close()

def main():
    ensure_dir()
    log = load_trainer_state()
    plot_training_loss(log)
    model, tokenizer, eval_set, device = load_model_and_data()
    preds, trues = run_inference(model, tokenizer, eval_set, device)
    # Classification report text
    report = classification_report(trues, (preds>=THRESHOLD).astype(int), target_names=LABEL_NAMES, digits=4)
    with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
        f.write(report)
    # Save heatmap of report
    from sklearn.metrics import classification_report as cr_dict
    rpt = cr_dict(trues, (preds>=THRESHOLD).astype(int), target_names=LABEL_NAMES, output_dict=True, digits=4)
    rows = LABEL_NAMES + ["micro avg", "macro avg", "weighted avg"]
    data = [[rpt[r][c] for c in ["precision","recall","f1-score"]] for r in rows]
    plt.figure(figsize=(8,4))
    sns.heatmap(data, annot=True, fmt=".4f", cmap='Blues', xticklabels=["Prec","Rec","F1"], yticklabels=rows)
    plt.title('Classification Report')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/classification_report_small.png", dpi=300)
    plt.close()
    plot_confusion(trues, preds)
    plot_roc(trues, preds)
    print('Report generation complete.')

if __name__ == '__main__':
    main()
