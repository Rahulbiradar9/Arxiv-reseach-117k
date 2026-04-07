"""
Full Model Evaluation Report – Generates all required graphs and text files for the
portfolio / paper. Uses the actual trainer_state.json and runs inference on the
full evaluation split.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for saving files only
import matplotlib.pyplot as plt
import seaborn as sns
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

# ---------------------------------------------------------------------------
# Paths – resolved relative to this file's location (model_training directory)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "bert-multi-label-model", "final")
CHECKPOINT_STATE = os.path.join(
    BASE_DIR, "bert-multi-label-model", "checkpoint-6616", "trainer_state.json"
)
DATA_FILE = os.path.join(BASE_DIR, "data", "processed_dataset.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "screenshots")
LABEL_NAMES = ["AI", "Networks", "Security", "Systems"]
THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Plot style – clean, professional, no emojis
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#dddddd",
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.color": "#cccccc",
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

# Colors – a small, harmonious palette
PALETTE = {
    "primary": "#2563eb",
    "secondary": "#7c3aed",
    "accent": "#0891b2",
    "success": "#059669",
    "warning": "#d97706",
    "danger": "#dc2626",
    "class": ["#2563eb", "#7c3aed", "#0891b2", "#059669"],
}

# ---------------------------------------------------------------------------
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

# ---------------------------------------------------------------------------
def load_trainer_state():
    print("Loading trainer_state.json …")
    with open(CHECKPOINT_STATE, "r") as f:
        state = json.load(f)
    return state["log_history"]

# ---------------------------------------------------------------------------
def load_model_and_eval_set():
    print("Loading model and tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=4, problem_type="multi_label_classification"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("Loading processed dataset …")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    # 10 % hold‑out for evaluation
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    eval_set = dataset["test"]
    print(f"Evaluation set size: {len(eval_set)} samples")
    return model, tokenizer, eval_set, device

# ---------------------------------------------------------------------------
def run_inference(model, tokenizer, eval_set, device, batch_size=64):
    texts = eval_set["text"]
    labels = eval_set["multi_hot_vector"]
    all_preds = []
    all_labels = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]
        inputs = tokenizer(
            batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = torch.sigmoid(model(**inputs).logits).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(np.array(batch_labels))
    predictions = np.concatenate(all_preds, axis=0)
    true_labels = np.concatenate(all_labels, axis=0)
    return predictions, true_labels

# ---------------------------------------------------------------------------
# Plot 1 – Training loss curve (all steps)
# ---------------------------------------------------------------------------
def plot_training_loss(log_history):
    train = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    steps = [e["step"] for e in train]
    loss = [e["loss"] for e in train]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, loss, color=PALETTE["primary"], linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss over steps")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"), dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 2 – Training vs Validation loss (step‑wise)
# ---------------------------------------------------------------------------
def plot_train_vs_val(log_history):
    train = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_ = [e for e in log_history if "eval_loss" in e]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([e["step"] for e in train], [e["loss"] for e in train], label="Train", color=PALETTE["primary"])
    ax.plot([e["step"] for e in eval_], [e["eval_loss"] for e in eval_], label="Validation", color=PALETTE["danger"], marker="o")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "train_vs_val_loss.png"), dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 3 – Learning‑rate schedule
# ---------------------------------------------------------------------------
def plot_lr(log_history):
    lr_entries = [e for e in log_history if "learning_rate" in e]
    steps = [e["step"] for e in lr_entries]
    lrs = [e["learning_rate"] for e in lr_entries]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(steps, lrs, color=PALETTE["secondary"], linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning‑rate schedule")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1e}"))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "learning_rate.png"), dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 4 – Evaluation metrics per epoch (F1‑micro, F1‑macro, precision, recall)
# ---------------------------------------------------------------------------
def plot_eval_metrics(log_history):
    eval_entries = [e for e in log_history if "eval_f1_macro" in e]
    epochs = [int(e["epoch"]) for e in eval_entries]
    metrics = {
        "F1‑micro": [e["eval_f1_micro"] for e in eval_entries],
        "F1‑macro": [e["eval_f1_macro"] for e in eval_entries],
        "Precision": [e["eval_precision"] for e in eval_entries],
        "Recall": [e["eval_recall"] for e in eval_entries],
    }
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["success"], PALETTE["warning"]]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(epochs))
    width = 0.18
    for i, (name, vals) in enumerate(metrics.items()):
        ax.bar(x + i * width, vals, width, label=name, color=colors[i])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"Epoch {e}" for e in epochs])
    ax.set_ylabel("Score")
    ax.set_title("Evaluation metrics per epoch")
    ax.set_ylim(0.98, 1.002)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eval_metrics.png"), dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 5 – Classification report (heat‑map) and raw text file
# ---------------------------------------------------------------------------
def generate_classification_report(true_labels, pred_probs):
    y_pred = (pred_probs >= THRESHOLD).astype(int)
    report_text = classification_report(
        true_labels, y_pred, target_names=LABEL_NAMES, digits=4
    )
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report_text)
    # Heat‑map (precision, recall, f1)
    from sklearn.metrics import classification_report as cr_dict
    rpt = cr_dict(true_labels, y_pred, target_names=LABEL_NAMES, output_dict=True, digits=4)
    rows = LABEL_NAMES + ["micro avg", "macro avg", "weighted avg"]
    data = [[rpt[r][c] for c in ["precision", "recall", "f1-score"]] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        data,
        annot=True,
        fmt=".4f",
        cmap="Blues",
        xticklabels=["Precision", "Recall", "F1"],
        yticklabels=rows,
        ax=ax,
    )
    ax.set_title("Classification report heat‑map")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "classification_report.png"), dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 6 – Per‑class confusion matrices (4‑panel)
# ---------------------------------------------------------------------------
def plot_confusion_matrices(true_labels, pred_probs):
    y_pred = (pred_probs >= THRESHOLD).astype(int)
    mcm = multilabel_confusion_matrix(true_labels, y_pred)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for i, (mat, label) in enumerate(zip(mcm, LABEL_NAMES)):
        ax = axes[i]
        sns.heatmap(
            mat,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"],
            ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion matrices – per class", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 7 – ROC curves (multi‑label)
# ---------------------------------------------------------------------------
def plot_roc_curves(true_labels, pred_probs):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (label, color) in enumerate(zip(LABEL_NAMES, PALETTE["class"])):
        fpr, tpr, _ = roc_curve(true_labels[:, i], pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves – multi‑label")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 8 – Precision‑Recall curves (multi‑label)
# ---------------------------------------------------------------------------
def plot_pr_curves(true_labels, pred_probs):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (label, color) in enumerate(zip(LABEL_NAMES, PALETTE["class"])):
        prec, rec, _ = precision_recall_curve(true_labels[:, i], pred_probs[:, i])
        ap = average_precision_score(true_labels[:, i], pred_probs[:, i])
        ax.plot(rec, prec, label=f"{label} (AP={ap:.3f})", color=color, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision‑Recall curves – multi‑label")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "precision_recall_curves.png"), dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------------
def main():
    ensure_output_dir()
    log_history = load_trainer_state()
    # 1‑4: trainer‑state based plots
    plot_training_loss(log_history)
    plot_train_vs_val(log_history)
    plot_lr(log_history)
    plot_eval_metrics(log_history)
    # 5‑8: inference based plots
    model, tokenizer, eval_set, device = load_model_and_eval_set()
    preds, trues = run_inference(model, tokenizer, eval_set, device)
    generate_classification_report(trues, preds)
    plot_confusion_matrices(trues, preds)
    plot_roc_curves(trues, preds)
    plot_pr_curves(trues, preds)
    print("Report generation complete. Files are in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
