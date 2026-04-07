"""
Generate Model Training Report
================================
Produces all evaluation graphs and classification reports from the
trained DistilBERT multi-label ArXiv classifier.

Outputs are saved to: ../screenshots/
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving only
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "../bert-multi-label-model/final"
CHECKPOINT_STATE = "../bert-multi-label-model/checkpoint-6616/trainer_state.json"
DATA_FILE = "../data/processed_dataset.json"
OUTPUT_DIR = "../screenshots"
LABEL_NAMES = ["AI", "Networks", "Security", "Systems"]
THRESHOLD = 0.5

# Plot style
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#cccccc",
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

COLORS = {
    "primary": "#2563eb",
    "secondary": "#7c3aed",
    "accent": "#0891b2",
    "success": "#059669",
    "warning": "#d97706",
    "danger": "#dc2626",
    "classes": ["#2563eb", "#7c3aed", "#0891b2", "#059669"],
}


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")


def load_trainer_state():
    print(f"Loading trainer state from {CHECKPOINT_STATE}...")
    with open(CHECKPOINT_STATE, "r") as f:
        state = json.load(f)
    return state["log_history"]


def load_model_and_data():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=4, problem_type="multi_label_classification"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"  -> Device: {device}")

    print(f"Loading dataset from {DATA_FILE}...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    eval_dataset = dataset["test"]
    print(f"  -> Eval samples: {len(eval_dataset)}")

    return model, tokenizer, eval_dataset, device


def run_inference(model, tokenizer, eval_dataset, device, batch_size=64):
    """Run inference on eval set and return raw predictions + true labels."""
    print("Running inference on evaluation set...")
    all_preds = []
    all_labels = []

    texts = eval_dataset["text"]
    labels = eval_dataset["multi_hot_vector"]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]

        inputs = tokenizer(
            batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()

        all_preds.append(probs)
        all_labels.append(np.array(batch_labels))

        if (i // batch_size + 1) % 20 == 0:
            print(f"  -> Processed {min(i + batch_size, len(texts))}/{len(texts)} samples")

    predictions = np.concatenate(all_preds, axis=0)
    true_labels = np.concatenate(all_labels, axis=0)
    print(f"  -> Inference complete. Shape: {predictions.shape}")
    return predictions, true_labels


# ---------------------------------------------------------------------------
# Plot 1: Training Loss Curve
# ---------------------------------------------------------------------------
def plot_training_loss(log_history):
    print("Generating: Training Loss Curve...")
    train_steps = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    steps = [e["step"] for e in train_steps]
    losses = [e["loss"] for e in train_steps]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, color=COLORS["primary"], linewidth=2, label="Training Loss")
    ax.fill_between(steps, losses, alpha=0.08, color=COLORS["primary"])

    # Mark epoch boundaries
    for e in log_history:
        if "eval_loss" in e:
            ax.axvline(x=e["step"], color="#999999", linestyle="--", alpha=0.5, linewidth=1)
            ax.text(e["step"], max(losses) * 0.95, f'Epoch {int(e["epoch"])}',
                    ha="center", fontsize=9, color="#666666")

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Over Steps")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/training_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved: training_loss.png")


# ---------------------------------------------------------------------------
# Plot 2: Training vs Validation Loss
# ---------------------------------------------------------------------------
def plot_train_vs_val_loss(log_history):
    print("Generating: Training vs Validation Loss...")
    train_steps = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_steps = [e for e in log_history if "eval_loss" in e]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        [e["step"] for e in train_steps],
        [e["loss"] for e in train_steps],
        color=COLORS["primary"], linewidth=2, label="Training Loss", marker="", markersize=4,
    )
    ax.plot(
        [e["step"] for e in eval_steps],
        [e["eval_loss"] for e in eval_steps],
        color=COLORS["danger"], linewidth=2, label="Validation Loss",
        marker="o", markersize=8, markerfacecolor="white", markeredgewidth=2,
    )

    for e in eval_steps:
        ax.annotate(
            f'{e["eval_loss"]:.4f}',
            xy=(e["step"], e["eval_loss"]),
            xytext=(0, 14), textcoords="offset points",
            fontsize=10, ha="center", color=COLORS["danger"], fontweight="bold",
        )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/train_vs_val_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved: train_vs_val_loss.png")


# ---------------------------------------------------------------------------
# Plot 3: Learning Rate Schedule
# ---------------------------------------------------------------------------
def plot_learning_rate(log_history):
    print("Generating: Learning Rate Schedule...")
    lr_entries = [e for e in log_history if "learning_rate" in e]
    steps = [e["step"] for e in lr_entries]
    lrs = [e["learning_rate"] for e in lr_entries]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, lrs, color=COLORS["secondary"], linewidth=2)
    ax.fill_between(steps, lrs, alpha=0.08, color=COLORS["secondary"])
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/learning_rate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved: learning_rate.png")


# ---------------------------------------------------------------------------
# Plot 4: F1 / Precision / Recall Over Epochs
# ---------------------------------------------------------------------------
def plot_eval_metrics(log_history):
    print("Generating: Evaluation Metrics...")
    eval_entries = [e for e in log_history if "eval_f1_macro" in e]
    epochs = [int(e["epoch"]) for e in eval_entries]

    metrics = {
        "F1 Micro": [e["eval_f1_micro"] for e in eval_entries],
        "F1 Macro": [e["eval_f1_macro"] for e in eval_entries],
        "Precision": [e["eval_precision"] for e in eval_entries],
        "Recall": [e["eval_recall"] for e in eval_entries],
    }

    colors_list = [COLORS["primary"], COLORS["secondary"], COLORS["success"], COLORS["warning"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(epochs))
    width = 0.18

    for i, (name, values) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, values, width, label=name, color=colors_list[i], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics Per Epoch")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"Epoch {e}" for e in epochs])
    ax.set_ylim(0.98, 1.002)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/eval_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved: eval_metrics.png")


# ---------------------------------------------------------------------------
# Plot 5: Classification Report (as heatmap)
# ---------------------------------------------------------------------------
def plot_classification_report(true_labels, predictions):
    print("Generating: Classification Report...")

    y_pred = (predictions >= THRESHOLD).astype(int)

    # Print text report
    report_text = classification_report(true_labels, y_pred, target_names=LABEL_NAMES, digits=4)
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report_text)
    print("=" * 60 + "\n")

    # Save text report
    with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
        f.write(report_text)

    # Build heatmap from report
    from sklearn.metrics import classification_report as cr_dict
    report = cr_dict(true_labels, y_pred, target_names=LABEL_NAMES, output_dict=True, digits=4)

    rows = LABEL_NAMES + ["micro avg", "macro avg", "weighted avg", "samples avg"]
    cols = ["precision", "recall", "f1-score", "support"]
    data = []
    for row_name in rows:
        if row_name in report:
            data.append([report[row_name].get(c, 0) for c in cols])

    data_arr = np.array(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    # Separate support column (integer) from ratio columns for formatting
    display_data = data_arr.copy()
    annot_labels = []
    for row in data_arr:
        row_labels = []
        for j, val in enumerate(row):
            if j == 3:  # support column
                row_labels.append(f"{int(val)}")
            else:
                row_labels.append(f"{val:.4f}")
        annot_labels.append(row_labels)

    sns.heatmap(
        data_arr[:, :3],  # Only precision, recall, f1
        annot=np.array(annot_labels)[:, :3],
        fmt="",
        cmap="Blues",
        xticklabels=["Precision", "Recall", "F1-Score"],
        yticklabels=rows,
        vmin=0.95, vmax=1.0,
        linewidths=0.5,
        linecolor="#eeeeee",
        ax=ax,
    )
    ax.set_title("Classification Report Heatmap")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/classification_report.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved: classification_report.png")


# ---------------------------------------------------------------------------
# Plot 6: Per-Class Confusion Matrices
# ---------------------------------------------------------------------------
def plot_confusion_matrices(true_labels, predictions):
    print("Generating: Confusion Matrices...")

    y_pred = (predictions >= THRESHOLD).astype(int)
    mcm = multilabel_confusion_matrix(true_labels, y_pred)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for i, (matrix, label, color) in enumerate(zip(mcm, LABEL_NAMES, COLORS["classes"])):
        ax = axes[i]
        sns.heatmap(
            matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"],
            linewidths=0.5, linecolor="#eeeeee", ax=ax,
            annot_kws={"size": 14, "fontweight": "bold"},
        )
        ax.set_title(f"{label}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices — Per Class", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved: confusion_matrices.png")


# ---------------------------------------------------------------------------
# Plot 7: ROC Curves (Multi-label)
# ---------------------------------------------------------------------------
def plot_roc_curves(true_labels, predictions):
    print("Generating: ROC Curves...")

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, (label, color) in enumerate(zip(LABEL_NAMES, COLORS["classes"])):
        fpr, tpr, _ = roc_curve(true_labels[:, i], predictions[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label}  (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], color="#cccccc", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Multi-Label")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved: roc_curves.png")


# ---------------------------------------------------------------------------
# Plot 8: Precision-Recall Curves
# ---------------------------------------------------------------------------
def plot_precision_recall_curves(true_labels, predictions):
    print("Generating: Precision-Recall Curves...")

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, (label, color) in enumerate(zip(LABEL_NAMES, COLORS["classes"])):
        prec, rec, _ = precision_recall_curve(true_labels[:, i], predictions[:, i])
        ap = average_precision_score(true_labels[:, i], predictions[:, i])
        ax.plot(rec, prec, color=color, linewidth=2, label=f"{label}  (AP = {ap:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Multi-Label")
    ax.legend(loc="lower left")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/precision_recall_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved: precision_recall_curves.png")


# ---------------------------------------------------------------------------
# Plot 9: Gradient Norms
# ---------------------------------------------------------------------------
def plot_gradient_norms(log_history):
    print("Generating: Gradient Norms...")

    grad_entries = [e for e in log_history if "grad_norm" in e]
    steps = [e["step"] for e in grad_entries]
    norms = [e["grad_norm"] for e in grad_entries]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, norms, color=COLORS["accent"], linewidth=2, marker="o", markersize=4)
    ax.fill_between(steps, norms, alpha=0.08, color=COLORS["accent"])
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm During Training")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/gradient_norms.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved: gradient_norms.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  MODEL TRAINING REPORT GENERATOR")
    print("=" * 60)
    print()

    ensure_output_dir()

    # ---- Phase 1: Plots from trainer_state.json (no GPU needed) ----
    log_history = load_trainer_state()
    plot_training_loss(log_history)
    plot_train_vs_val_loss(log_history)
    plot_learning_rate(log_history)
    plot_eval_metrics(log_history)
    plot_gradient_norms(log_history)

    # ---- Phase 2: Plots requiring model inference on eval set ----
    model, tokenizer, eval_dataset, device = load_model_and_data()
    predictions, true_labels = run_inference(model, tokenizer, eval_dataset, device)

    plot_classification_report(true_labels, predictions)
    plot_confusion_matrices(true_labels, predictions)
    plot_roc_curves(true_labels, predictions)
    plot_precision_recall_curves(true_labels, predictions)

    # ---- Summary ----
    print()
    print("=" * 60)
    print("  ALL REPORTS GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"\n  Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Files generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"    - {f}  ({size / 1024:.1f} KB)")
    print()


if __name__ == "__main__":
    main()
