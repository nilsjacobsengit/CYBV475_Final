"""
evaluate.py
===========
Loads the fine-tuned model and produces a detailed evaluation report:
  - Accuracy, Precision, Recall, F1
  - Confusion matrix
  - Confidence distribution plots (saved to ../results/)
  - Top misclassified examples (most useful for your write-up)

Run after fine_tune.py has finished:
    python evaluate.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_DIR   = "../results/model"
DATA_PATH   = "../dataset/reviews.csv"
RESULTS_DIR = "../results"
MAX_LEN     = 128
SEED        = 42

LABEL2ID = {"real": 0, "fake": 1}
ID2LABEL = {0: "real", 1: "fake"}

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def load_eval_data():
    df = pd.read_csv(DATA_PATH)[["review_text", "label"]].dropna()
    df["label_id"] = df["label"].map(LABEL2ID)
    _, eval_df = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df["label_id"])
    return eval_df.reset_index(drop=True)


def get_predictions(model, tokenizer, texts, device):
    all_probs = []
    model.eval()
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=MAX_LEN, padding=True).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        all_probs.append(probs)
    return np.array(all_probs)   # shape (N, 2)


# ─────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["real", "fake"],
                yticklabels=["real", "fake"], ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix — Fake Review Detector")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_confidence_distributions(df_eval, probs, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (label_str, label_id) in zip(axes, [("real", 0), ("fake", 1)]):
        mask = df_eval["label_id"] == label_id
        confs = probs[mask, label_id] * 100
        ax.hist(confs, bins=15, color=("#4CAF50" if label_id == 0 else "#F44336"),
                alpha=0.75, edgecolor="white")
        ax.set_title(f"Confidence on TRUE {label_str.upper()} reviews")
        ax.set_xlabel("Model confidence for correct class (%)")
        ax.set_ylabel("Count")
        ax.axvline(50, color="grey", linestyle="--", linewidth=1, label="50% threshold")
        ax.legend()
    fig.suptitle("Confidence Distribution by True Class", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_roc_curve(y_true, probs_fake, save_path):
    fpr, tpr, _ = roc_curve(y_true, probs_fake)
    auc = roc_auc_score(y_true, probs_fake)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#1565C0", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Fake Review Detector")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Fake Review Detector — Detailed Evaluation")
    print("=" * 55)

    # ── Load data ──────────────────────────────────────────
    df_eval = load_eval_data()
    texts   = df_eval["review_text"].tolist()
    y_true  = df_eval["label_id"].values

    print(f"\n📊 Evaluation set: {len(df_eval)} samples")

    # ── Load model ─────────────────────────────────────────
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    print(f"🤖 Model loaded from {MODEL_DIR}  (device: {device})\n")

    # ── Predictions ────────────────────────────────────────
    probs  = get_predictions(model, tokenizer, texts, device)
    y_pred = np.argmax(probs, axis=-1)

    # ── Classification report ──────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probs[:, 1])

    print(f"{'─'*45}")
    print(f"  Accuracy : {acc*100:.1f}%")
    print(f"  ROC-AUC  : {auc:.3f}")
    print(f"{'─'*45}")
    print("\n" + classification_report(y_true, y_pred, target_names=["real", "fake"]))

    # ── Plots ──────────────────────────────────────────────
    print("📊 Generating plots …")
    plot_confusion_matrix(
        y_true, y_pred,
        os.path.join(RESULTS_DIR, "confusion_matrix.png")
    )
    plot_confidence_distributions(
        df_eval, probs,
        os.path.join(RESULTS_DIR, "confidence_distributions.png")
    )
    plot_roc_curve(
        y_true, probs[:, 1],
        os.path.join(RESULTS_DIR, "roc_curve.png")
    )

    # ── Misclassified examples ─────────────────────────────
    df_eval = df_eval.copy()
    df_eval["predicted"]    = [ID2LABEL[p] for p in y_pred]
    df_eval["prob_fake"]    = probs[:, 1]
    df_eval["prob_real"]    = probs[:, 0]
    df_eval["correct"]      = df_eval["label_id"] == y_pred
    df_eval["confidence"]   = np.max(probs, axis=1)

    misclassified = df_eval[~df_eval["correct"]].sort_values("confidence", ascending=False)

    print(f"\n⚠️  Misclassified Examples ({len(misclassified)} total):")
    print(f"{'─'*55}")
    for _, row in misclassified.head(10).iterrows():
        snippet = row["review_text"][:90] + ("…" if len(row["review_text"]) > 90 else "")
        print(f"  True: {row['label']:4s} | Pred: {row['predicted']:4s} "
              f"| Conf: {row['confidence']*100:.1f}%")
        print(f"  \"{snippet}\"")
        print()

    # ── Save misclassified to CSV ──────────────────────────
    mis_path = os.path.join(RESULTS_DIR, "misclassified.csv")
    misclassified.to_csv(mis_path, index=False)
    print(f"  Saved full misclassified list: {mis_path}")

    print("\n✅ Evaluation complete!")
    print(f"   Plots and CSVs saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
