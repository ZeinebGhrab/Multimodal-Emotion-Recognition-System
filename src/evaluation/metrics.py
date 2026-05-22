"""
src/evaluation/metrics.py
──────────────────────────
Comprehensive evaluation for all emotion recognition models.

Produces:
  - Accuracy, Precision, Recall, F1 (macro + per-class)
  - Confusion matrix (normalized + absolute)
  - Training curves (loss + accuracy over epochs)
  - Model comparison bar chart
  - Per-class performance breakdown
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ─── Core metrics ─────────────────────────────────────────────────────────────

def compute_metrics(y_true: list, y_pred: list,
                    labels: list = EMOTION_LABELS) -> dict:
    """
    Compute a full suite of classification metrics.

    Returns a dict with:
      accuracy, macro_precision, macro_recall, macro_f1,
      per_class (list of dicts per class),
      confusion_matrix (2-D list)

    FIX: passes labels=all_labels to precision_recall_fscore_support so that
    classes absent from y_pred (e.g. 'disgust'/'neutral' when using the NLP
    dataset) are still included with score=0, preventing an IndexError.
    """
    acc = accuracy_score(y_true, y_pred)

    num_classes = len(labels)
    all_labels  = list(range(num_classes))   # ← FIX: force all 7 indices

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=all_labels, zero_division=0
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=all_labels, zero_division=0
    )

    # Macro F1 restricted to classes that actually appear in the test set.
    # When the text dataset (dair-ai/emotion) is used, 'disgust' and 'neutral'
    # have no samples, so their F1=0 drags the full macro down artificially.
    # This metric gives the honest picture of performance on present classes.
    present_labels = [i for i in all_labels if support[i] > 0]
    if present_labels and len(present_labels) < num_classes:
        _, _, macro_f1_present, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", labels=present_labels, zero_division=0
        )
    else:
        macro_f1_present = float(macro_f1)   # all classes present — no difference

    cm = confusion_matrix(y_true, y_pred, labels=all_labels).tolist()

    per_class = [
        {
            "emotion":   labels[i],
            "precision": float(prec[i]),
            "recall":    float(rec[i]),
            "f1":        float(f1[i]),
            "support":   int(support[i])
        }
        for i in range(num_classes)
    ]

    report = classification_report(
        y_true, y_pred, target_names=labels, labels=all_labels, zero_division=0
    )
    print(report)

    missing = [labels[i] for i in all_labels if support[i] == 0]
    if missing:
        print(f"[Metrics] Zero-support classes (excluded from macro_f1_present): "
              f"{missing}")
        print(f"[Metrics] macro_f1_all={macro_f1*100:.2f}%  "
              f"macro_f1_present={macro_f1_present*100:.2f}%\n")

    return {
        "accuracy":             float(acc),
        "macro_precision":      float(macro_prec),
        "macro_recall":         float(macro_rec),
        "macro_f1":             float(macro_f1),          # all 7 classes
        "macro_f1_present":     float(macro_f1_present),  # present classes only
        "zero_support_classes": missing,
        "per_class":            per_class,
        "confusion_matrix":     cm,
        "classification_report": report
    }


def save_metrics(metrics: dict, path: str):
    """Save metrics dict as JSON (excluding the classification report string)."""
    data = {k: v for k, v in metrics.items() if k != "classification_report"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Metrics] Saved to {path}")


# ─── Plotting ─────────────────────────────────────────────────────────────────

_PALETTE = {
    "bg":      "#0F1117",
    "surface": "#1C1F2A",
    "accent":  "#4F8EF7",
    "success": "#4ADE80",
    "warning": "#FBBF24",
    "danger":  "#F87171",
    "text":    "#E2E8F0",
    "muted":   "#94A3B8",
}


def _style():
    """Apply dark publication-quality style."""
    plt.rcParams.update({
        "figure.facecolor": _PALETTE["bg"],
        "axes.facecolor":   _PALETTE["surface"],
        "axes.edgecolor":   _PALETTE["muted"],
        "axes.labelcolor":  _PALETTE["text"],
        "text.color":       _PALETTE["text"],
        "xtick.color":      _PALETTE["muted"],
        "ytick.color":      _PALETTE["muted"],
        "grid.color":       "#2D3748",
        "grid.linestyle":   "--",
        "grid.linewidth":   0.5,
        "font.family":      "DejaVu Sans",
        "font.size":        11,
    })


def plot_confusion_matrix(cm, labels, title="Confusion Matrix",
                           save_path=None, normalize=True):
    """
    Plot a heatmap of the confusion matrix.

    Args:
        cm         : 2-D list (output of compute_metrics)
        normalize  : normalise rows so values are recall per class
    """
    _style()
    cm_arr = np.array(cm, dtype=float)
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_arr   = np.divide(cm_arr, row_sums, where=row_sums != 0)
        fmt, vmax = ".2f", 1.0
    else:
        fmt, vmax = "d", cm_arr.max()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_arr, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.4, linecolor="#2D3748",
        ax=ax, cbar_kws={"shrink": 0.8},
        vmin=0, vmax=vmax
    )
    ax.set_title(title, fontsize=14, pad=16, color=_PALETTE["text"])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Confusion matrix → {save_path}")
    plt.close()


def plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                          model_name="Model", save_path=None):
    """Plot loss and accuracy curves for train and validation sets."""
    _style()
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss
    ax1.plot(epochs, train_losses, color=_PALETTE["accent"], lw=2, label="Train loss")
    ax1.plot(epochs, val_losses, color=_PALETTE["warning"], lw=2,
             linestyle="--", label="Val loss")
    ax1.set_title("Loss", fontsize=13)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    # Accuracy
    ax2.plot(epochs, [a*100 for a in train_accs], color=_PALETTE["success"],
             lw=2, label="Train acc")
    ax2.plot(epochs, [a*100 for a in val_accs], color=_PALETTE["danger"],
             lw=2, linestyle="--", label="Val acc")
    ax2.set_title("Accuracy", fontsize=13)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    fig.suptitle(f"Training Curves — {model_name}", fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Training curves → {save_path}")
    plt.close()


def plot_model_comparison(results: dict, save_path: str):
    """
    Bar chart comparing models on Accuracy, Macro-F1, Macro-Precision, Macro-Recall.

    Args:
        results : {model_name: metrics_dict, ...}
    """
    _style()
    models = list(results.keys())
    metrics = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    labels  = ["Accuracy", "Macro F1", "Macro Precision", "Macro Recall"]
    colors  = [_PALETTE["accent"], _PALETTE["success"],
               _PALETTE["warning"], _PALETTE["danger"]]

    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        values = [results[m].get(metric, 0) * 100 for m in models]
        bars = ax.bar(x + i * width, values, width, label=label,
                      color=color, alpha=0.85, edgecolor="none")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=8, color=_PALETTE["text"])

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Model Comparison — Emotion Recognition", fontsize=14)
    ax.legend(framealpha=0.3)
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Model comparison → {save_path}")


def plot_per_class_f1(metrics_dict: dict, model_name: str,
                      labels: list, save_path: str):
    """Horizontal bar chart of per-class F1 scores."""
    _style()
    f1_scores = [c["f1"] * 100 for c in metrics_dict["per_class"]]
    colors = [_PALETTE["success"] if s >= 70 else
              _PALETTE["warning"] if s >= 50 else
              _PALETTE["danger"] for s in f1_scores]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, f1_scores, color=colors, edgecolor="none", height=0.6)

    for bar, val in zip(bars, f1_scores):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, color=_PALETTE["text"])

    ax.axvline(x=70, color=_PALETTE["muted"], linestyle="--", lw=1, alpha=0.6)
    ax.set_xlim(0, 105)
    ax.set_xlabel("F1 Score (%)")
    ax.set_title(f"Per-class F1 — {model_name}", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Per-class F1 → {save_path}")
    plt.close()