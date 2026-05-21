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
                    labels: list[str] = EMOTION_LABELS) -> dict:
    """
    Compute a full suite of classification metrics.

    Returns a dict with:
      accuracy, macro_precision, macro_recall, macro_f1,
      per_class (list of dicts per class),
      confusion_matrix (2-D list)
    """
    acc = accuracy_score(y_true, y_pred)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred).tolist()

    per_class = [
        {
            "emotion":   labels[i],
            "precision": float(prec[i]),
            "recall":    float(rec[i]),
            "f1":        float(f1[i]),
            "support":   int(support[i])
        }
        for i in range(len(labels))
    ]

    report = classification_report(y_true, y_pred, target_names=labels)
    print(report)

    return {
        "accuracy":         float(acc),
        "macro_precision":  float(macro_prec),
        "macro_recall":     float(macro_rec),
        "macro_f1":         float(macro_f1),
        "per_class":        per_class,
        "confusion_matrix": cm,
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


def plot_confusion_matrix(cm: list[list], labels: list[str],
                           title: str, save_path: str, normalize: bool = True):
    """
    Plot a heatmap of the confusion matrix.

    Args:
        cm         : 2-D list (output of compute_metrics)
        normalize  : normalise rows so values are recall per class
    """
    _style()
    cm_arr = np.array(cm, dtype=float)
    if normalize:
        cm_arr = cm_arr / (cm_arr.sum(axis=1, keepdims=True) + 1e-8)
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Confusion matrix → {save_path}")


def plot_training_curves(train_losses: list, val_losses: list,
                          train_accs: list, val_accs: list,
                          model_name: str, save_path: str):
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training curves → {save_path}")


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
                      labels: list[str], save_path: str):
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Per-class F1 → {save_path}")


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(0)

    n = 500
    y_true = [random.randint(0, 6) for _ in range(n)]
    y_pred = [t if random.random() > 0.35 else random.randint(0, 6) for t in y_true]

    m = compute_metrics(y_true, y_pred)
    print(f"\nAccuracy : {m['accuracy']*100:.2f}%")
    print(f"Macro F1 : {m['macro_f1']*100:.2f}%")

    os.makedirs("outputs/figures", exist_ok=True)
    plot_confusion_matrix(m["confusion_matrix"], EMOTION_LABELS,
                           "Demo Confusion Matrix", "outputs/figures/demo_cm.png")

    # Simulated training curves
    tl = [2.0 * (0.85**e) for e in range(20)]
    vl = [2.1 * (0.87**e) for e in range(20)]
    ta = [0.3 + 0.6 * (1 - 0.85**e) for e in range(20)]
    va = [0.28 + 0.55 * (1 - 0.87**e) for e in range(20)]

    plot_training_curves(tl, vl, ta, va, "Demo CNN", "outputs/figures/demo_curves.png")

    # Model comparison with simulated data
    fake_results = {
        "CNN":               {"accuracy": 0.65, "macro_f1": 0.62, "macro_precision": 0.63, "macro_recall": 0.62},
        "ViT":               {"accuracy": 0.68, "macro_f1": 0.66, "macro_precision": 0.67, "macro_recall": 0.66},
        "Bi-LSTM":           {"accuracy": 0.70, "macro_f1": 0.68, "macro_precision": 0.69, "macro_recall": 0.68},
        "BERT":              {"accuracy": 0.78, "macro_f1": 0.76, "macro_precision": 0.77, "macro_recall": 0.76},
        "Early Fusion":      {"accuracy": 0.80, "macro_f1": 0.79, "macro_precision": 0.79, "macro_recall": 0.79},
        "Late Fusion":       {"accuracy": 0.81, "macro_f1": 0.80, "macro_precision": 0.80, "macro_recall": 0.80},
        "Attention Fusion":  {"accuracy": 0.83, "macro_f1": 0.82, "macro_precision": 0.82, "macro_recall": 0.82},
    }
    plot_model_comparison(fake_results, "outputs/figures/model_comparison.png")
    print("\nAll demo plots generated in outputs/figures/")
