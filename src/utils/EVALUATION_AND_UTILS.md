📈 src/evaluation/ — Metrics & Visualization
=============================================

## Overview

The evaluation module provides a unified metric computation function and
four publication-quality visualization functions used by all training scripts.

```
src/evaluation/
├── __init__.py
└── metrics.py     compute_metrics() | plot_* functions
```

---

## metrics.py

### compute_metrics()

```python
from src.evaluation.metrics import compute_metrics

metrics = compute_metrics(y_true, y_pred, labels=EMOTION_LABELS)
```

**Returns a dictionary with:**

| Key                     | Type       | Description                                       |
|-------------------------|------------|---------------------------------------------------|
| `accuracy`              | float      | Overall accuracy                                  |
| `macro_precision`       | float      | Macro-averaged precision (all 7 classes)          |
| `macro_recall`          | float      | Macro-averaged recall (all 7 classes)             |
| `macro_f1`              | float      | Macro F1 including zero-support classes           |
| `macro_f1_present`      | float      | Macro F1 excluding zero-support classes           |
| `zero_support_classes`  | list[str]  | Classes with no test samples (e.g. `["disgust","neutral"]` for text models) |
| `per_class`             | list[dict] | Per-class precision, recall, F1, support          |
| `confusion_matrix`      | list[list] | Raw confusion matrix as 2-D list                  |
| `classification_report` | str        | sklearn text report                               |

**Key design decision — `macro_f1_present`:**

When text models trained on dair-ai/emotion are evaluated, `disgust` and
`neutral` have zero support. Including them in macro F1 gives an
artificially low score (64–65%) that misrepresents the model's actual
capability. `macro_f1_present` excludes zero-support classes and reflects
performance on the classes the model was actually trained on.

```python
# Example output for BERT
{
  "accuracy": 0.9575,
  "macro_f1": 0.6526,              # dragged down by disgust (0.00) + neutral (0.00)
  "macro_f1_present": 0.9136,      # honest metric: 5/7 classes
  "zero_support_classes": ["disgust", "neutral"]
}
```

**Bug fix applied — `labels=all_labels`:**
`precision_recall_fscore_support` is called with `labels=list(range(7))` to
force all 7 class indices to appear in the output, even if some have no
predicted samples. Without this, the function returns only as many rows as
there are predicted classes, causing an `IndexError` when `per_class` is
indexed to 7 positions.

---

### Visualization Functions

All plots use a consistent **dark theme** defined by `_PALETTE`:

```python
_PALETTE = {
    "bg":      "#0F1117",
    "surface": "#1C1F2A",
    "accent":  "#4F8EF7",   # blue  — train curves
    "success": "#4ADE80",   # green — accuracy
    "warning": "#FBBF24",   # amber — val curves
    "danger":  "#F87171",   # red   — poor F1
    "text":    "#E2E8F0",
    "muted":   "#94A3B8",
}
```

#### plot_confusion_matrix()

```python
plot_confusion_matrix(
    cm=metrics["confusion_matrix"],
    labels=emotion_names,
    title="CNN — Confusion Matrix",
    save_path="outputs/figures/cnn_cm.png",
    normalize=True,   # rows sum to 1 → shows recall per class
)
```

- Normalized by default (row sums = 1 → recall per class)
- Uses seaborn heatmap with `cmap="Blues"`
- Saved at 150 DPI

#### plot_training_curves()

```python
plot_training_curves(
    train_losses, val_losses,
    train_accs,   val_accs,
    model_name="BERT Classifier",
    save_path="outputs/figures/bert_curves.png",
)
```

Two-panel figure:
- Left: cross-entropy loss over epochs (train + val)
- Right: accuracy % over epochs (train + val)

#### plot_model_comparison()

```python
plot_model_comparison(
    results={"CNN": metrics_cnn, "BERT": metrics_bert, ...},
    save_path="outputs/figures/comparison.png",
)
```

Grouped bar chart: 4 bars per model (Accuracy, Macro F1, Precision, Recall).
Value labels printed above each bar.

#### plot_per_class_f1()

```python
plot_per_class_f1(
    metrics_dict=metrics,
    model_name="ResNet-50",
    labels=emotion_names,
    save_path="outputs/figures/cnn_per_class.png",
)
```

Horizontal bar chart. Color coding:
- 🟢 Green: F1 ≥ 0.70 (good)
- 🟡 Amber: F1 ≥ 0.50 (moderate)
- 🔴 Red: F1 < 0.50 (poor)

Reference dashed line drawn at F1 = 0.70.

---

## save_metrics()

```python
from src.evaluation.metrics import save_metrics
save_metrics(metrics, path="outputs/checkpoints/bert_run/test_metrics.json")
```

Saves all keys except `classification_report` (which is a long text string).
The JSON is loaded by `scripts/compare_models.py` to aggregate results.

---

---

🛑 src/utils/ — Training Utilities
====================================

## Overview

Shared utilities used across all training scripts.

```
src/utils/
├── __init__.py
└── early_stopping.py     EarlyStopping callback
```

---

## early_stopping.py — EarlyStopping

A generic, reusable early-stopping callback for any PyTorch training loop.

### Usage

```python
from src.utils.early_stopping import EarlyStopping

es = EarlyStopping(
    patience=3,
    min_delta=0.001,
    mode="min",          # "min" for loss, "max" for accuracy / F1
    restore_best=True,   # reload best checkpoint on stop
    verbose=True,
)

for epoch in range(epochs):
    val_loss, val_acc, _, _ = evaluate(...)

    if es(val_loss, model, "outputs/checkpoints/best.pt"):
        print(f"Early stopping at epoch {epoch}")
        break
# Best weights already loaded back into model
```

### Parameters

| Parameter      | Type    | Default | Description                                              |
|----------------|---------|---------|----------------------------------------------------------|
| `patience`     | int     | 7       | Epochs to wait after last improvement before stopping    |
| `min_delta`    | float   | 1e-3    | Minimum change to qualify as an improvement              |
| `mode`         | str     | `"min"` | `"min"` for loss, `"max"` for accuracy/F1               |
| `restore_best` | bool    | True    | Reload best weights into model when stopping             |
| `verbose`      | bool    | True    | Print improvement / counter messages                     |

### Checkpoint Saving

```
If checkpoint_path is provided:
  → save model.state_dict() to disk on each improvement
  → restore from disk when stopping

If checkpoint_path is None:
  → save state_dict() to in-memory BytesIO buffer
  → restore from buffer (useful for quick experiments, no disk I/O)
```

### Reset for Cross-Validation

```python
es.reset()    # resets counter, best_score, early_stop flag
# Use in k-fold loops between folds
```

### Measured Stopping Epochs

| Model          | Stopped at | Max allowed | Time saved |
|----------------|------------|-------------|------------|
| CNN (ResNet-50)| 15         | 30          | 50%        |
| BERT           | 5          | 10          | 50%        |
| BiLSTM + GloVe | 14         | 30          | 53%        |
| Fusion (typical) | 15–18    | 20          | ~15%       |

Early stopping consistently saves 40–55% of training time without any
accuracy penalty — best weights are always restored before evaluation.

### Console Output (verbose=True)

```
  [EarlyStopping] Improvement ↓  best=0.42150  Δ=—  (counter reset)
  [EarlyStopping] No improvement ↓ (1/3)  best=0.42150  current=0.43200
  [EarlyStopping] No improvement ↓ (2/3)  best=0.42150  current=0.44100
  [EarlyStopping] No improvement ↓ (3/3)  best=0.42150  current=0.45300
  → Training stops; best weights restored from outputs/checkpoints/best.pt
```

---

Last Updated: 23/05/2026
Status: Active ✓