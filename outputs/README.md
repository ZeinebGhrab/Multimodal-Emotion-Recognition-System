📈 outputs/ — Models, Results & Artifacts
==========================================

## Overview

The `outputs/` folder stores all artifacts produced by training runs.
It is the only folder that persists between sessions (not in `.gitignore`
except for `.pt` model weights and checkpoint directories).

```
outputs/
│
├── checkpoints/                    Per-run model weights and metrics
│   ├── cnn_20260521_120000/
│   │   ├── best_cnn.pt             Best CNN weights (saved by EarlyStopping)
│   │   └── test_metrics.json       Accuracy, F1, per-class scores
│   │
│   ├── bert_20260521_130000/
│   │   ├── best_bert.pt
│   │   └── test_metrics.json
│   │
│   ├── lstm_20260521_140000/
│   │   ├── best_lstm.pt
│   │   └── test_metrics.json
│   │
│   └── attention_20260521_150000/
│       ├── best_model.pt           Best fusion model weights
│       ├── test_metrics.json
│       └── sample_report.json      Demo GenAI report from train_multimodal.py
│
├── figures/                        PNG plots produced by training scripts
│   ├── cnn_<timestamp>_curves.png          Loss + accuracy over epochs
│   ├── cnn_<timestamp>_cm.png              Confusion matrix
│   ├── bert_<timestamp>_curves.png
│   ├── bert_<timestamp>_cm.png
│   ├── lstm_<timestamp>_curves.png
│   ├── lstm_<timestamp>_cm.png
│   └── screenshots/                        Terminal screenshots (manual)
│       ├── bert_training_terminal.png
│       ├── cnn_training_terminal.png
│       └── lstm_training_terminal.png
│
└── reports/                        Aggregated results and JSON emotion reports
    ├── model_comparison.png         Bar chart (compare_models.py)
    ├── comparison_summary.json      Best model, all metrics
    └── report_<emotion>.json        Individual emotion reports (genai module)
```

---

## checkpoints/

### Naming Convention

Each training run creates a timestamped subdirectory:

```
<model_type>_<YYYYMMDD>_<HHMMSS>/
```

Examples:
```
cnn_20260521_120000/
bert_20260521_130000/
lstm_20260521_140000/
early_20260521_150000/
late_20260521_150000/
attention_20260521_150000/
```

The `model_type` prefix is used by `compare_models.py` to infer which
model each run represents.

### best_*.pt

Saved automatically by `EarlyStopping` whenever validation loss improves.

| File              | Created by          | Contains                     | Size    |
|-------------------|---------------------|------------------------------|---------|
| `best_cnn.pt`     | `train_cnn.py`      | ResNet-50 `state_dict`       | ~90 MB  |
| `best_bert.pt`    | `train_bert.py`     | BERT `state_dict`            | ~420 MB |
| `best_lstm.pt`    | `train_lstm.py`     | BiLSTM `state_dict`          | ~12 MB  |
| `best_model.pt`   | `train_multimodal.py`| Fusion `state_dict`         | ~510 MB |

> `.pt` files are in `.gitignore` — they must be regenerated or
> downloaded separately.

### test_metrics.json

Written by each training script after final evaluation.
Loaded by `compare_models.py` to produce the comparison chart.

```json
{
  "accuracy": 0.6649,
  "macro_precision": 0.6850,
  "macro_recall": 0.6527,
  "macro_f1": 0.6104,
  "macro_f1_present": 0.6104,
  "zero_support_classes": [],
  "per_class": [
    {"emotion": "angry",    "precision": 0.60, "recall": 0.58, "f1": 0.59, "support": 958},
    {"emotion": "disgust",  "precision": 0.76, "recall": 0.26, "f1": 0.39, "support": 111},
    {"emotion": "fear",     "precision": 0.55, "recall": 0.39, "f1": 0.46, "support": 1024},
    {"emotion": "happy",    "precision": 0.87, "recall": 0.87, "f1": 0.87, "support": 1774},
    {"emotion": "neutral",  "precision": 0.59, "recall": 0.69, "f1": 0.64, "support": 1233},
    {"emotion": "sad",      "precision": 0.52, "recall": 0.59, "f1": 0.55, "support": 1247},
    {"emotion": "surprise", "precision": 0.76, "recall": 0.79, "f1": 0.78, "support": 831}
  ],
  "confusion_matrix": [[...], ...]
}
```

---

## figures/

All plots are saved at **150 DPI** with `bbox_inches="tight"`.

| File pattern                    | Content                           | Produced by          |
|---------------------------------|-----------------------------------|----------------------|
| `<run>_curves.png`              | Loss + accuracy over epochs        | All `train_*.py`     |
| `<run>_cm.png`                  | Normalized confusion matrix        | All `train_*.py`     |
| `per_class_heatmap.png`         | F1 heatmap across models           | `compare_models.py`  |
| `screenshots/*.png`             | Terminal output (manual captures)  | Manual               |

### Training Curves Plot

Two panels:
- Left: cross-entropy loss (train blue, val amber)
- Right: accuracy % (train green, val red)

Early stopping epoch is visible as the last data point.

### Confusion Matrix Plot

- Normalized by row (shows recall per class)
- Blue colormap (`cmap="Blues"`)
- Cell annotations with 2 decimal places
- Rows = true class, Columns = predicted class

---

## reports/

### model_comparison.png

Grouped bar chart produced by `compare_models.py`.
4 bars per model: Accuracy, Macro F1, Macro Precision, Macro Recall.
Value labels printed above each bar.

### comparison_summary.json

```json
{
  "models": [
    {
      "model": "Attention Fusion",
      "accuracy": 0.83,
      "macro_f1": 0.82,
      "precision": 0.83,
      "recall": 0.82
    },
    ...
  ],
  "best_accuracy": 0.83,
  "best_model": "Attention Fusion"
}
```

### report_<emotion>.json

Individual emotion reports generated by `src/genai/report_generator.py`.
See `GENAI.md` for the full schema.

---

## Loading Saved Models

### CNN

```python
from src.models.cnn_model import EmotionCNN

model = EmotionCNN(num_classes=7, pretrained=False)
model.load_state_dict(
    torch.load("outputs/checkpoints/cnn_20260521/best_cnn.pt",
               map_location="cpu")
)
model.eval()
```

### BERT

```python
from src.models.lstm_model import BERTClassifier

model = BERTClassifier(model_name="bert-base-uncased", num_classes=7)
model.load_state_dict(
    torch.load("outputs/checkpoints/bert_20260521/best_bert.pt",
               map_location="cpu")
)
model.eval()
```

### Fusion (via FastAPI env vars)

```bash
CNN_CHECKPOINT=outputs/checkpoints/cnn_20260521/best_cnn.pt
BERT_CHECKPOINT=outputs/checkpoints/bert_20260521/best_bert.pt
FUSION_CHECKPOINT=outputs/checkpoints/attention_20260521/best_model.pt
uvicorn api.app:app --port 8000
```

---

## File Size Overview

```
Large files (model weights):
├── best_bert.pt               ~420 MB
├── best_model.pt (fusion)     ~510 MB
├── best_cnn.pt                ~90 MB
├── best_lstm.pt               ~12 MB

Medium files:
├── test_metrics.json          ~5 KB per run
├── comparison_summary.json    ~10 KB

Figures:
├── *_curves.png               ~400 KB each
├── *_cm.png                   ~300 KB each
├── model_comparison.png       ~500 KB
```

---

## .gitignore for outputs/

```gitignore
outputs/checkpoints          # entire checkpoints directory (large .pt files)
*.pt                         # model weights anywhere
*.bin                        # HuggingFace weight files
```

Figures and JSON reports are **not** gitignored — they are small and
useful for documentation.

---

Last Updated: 23/05/2026
Status: Active ✓