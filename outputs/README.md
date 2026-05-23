# outputs/ — Models, Results & Artifacts

The `outputs/` folder stores all artifacts produced by training runs. Figures and JSON reports are committed; model weights (`.pt`) are not.

```
outputs/
│
├── checkpoints/                    Per-run model weights and metrics
│   ├── cnn_<YYYYMMDD_HHMMSS>/
│   │   ├── best_cnn.pt             Best CNN weights (EarlyStopping)
│   │   └── test_metrics.json
│   │
│   ├── bert_<YYYYMMDD_HHMMSS>/
│   │   ├── best_bert.pt
│   │   └── test_metrics.json
│   │
│   ├── lstm_<YYYYMMDD_HHMMSS>/
│   │   ├── best_lstm.pt
│   │   └── test_metrics.json
│   │
│   └── attention_<YYYYMMDD_HHMMSS>/
│       ├── best_model.pt           Best fusion weights
│       ├── test_metrics.json
│       └── sample_report.json      Demo GenAI emotion report
│
├── figures/
│   ├── cnn_<timestamp>_curves.png          Training curves (loss + accuracy)
│   ├── cnn_<timestamp>_cm.png              Confusion matrix
│   ├── bert_<timestamp>_curves.png
│   ├── bert_<timestamp>_cm.png
│   ├── lstm_<timestamp>_curves.png
│   ├── lstm_<timestamp>_cm.png
│   ├── attention_<timestamp>_curves.png
│   ├── attention_<timestamp>_cm.png
│   └── screenshots/                        Terminal training logs (manual)
│
└── reports/
    ├── model_comparison.png         Grouped bar chart (compare_models.py)
    ├── comparison_summary.json      Best model + all metrics
    └── report_<emotion>.json        Individual GenAI emotion reports
```

---

## checkpoints/

### Naming Convention

Each training run creates a timestamped sub-directory:

```
<model_type>_<YYYYMMDD>_<HHMMSS>/
```

`model_type` prefixes: `cnn`, `bert`, `lstm`, `early`, `late`, `attention`

### Model Weight Files

| File | Created by | Size |
|------|-----------|------|
| `best_cnn.pt` | `train_cnn.py` | ~90 MB |
| `best_bert.pt` | `train_bert.py` | ~420 MB |
| `best_lstm.pt` | `train_lstm.py` | ~12 MB |
| `best_model.pt` | `train_multimodal.py` | ~510 MB |

All `.pt` files are gitignored and must be regenerated or downloaded separately.

### test_metrics.json

Written after final evaluation. Loaded by `compare_models.py` to build the comparison chart.

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

| File pattern | Content | Produced by |
|-------------|---------|-------------|
| `<run>_curves.png` | Loss + accuracy over epochs | All `train_*.py` |
| `<run>_cm.png` | Normalised confusion matrix | All `train_*.py` |
| `screenshots/*.png` | Terminal output | Manual captures |

**Training curves:** Two panels — loss (train/val) and accuracy (train/val). Early stopping epoch is visible as the last data point.

**Confusion matrix:** Normalised by row (shows recall per class). Blue colormap, 2-decimal cell annotations.

---

## reports/

### model_comparison.png

Grouped bar chart from `compare_models.py` — 4 bars per model (Accuracy, Macro F1, Precision, Recall) with value labels above each bar.

### comparison_summary.json

```json
{
  "models": [
    {
      "model": "Attention Fusion",
      "accuracy": 0.977,
      "macro_f1": 0.9748,
      "precision": 0.977,
      "recall": 0.9748
    }
  ],
  "best_accuracy": 0.977,
  "best_model": "Attention Fusion"
}
```

---

## Loading Saved Models

### CNN

```python
from src.models.cnn_model import EmotionCNN

model = EmotionCNN(num_classes=7, pretrained=False)
model.load_state_dict(
    torch.load("outputs/checkpoints/cnn_20260521/best_cnn.pt", map_location="cpu")
)
model.eval()
```

### BERT

```python
from src.models.lstm_model import BERTClassifier

model = BERTClassifier(model_name="bert-base-uncased", num_classes=7)
model.load_state_dict(
    torch.load("outputs/checkpoints/bert_20260521/best_bert.pt", map_location="cpu")
)
model.eval()
```

### Fusion (via FastAPI environment variables)

```bash
CNN_CHECKPOINT=outputs/checkpoints/cnn_20260521/best_cnn.pt \
BERT_CHECKPOINT=outputs/checkpoints/bert_20260521/best_bert.pt \
FUSION_CHECKPOINT=outputs/checkpoints/attention_20260521/best_model.pt \
uvicorn api.app:app --port 8000
```

---

## .gitignore Rules

```gitignore
outputs/checkpoints    # large .pt model weight files
*.pt
*.bin                  # HuggingFace weight files
```

Figures (PNG) and JSON reports are **not** gitignored — they are small and useful for documentation.

---

*Last Updated: 23/05/2026 — Status: Active ✓*
