# scripts/ — Training & Preprocessing Scripts

All runnable scripts for data preparation, model training, and evaluation. Each script reads `configs/config.yaml` at start-up and accepts CLI flags to override any value.

```
scripts/
├── preprocess_all.py      One-shot data preparation (both modalities)
├── train_cnn.py           Train ResNet-50 on FER2013
├── train_bert.py          Fine-tune BERT on dair-ai/emotion
├── train_lstm.py          Train BiLSTM + GloVe on dair-ai/emotion
├── train_multimodal.py    Train multimodal Attention Fusion model
└── compare_models.py      Compare all saved checkpoints
```

---

## Recommended Execution Order

```bash
# 1. Prepare all data
python scripts/preprocess_all.py

# 2. Train unimodal models
python scripts/train_cnn.py
python scripts/train_bert.py
python scripts/train_lstm.py        # optional — only for BiLSTM

# 3. Train fusion (pass the checkpoints from step 2)
python scripts/train_multimodal.py \
  --fusion attention \
  --cnn_checkpoint  outputs/checkpoints/cnn_<date>/best_cnn.pt \
  --bert_checkpoint outputs/checkpoints/bert_<date>/best_bert.pt \
  --no_finetune_encoders

# 4. Compare all models
python scripts/compare_models.py
```

---

## preprocess_all.py

One-shot script that prepares all data before training.

```bash
python scripts/preprocess_all.py               # Full pipeline
python scripts/preprocess_all.py --skip_glove  # Skip GloVe (BERT-only training)
python scripts/preprocess_all.py --data_dir /path/to/data
```

**What it does:**

```
Step 1 — Validate FER2013
  └─ Auto-detect format (folder-based or CSV)
  └─ Print class distribution per split
  └─ Assert pixel format (48×48 grayscale)

Step 2 — Preprocess dair-ai/emotion
  └─ Auto-download from HuggingFace (if not present)
  └─ Clean text (lowercase, strip URLs / mentions / special chars)
  └─ Map NLP labels → FER indices
  └─ Save emotion_train.csv / emotion_val.csv / emotion_test.csv

Step 3 — Download GloVe (optional)
  └─ Fetch glove.6B.zip from Stanford NLP (~822 MB unzipped)
  └─ Extract glove.6B.100d.txt
  └─ Remove the zip archive
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/config.yaml` | Config file path |
| `--data_dir` | `cfg.paths.data_raw` | Override raw data directory |
| `--skip_glove` | False | Skip GloVe download |
| `--val_ratio` | `0.1` | Fraction of training data for validation |

---

## train_cnn.py

Train a ResNet-50 classifier on FER2013 facial expression images.

```bash
python scripts/train_cnn.py                                         # Defaults from config
python scripts/train_cnn.py --epochs 40 --lr 5e-5 --patience 10   # Custom run
python scripts/train_cnn.py --batch_size 32 --backbone_lr_factor 0.05
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `cfg.cnn.epochs` | Number of training epochs |
| `--batch_size` | `cfg.cnn.batch_size` | Batch size |
| `--lr` | `cfg.cnn.learning_rate` | Learning rate for classifier head |
| `--weight_decay` | `cfg.cnn.weight_decay` | L2 regularisation |
| `--backbone_lr_factor` | `0.1` | LR multiplier for pretrained backbone |
| `--patience` | `cfg.training.early_stopping.patience` | Early stopping patience |
| `--val_ratio` | `0.1` | Validation split ratio |
| `--device` | `auto` | `auto / cuda / cpu / mps` |

**Outputs:**

```
outputs/checkpoints/cnn_<YYYYMMDD_HHMMSS>/
  ├── best_cnn.pt          Best weights (EarlyStopping)
  └── test_metrics.json    Accuracy, F1, per-class scores, confusion matrix

outputs/figures/
  ├── cnn_<timestamp>_curves.png
  └── cnn_<timestamp>_cm.png
```

**Measured results — FER2013 test set:**

| Metric | Value |
|--------|-------|
| Test accuracy | **66.49%** |
| Macro F1 | **61.04%** |
| Early stopped at | Epoch 15 / 30 |
| Trainable params | 24,560,711 |

---

## train_bert.py

Fine-tune `bert-base-uncased` on the dair-ai/emotion text dataset.

```bash
python scripts/train_bert.py
python scripts/train_bert.py --epochs 10 --lr 2e-5 --weight_decay 0.005

# Use pre-split CSV files
python scripts/train_bert.py \
  --train_csv data/raw/emotion_train.csv \
  --val_csv   data/raw/emotion_val.csv \
  --test_csv  data/raw/emotion_test.csv
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `cfg.bert.epochs` | Max training epochs |
| `--batch_size` | `cfg.bert.batch_size` | Batch size |
| `--lr` | `cfg.bert.learning_rate` | AdamW learning rate |
| `--weight_decay` | `cfg.bert.weight_decay` | L2 regularisation |
| `--patience` | `cfg.training.early_stopping.patience` | Early stopping patience |
| `--train_csv` | `data/raw/emotion_train.csv` | Training data CSV |
| `--val_csv` | `data/raw/emotion_val.csv` | Validation data CSV |
| `--test_csv` | `data/raw/emotion_test.csv` | Test data CSV |
| `--device` | `auto` | `auto / cuda / cpu / mps` |

**Outputs:**

```
outputs/checkpoints/bert_<YYYYMMDD_HHMMSS>/
  ├── best_bert.pt
  └── test_metrics.json
```

**Measured results — dair-ai/emotion test set:**

| Metric | Value |
|--------|-------|
| Test accuracy | **95.75%** |
| Macro F1 (5 present classes) | **91.36%** |
| Early stopped at | Epoch 5 / 10 |
| Trainable params | 109,680,903 |

> `disgust` and `neutral` have zero support in dair-ai/emotion — their F1 is 0.00. `macro_f1_present` (which excludes them) is the meaningful metric.

---

## train_lstm.py

Train a BiLSTM + GloVe classifier on the dair-ai/emotion text dataset.

```bash
python scripts/train_lstm.py
python scripts/train_lstm.py --no_glove          # Random embeddings (faster, lower accuracy)
python scripts/train_lstm.py --hidden_size 512   # Larger model
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `cfg.lstm.epochs` | Max training epochs |
| `--batch_size` | `cfg.lstm.batch_size` | Batch size |
| `--lr` | `cfg.lstm.learning_rate` | AdamW learning rate |
| `--weight_decay` | `cfg.lstm.weight_decay` | L2 regularisation |
| `--hidden_size` | `cfg.lstm.hidden_size` | LSTM hidden units per direction |
| `--patience` | `cfg.training.early_stopping.patience` | Early stopping patience |
| `--no_glove` | False | Skip GloVe; use random embeddings |
| `--device` | `auto` | `auto / cuda / cpu / mps` |

**Outputs:**

```
outputs/checkpoints/lstm_<YYYYMMDD_HHMMSS>/
  ├── best_lstm.pt
  └── test_metrics.json
```

**Measured results — dair-ai/emotion test set:**

| Metric | Value |
|--------|-------|
| Test accuracy | **95.50%** |
| Macro F1 (5 present classes) | **89.61%** |
| Early stopped at | Epoch 14 / 30 |
| Trainable params | 3,055,271 |
| GloVe vocabulary coverage | 7,346 / 7,400 (99.3%) |

---

## train_multimodal.py

Train the multimodal Attention Fusion model combining ResNet-50 + BERT. Optionally loads pre-trained unimodal checkpoints and freezes the encoders to train only the fusion head.

```bash
# Full fine-tuning (train everything)
python scripts/train_multimodal.py --fusion attention

# Recommended: freeze pretrained encoders, train fusion head only (faster)
python scripts/train_multimodal.py \
  --fusion attention \
  --cnn_checkpoint  outputs/checkpoints/cnn_20260522_171658/best_cnn.pt \
  --bert_checkpoint outputs/checkpoints/bert_20260522_165038/best_bert.pt \
  --no_finetune_encoders \
  --epochs 30

# Try other fusion strategies
python scripts/train_multimodal.py --fusion early
python scripts/train_multimodal.py --fusion late
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--fusion` | `attention` | Fusion strategy: `early / late / attention` |
| `--epochs` | `cfg.fusion.epochs` | Max training epochs |
| `--batch_size` | `cfg.fusion.batch_size` | Batch size |
| `--lr` | `cfg.fusion.learning_rate` | Learning rate |
| `--encoder_lr_factor` | `0.1` | LR multiplier for encoder parameters |
| `--weight_decay_enc` | `cfg.fusion.weight_decay_encoders` | L2 for encoder params |
| `--weight_decay_fusion` | `cfg.fusion.weight_decay_fusion` | L2 for fusion head |
| `--patience` | `cfg.training.early_stopping.patience` | Early stopping patience |
| `--cnn_checkpoint` | None | Path to pre-trained CNN weights |
| `--bert_checkpoint` | None | Path to pre-trained BERT weights |
| `--no_finetune_encoders` | False | Freeze encoders (requires checkpoints) |
| `--device` | `auto` | `auto / cuda / cpu / mps` |

**Data split:**

```
FER2013:
  train/ → 90% training + 10% validation (carved out automatically)
  test/  → held-out evaluation set

Text:
  emotion_train.csv → training pairs
  emotion_val.csv   → validation pairs
  emotion_test.csv  → test pairs

Note: disgust and neutral have no NLP counterpart in dair-ai/emotion.
      Placeholder texts are used so all 7 FER classes are represented.
```

**Outputs:**

```
outputs/checkpoints/<fusion>_<YYYYMMDD_HHMMSS>/
  ├── best_model.pt         Best fusion weights
  ├── test_metrics.json     Final evaluation metrics
  └── sample_report.json    Demo GenAI emotion report

outputs/figures/
  ├── <run>_curves.png
  └── <run>_cm.png
```

---

## compare_models.py

Loads all `test_metrics.json` files from `outputs/checkpoints/` and produces a grouped bar chart.

```bash
python scripts/compare_models.py
python scripts/compare_models.py --checkpoint_dir outputs/checkpoints
python scripts/compare_models.py --output outputs/reports/my_comparison.png
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint_dir` | `outputs/checkpoints` | Directory with run folders |
| `--output` | `outputs/reports/comparison.png` | Output chart path |
| `--config` | `configs/config.yaml` | Config file path |

**Outputs:**

```
outputs/reports/
  ├── model_comparison.png       Grouped bar chart (4 metrics × all models)
  └── comparison_summary.json    Best model name + full metrics table
```

---

*Last Updated: 23/05/2026 — Status: Active ✓*
