# 📜 scripts/ — Training & Preprocessing Scripts

## Overview

All runnable scripts for data preparation, model training, and comparison.
Each script is self-contained: it reads `configs/config.yaml`, accepts CLI
flags to override any value, and saves outputs to `outputs/`.

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

## preprocess_all.py

One-shot script that prepares all data for training.

```bash
# Full pipeline (FER2013 must be downloaded manually first)
python scripts/preprocess_all.py

# Skip GloVe download (not needed for BERT-only training)
python scripts/preprocess_all.py --skip_glove

# Override raw data directory
python scripts/preprocess_all.py --data_dir /path/to/data
```

### What it does

```
Step 1 — Validate FER2013
  └─ Auto-detect format (folder-based or CSV)
  └─ Print class distribution per split
  └─ Assert pixel format (48×48 grayscale)

Step 2 — Preprocess Emotion NLP dataset
  └─ Auto-download dair-ai/emotion from HuggingFace (if not present)
  └─ Clean text (lowercase, strip URLs / mentions / special chars)
  └─ Map NLP labels → FER indices
  └─ Save emotion_train.csv / emotion_val.csv / emotion_test.csv

Step 3 — Download GloVe embeddings (optional)
  └─ Fetch glove.6B.zip from Stanford NLP (~822 MB)
  └─ Extract glove.6B.100d.txt
  └─ Remove the zip archive
```

### CLI flags

| Flag            | Default                     | Description                              |
|-----------------|-----------------------------|------------------------------------------|
| `--config`      | `configs/config.yaml`       | Path to config file                      |
| `--data_dir`    | `cfg["paths"]["data_raw"]`  | Override raw data directory              |
| `--skip_glove`  | False                       | Skip GloVe download                      |
| `--val_ratio`   | `0.1`                       | Fraction of training data for validation |

---

## train_cnn.py

Train a ResNet-50 classifier on FER2013 facial expression images.

```bash
# Default training (reads all settings from config.yaml)
python scripts/train_cnn.py

# Custom hyperparameters
python scripts/train_cnn.py --epochs 40 --lr 5e-5 --weight_decay 0.0005 --patience 5

# Override batch size and backbone learning rate factor
python scripts/train_cnn.py --batch_size 32 --backbone_lr_factor 0.05
```

### CLI flags

| Flag                  | Default                                | Description                          |
|-----------------------|----------------------------------------|--------------------------------------|
| `--epochs`            | `cfg.cnn.epochs`                       | Number of training epochs            |
| `--batch_size`        | `cfg.cnn.batch_size`                   | Batch size                           |
| `--lr`                | `cfg.cnn.learning_rate`                | Learning rate for classifier head    |
| `--weight_decay`      | `cfg.cnn.weight_decay`                 | L2 regularisation                    |
| `--backbone_lr_factor`| `0.1`                                  | LR multiplier for pretrained backbone|
| `--patience`          | `cfg.training.early_stopping.patience` | Early stopping patience              |
| `--val_ratio`         | `0.1`                                  | Validation split ratio               |
| `--config`            | `configs/config.yaml`                  | Config file path                     |
| `--device`            | `auto`                                 | `auto / cuda / cpu / mps`            |

### Outputs

```
outputs/checkpoints/cnn_<YYYYMMDD_HHMMSS>/
  ├── best_cnn.pt          Best weights (saved by EarlyStopping)
  └── test_metrics.json    Accuracy, F1, per-class scores, confusion matrix

outputs/figures/
  ├── cnn_<timestamp>_curves.png   Loss + accuracy training curves
  └── cnn_<timestamp>_cm.png       Confusion matrix (normalised)
```

### Measured results (FER2013 test set)

| Metric           | Value       |
|------------------|-------------|
| Test accuracy    | **66.49%**  |
| Macro F1         | **61.04%**  |
| Early stopped at | Epoch 15/30 |
| Trainable params | 24,560,711  |

---

## train_bert.py

Fine-tune `bert-base-uncased` on the dair-ai/emotion text dataset.

```bash
# Default training
python scripts/train_bert.py

# Custom hyperparameters
python scripts/train_bert.py --epochs 10 --lr 2e-5 --weight_decay 0.005 --patience 5

# Use pre-split CSV files
python scripts/train_bert.py \
  --train_csv data/raw/emotion_train.csv \
  --val_csv   data/raw/emotion_val.csv \
  --test_csv  data/raw/emotion_test.csv
```

### CLI flags

| Flag             | Default                                | Description                |
|------------------|----------------------------------------|----------------------------|
| `--epochs`       | `cfg.bert.epochs`                      | Max training epochs        |
| `--batch_size`   | `cfg.bert.batch_size`                  | Batch size                 |
| `--lr`           | `cfg.bert.learning_rate`               | AdamW learning rate        |
| `--weight_decay` | `cfg.bert.weight_decay`                | L2 regularisation          |
| `--patience`     | `cfg.training.early_stopping.patience` | Early stopping patience    |
| `--train_csv`    | `data/raw/emotion_train.csv`           | Training data CSV          |
| `--val_csv`      | `data/raw/emotion_val.csv`             | Validation data CSV        |
| `--test_csv`     | `data/raw/emotion_test.csv`            | Test data CSV              |
| `--config`       | `configs/config.yaml`                  | Config file path           |
| `--device`       | `auto`                                 | `auto / cuda / cpu / mps`  |

### Outputs

```
outputs/checkpoints/bert_<YYYYMMDD_HHMMSS>/
  ├── best_bert.pt
  └── test_metrics.json
```

### Measured results (dair-ai/emotion test set)

| Metric                       | Value      |
|------------------------------|------------|
| Test accuracy                | **95.75%** |
| Macro F1 (5 present classes) | **91.36%** |
| Early stopped at             | Epoch 5/10 |
| Trainable params             | 109,680,903|

> `disgust` and `neutral` have zero support in dair-ai/emotion — their F1 is 0.00.
> The `macro_f1_present` metric excludes them and is the meaningful comparison metric.

---

## train_lstm.py

Train a BiLSTM + GloVe classifier on the dair-ai/emotion text dataset.

```bash
# Default training
python scripts/train_lstm.py

# Without GloVe (uses random embeddings)
python scripts/train_lstm.py --no_glove

# Custom hyperparameters
python scripts/train_lstm.py --epochs 30 --lr 0.001 --hidden_size 512
```

### CLI flags

| Flag             | Default                                | Description                         |
|------------------|----------------------------------------|-------------------------------------|
| `--epochs`       | `cfg.lstm.epochs`                      | Max training epochs                 |
| `--batch_size`   | `cfg.lstm.batch_size`                  | Batch size                          |
| `--lr`           | `cfg.lstm.learning_rate`               | AdamW learning rate                 |
| `--weight_decay` | `cfg.lstm.weight_decay`                | L2 regularisation                   |
| `--hidden_size`  | `cfg.lstm.hidden_size`                 | LSTM hidden units per direction     |
| `--patience`     | `cfg.training.early_stopping.patience` | Early stopping patience             |
| `--no_glove`     | False                                  | Skip GloVe; use random embeddings   |
| `--config`       | `configs/config.yaml`                  | Config file path                    |
| `--device`       | `auto`                                 | `auto / cuda / cpu / mps`           |

### Outputs

```
outputs/checkpoints/lstm_<YYYYMMDD_HHMMSS>/
  ├── best_lstm.pt
  └── test_metrics.json
```

### Measured results (dair-ai/emotion test set)

| Metric                       | Value                 |
|------------------------------|-----------------------|
| Test accuracy                | **95.50%**            |
| Macro F1 (5 present classes) | **89.61%**            |
| Early stopped at             | Epoch 14/30           |
| Trainable params             | 3,055,271             |
| GloVe vocabulary coverage    | 7,346 / 7,400 (99.3%) |

---

## train_multimodal.py

Train the multimodal Attention Fusion model combining ResNet-50 + BERT.
Optionally loads pre-trained unimodal checkpoints and freezes the encoders.

```bash
# Full fine-tuning (train everything from scratch)
python scripts/train_multimodal.py --fusion attention

# Freeze encoders — train fusion head only (fastest)
python scripts/train_multimodal.py \
  --fusion attention \
  --cnn_checkpoint  outputs/checkpoints/cnn_20260522_171658/best_cnn.pt \
  --bert_checkpoint outputs/checkpoints/bert_20260522_165038/best_bert.pt \
  --no_finetune_encoders \
  --epochs 30

# Try a different fusion strategy
python scripts/train_multimodal.py --fusion early
python scripts/train_multimodal.py --fusion late
```

### CLI flags

| Flag                     | Default                              | Description                           |
|--------------------------|--------------------------------------|---------------------------------------|
| `--fusion`               |`attention`                           |Fusion strategy: `early/late/attention`|
| `--epochs`               |`cfg.fusion.epochs`                   | Max training epochs                   |
| `--batch_size`           |`cfg.fusion.batch_size`               | Batch size                            |
| `--lr`                   |`cfg.fusion.learning_rate`            | Learning rate                         |
| `--encoder_lr_factor`    |`0.1`                                 | LR multiplier for encoder parameters  |
| `--weight_decay_enc`     |`cfg.fusion.weight_decay_encoders`    | L2 for encoder params                 |
| `--weight_decay_fusion`  |`cfg.fusion.weight_decay_fusion`      | L2 for fusion head params             |
| `--patience`             |`cfg.training.early_stopping.patience`| Early stopping patience               |
| `--val_ratio`            |`0.1`                                 |Fraction of FER2013/train used for val |
| `--cnn_checkpoint`       |None                                  | Path to pre-trained CNN weights       |
| `--bert_checkpoint`      |None                                  | Path to pre-trained BERT weights      |
| `--no_finetune_encoders` |False                                 | Freeze encoders (requires checkpoints)|
| `--config`               |`configs/config.yaml`                 | Config file path                      |
| `--device`               |`auto`                                | `auto / cuda / cpu / mps`             |

### Data split strategy

```
FER2013 images:
  fer2013/train/  → 90% → training
  fer2013/train/  → 10% → validation   (carved out, never seen in train)
  fer2013/test/   → 100% → test        (held out, evaluated once at end)

Text data:
  emotion_train.csv → training pairs
  emotion_val.csv   → validation pairs
  emotion_test.csv  → test pairs

Note: disgust and neutral have no NLP data → default placeholder texts
      are used ("I feel disgusted", "feeling calm and neutral", etc.)
      so all 7 emotion classes are represented in training.
```

### Outputs

```
outputs/checkpoints/<fusion>_<YYYYMMDD_HHMMSS>/
  ├── best_model.pt         Best fusion weights
  ├── test_metrics.json     Final evaluation metrics
  └── sample_report.json    Demo GenAI report

outputs/figures/
  ├── <run>_curves.png
  └── <run>_cm.png
```

---

## compare_models.py

Load all `test_metrics.json` files from `outputs/checkpoints/` and
produce a grouped bar chart comparing all models.

```bash
python scripts/compare_models.py

# Specify checkpoint directory
python scripts/compare_models.py --checkpoint_dir outputs/checkpoints

# Save comparison to custom path
python scripts/compare_models.py --output outputs/reports/my_comparison.png
```

### CLI flags

| Flag               | Default                          | Description                     |
|--------------------|----------------------------------|---------------------------------|
| `--checkpoint_dir` | `outputs/checkpoints`            | Directory containing run folders|
| `--output`         | `outputs/reports/comparison.png` | Output chart path               |
| `--config`         | `configs/config.yaml`            | Config file path                |

### Output

```
outputs/reports/
  ├── model_comparison.png         Grouped bar chart (4 metrics per model)
  └── comparison_summary.json      Best model name + all metrics table
```

---

## Running All Scripts in Order

```bash
# 1. Prepare data
python scripts/preprocess_all.py

# 2. Train unimodal models
python scripts/train_cnn.py
python scripts/train_bert.py
python scripts/train_lstm.py    # optional — only needed if using BiLSTM

# 3. Train fusion model (use the checkpoints from step 2)
python scripts/train_multimodal.py \
  --fusion attention \
  --cnn_checkpoint  outputs/checkpoints/cnn_<date>/best_cnn.pt \
  --bert_checkpoint outputs/checkpoints/bert_<date>/best_bert.pt \
  --no_finetune_encoders

# 4. Compare all models
python scripts/compare_models.py
```

---

Last Updated: 23/05/2026<br>
Status: Active ✓
