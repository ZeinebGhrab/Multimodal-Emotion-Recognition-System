📜 scripts/ — Training & Utility Scripts
==========================================

## Overview

The `scripts/` folder contains six standalone Python scripts that
cover the full ML pipeline from data preparation to model comparison.
Each script reads from `configs/config.yaml` and accepts CLI flags to
override any parameter without editing the config file.

```
scripts/
├── preprocess_all.py      Data preparation (FER2013 + NLP + GloVe)
├── train_cnn.py           ResNet-50 training
├── train_bert.py          BERT text classifier training
├── train_lstm.py          BiLSTM + GloVe training
├── train_multimodal.py    Fusion model training
└── compare_models.py      Aggregate results + comparison chart
```

---

## preprocess_all.py

### Purpose

One-shot data preparation for both modalities.

### What it does

```
Step 1 — FER2013 validation
  └─ Auto-detect format (folder vs CSV)
  └─ Print class distribution and pixel format check

Step 2 — Emotion NLP dataset
  └─ Auto-download dair-ai/emotion from HuggingFace if not present
  └─ Clean text, unify labels, create train/val/test CSV splits
  └─ Save emotion_train/val/test.csv to data/raw/

Step 3 — GloVe download (optional)
  └─ Download glove.6B.zip (~822 MB) from Stanford NLP
  └─ Extract glove.6B.100d.txt
```

### Usage

```bash
python scripts/preprocess_all.py                     # full pipeline
python scripts/preprocess_all.py --skip_glove        # skip GloVe download
python scripts/preprocess_all.py --data_dir /path    # custom data directory
```

### CLI Arguments

| Flag           | Default         | Description                              |
|----------------|-----------------|------------------------------------------|
| `--config`     | `configs/config.yaml` | Config file path                   |
| `--data_dir`   | from config     | Override `paths.data_raw`               |
| `--skip_glove` | False           | Skip GloVe download (BERT-only training) |
| `--val_ratio`  | 0.1             | Fraction of NLP train set used for val  |

---

## train_cnn.py

### Purpose

Fine-tune ResNet-50 on FER2013 for facial expression recognition.

### Training Loop Summary

```
Config → FER2013 DataLoaders → EmotionCNN → AdamW (two-group) →
CosineAnnealing + warmup → EarlyStopping (patience=3) →
Best checkpoint → Test evaluation → Confusion matrix + curves
```

### Usage

```bash
python scripts/train_cnn.py                          # default config
python scripts/train_cnn.py --epochs 40 --lr 5e-5   # override epochs/LR
python scripts/train_cnn.py --amp                    # AMP (GPU only)
python scripts/train_cnn.py --no_pretrain            # train from scratch
python scripts/train_cnn.py --freeze_bn              # freeze BatchNorm
```

### CLI Arguments

| Flag             | Default    | Description                                     |
|------------------|------------|-------------------------------------------------|
| `--epochs`       | 30         | Max training epochs                             |
| `--lr`           | 1e-4       | Head learning rate                              |
| `--batch_size`   | 64         | Batch size                                      |
| `--dropout`      | 0.5        | Dropout in classifier head                      |
| `--weight_decay` | 1e-4       | L2 regularisation (both param groups)           |
| `--patience`     | 3          | Early stopping patience                         |
| `--amp`          | False      | Automatic mixed precision (GPU only)            |
| `--freeze_bn`    | False      | Freeze BatchNorm layers                         |
| `--no_pretrain`  | False      | Random init (no ImageNet weights)               |
| `--device`       | auto       | `cuda` / `cpu` / `mps`                         |

### Output Artifacts

```
outputs/checkpoints/cnn_<timestamp>/
├── best_cnn.pt            Best model weights
└── test_metrics.json      Accuracy, Macro F1, per-class scores

outputs/figures/
├── cnn_<timestamp>_curves.png
└── cnn_<timestamp>_cm.png
```

### Measured Results

```
Test Accuracy : 66.49%
Macro F1      : 61.04%
Best epoch    : 15 / 30  (early stopping saved 50% training time)
```

---

## train_bert.py

### Purpose

Fine-tune `bert-base-uncased` on dair-ai/emotion for text emotion classification.

### Training Loop Summary

```
Config → BERTEmotionDataset → BERTClassifier → AdamW (no-decay groups) →
LinearSchedule + warmup → EarlyStopping → Best checkpoint → Evaluation
```

### Usage

```bash
python scripts/train_bert.py
python scripts/train_bert.py --epochs 10 --lr 2e-5 --batch_size 32
python scripts/train_bert.py --model_name distilbert-base-uncased   # lighter model
python scripts/train_bert.py --weight_decay 0.005 --patience 4
```

### CLI Arguments

| Flag             | Default              | Description                             |
|------------------|----------------------|-----------------------------------------|
| `--model_name`   | bert-base-uncased    | HuggingFace model identifier            |
| `--epochs`       | 10                   | Max training epochs                     |
| `--lr`           | 2e-5                 | Learning rate                           |
| `--batch_size`   | 32                   | Batch size                              |
| `--max_length`   | 128                  | Token sequence length                   |
| `--warmup_steps` | 500                  | Linear warmup steps                     |
| `--weight_decay` | 0.01                 | L2 penalty (BERT layers)                |
| `--patience`     | 3                    | Early stopping patience                 |
| `--device`       | auto                 | `cuda` / `cpu` / `mps`                 |

### Measured Results

```
Test Accuracy   : 95.75%
Macro F1 (all 7): 65.26%  ← includes disgust (0) and neutral (0)
Macro F1 (5 present): 91.36%
Best epoch      : 5 / 10
Trainable params: 109,680,903
```

---

## train_lstm.py

### Purpose

Train a BiLSTM + GloVe classifier on dair-ai/emotion.
Faster and lighter than BERT — good for resource-constrained environments.

### Training Loop Summary

```
Config → Vocabulary → GloVe matrix → BiLSTMClassifier →
AdamW → CosineAnnealing → EarlyStopping → Evaluation
```

### Usage

```bash
python scripts/train_lstm.py
python scripts/train_lstm.py --epochs 30 --lr 1e-3 --batch_size 128
python scripts/train_lstm.py --no_glove       # random embeddings
python scripts/train_lstm.py --freeze_emb     # freeze GloVe weights
```

### CLI Arguments

| Flag             | Default | Description                                |
|------------------|---------|--------------------------------------------|
| `--epochs`       | 30      | Max training epochs                        |
| `--lr`           | 1e-3    | Learning rate                              |
| `--batch_size`   | 128     | Batch size                                 |
| `--weight_decay` | 1e-4    | L2 regularisation                          |
| `--patience`     | 3       | Early stopping patience                    |
| `--no_glove`     | False   | Skip GloVe loading — use random embeddings |
| `--freeze_emb`   | False   | Freeze embedding layer                     |

### Measured Results

```
Test Accuracy        : 95.50%
Macro F1 (5 present) : 89.61%
Best epoch           : 14 / 30
Trainable params     : 3,055,271
GloVe coverage       : 7,346 / 7,400 (99.3%)
```

---

## train_multimodal.py

### Purpose

Train the multimodal fusion model (Early / Late / Attention) combining
a pre-trained CNN encoder and a pre-trained BERT encoder.

### Training Loop Summary

```
Load CNN checkpoint → Load BERT checkpoint → MultimodalEmotionModel →
AdamW (two weight-decay groups) → CosineAnnealing → EarlyStopping →
Evaluation → GenAI demo report
```

> **Note:** Uses `DummyMultimodalDataset` by default (random tensors).
> Replace with a real aligned image+text dataset for actual training.
> Suitable real datasets: CMU-MOSI, CMU-MOSEI, or FER2013 images
> paired with BLIP-2 / LLaVA-generated captions.

### Usage

```bash
# Train with Attention Fusion using pre-trained encoders
python scripts/train_multimodal.py \
    --fusion attention \
    --cnn_checkpoint  outputs/checkpoints/cnn_20260521/best_cnn.pt \
    --bert_checkpoint outputs/checkpoints/bert_20260521/best_bert.pt

# Freeze encoders, train only fusion head (faster)
python scripts/train_multimodal.py \
    --fusion attention \
    --no_finetune_encoders \
    --epochs 30

# Early or Late fusion
python scripts/train_multimodal.py --fusion early
python scripts/train_multimodal.py --fusion late
```

### CLI Arguments

| Flag                    | Default      | Description                                   |
|-------------------------|--------------|-----------------------------------------------|
| `--fusion`              | `attention`  | Fusion strategy: `early` / `late` / `attention` |
| `--epochs`              | 20           | Max epochs                                    |
| `--batch_size`          | 32           | Batch size                                    |
| `--lr`                  | 1e-4         | Fusion head learning rate                     |
| `--encoder_lr_factor`   | 0.1          | LR multiplier for pretrained encoders         |
| `--weight_decay_enc`    | 1e-4         | L2 for encoders                               |
| `--weight_decay_fusion` | 1e-3         | L2 for fusion head                            |
| `--patience`            | 3            | Early stopping patience                       |
| `--no_finetune_encoders`| False        | Freeze encoders — train fusion head only      |
| `--cnn_checkpoint`      | None         | Path to trained CNN weights                   |
| `--bert_checkpoint`     | None         | Path to trained BERT weights                  |

---

## compare_models.py

### Purpose

Aggregate test metrics from all training runs and produce a comparison
bar chart and summary JSON.

### How it Works

```
1. Walk outputs/checkpoints/**/test_metrics.json
2. Infer model type from run directory prefix (cnn_, bert_, lstm_, attention_, ...)
3. Print comparison table to stdout
4. Plot grouped bar chart (Accuracy + Macro F1 per model)
5. Save outputs/reports/model_comparison.png
6. Save outputs/reports/comparison_summary.json
```

If no checkpoints are found, uses the expected figures from the README
for illustration.

### Usage

```bash
python scripts/compare_models.py
python scripts/compare_models.py --output_dir outputs/comparison
```

### Output

```
Comparison table (stdout):
──────────────────────────────────────────────────────────────
Model                   Accuracy   Macro F1  Precision   Recall
──────────────────────────────────────────────────────────────
CNN (ResNet-50)          65.00%     63.00%    64.00%     63.00%
ViT-B/16                 68.00%     67.00%    68.00%     67.00%
BiLSTM + GloVe           70.00%     69.00%    70.00%     69.00%
BERT                     78.00%     77.00%    78.00%     77.00%
Early Fusion             80.00%     79.00%    80.00%     79.00%
Late Fusion              81.00%     80.00%    81.00%     80.00%
Attention Fusion         83.00%     82.00%    83.00%     82.00%
──────────────────────────────────────────────────────────────
🏆  Best model: Attention Fusion  (Accuracy: 83.00%,  F1: 82.00%)

Files saved:
outputs/reports/model_comparison.png
outputs/reports/comparison_summary.json
```

---

## Common Patterns Across Scripts

All training scripts share the same structure:

```python
# 1. Parse args (CLI overrides)
args = parse_args()

# 2. Load config
with open(args.config) as f:
    cfg = yaml.safe_load(f)

# 3. Resolve final values (CLI flag > config value)
epochs = args.epochs or cfg["cnn"]["epochs"]

# 4. Create output directory with timestamp
run_id  = f"cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
out_dir = Path(cfg["paths"]["checkpoints"]) / run_id

# 5. Data → Model → Loss → Optimizer → Scheduler → EarlyStopping

# 6. Training loop with EarlyStopping

# 7. Evaluate on test set → save metrics JSON

# 8. Plot confusion matrix + training curves
```

---

Last Updated: 23/05/2026
Status: Active ✓