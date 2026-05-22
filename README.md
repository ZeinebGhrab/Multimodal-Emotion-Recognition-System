# Multimodal Emotion Recognition System

> **Deep Learning + Generative AI — Academic Project**
> Detect human emotions from facial images (FER2013) and text (tweets), fuse both modalities, and generate an AI-powered emotional report via the Claude API.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Datasets](#3-datasets)
4. [Models & Architecture](#4-models--architecture)
   - [Image Models](#41-image-models)
   - [Text Models](#42-text-models)
   - [Fusion Strategies](#43-fusion-strategies)
5. [Regularisation](#5-regularisation)
   - [Weight Decay (L2)](#51-weight-decay-l2)
   - [Early Stopping](#52-early-stopping)
6. [GenAI Module](#6-genai-module)
7. [FastAPI Inference Server](#7-fastapi-inference-server)
8. [Quick Start](#8-quick-start)
9. [Configuration Reference](#9-configuration-reference)
10. [Environment Variables](#10-environment-variables)
11. [Tech Stack](#11-tech-stack)
12. [Expected Results](#12-expected-results)
13. [Authors](#13-authors)

---

## 1. Project Overview

This system recognises **7 discrete emotions** — `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise` — from two complementary input modalities:

| Modality | Input | Model |
|----------|-------|-------|
| **Vision** | 48×48 grayscale face image (FER2013) | ResNet-50 · ViT-B/16 |
| **Language** | Short emotion-bearing text (tweet/caption) | BiLSTM+GloVe · BERT |
| **Fusion** | Image + Text jointly | Early · Late · Attention |

The best-performing configuration (**Attention Fusion**) reaches ~83% accuracy on the FER2013 test split, a +18 pp improvement over the CNN baseline alone.

A **GenAI module** (`src/genai/report_generator.py`) takes the predicted emotion and probability scores and calls the **Claude API** to produce a personalised, clinically-informed emotional report in JSON format.

---

## 2. Project Structure

```
multimodal_emotion_recognition/
│
├── configs/
│   └── config.yaml                  # All hyperparameters, paths, regularisation
│
├── data/
│   ├── raw/                         # Original datasets (not committed)
│   └── processed/                   # Preprocessed tensors / CSVs
│
├── src/
│   ├── preprocessing/
│   │   ├── image_preprocessing.py   # FER2013 Dataset + augmentation transforms
│   │   └── text_preprocessing.py    # BERT/LSTM tokenizers, Vocabulary, GloVe loader
│   │
│   ├── models/
│   │   ├── cnn_model.py             # ResNet-50 + classifier head
│   │   ├── vit_model.py             # Vision Transformer ViT-B/16
│   │   └── lstm_model.py            # BiLSTM + GloVe  |  BERT classifier
│   │
│   ├── fusion/
│   │   └── fusion_models.py         # EarlyFusion | LateFusion | AttentionFusion
│   │
│   ├── evaluation/
│   │   └── metrics.py               # Accuracy, F1, confusion matrix, training curves
│   │
│   ├── utils/
│   │   └── early_stopping.py        # Reusable EarlyStopping callback
│   │
│   └── genai/
│       └── report_generator.py      # Claude-powered emotion reports
│
├── api/
│   └── app.py                       # FastAPI inference server (3 endpoints)
│
├── scripts/
│   ├── preprocess_all.py            # One-shot data preparation
│   ├── train_cnn.py                 # Train ResNet-50
│   ├── train_lstm.py                # Train BiLSTM + GloVe
│   ├── train_bert.py                # Train BERT classifier
│   ├── train_multimodal.py          # Train multimodal fusion model
│   └── compare_models.py            # Compare all saved checkpoints
│
├── notebooks/
│   └── full_pipeline.ipynb          # End-to-end Jupyter walkthrough
│
├── outputs/
│   ├── checkpoints/                 # Saved model weights (.pt)
│   ├── figures/                     # Plots, confusion matrices, training curves
│   └── reports/                     # Generated JSON emotion reports
│
└── requirements.txt
```

---

## 3. Datasets

| Dataset | Task | Classes | Size | Source |
|---------|------|---------|------|--------|
| **FER2013** | Facial expression recognition | 7 | ~35 000 images | [Kaggle — msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013) |
| **dair-ai/emotion** | Text emotion classification | 6 → 7 (remapped) | ~20 000 sentences | [HuggingFace](https://huggingface.co/datasets/dair-ai/emotion) |
| **GloVe 6B** | Word embeddings (100d) | — | 400 000 tokens | [Stanford NLP](https://nlp.stanford.edu/data/glove.6B.zip) |

### FER2013 — Supported Formats

The preprocessing pipeline auto-detects two layouts:

```
# Format A — Folder-based (most common on Kaggle)
data/raw/fer2013/
  train/  angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
  test/   angry/ ...

# Format B — Original CSV
data/raw/fer2013.csv   (columns: emotion, pixels, Usage)
```

### Label Mapping — NLP → FER

Since `dair-ai/emotion` uses 6 labels and FER2013 uses 7, the following mapping is applied:

| NLP label | → FER label |
|-----------|-------------|
| `joy` | `happy` |
| `sadness` | `sad` |
| `anger` | `angry` |
| `fear` | `fear` |
| `surprise` | `surprise` |
| `love` | `happy` *(closest positive)* |

---

## 4. Models & Architecture

### 4.1 Image Models

#### ResNet-50 (`src/models/cnn_model.py`)

```
Input (B, 3, 224, 224)
  → ResNet-50 backbone (ImageNet pretrained, layers frozen at 0.1× LR)
  → GlobalAvgPool → (B, 2048)
  → Dropout(0.5) → Linear(2048→512) → ReLU
  → Dropout(0.25) → Linear(512→7)
  → logits (B, 7)
```

**Design choices:**
- Transfer learning from ImageNet — facial textures share low-level features with natural images.
- Two-group AdamW: backbone at `lr × 0.1`, head at `lr` to preserve pretrained representations.
- `freeze_bn` option freezes BatchNorm stats during fine-tuning (useful with small batch sizes).

#### Vision Transformer — ViT-B/16 (`src/models/vit_model.py`)

```
Input (B, 3, 224, 224)
  → Patch embedding: 196 patches of 16×16
  → 12 Transformer encoder layers (self-attention)
  → [CLS] token → (B, 768)
  → Dropout(0.1) → Linear(768→256) → GELU
  → Dropout(0.05) → Linear(256→7)
  → logits (B, 7)
```

**Why ViT over CNN?**
Self-attention captures global facial structure (e.g. simultaneous brow + mouth context), while CNNs are inherently local. ViT outperforms ResNet-50 when the pretrained corpus is large (ImageNet-21k).

---

### 4.2 Text Models

#### BiLSTM + GloVe (`src/models/lstm_model.py` → `BiLSTMClassifier`)

```
Input token IDs (B, T)
  → Embedding(vocab_size, 100) [GloVe-init, optionally frozen]
  → BiLSTM(256 units/dir, 2 layers)  →  (B, T, 512)
  → AttentionPooling                 →  (B, 512)   ← learns which tokens carry emotion
  → LayerNorm → Dropout(0.4) → Linear(512→7)
  → logits (B, 7)
```

**AttentionPooling** — a learnable 1-layer scorer selects emotionally salient tokens instead of using last-step or mean pooling.

#### BERT Classifier (`src/models/lstm_model.py` → `BERTClassifier`)

```
Input token IDs + attention mask (B, T=128)
  → bert-base-uncased (12 layers, 768 hidden)
  → [CLS] token  →  (B, 768)
  → Dropout(0.3) → Linear(768→256) → ReLU
  → Dropout(0.15) → Linear(256→7)
  → logits (B, 7)
```

**Why [CLS]?** BERT's pre-training objective trains [CLS] to aggregate sentence-level semantics across all 12 attention layers, making it the natural representation for classification.

**Optimizer — no-decay exemptions:** `bias` and `LayerNorm.weight` parameters are excluded from weight decay, following the original BERT fine-tuning recipe.

---

### 4.3 Fusion Strategies

All three strategies are implemented in `src/fusion/fusion_models.py` and wrapped by `MultimodalEmotionModel`.

#### Early Fusion (Concatenation)

```
img_feats (B, 2048)  →  Linear(2048→512) → ReLU  ─┐
                                                    ├─ cat → (B, 1024)
txt_feats (B, 768)   →  Linear(768→512)  → ReLU  ─┘
  → MLP(1024→512→256→7) → logits
```

Simple and fast. Lets the network freely learn cross-modal interactions, but provides no explicit alignment mechanism.

#### Late Fusion (Learned Ensemble)

```
CNN  →  img_head  →  P_img (B, 7)  ─┐
                                     ├─ MLP(14→64→7) → logits
BERT →  txt_head  →  P_txt (B, 7)  ─┘

# Alternative: weighted combination
P_final = σ(α) · P_img + (1 − σ(α)) · P_txt   (α learned per class)
```

Effective when unimodal models are already strong. Two modes: `mlp` (default) or `weighted`.

#### Attention Fusion ⭐ (Best)

```
img_feats  →  proj  →  img_h (B, 512)
txt_feats  →  proj  →  txt_h (B, 512)

img_h ──CrossAttention(Q=img, KV=txt)──► img_ctx
txt_h ──CrossAttention(Q=txt, KV=img)──► txt_ctx

img_gate = σ(W · img_h)
txt_gate = σ(W · txt_h)

img_out = gate · img_ctx + (1−gate) · img_h   # soft blend
txt_out = gate · txt_ctx + (1−gate) · txt_h

[img_out ‖ txt_out] → MLP(1024→512→256→7) → logits
```

Bidirectional cross-attention explicitly models *which image regions are relevant to which text tokens* and vice versa. The gating mechanism prevents over-reliance on one modality when the other is noisy.

---

## 5. Regularisation

### 5.1 Weight Decay (L2)

Weight decay is now **fully configurable** via `config.yaml` and propagated to every optimizer. Values are chosen per model based on standard practice:

| Model | `weight_decay` | Rationale |
|-------|----------------|-----------|
| CNN (ResNet-50) | `0.0001` | Backbone already regularised by BN + dropout |
| ViT-B/16 | `0.01` | Standard transformer value |
| BiLSTM | `0.0001` | Lightweight model — mild L2 |
| BERT | `0.01` | Official BERT fine-tuning recommendation |
| Fusion — encoders | `0.0001` | Pre-trained layers, lighter penalty |
| Fusion — head | `0.001` | New head, stronger regularisation |

**BERT no-decay exemptions** — `bias` and `LayerNorm.weight` are excluded from weight decay (applying L2 to these provides no benefit and can destabilise training):

```python
# src/models/lstm_model.py — build_bert_optimizer()
optimizer_groups = [
    {"params": bert_decay_params,    "weight_decay": 0.01},
    {"params": bert_no_decay_params, "weight_decay": 0.0},   # bias, LayerNorm.weight
    {"params": classifier_params,    "weight_decay": 0.01},
]
```

To override at runtime:
```bash
python scripts/train_bert.py --weight_decay 0.005
python scripts/train_cnn.py  --weight_decay 0.0005
```

---

### 5.2 Early Stopping

Automatic training termination when validation performance stops improving. Implemented in `src/utils/early_stopping.py` and integrated into all four training scripts.

**How it works:**

```
End of epoch N:
  metric = val_loss  (or val_acc)
  if improvement > min_delta:
      best_score ← metric
      counter ← 0
      save checkpoint
  else:
      counter += 1
      if counter == patience:
          restore best weights
          stop training
```

**Configuration** (`configs/config.yaml`):

```yaml
training:
  early_stopping:
    patience: 7          # Epochs without improvement before stopping
    min_delta: 0.001     # Minimum change to count as improvement
    monitor: "val_loss"  # "val_loss" (mode=min) | "val_acc" (mode=max)
    restore_best: true   # Reload best weights automatically on stop
```

**Standalone usage:**

```python
from src.utils.early_stopping import EarlyStopping

es = EarlyStopping(patience=7, min_delta=0.001, mode="min")

for epoch in range(max_epochs):
    val_loss, val_acc, _, _ = evaluate(model, val_loader, ...)

    if es(val_loss, model, checkpoint_path="outputs/checkpoints/best.pt"):
        print(f"Stopped at epoch {epoch} — best val_loss={es.best_score:.4f}")
        break
# Best weights are automatically reloaded into `model`
```

**Console output:**
```
  [EarlyStopping] Improvement ↓  best=0.43100  Δ=0.01200  (counter reset)
  [EarlyStopping] No improvement ↓ (1/7)  best=0.43100  current=0.44300
  [EarlyStopping] No improvement ↓ (7/7)  best=0.43100  current=0.45100
  [CNN Train] Early stopping triggered at epoch 18.
  [EarlyStopping] Best weights restored from outputs/checkpoints/best_cnn.pt
```

To override patience at runtime:
```bash
python scripts/train_cnn.py  --patience 10
python scripts/train_bert.py --patience 5 --config configs/config.yaml
```

---

## 6. GenAI Module

`src/genai/report_generator.py` generates a structured emotional report for each prediction.

**Two modes:**

| Mode | Trigger | Output |
|------|---------|--------|
| **LLM** (Claude API) | `ANTHROPIC_API_KEY` set | Rich, personalised NL report |
| **Rule-based** | No API key | Template-based fallback |

**Report fields (JSON):**

```json
{
  "timestamp": "2024-05-21T14:32:00",
  "emotion": "sad",
  "confidence": 0.812,
  "top_emotions": [
    {"emotion": "sad",     "score": 0.812},
    {"emotion": "neutral", "score": 0.103},
    {"emotion": "fear",    "score": 0.051}
  ],
  "description": "...",
  "psychological_insight": "...",
  "physical_signals": "...",
  "recommendations": ["...", "...", "..."],
  "wellbeing_tip": "...",
  "source": "claude:claude-sonnet-4-20250514"
}
```

**CLI usage:**

```bash
export ANTHROPIC_API_KEY=sk-ant-...

python src/genai/report_generator.py \
  --emotion sad \
  --text "I feel completely overwhelmed and can't focus." \
  --save outputs/reports/my_report.json
```

---

## 7. FastAPI Inference Server

```bash
uvicorn api.app:app --reload --port 8000
```

### Endpoints

| Method | Path | Input | Description |
|--------|------|-------|-------------|
| `GET` | `/health` | — | Liveness check + model status |
| `GET` | `/classes` | — | List of 7 emotion class names |
| `POST` | `/predict/text` | JSON body | Emotion from text |
| `POST` | `/predict/image` | Form file upload | Emotion from face image |
| `POST` | `/predict/multimodal` | File + form text | Emotion from image + text |

### Example Requests

```bash
# Text prediction
curl -X POST http://localhost:8000/predict/text \
     -H "Content-Type: application/json" \
     -d '{"text": "I am so excited today!", "include_report": false}'

# Image prediction
curl -X POST http://localhost:8000/predict/image \
     -F "file=@face.jpg"

# Multimodal prediction (most accurate)
curl -X POST http://localhost:8000/predict/multimodal \
     -F "file=@face.jpg" \
     -F "text=I feel wonderful today" \
     -F "include_report=true"
```

### Example Response

```json
{
  "emotion": "happy",
  "confidence": 0.8732,
  "scores": {
    "angry":   0.0124,
    "disgust": 0.0031,
    "fear":    0.0098,
    "happy":   0.8732,
    "neutral": 0.0701,
    "sad":     0.0212,
    "surprise":0.0102
  },
  "report": { "..." }
}
```

**Swagger UI:** `http://localhost:8000/docs`

---

## 8. Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Download & preprocess data

FER2013 must be downloaded manually from Kaggle (see [Datasets](#3-datasets)).
The Emotion NLP dataset is downloaded automatically from HuggingFace.

```bash
python scripts/preprocess_all.py
# Skip GloVe download if not using BiLSTM:
python scripts/preprocess_all.py --skip_glove
```

### 3 — Train individual models

```bash
# ResNet-50 (image only)
python scripts/train_cnn.py

# BERT classifier (text only)
python scripts/train_bert.py

# BiLSTM + GloVe (text only)
python scripts/train_lstm.py
```

CLI overrides are available for every key hyperparameter:
```bash
python scripts/train_cnn.py \
  --epochs 40 \
  --lr 5e-5 \
  --weight_decay 0.0005 \
  --patience 10 \
  --amp
```

### 4 — Train multimodal fusion

```bash
# Recommended: attention fusion with pretrained encoders
python scripts/train_multimodal.py \
  --fusion attention \
  --cnn_checkpoint  outputs/checkpoints/cnn_20240521/best_cnn.pt \
  --bert_checkpoint outputs/checkpoints/bert_20240521/best_bert.pt

# Freeze encoders — train only the fusion head
python scripts/train_multimodal.py \
  --fusion attention \
  --no_finetune_encoders \
  --epochs 30
```

### 5 — Compare all models

```bash
python scripts/compare_models.py
# Outputs: outputs/reports/model_comparison.png
#          outputs/reports/comparison_summary.json
```

### 6 — Generate emotion report (GenAI)

```bash
export ANTHROPIC_API_KEY=sk-ant-...

python src/genai/report_generator.py \
  --emotion happy \
  --text "I feel great today!"
```

### 7 — Start the API server

```bash
uvicorn api.app:app --reload --port 8000
```

---

## 9. Configuration Reference

All hyperparameters are centralised in `configs/config.yaml`.

```yaml
# ── CNN ──────────────────────────────────────────────────────
cnn:
  backbone: "resnet50"
  pretrained: true
  dropout: 0.5
  learning_rate: 0.0001
  weight_decay: 0.0001         # L2 regularisation
  backbone_lr_factor: 0.1      # Backbone LR = lr × this
  batch_size: 64
  epochs: 30
  scheduler: "cosine"

# ── BERT ─────────────────────────────────────────────────────
bert:
  model_name: "bert-base-uncased"
  dropout: 0.3
  learning_rate: 0.00002
  weight_decay: 0.01
  no_decay_params: ["bias", "LayerNorm.weight"]
  batch_size: 32
  epochs: 10
  warmup_steps: 500

# ── Fusion ───────────────────────────────────────────────────
fusion:
  type: "attention"            # early | late | attention
  hidden_dim: 512
  dropout: 0.3
  learning_rate: 0.0001
  weight_decay_encoders: 0.0001
  weight_decay_fusion: 0.001
  attention:
    num_heads: 8
    d_model: 512

# ── Early Stopping ────────────────────────────────────────────
training:
  early_stopping:
    patience: 7
    min_delta: 0.001
    monitor: "val_loss"        # val_loss | val_acc
    restore_best: true

# ── GenAI ─────────────────────────────────────────────────────
genai:
  model: "claude-sonnet-4-20250514"
  max_tokens: 600
  temperature: 0.7
```

---

## 10. Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | For GenAI | — | Claude API key for emotion reports |
| `CNN_CHECKPOINT` | For API | `outputs/checkpoints/best_cnn.pt` | CNN weights path |
| `BERT_CHECKPOINT` | For API | `outputs/checkpoints/best_bert.pt` | BERT weights path |
| `FUSION_CHECKPOINT` | For API | `outputs/checkpoints/best_fusion.pt` | Fusion model path |
| `FUSION_TYPE` | For API | `attention` | Fusion strategy (`early`/`late`/`attention`) |
| `BERT_MODEL` | For API | `bert-base-uncased` | HuggingFace model name |

---

## 11. Tech Stack

| Layer | Tools |
|-------|-------|
| Deep Learning | PyTorch 2.x |
| NLP | HuggingFace Transformers, Tokenizers |
| Vision | torchvision, timm (ViT), OpenCV, Pillow |
| Data | Pandas, NumPy, scikit-learn, HuggingFace Datasets |
| Visualisation | Matplotlib, Seaborn |
| GenAI Reports | Anthropic Claude API (`anthropic>=0.23.0`) |
| API Serving | FastAPI + Uvicorn |
| Config | PyYAML |
| Logging | TensorBoard, rich, tqdm |

---

## 12. Expected Results

| Model | Modality | Backbone | Accuracy | Macro F1 |
|-------|----------|----------|----------|----------|
| CNN Baseline | Image | ResNet-50 | ~65% | ~63% |
| Vision Transformer | Image | ViT-B/16 | ~68% | ~67% |
| BiLSTM + GloVe | Text | GloVe 100d | ~70% | ~69% |
| BERT Classifier | Text | bert-base-uncased | ~78% | ~77% |
| Early Fusion | Image + Text | CNN + BERT | ~80% | ~79% |
| Late Fusion | Image + Text | Ensemble | ~81% | ~80% |
| **Attention Fusion** | **Image + Text** | **Cross-Attention** | **~83%** | **~82%** |

> Results are indicative. Actual numbers depend on data splits, random seed, and number of training epochs. Early stopping with `patience=7` on `val_loss` typically stops CNN training around epoch 18–22 and BERT around epoch 7–9.

---

## 13. 👤 Author

**Zeineb Ghrab**  
Data & Decisional Systems Engineer Student  
National School of Electronics and Telecommunications of Sfax (ENET'Com)