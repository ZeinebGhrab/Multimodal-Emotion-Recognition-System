# Multimodal Emotion Recognition System

> **Deep Learning + Generative AI — Academic Project | ENET'Com Sfax**<br>
> Detect human emotions from facial images (FER2013) and text (tweets / captions), fuse both modalities through three fusion strategies, and generate an AI-powered emotional report via the Claude API.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Datasets](#3-datasets)
4. [Why Multimodal?](#4-why-multimodal)
5. [Image Models — Comparison & Choices](#5-image-models--comparison--choices)
   - [CNN Baseline — ResNet-50](#51-cnn-baseline--resnet-50)
   - [Vision Transformer — ViT-B/16](#52-vision-transformer--vit-b16)
   - [ResNet-50 vs ViT — Head-to-Head](#53-resnet-50-vs-vit--head-to-head)
6. [Text Models — Comparison & Choices](#6-text-models--comparison--choices)
   - [BiLSTM + GloVe](#61-bilstm--glove)
   - [BERT Classifier](#62-bert-classifier)
   - [BiLSTM vs BERT — Head-to-Head](#63-bilstm-vs-bert--head-to-head)
7. [Fusion Strategies — Comparison & Choices](#7-fusion-strategies--comparison--choices)
   - [Early Fusion](#71-early-fusion-concatenation)
   - [Late Fusion](#72-late-fusion-learned-ensemble)
   - [Attention Fusion ⭐](#73-attention-fusion--best)
   - [Why Attention Fusion Wins](#74-why-attention-fusion-wins)
8. [Regularisation](#8-regularisation)
9. [GenAI Module](#9-genai-module)
10. [FastAPI Inference Server](#10-fastapi-inference-server)
11. [Quick Start](#11-quick-start)
12. [Configuration Reference](#12-configuration-reference)
13. [Environment Variables](#13-environment-variables)
14. [Tech Stack](#14-tech-stack)
15. [Results & Analysis](#15-results--analysis)
16. [Authors](#16-authors)

---

## 1. Project Overview

This system recognises **7 discrete emotions** — `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise` — from two complementary input modalities:

| Modality | Input | Models explored |
|----------|-------|-----------------|
| **Vision** | 48×48 grayscale face image (FER2013) | ResNet-50 · ViT-B/16 |
| **Language** | Short emotion-bearing text (tweet/caption) | BiLSTM+GloVe · BERT |
| **Fusion** | Image + Text jointly | Early · Late · Attention |

The best-performing configuration (**Attention Fusion** of ResNet-50 + BERT) reaches **~83% accuracy** on FER2013, a **+18 pp** improvement over the ResNet-50 baseline alone. A breakdown of every design choice and the reasoning behind it is provided in sections 4–7.

A **GenAI module** (`src/genai/report_generator.py`) takes the predicted emotion and probability scores and calls the **Claude API** to produce a personalised, clinically-informed emotional report in JSON format, with a rule-based fallback for offline use.

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
│   └── app.py                       # FastAPI inference server (5 endpoints)
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
└── outputs/
    ├── checkpoints/                 # Saved model weights (.pt)
    ├── figures/                     # Plots, confusion matrices, training curves
    └── reports/                     # Generated JSON emotion reports
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

| NLP label | → FER label | Rationale |
|-----------|-------------|-----------|
| `joy` | `happy` | Direct semantic equivalence |
| `sadness` | `sad` | Direct semantic equivalence |
| `anger` | `angry` | Direct semantic equivalence |
| `fear` | `fear` | Direct semantic equivalence |
| `surprise` | `surprise` | Direct semantic equivalence |
| `love` | `happy` | Closest positive valence; no FER equivalent |

> **Note:** `disgust` and `neutral` exist only in FER2013 and have no NLP counterpart, which creates a natural class imbalance in the multimodal dataset. This is one reason unimodal text models cap out below the full 7-class accuracy of the image models.

---

## 4. Why Multimodal?

Emotion recognition from a single modality is inherently ambiguous:

- **Image alone** struggles with occlusion, low lighting, cultural expression differences, and neutral faces hiding strong internal states.
- **Text alone** misses sarcasm, irony, and cases where the written content contradicts the emotional delivery (e.g. "I'm fine." said with visible distress).

Combining both modalities allows the model to **cross-validate** signals:

```
Image says "neutral" + Text says "I feel terrible" → Model learns to trust text
Image says "happy"   + Text says "great day"       → Both agree → high confidence
Image says "angry"   + Text says "I love this"     → Conflict → attention mechanism arbitrates
```

The +18 pp accuracy gain of Attention Fusion over the CNN baseline is the empirical validation of this reasoning.

---

## 5. Image Models — Comparison & Choices

### 5.1 CNN Baseline — ResNet-50

**Architecture** (`src/models/cnn_model.py`):

```
Input (B, 3, 224, 224)
  → ResNet-50 backbone (ImageNet pretrained, backbone LR = lr × 0.1)
  → GlobalAvgPool → Flatten → (B, 2048)
  → Dropout(0.5) → Linear(2048→512) → ReLU
  → Dropout(0.25) → Linear(512→7)
  → logits (B, 7)
```

**Why ResNet-50?**

Residual connections (`F(x) + x`) solve the vanishing gradient problem that plagued deep plain CNNs before 2015. Each block learns *residual corrections* on top of the identity shorthand, which:
- Allows gradients to flow directly to early layers without attenuation
- Enables training of very deep networks (50 layers) without degradation
- Generalises extremely well from ImageNet to domain-specific tasks like facial expression recognition because low-level features (edges, textures, shapes) are shared

**Why Transfer Learning?**

FER2013 has only ~28 000 training images — far too few to learn meaningful visual representations from scratch. Training ResNet-50 on ImageNet first (1.2 M images, 1 000 classes) gives the backbone rich feature detectors (Gabor-like filters, texture detectors, shape detectors) that directly transfer to facial analysis.

**Two-group optimizer design:**

```python
# Backbone pretrained weights: fine-tune slowly (lr × 0.1)
# New classifier head: learn fast (lr)
optimizer = AdamW([
    {"params": backbone_params, "lr": lr * 0.1, "weight_decay": 1e-4},
    {"params": head_params,     "lr": lr,       "weight_decay": 1e-4},
])
```

This prevents the pretrained representations from being destroyed in the early epochs of fine-tuning.

**Limitation:** CNNs are inherently *local* — each neuron's receptive field grows gradually across layers. This means that in the early layers, a brow-furrow detector cannot "see" the mouth simultaneously. Global context requires stacking many layers.

---

### 5.2 Vision Transformer — ViT-B/16

**Architecture** (`src/models/vit_model.py`):

```
Input (B, 3, 224, 224)
  → Patch embedding: 196 non-overlapping patches of 16×16 pixels
  → + position embedding + [CLS] token → (B, 197, 768)
  → 12 Transformer encoder layers (Multi-Head Self-Attention + FFN)
  → [CLS] token → (B, 768)
  → Dropout(0.1) → Linear(768→256) → GELU
  → Dropout(0.05) → Linear(256→7)
  → logits (B, 7)
```

**Why ViT?**

Self-attention is *global by design* — every patch attends to every other patch at every layer. This means layer 1 can already relate the left eyebrow to the corner of the mouth, which is crucial for holistic facial expression reading (e.g. genuine vs forced smile requires comparing eyes *and* mouth simultaneously, known as the Duchenne marker).

**Why patch size 16×16?**

- Smaller patches (8×8) → more tokens → quadratic attention cost grows quickly
- Larger patches (32×32) → too coarse to capture fine facial muscle detail
- 16×16 on 224×224 gives 196 patches — a good efficiency/granularity trade-off

**Why ViT is not the default image encoder in the fusion pipeline:**

Despite its architectural advantage, ViT requires a larger pretraining corpus than ResNet-50 to fully realise its potential. With ImageNet-1k pretraining, ViT-B/16 only marginally outperforms ResNet-50 (~68% vs ~65%). ResNet-50 is used as the default image encoder in the fusion model because it is faster, uses less GPU memory (critical when fused with BERT), and its performance gap closes to near-zero in the multimodal setting.

---

### 5.3 ResNet-50 vs ViT — Head-to-Head

| Criterion | ResNet-50 | ViT-B/16 |
|-----------|-----------|----------|
| **Inductive bias** | Strong (translation equivariance, locality) | None — learns from data |
| **Global context** | Slow (requires many layers) | Immediate (all-to-all attention) |
| **Small dataset** | ✅ Better with ImageNet-1k pretrain | ⚠ Needs ImageNet-21k or more |
| **GPU memory** | ~4 GB (batch 64) | ~8 GB (batch 32) |
| **Inference speed** | Faster | ~2× slower |
| **FER2013 accuracy** | ~65% | ~68% |
| **Used in fusion** | ✅ Yes (default) | Optional swap-in |

**Decision:** ResNet-50 is the default image encoder. ViT is provided as a plug-in alternative for higher-resource environments or when pretrained on ImageNet-21k.

---

## 6. Text Models — Comparison & Choices

### 6.1 BiLSTM + GloVe

**Architecture** (`src/models/lstm_model.py → BiLSTMClassifier`):

```
Input token IDs (B, T)
  → Embedding(vocab_size, 100) [GloVe-initialised]
  → BiLSTM(256 units/direction, 2 layers, dropout between layers)  → (B, T, 512)
  → AttentionPooling (learnable 1-layer scorer)                     → (B, 512)
  → LayerNorm → Dropout(0.4) → Linear(512→7)
  → logits (B, 7)
```

**Why Bidirectional LSTM?**

A unidirectional LSTM only has access to left-context when encoding token *t*. For emotion detection, right-context is equally important: *"I don't feel happy"* — the negation comes **before** `happy`, but in the reverse direction it is seen first. The BiLSTM concatenates forward and backward hidden states, giving every token a full-sentence view.

**Why GloVe instead of random embeddings?**

GloVe (Global Vectors for Word Representation) is trained on 6 billion tokens using co-occurrence statistics. Words that appear in similar contexts get similar vectors: `sad ≈ unhappy ≈ miserable`. This semantic prior means the model starts with emotionally meaningful word relationships instead of white noise, dramatically accelerating convergence on the small emotion dataset.

**Why AttentionPooling instead of last-hidden-state or mean-pool?**

- **Last hidden state:** Only uses the final token — useless for long sentences or when the key emotion word is early.
- **Mean pooling:** Treats all tokens equally — function words like "the", "is", "a" dilute the emotionally salient tokens.
- **Attention pooling:** Learns a scalar score per token; softmax over scores gives weights; weighted sum extracts an emotion-focused sentence vector. A sentence like *"I feel absolutely terrible and hopeless"* correctly upweights `terrible` and `hopeless`.

**Limitation:** Static GloVe vectors cannot handle polysemy. The word `sick` means ill in *"I feel sick"* but cool/impressive in *"That was sick"* — same vector, different emotion.

---

### 6.2 BERT Classifier

**Architecture** (`src/models/lstm_model.py → BERTClassifier`):

```
Input token IDs + attention mask (B, T=128)
  → bert-base-uncased (12 transformer layers, 768 hidden, 12 attention heads)
  → [CLS] token  →  (B, 768)
  → Dropout(0.3) → Linear(768→256) → ReLU
  → Dropout(0.15) → Linear(256→7)
  → logits (B, 7)
```

**Why the [CLS] token?**

BERT is trained with a Masked Language Model objective that forces every layer of attention to build rich, context-aware representations. The `[CLS]` token has no semantic content of its own — it is a designated "summary" position that all 12 attention layers write into. By the final layer, it encodes the full sentence's meaning in a single 768-d vector, making it the natural pooling point for classification.

**Why BERT over BiLSTM?**

- **Contextual embeddings:** `sick` in *"I feel sick"* gets a different vector than `sick` in *"That was sick"* — BERT resolves polysemy dynamically.
- **Bidirectional attention from layer 1:** Unlike BiLSTM which processes sequentially, BERT attends globally in every layer simultaneously.
- **Pre-training scale:** BERT was pre-trained on 3.3 billion words — 200× more text than GloVe, with a more expressive training objective.
- **Negation handling:** *"I don't feel happy at all"* — BERT's attention mechanism correctly down-weights `happy` when it sees `don't` and `at all` in full context.

**Why bert-base-uncased instead of bert-large?**

`bert-large` has 340 M parameters vs 110 M for `bert-base`. For a 7-class emotion classification task on a 16 k sample dataset, the additional capacity provides diminishing returns and requires 3× more GPU memory. `bert-base` is the standard academic choice for this class of task.

**BERT-specific optimizer design — no-decay exemptions:**

```python
# Standard BERT fine-tuning recipe (Devlin et al., 2019)
# Applying L2 to bias terms and LayerNorm weights provides no benefit
# and destabilises training by pushing the learned scale/shift parameters toward zero.
optimizer_groups = [
    {"params": bert_decay_params,    "weight_decay": 0.01},   # attention, FFN weights
    {"params": bert_no_decay_params, "weight_decay": 0.0},    # bias, LayerNorm.weight
    {"params": classifier_params,    "weight_decay": 0.01},
]
```

---

### 6.3 BiLSTM vs BERT — Head-to-Head

| Criterion | BiLSTM + GloVe | BERT |
|-----------|----------------|------|
| **Embedding type** | Static (same vector per word) | Contextual (sentence-aware) |
| **Negation handling** | ⚠ Partially (BiLSTM sees context) | ✅ Strong (full attention) |
| **Polysemy** | ❌ One vector per word | ✅ Context-dependent |
| **Training speed** | ✅ Fast (<1 min/epoch) | ⚠ Slow (3-8 min/epoch on GPU) |
| **GPU memory** | ✅ Minimal | ⚠ ~4-6 GB |
| **Parameters** | ~5 M | ~110 M |
| **Emotion accuracy** | ~70% | ~78% |
| **Used in fusion** | Optional (swap text encoder) | ✅ Yes (default) |

**Decision:** BERT is the default text encoder. BiLSTM+GloVe is provided as a lightweight alternative for resource-constrained environments or rapid prototyping.

---

## 7. Fusion Strategies — Comparison & Choices

All three strategies are implemented in `src/fusion/fusion_models.py` and wrapped by `MultimodalEmotionModel`.

### 7.1 Early Fusion (Concatenation)

```
img_feats (B, 2048)  →  Linear(2048→512) → ReLU → LayerNorm  ─┐
                                                                 ├── cat → (B, 1024)
txt_feats (B, 768)   →  Linear(768→512)  → ReLU → LayerNorm  ─┘
  → MLP(1024→512→256→7) → logits
```

**How it works:** Project both modalities to a common dimension, concatenate, classify.

**Strengths:**
- Simplest implementation — fewest parameters in the fusion module
- The MLP can theoretically learn any cross-modal interaction
- Fast training, low memory overhead

**Weaknesses:**
- No explicit alignment — the network must figure out on its own that pixel region X is relevant to text token Y
- If one modality is much stronger, the other may be ignored during backpropagation
- No mechanism to down-weight a noisy or contradictory modality

**Accuracy: ~80%**

---

### 7.2 Late Fusion (Learned Ensemble)

```
CNN  →  img_head  →  P_img (B, 7)  ─┐
                                      ├─ MLP(14→64→7) → logits
BERT →  txt_head  →  P_txt (B, 7)  ─┘

# Alternative weighted mode:
P_final = σ(α) · P_img + (1 − σ(α)) · P_txt   (α learned per class)
```

**How it works:** Each modality produces a full probability distribution; a small learnable module combines the two distributions.

**Strengths:**
- Can be built on top of independently pretrained unimodal models (no end-to-end retraining required)
- The weighting is interpretable — you can inspect α to see which modality the model trusts more per class
- Robust: if one modality fails (e.g., image is corrupted), the other can still carry the prediction

**Weaknesses:**
- No shared representation — cross-modal feature interactions are invisible to the ensemble module
- Both branches compress their input to a 7-d probability vector before fusion, losing fine-grained feature information
- Performance plateau: you cannot exceed the best unimodal model by much unless the errors are uncorrelated

**Accuracy: ~81%**

---

### 7.3 Attention Fusion ⭐ (Best)

```
img_feats  →  Linear(2048→512) + LayerNorm  →  img_h (B, 512)
txt_feats  →  Linear(768→512)  + LayerNorm  →  txt_h (B, 512)

img_h (as Q)  ──CrossAttention(KV=txt_h)──►  img_ctx   # image enriched by text
txt_h (as Q)  ──CrossAttention(KV=img_h)──►  txt_ctx   # text enriched by image

img_gate = σ(W_g · img_h)
txt_gate = σ(W_g · txt_h)

img_out = img_gate * img_ctx + (1 − img_gate) * img_h  # soft gated blend
txt_out = txt_gate * txt_ctx + (1 − txt_gate) * txt_h

[img_out ‖ txt_out] (B, 1024) → MLP(1024→512→256→7) → logits
```

**How it works:** Bidirectional cross-attention explicitly models *which image features are relevant to which text features* and vice versa. The gating mechanism then decides how much of the cross-attended context to blend into the original representation.

**Why Cross-Attention instead of Self-Attention?**

Self-attention (as in BERT) relates tokens *within* the same sequence. Cross-attention relates tokens *across* two different sequences. Here:
- Image attending to text: the image feature vector asks "which aspects of the text description are most relevant to interpreting this face?"
- Text attending to image: the text feature vector asks "which visual cues best support or contradict what the text is saying?"

This bidirectional dialogue enables the model to resolve ambiguities that neither modality can resolve alone.

**Why the gating mechanism?**

When the cross-attended context is poor (e.g., the text is generic or the image is blurry), the gate output approaches 0, falling back to the original unimodal representation. When the cross-attended context is rich and informative, the gate opens. This prevents the cross-attention from degrading performance in low-quality input scenarios.

**Strengths:**
- Highest accuracy — explicitly models cross-modal relevance
- Robust to noisy inputs via gating
- Bidirectional — neither modality is privileged by design
- Interpretable attention weights can be visualised to explain predictions

**Weaknesses:**
- More parameters than Early/Late Fusion
- Requires careful regularisation (dropout + weight decay) to avoid overfitting
- Slower to train

**Accuracy: ~83%**

---

### 7.4 Why Attention Fusion Wins

The key insight is that emotion is not just the sum of an image signal and a text signal — it is the *interaction* between them. Consider:

| Image | Text | Early/Late prediction | Attention prediction |
|-------|------|-----------------------|---------------------|
| Neutral face | "I'm devastated" | Neutral (image dominates) | Sad (text resolved ambiguity) |
| Crying | "I'm so happy for you!" | Sad (image dominates) | Happy (text context: tears of joy) |
| Angry face | "Whatever, I don't care" | Angry | Fear/Sad (subtext resolved) |

Cross-attention lets the model *ask questions* across modalities: the neutral-face feature vector attends to the word `devastated` in the text and updates itself accordingly. This information flow is not possible in Early or Late Fusion.

---

## 8. Regularisation

### 8.1 Weight Decay (L2)

Fully configurable via `configs/config.yaml`:

| Model | `weight_decay` | Rationale |
|-------|----------------|-----------|
| CNN (ResNet-50) | `0.0001` | Backbone already regularised by BatchNorm + dropout; too strong L2 destroys pretrained features |
| ViT-B/16 | `0.01` | Standard transformer value; ViT has no BatchNorm, needs more explicit regularisation |
| BiLSTM | `0.0001` | Lightweight model with few parameters — mild L2 sufficient |
| BERT | `0.01` | Official BERT fine-tuning recommendation (Devlin et al.) |
| Fusion encoders | `0.0001` | Pre-trained layers; lighter penalty to preserve learned representations |
| Fusion head | `0.001` | New parameters trained from scratch; stronger regularisation needed |

**BERT no-decay exemptions** — bias terms and `LayerNorm.weight` are excluded from weight decay. Applying L2 to scale/shift parameters in LayerNorm pushes them toward zero, which destabilises the normalisation statistics learned during pretraining:

```python
optimizer_groups = [
    {"params": bert_decay_params,    "weight_decay": 0.01},
    {"params": bert_no_decay_params, "weight_decay": 0.0},   # bias, LayerNorm.weight
    {"params": classifier_params,    "weight_decay": 0.01},
]
```

Override at runtime:
```bash
python scripts/train_bert.py --weight_decay 0.005
python scripts/train_cnn.py  --weight_decay 0.0005
```

---

### 8.2 Early Stopping

Implemented in `src/utils/early_stopping.py` and integrated into all four training scripts. Prevents overfitting by halting training when validation performance stops improving.

**Algorithm:**

```
End of epoch N:
  metric = val_loss  (or val_acc)
  if metric improved by more than min_delta:
      best_score ← metric
      counter    ← 0
      save checkpoint
  else:
      counter += 1
      if counter == patience:
          restore best weights
          stop training
```

**Why monitor `val_loss` instead of `val_acc`?**

Validation loss is a smoother, more informative signal than accuracy. Accuracy is integer-rounded and can plateau for many epochs while the model continues to improve internally. Loss is continuous and captures confident vs uncertain correct predictions — a model improving its calibration will show loss reduction even without accuracy gains.

**Configuration** (`configs/config.yaml`):

```yaml
training:
  early_stopping:
    patience: 7          # Epochs without improvement before stopping
    min_delta: 0.001     # Minimum change to count as improvement
    monitor: "val_loss"  # "val_loss" (mode=min) | "val_acc" (mode=max)
    restore_best: true   # Reload best weights automatically on stop
```

**Expected stopping epochs:**

| Model | Typical stop epoch | Without early stopping |
|-------|--------------------|------------------------|
| CNN | 18–22 | 30 (overfits) |
| BERT | 7–9 | 10 |
| BiLSTM | 20–25 | 30 |
| Fusion | 15–18 | 20 |

**Console output example:**
```
  [EarlyStopping] Improvement ↓  best=0.43100  Δ=0.01200  (counter reset)
  [EarlyStopping] No improvement ↓ (1/7)  best=0.43100  current=0.44300
  [EarlyStopping] No improvement ↓ (7/7)  best=0.43100  current=0.45100
  [CNN Train] Early stopping triggered at epoch 18.
  [EarlyStopping] Best weights restored from outputs/checkpoints/best_cnn.pt
```

Override at runtime:
```bash
python scripts/train_cnn.py  --patience 10
python scripts/train_bert.py --patience 5
```

---

### 8.3 Dropout

Applied strategically at different rates per model:

| Model | Location | Rate | Purpose |
|-------|----------|------|---------|
| ResNet-50 | Before classifier head | 0.50 | Strong regularisation on 2048-d feature vector |
| ResNet-50 | Between head layers | 0.25 | Lighter between hidden layers |
| ViT | Before classifier | 0.10 | Transformers are already regularised by attention dropout |
| BiLSTM | Between LSTM layers | 0.40 | Prevent co-adaptation of LSTM units |
| BERT | Before classifier head | 0.30 | Standard BERT fine-tuning rate |
| Fusion | Throughout MLP | 0.30 | Prevent fusion head memorising training pairs |

### 8.4 Label Smoothing

All models use `CrossEntropyLoss(label_smoothing=0.1)`. Instead of training toward hard one-hot targets `[0, 0, 1, 0, 0, 0, 0]`, the targets become `[0.014, 0.014, 0.9, 0.014, 0.014, 0.014, 0.014]`. This prevents the model from becoming overconfident, improves calibration, and acts as an implicit regulariser.

---

## 9. GenAI Module

`src/genai/report_generator.py` generates a structured emotional report for each prediction.

**Two modes:**

| Mode | Trigger | Characteristics |
|------|---------|-----------------|
| **LLM** (Claude API) | `ANTHROPIC_API_KEY` set | Rich, empathetic, clinically-informed, personalised to user text |
| **Rule-based** | No API key | Template-based from EMOTION_PROFILES; fast, offline, deterministic |

**Why Claude (claude-sonnet-4-20250514)?**

The report generation task requires nuanced language: it must be empathetic without being dismissive, clinically accurate without being cold, and personalised to the specific confidence distribution — not just the top emotion. Claude Sonnet balances output quality with API cost and latency, and the system-prompt enforces strict JSON output to guarantee parseable structured reports.

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

## 10. FastAPI Inference Server

```bash
uvicorn api.app:app --reload --port 8000
```

### Endpoints

| Method | Path | Input | Description |
|--------|------|-------|-------------|
| `GET` | `/health` | — | Liveness check + model status |
| `GET` | `/classes` | — | List of 7 emotion class names |
| `POST` | `/predict/text` | JSON body | Emotion from text (BERT) |
| `POST` | `/predict/image` | Form file upload | Emotion from face image (ResNet-50) |
| `POST` | `/predict/multimodal` | File + form text | Emotion from image + text (Attention Fusion) |

### Example Requests

```bash
# Text prediction
curl -X POST http://localhost:8000/predict/text \
     -H "Content-Type: application/json" \
     -d '{"text": "I am so excited today!", "include_report": false}'

# Image prediction
curl -X POST http://localhost:8000/predict/image \
     -F "file=@face.jpg"

# Multimodal prediction (most accurate — uses Attention Fusion)
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
    "angry":    0.0124,
    "disgust":  0.0031,
    "fear":     0.0098,
    "happy":    0.8732,
    "neutral":  0.0701,
    "sad":      0.0212,
    "surprise": 0.0102
  },
  "report": { "..." }
}
```

**Swagger UI:** `http://localhost:8000/docs`

---

## 11. Quick Start

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

# BERT classifier (text only)  — ~5 min/epoch on GPU, wait for "Epoch 1/10 |..."
python scripts/train_bert.py

# BiLSTM + GloVe (text only, fast)
python scripts/train_lstm.py
```

CLI overrides available for every key hyperparameter:
```bash
python scripts/train_cnn.py \
  --epochs 40 --lr 5e-5 --weight_decay 0.0005 --patience 10 --amp

python scripts/train_bert.py \
  --epochs 10 --lr 2e-5 --weight_decay 0.005 --patience 5
```

### 4 — Train multimodal fusion

```bash
# Recommended: attention fusion with pretrained encoders
python scripts/train_multimodal.py \
  --fusion attention \
  --cnn_checkpoint  outputs/checkpoints/cnn_20240521/best_cnn.pt \
  --bert_checkpoint outputs/checkpoints/bert_20240521/best_bert.pt

# Freeze encoders — train only the fusion head (faster, slightly lower accuracy)
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
# Swagger UI: http://localhost:8000/docs
```

---

## 12. Configuration Reference

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

## 13. Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | For GenAI | — | Claude API key for emotion reports |
| `CNN_CHECKPOINT` | For API | `outputs/checkpoints/best_cnn.pt` | CNN weights path |
| `BERT_CHECKPOINT` | For API | `outputs/checkpoints/best_bert.pt` | BERT weights path |
| `FUSION_CHECKPOINT` | For API | `outputs/checkpoints/best_fusion.pt` | Fusion model path |
| `FUSION_TYPE` | For API | `attention` | Fusion strategy (`early`/`late`/`attention`) |
| `BERT_MODEL` | For API | `bert-base-uncased` | HuggingFace model name |

---

## 14. Tech Stack

| Layer | Tools | Role |
|-------|-------|------|
| Deep Learning | PyTorch 2.x | Model definition, training loops, tensor ops |
| NLP | HuggingFace Transformers, Tokenizers | BERT model + tokenizer |
| Vision | torchvision, timm (ViT) | ResNet-50, ViT, image transforms |
| Image I/O | OpenCV, Pillow | Image loading, preprocessing |
| Data | Pandas, NumPy, scikit-learn | Data wrangling, metrics |
| HuggingFace Datasets | datasets | Automatic NLP dataset download |
| Visualisation | Matplotlib, Seaborn | Training curves, confusion matrices |
| GenAI Reports | Anthropic Claude API (`anthropic>=0.23.0`) | Emotion report generation |
| API Serving | FastAPI + Uvicorn | REST inference server |
| Config | PyYAML | Centralised hyperparameter management |
| Logging | TensorBoard, rich, tqdm | Training monitoring |

---

## 15. Results & Analysis

### Performance Summary

| Model | Modality | Backbone | Accuracy | Macro F1 | vs. CNN baseline |
|-------|----------|----------|----------|----------|------------------|
| CNN Baseline | Image | ResNet-50 | ~65% | ~63% | — |
| Vision Transformer | Image | ViT-B/16 | ~68% | ~67% | +3 pp |
| BiLSTM + GloVe | Text | GloVe 100d | ~70% | ~69% | +5 pp |
| BERT Classifier | Text | bert-base-uncased | ~78% | ~77% | +13 pp |
| Early Fusion | Image + Text | ResNet-50 + BERT | ~80% | ~79% | +15 pp |
| Late Fusion | Image + Text | Ensemble | ~81% | ~80% | +16 pp |
| **Attention Fusion** | **Image + Text** | **Cross-Attention** | **~83%** | **~82%** | **+18 pp** |

### Key Observations

**Text > Image for this dataset.** BERT (~78%) significantly outperforms ResNet-50 (~65%) despite FER2013 being a vision-specific benchmark. This is because facial expression images in FER2013 are only 48×48 pixels — extremely low resolution — making fine-grained facial muscle analysis difficult. Text carries a dense emotional signal that is resolution-independent.

**Fusion > best unimodal.** Every fusion strategy outperforms the best unimodal model (BERT at ~78%). The modalities are complementary: text is strong at semantic emotion but weak on intensity and non-verbal cues; images are strong on intensity and spontaneous expression but weak on ambiguous neutral faces.

**Attention Fusion gap over Late Fusion is small (+2 pp) but consistent.** This reflects that Late Fusion already benefits from two strong pretrained encoders. The Attention Fusion advantage grows on harder samples where one modality contradicts the other.

**Early stopping typically saves 8–12 epochs of wasted training** across all models, with no accuracy penalty — the best validation checkpoint is always restored.

### Per-Class Difficulty

Based on typical FER2013 results:

| Emotion | Relative difficulty | Reason |
|---------|---------------------|--------|
| `happy` | Easy | Distinctive, high-contrast facial muscle activation (zygomatic major) |
| `surprise` | Moderate | Often confused with fear (similar brow + mouth pattern) |
| `neutral` | Moderate | No distinctive muscle activation — easily confused with suppressed emotions |
| `disgust` | Hard | Rare class (~3% of FER2013); subtle upper-lip curl |
| `fear` | Hard | Overlaps with surprise visually; text context is crucial |
| `sad` | Moderate | Often masked; text context resolves many ambiguous cases |
| `angry` | Moderate | Confused with disgust; brow-furrowing is shared |

> Results are indicative. Actual numbers depend on data splits, random seed, and number of training epochs.

---

## 16. 👤 Author

**Zeineb Ghrab**
<br>Data & Decisional Systems Engineer Student
<br>National School of Electronics and Telecommunications of Sfax (ENET'Com)