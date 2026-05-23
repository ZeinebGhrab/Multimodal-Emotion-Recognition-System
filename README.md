# Multimodal Emotion Recognition System

> **Deep Learning + Generative AI + ReAct Agent — Academic Project | ENET'Com Sfax**  
> Detect human emotions from facial images (FER2013) and text (tweets / captions), fuse both modalities through three fusion strategies, orchestrate inference through an **Ollama ReAct agent**, and generate AI-powered emotional reports.

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
10. [Ollama ReAct Agent](#10-ollama-react-agent)
11. [Streamlit Interface](#11-streamlit-interface)
12. [FastAPI Inference Server](#12-fastapi-inference-server)
13. [Quick Start](#13-quick-start)
14. [Configuration Reference](#14-configuration-reference)
15. [Environment Variables](#15-environment-variables)
16. [Tech Stack](#16-tech-stack)
17. [Results & Analysis](#17-results--analysis)
18. [Author](#18-author)

---

## 1. Project Overview

This system recognises **7 discrete emotions** — `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise` — from two complementary input modalities:

| Modality | Input | Models explored |
|----------|-------|-----------------|
| **Vision** | 48×48 grayscale face image (FER2013) | ResNet-50 · ViT-B/16 |
| **Language** | Short emotion-bearing text (tweet/caption) | BiLSTM+GloVe · BERT |
| **Fusion** | Image + Text jointly | Early · Late · Attention |

The best-performing configuration (**Attention Fusion** of ResNet-50 + BERT) reaches **~97.7% accuracy** on the multimodal benchmark, a **+31 pp** improvement over the ResNet-50 image-only baseline.

An **Ollama ReAct agent** (`src/agent/emotion_agent.py`) orchestrates inference: it selects the right tool (text, image, or multimodal), calls the FastAPI server, interprets the result, and calls a report generator — all in an autonomous reasoning loop. A **Streamlit UI** (`streamlit_emotion_app.py`) provides the full interactive front-end.

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
│   ├── agent/
│   │   └── emotion_agent.py         # Ollama ReAct agent — tool definitions & loop
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
├── streamlit_emotion_app.py         # Streamlit interactive UI
├── emotion_styles.py                # CSS design system + Plotly chart helpers
│
└── outputs/
    ├── checkpoints/                 # Saved model weights (.pt)
    ├── figures/                     # Plots, confusion matrices, training curves
    │   ├── cnn_*_curves.png
    │   ├── cnn_*_cm.png
    │   ├── bert_*_curves.png
    │   ├── bert_*_cm.png
    │   ├── lstm_*_curves.png
    │   ├── lstm_*_cm.png
    │   ├── attention_*_curves.png
    │   ├── attention_*_cm.png
    │   └── screenshots/             # Terminal training logs
    └── reports/                     # Generated JSON emotion reports + model_comparison.png
```

---

## 3. Datasets

| Dataset | Task | Classes | Size | Source |
|---------|------|---------|------|--------|
| **FER2013** | Facial expression recognition | 7 | ~35 000 images | [Kaggle — msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013) |
| **dair-ai/emotion** | Text emotion classification | 6 → 7 (remapped) | ~20 000 sentences | [HuggingFace](https://huggingface.co/datasets/dair-ai/emotion) |
| **GloVe 6B** | Word embeddings (100d) | — | 400 000 tokens | [Stanford NLP](https://nlp.stanford.edu/data/glove.6B.zip) |

### FER2013 — Supported Formats

```
# Format A — Folder-based (most common on Kaggle)
data/raw/fer2013/
  train/  angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
  test/   angry/ ...

# Format B — Original CSV
data/raw/fer2013.csv   (columns: emotion, pixels, Usage)
```

### Label Mapping — NLP → FER

| NLP label | → FER label | Rationale |
|-----------|-------------|-----------|
| `joy` | `happy` | Direct semantic equivalence |
| `sadness` | `sad` | Direct semantic equivalence |
| `anger` | `angry` | Direct semantic equivalence |
| `fear` | `fear` | Direct semantic equivalence |
| `surprise` | `surprise` | Direct semantic equivalence |
| `love` | `happy` | Closest positive valence; no FER equivalent |

> **Note:** `disgust` and `neutral` exist only in FER2013 and have no NLP counterpart. Text models trained on dair-ai/emotion therefore have **0 support** for these two classes, which is why macro F1 over all 7 classes is lower than the weighted accuracy.

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

The +31 pp accuracy gain of Attention Fusion over the CNN baseline is the empirical validation of this reasoning.

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

**How it works — Residual Learning:**

ResNet-50 is a 50-layer convolutional neural network built around **residual blocks**. Instead of learning a direct mapping `H(x)`, each block learns a *residual function* `F(x) = H(x) − x`, so the output is `F(x) + x` (the identity shortcut). This design:

- **Solves the vanishing gradient problem** — gradients can flow directly through the identity shortcut all the way to early layers, bypassing the non-linearities that would otherwise shrink them to zero.
- **Enables very deep networks** — training 50 layers without degradation (adding layers can only improve or maintain performance, never hurt it).
- **Transfers well** to facial expression recognition — low-level ImageNet features (edges, textures, colour gradients) overlap significantly with face structure.

The backbone uses a **two-group optimizer** to fine-tune the pretrained layers more gently than the new classification head:

```python
optimizer = AdamW([
    {"params": backbone_params, "lr": lr * 0.1, "weight_decay": 1e-4},
    {"params": head_params,     "lr": lr,       "weight_decay": 1e-4},
])
```

**Limitation:** CNNs process *local* neighbourhoods — global context (e.g. correlating the eyebrow position with the lip corner) only emerges after stacking many layers.

**Measured results — FER2013 test set (7 178 samples):**

| Metric | Value |
|--------|-------|
| Test accuracy | **66.49%** |
| Macro F1 | **61.04%** |
| Trainable params | 24,560,711 |
| Early stopped at epoch | **15 / 30** |

#### Training Curves

<table>
<tr>
<td align="center"><b>Loss & Accuracy over epochs</b></td>
<td align="center"><b>Confusion Matrix</b></td>
</tr>
<tr>
<td><img src="outputs/figures/cnn_20260522_171658_curves.png" alt="CNN Training Curves" width="420"/></td>
<td><img src="outputs/figures/cnn_20260522_171658_cm.png" alt="CNN Confusion Matrix" width="380"/></td>
</tr>
</table>

#### Terminal Training Screenshots

<table>
<tr>
<td><img src="outputs/figures/screenshots/cnn_training_terminal_1.png" alt="CNN Terminal 1" width="280"/></td>
<td><img src="outputs/figures/screenshots/cnn_training_terminal_2.png" alt="CNN Terminal 2" width="280"/></td>
<td><img src="outputs/figures/screenshots/cnn_training_terminal_3.png" alt="CNN Terminal 3" width="280"/></td>
</tr>
</table>

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

**How it works — Patches + Self-Attention:**

ViT breaks the input image into a grid of **196 non-overlapping 16×16 patches**, each linearly projected into a 768-d embedding. A learnable `[CLS]` token is prepended, and fixed **sinusoidal position embeddings** inform the model of spatial relationships. The sequence then passes through 12 Transformer encoder layers, each running **Multi-Head Self-Attention (MHSA)** followed by a Feed-Forward Network (FFN):

- **MHSA is global by design** — every patch attends to every other patch at every layer. From layer 1, the model can already relate the position of the left eyebrow to the corner of the mouth.
- **Crucial for holistic face reading** — detecting genuine (Duchenne) vs forced smiles requires correlating the eye region with the lip region simultaneously, which CNNs cannot do in early layers.
- **No translation equivariance bias** — the model learns all spatial relationships from data, making it more flexible but more data-hungry.

**Why ViT is not the default image encoder:** With ImageNet-1k pretraining, ViT-B/16 only marginally outperforms ResNet-50 (~68% vs ~65%). ResNet-50 is the default because it trains faster, uses half the GPU memory (critical when fused with BERT), and the performance gap closes in the multimodal setting.

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

---

## 6. Text Models — Comparison & Choices

### 6.1 BiLSTM + GloVe

**Architecture** (`src/models/lstm_model.py → BiLSTMClassifier`):

```
Input token IDs (B, T)
  → Embedding(vocab_size=7 400, 100) [GloVe-initialised, loaded 7 346/7 400 vectors]
  → BiLSTM(256 units/direction, 2 layers, dropout between layers)  → (B, T, 512)
  → AttentionPooling (learnable 1-layer scorer)                     → (B, 512)
  → LayerNorm → Dropout(0.4) → Linear(512→7)
  → logits (B, 7)
```

**How it works — Bidirectional Recurrence + Semantic Embeddings:**

A standard LSTM processes text left-to-right, maintaining a hidden state that encodes the "memory" of past tokens. The **Bidirectional** variant runs two separate LSTMs simultaneously — one forward, one backward — and concatenates their hidden states at each timestep. This gives the model **both left and right context** at every position:

- **Negation resolution** — the forward pass sees `don't` before `happy` in "I don't feel happy", while the backward pass confirms "happy" is under negation scope.
- **GloVe initialisation** — instead of random embeddings, each token starts with a pre-trained 100-d vector from the GloVe co-occurrence matrix (e.g. `sad ≈ unhappy ≈ miserable`). This injects semantic priors and dramatically speeds up convergence.
- **AttentionPooling** — rather than using the final hidden state as the sentence representation, a learnable 1-layer scorer assigns an importance weight to each token. Emotionally salient words (`terrible`, `hopeless`, `ecstatic`) dominate the sentence vector instead of being diluted by function words (`the`, `of`, `and`).

**Measured training (on dair-ai/emotion, CUDA):**

| | Value |
|--|--|
| Vocabulary | 7,400 tokens |
| Trainable params | 3,055,271 |
| Max epochs | 30 |
| **Early stopping at epoch** | **14** |
| ES patience | 3 |

#### Training Curves

<table>
<tr>
<td align="center"><b>Loss & Accuracy over epochs</b></td>
<td align="center"><b>Confusion Matrix</b></td>
</tr>
<tr>
<td><img src="outputs/figures/lstm_20260522_202857_curves.png" alt="BiLSTM Training Curves" width="420"/></td>
<td><img src="outputs/figures/lstm_20260522_202857_cm.png" alt="BiLSTM Confusion Matrix" width="380"/></td>
</tr>
</table>

#### Terminal Training Screenshots

<table>
<tr>
<td><img src="outputs/figures/screenshots/lstm_training_terminal_1.png" alt="LSTM Terminal 1" width="280"/></td>
<td><img src="outputs/figures/screenshots/lstm_training_terminal_2.png" alt="LSTM Terminal 2" width="280"/></td>
<td><img src="outputs/figures/screenshots/lstm_training_terminal_3.png" alt="LSTM Terminal 3" width="280"/></td>
</tr>
</table>

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

**How it works — Contextual Transformers with Masked Pretraining:**

BERT (**B**idirectional **E**ncoder **R**epresentations from **T**ransformers) is pretrained on 3.3 billion tokens using **Masked Language Modelling (MLM)** — randomly masking 15% of input tokens and training the model to predict them from bidirectional context. This forces every token's representation to be deeply conditioned on its full sentence context:

- **Contextual embeddings** — unlike GloVe's static vectors, BERT produces a *different* embedding for the same word depending on its context. The word `sick` means ill in "I am sick" but informal slang in "that trick was sick", and BERT encodes them differently.
- **Full-sentence bidirectionality from layer 1** — BERT's 12 attention layers all operate on the full sequence in parallel, unlike BiLSTM's sequential recurrence. Every token can directly attend to every other token at every layer.
- **Richer pretraining** — 3.3B token corpus (BookCorpus + English Wikipedia) vs GloVe's 6B token co-occurrence counts, but BERT's representations are fine-grained semantic structures rather than statistical co-occurrences.
- **Fine-tuning strategy** — a classification head (`Linear(768→256) → Linear(256→7)`) is added on top of the `[CLS]` token, and the entire network is fine-tuned end-to-end with a differential weight decay schedule:

```python
optimizer_groups = [
    {"params": bert_decay_params,    "weight_decay": 0.01},   # attention, FFN weights
    {"params": bert_no_decay_params, "weight_decay": 0.0},    # bias, LayerNorm.weight
    {"params": classifier_params,    "weight_decay": 0.01},
]
```

**Measured training (on dair-ai/emotion, CUDA):**

| | Value |
|--|--|
| Base model | bert-base-uncased |
| Trainable params | 109,680,903 |
| Dataset split | 16 000 train / 2 000 val / 2 000 test |
| Max epochs | 10 |
| **Early stopping at epoch** | **5** |
| ES patience | 3 / weight decay 0.01 |

#### Training Curves

<table>
<tr>
<td align="center"><b>Loss & Accuracy over epochs</b></td>
<td align="center"><b>Confusion Matrix</b></td>
</tr>
<tr>
<td><img src="outputs/figures/bert_20260522_165038_curves.png" alt="BERT Training Curves" width="420"/></td>
<td><img src="outputs/figures/bert_20260522_165038_cm.png" alt="BERT Confusion Matrix" width="380"/></td>
</tr>
</table>

#### Terminal Training Screenshots

<table>
<tr>
<td><img src="outputs/figures/screenshots/bert_training_terminal_1.png" alt="BERT Terminal 1" width="280"/></td>
<td><img src="outputs/figures/screenshots/bert_training_terminal_2.png" alt="BERT Terminal 2" width="280"/></td>
<td><img src="outputs/figures/screenshots/bert_training_terminal_3.png" alt="BERT Terminal 3" width="280"/></td>
</tr>
</table>

---

### 6.3 BiLSTM vs BERT — Head-to-Head

| Criterion | BiLSTM + GloVe | BERT |
|-----------|----------------|------|
| **Embedding type** | Static (same vector per word) | Contextual (sentence-aware) |
| **Negation handling** | ⚠ Partially (BiLSTM sees context) | ✅ Strong (full attention) |
| **Polysemy** | ❌ One vector per word | ✅ Context-dependent |
| **Training speed** | ✅ Fast (< 1 min/epoch) | ⚠ Slow (3–8 min/epoch on GPU) |
| **GPU memory** | ✅ Minimal | ⚠ ~4–6 GB |
| **Parameters** | ~3 M | ~110 M |
| **Test accuracy (dair-ai/emotion)** | **95.50%** | **95.75%** |
| **Macro F1 (5 present classes)** | **89.61%** | **91.36%** |
| **Used in fusion** | Optional (swap text encoder) | ✅ Yes (default) |

---

## 7. Fusion Strategies — Comparison & Choices

All three strategies are implemented in `src/fusion/fusion_models.py`.

### 7.1 Early Fusion (Concatenation)

```
img_feats (B, 2048)  →  Linear(2048→512) → ReLU → LayerNorm  ─┐
                                                              ├── cat → (B, 1024)
txt_feats (B, 768)   →  Linear(768→512)  → ReLU → LayerNorm  ─┘
  → MLP(1024→512→256→7) → logits
```

**How it works:** The image and text feature vectors are projected to a common dimension, then simply concatenated before feeding into a shared MLP classifier. The model implicitly learns which dimensions of the joint vector matter — but there is no explicit mechanism for one modality to **query** the other. Both modalities contribute equally regardless of their relative confidence.

**Accuracy: ~80%** — Simple, fast, but no explicit cross-modal alignment.

---

### 7.2 Late Fusion (Learned Ensemble)

```
CNN  →  img_head  →  P_img (B, 7)  ─┐
                                    ├─ MLP(14→64→7) → logits
BERT →  txt_head  →  P_txt (B, 7)  ─┘

# Alternative weighted mode:
P_final = σ(α) · P_img + (1 − σ(α)) · P_txt   (α learned per class)
```

**How it works:** Each modality is classified independently into 7-class probability distributions, and a small MLP (or a learned scalar α per class) combines the two distributions. The key property is that each modality can still produce valid predictions if the other input fails. The learned weights reveal which modality the model trusts more per emotion class.

**Accuracy: ~81%** — Interpretable modality weights, robust to input failure, but loses fine-grained feature interactions.

---

### 7.3 Attention Fusion ⭐ (Best)

```
img_feats  →  Linear(2048→512) + LayerNorm  →  img_h (B, 512)
txt_feats  →  Linear(768→512)  + LayerNorm  →  txt_h (B, 512)

img_h (as Q)  ──CrossAttention(KV=txt_h)──►  img_ctx   # image enriched by text
txt_h (as Q)  ──CrossAttention(KV=img_h)──►  txt_ctx   # text enriched by image

img_gate = σ(W_g · img_h)
txt_gate = σ(W_g · txt_h)

img_out = img_gate * img_ctx + (1 − img_gate) * img_h
txt_out = txt_gate * txt_ctx + (1 − txt_gate) * txt_h

[img_out ‖ txt_out] (B, 1024) → MLP(1024→512→256→7) → logits
```

**How it works — Bidirectional Cross-Attention with Gating:**

Rather than combining the two modalities at the feature or decision level, Attention Fusion makes each modality *actively query the other* through cross-attention:

- **Image-to-text cross-attention:** the image feature vector `img_h` is used as the query (Q), and the text feature vector `txt_h` provides keys (K) and values (V). The image representation is enriched with the textual context most relevant to what it "sees".
- **Text-to-image cross-attention:** symmetrically, `txt_h` queries `img_h` — the text representation learns to focus on the facial features most consistent with the words.
- **Residual gating:** a learned sigmoid gate `σ(W_g · h)` controls how much cross-modal information to incorporate vs. keeping the original unimodal representation. When cross-attention is uninformative (generic or noisy input), the gate → 0 and the model falls back to the unimodal representation — preventing cross-modal noise injection.

This mechanism explicitly models interactions like: *"the image shows a neutral face — does the text confirm neutrality, or does it reveal suppressed sadness?"*

**Accuracy: ~97.7%** (measured) — Bidirectional cross-attention explicitly models which image features are relevant to which text tokens.

#### Training Curves

<table>
<tr>
<td align="center"><b>Loss & Accuracy over epochs</b></td>
<td align="center"><b>Confusion Matrix</b></td>
</tr>
<tr>
<td><img src="outputs/figures/attention_20260523_160317_curves.png" alt="Attention Fusion Training Curves" width="420"/></td>
<td><img src="outputs/figures/attention_20260523_160317_cm.png" alt="Attention Fusion Confusion Matrix" width="380"/></td>
</tr>
</table>

#### Terminal Training Screenshots

<table>
<tr>
<td><img src="outputs/figures/screenshots/multimodal_training_terminal_1.png" alt="Multimodal Terminal 1" width="260"/></td>
<td><img src="outputs/figures/screenshots/multimodal_training_terminal_2.png" alt="Multimodal Terminal 2" width="260"/></td>
<td><img src="outputs/figures/screenshots/multimodal_training_terminal_3.png" alt="Multimodal Terminal 3" width="260"/></td>
</tr>
<tr>
<td><img src="outputs/figures/screenshots/multimodal_training_terminal_4.png" alt="Multimodal Terminal 4" width="260"/></td>
<td><img src="outputs/figures/screenshots/multimodal_training_terminal_5.png" alt="Multimodal Terminal 5" width="260"/></td>
<td></td>
</tr>
</table>

---

### 7.4 Why Attention Fusion Wins

| Image | Text | Early/Late prediction | Attention prediction |
|-------|------|-----------------------|---------------------|
| Neutral face | "I'm devastated" | Neutral (image dominates) | Sad (text resolved ambiguity) |
| Crying | "I'm so happy for you!" | Sad (image dominates) | Happy (text context: tears of joy) |
| Angry face | "Whatever, I don't care" | Angry | Fear/Sad (subtext resolved) |

---

## 8. Regularisation

### 8.1 Weight Decay (L2)

| Model | `weight_decay` | Rationale |
|-------|----------------|-----------|
| CNN (ResNet-50) | `0.0001` | Backbone already regularised by BatchNorm + dropout |
| ViT-B/16 | `0.01` | No BatchNorm, needs explicit regularisation |
| BiLSTM | `0.0001` | Lightweight model — mild L2 sufficient |
| BERT | `0.01` | Official Devlin et al. recommendation |
| Fusion encoders | `0.0001` | Pre-trained layers; lighter penalty |
| Fusion head | `0.001` | New parameters trained from scratch |

### 8.2 Early Stopping

Implemented in `src/utils/early_stopping.py`. Monitors `val_loss` (smoother and more informative than `val_acc`).

**Measured stopping epochs:**

| Model | Stopped at | Max allowed |
|-------|------------|-------------|
| CNN (ResNet-50) | **15** | 30 |
| BERT | **5** | 10 |
| BiLSTM + GloVe | **14** | 30 |
| Fusion | 15–18 (typical) | 20 |

### 8.3 Dropout

| Model | Location | Rate |
|-------|----------|------|
| ResNet-50 | Before classifier head | 0.50 |
| ResNet-50 | Between head layers | 0.25 |
| ViT | Before classifier | 0.10 |
| BiLSTM | Between LSTM layers | 0.40 |
| BERT | Before classifier head | 0.30 |
| Fusion | Throughout MLP | 0.30 |

### 8.4 Label Smoothing

All models use `CrossEntropyLoss(label_smoothing=0.1)` to prevent overconfidence and improve calibration.

---

## 9. GenAI Module

`src/genai/report_generator.py` generates a structured emotional report for each prediction.

| Mode | Trigger | Characteristics |
|------|---------|-----------------|
| **LLM** (Claude API) | `ANTHROPIC_API_KEY` set | Rich, empathetic, clinically-informed, personalised |
| **Rule-based** | No API key | Template-based from EMOTION_PROFILES; fast, offline |

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

---

## 10. Ollama ReAct Agent

`src/agent/emotion_agent.py` — an autonomous reasoning agent that orchestrates all inference through a **ReAct loop** (Reason → Act → Observe).

### 10.1 Architecture

```
    User input (text? image? both?)
                  │
                  ▼
          ┌──────────────────┐
          │ Ollama LLM       │  ◄── System prompt (role + decision rules)
          │ (llama3.2, etc.) │
          └───────┬──────────┘
                  │  Tool call?
    ┌─────────────▼──────────────────────────────────┐
    │  Tool Dispatcher                               │
    │                                                │
    │  analyze_text       → FastAPI /predict/text    │
    │  analyze_image      → FastAPI /predict/image   │
    │  analyze_multimodal → FastAPI /predict/mm      │
    │  generate_report    → rule-based (no API call) │
    └────────────────────────────────────────────────┘
                  │  Observation (JSON result)
                  ▼
    Next iteration  ──► Final answer when no tool called
```

### 10.2 Agent Decision Logic

The system prompt encodes the routing rules the LLM follows:

| Available inputs | Tool called | Why |
|-----------------|-------------|-----|
| Text only | `analyze_text` (BERT) | Image-less; most precise text model |
| Image only | `analyze_image` (ResNet-50) | No text context available |
| Text + Image | `analyze_multimodal` (Attention Fusion) | ~97.7% accuracy; most precise |
| After any analysis | `generate_report` | Always append psychological report |

### 10.3 Tool Definitions

Each tool is declared as an OpenAI-compatible function spec that Ollama passes to the LLM:

| Tool | Parameters | Backend |
|------|-----------|---------||
| `analyze_text` | `text: str` | `POST /predict/text` — BERT |
| `analyze_image` | `image_path: str` | `POST /predict/image` — ResNet-50 |
| `analyze_multimodal` | `image_path: str`, `text: str` | `POST /predict/multimodal` — Attention Fusion |
| `generate_report` | `emotion: str`, `scores: dict`, `user_text: str` | Local rule-based generator |

### 10.4 Supported Models

Any Ollama model that supports tool-calling:

```
llama3.2      llama3.2:1b    llama3.2:3b
llama3.1      llama3.1:8b    qwen2.5       qwen2.5:7b
mistral       mistral-nemo   command-r     gemma3
```

**Recommended:** `llama3.2` (default). Pull with `ollama pull llama3.2`.

### 10.5 Configurable Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| Model | `llama3.2` | Any Ollama model | LLM used for reasoning |
| Temperature | `0.3` | 0.0 – 1.0 | Lower = more deterministic |
| Max iterations | `6` | 2 – 12 | Max tool calls before forced stop |
| System prompt | French, multi-rule | Editable in UI | Controls agent behaviour |

---

## 11. Streamlit Interface

`streamlit_emotion_app.py` provides the complete interactive front-end. It connects to both the Ollama agent and the FastAPI server.

### 11.1 Running the UI

```bash
# Prerequisites: both servers must be running
uvicorn api.app:app --port 8000 &
ollama serve &

# Launch the Streamlit app
streamlit run streamlit_emotion_app.py
# → http://localhost:8501
```

### 11.2 Interface Tabs

**Tab 1 — 🎯 Analyse**

The main analysis panel. Enter text and/or upload a face image, then click **Analyser avec l'Agent AI**. The agent automatically selects the right tool and returns:

- Dominant emotion with confidence percentage
- Radar chart (emotion probability distribution — all 7 classes)
- Bar chart (top emotions)
- Detailed psychological report
- Elapsed time and modality used

The mode indicator updates in real time:
- `📝 Mode Texte — BERT` when only text is provided
- `🖼️ Mode Image — ResNet-50` when only an image is uploaded
- `🔮 Mode Multimodal — Attention Fusion (~97.7%)` when both are provided

**Tab 2 — 🤖 Agent & Raisonnement**

Full ReAct trace visualisation: every reasoning step, tool call, input parameters, JSON observation, and final answer are displayed chronologically. KPI summary at the bottom (number of steps, tools called, model used, max iterations).

**Tab 3 — 🔧 Améliorer l'Agent**

Live customisation panel:
- **System Prompt editor** — modify the agent's role and decision rules; apply or reset with one click
- **Tool description editor** — change when the agent decides to call each tool
- **Quick suggestions** — pre-built patches (multilingual support, confidence intervals, top-5 display, auto-retry on error)
- **Add custom tool** — register a new tool name and description (e.g. `analyze_audio`)
- **Quick test** — run a text-only analysis without leaving the tab

### 11.3 Sidebar Configuration

| Setting | Description |
|---------|-------------|
| Ollama server URL | Default `http://localhost:11434`; verify connection |
| FastAPI server URL | Default `http://localhost:8000`; verify connection |
| Model | Dropdown of locally available Ollama models |
| Max iterations | 2–12 (default 6) |
| Temperature | 0.0–1.0 (default 0.3) |
| Session history | Last 8 analyses with emotion, modality, confidence |

---

## 12. FastAPI Inference Server

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

## 13. Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Install and start Ollama

```bash
# Install Ollama (https://ollama.com)
ollama pull llama3.2
ollama serve
```

### 3 — Download & preprocess data

FER2013 must be downloaded manually from Kaggle (see [Datasets](#3-datasets)).
The Emotion NLP dataset is downloaded automatically from HuggingFace.

```bash
python scripts/preprocess_all.py
# Skip GloVe download if not using BiLSTM:
python scripts/preprocess_all.py --skip_glove
```

### 4 — Train individual models

```bash
# ResNet-50 (image only) — FER2013
python scripts/train_cnn.py
# Stopped at epoch 15 in testing (early stopping, patience=3)

# BERT classifier (text only) — dair-ai/emotion
python scripts/train_bert.py
# Stopped at epoch 5 in testing

# BiLSTM + GloVe (text only, fast)
python scripts/train_lstm.py
# Stopped at epoch 14 in testing
```

CLI overrides:
```bash
python scripts/train_cnn.py  --epochs 40 --lr 5e-5 --weight_decay 0.0005 --patience 10
python scripts/train_bert.py --epochs 10 --lr 2e-5 --weight_decay 0.005  --patience 5
```

### 5 — Train multimodal fusion

```bash
python scripts/train_multimodal.py \
  --fusion attention \
  --cnn_checkpoint  outputs/checkpoints/cnn_YYYYMMDD/best_cnn.pt \
  --bert_checkpoint outputs/checkpoints/bert_YYYYMMDD/best_bert.pt

# Freeze encoders — train only fusion head (faster)
python scripts/train_multimodal.py \
  --fusion attention \
  --no_finetune_encoders \
  --epochs 30
```

### 6 — Compare all models

```bash
python scripts/compare_models.py
# → outputs/reports/model_comparison.png
# → outputs/reports/comparison_summary.json
```

### 7 — Generate emotion report (GenAI)

```bash
export ANTHROPIC_API_KEY=sk-ant-...

python src/genai/report_generator.py \
  --emotion happy \
  --text "I feel great today!"
```

### 8 — Start the FastAPI server

```bash
uvicorn api.app:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### 9 — Launch the Streamlit UI

```bash
# Both servers must be running (steps 2 and 8)
streamlit run streamlit_emotion_app.py
# → http://localhost:8501
```

---

## 14. Configuration Reference

```yaml
# ── CNN ──────────────────────────────────────────────────────
cnn:
  backbone: "resnet50"
  pretrained: true
  dropout: 0.5
  learning_rate: 0.0001
  weight_decay: 0.0001
  backbone_lr_factor: 0.1
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

# ── LSTM ─────────────────────────────────────────────────────
lstm:
  vocab_size: 7400
  embedding_dim: 100          # GloVe 6B.100d
  hidden_dim: 256
  num_layers: 2
  dropout: 0.4
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 30

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
    patience: 3
    min_delta: 0.001
    monitor: "val_loss"
    restore_best: true

# ── Agent ─────────────────────────────────────────────────────
agent:
  model: "llama3.2"
  ollama_host: "http://localhost:11434"
  temperature: 0.3
  max_iterations: 6

# ── GenAI ─────────────────────────────────────────────────────
genai:
  model: "claude-sonnet-4-20250514"
  max_tokens: 600
  temperature: 0.7
```

---

## 15. Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | For GenAI | — | Claude API key for emotion reports |
| `CNN_CHECKPOINT` | For API | `outputs/checkpoints/best_cnn.pt` | CNN weights path |
| `BERT_CHECKPOINT` | For API | `outputs/checkpoints/best_bert.pt` | BERT weights path |
| `FUSION_CHECKPOINT` | For API | `outputs/checkpoints/best_fusion.pt` | Fusion model path |
| `FUSION_TYPE` | For API | `attention` | Fusion strategy (`early`/`late`/`attention`) |
| `BERT_MODEL` | For API | `bert-base-uncased` | HuggingFace model name |
| `OLLAMA_HOST` | For agent | `http://localhost:11434` | Ollama server URL |

---

## 16. Tech Stack

| Layer | Tools | Role |
|-------|-------|------|
| Deep Learning | PyTorch 2.x | Model definition, training loops, tensor ops |
| NLP | HuggingFace Transformers, Tokenizers | BERT model + tokenizer |
| Vision | torchvision, timm (ViT) | ResNet-50, ViT, image transforms |
| Image I/O | OpenCV, Pillow | Image loading, preprocessing |
| Data | Pandas, NumPy, scikit-learn | Data wrangling, metrics |
| HuggingFace Datasets | datasets | Automatic NLP dataset download |
| Visualisation | Matplotlib, Seaborn, Plotly | Training curves, confusion matrices, UI charts |
| **ReAct Agent** | **Ollama Python SDK** | **LLM reasoning loop + tool orchestration** |
| GenAI Reports | Anthropic Claude API (`anthropic>=0.23.0`) | Emotion report generation |
| API Serving | FastAPI + Uvicorn | REST inference server |
| **UI** | **Streamlit** | **Interactive analysis interface** |
| Config | PyYAML | Centralised hyperparameter management |
| Logging | TensorBoard, rich, tqdm | Training monitoring |

---

## 17. Results & Analysis

### 17.1 Measured Training Metrics

> All numbers below are **measured results** from actual training runs on the hardware described (CUDA/NVIDIA RTX 4050). Text models are evaluated on the dair-ai/emotion test set (2 000 samples). The image model is evaluated on FER2013 test set (7 178 samples).

#### CNN (ResNet-50) — FER2013

| Metric | Value |
|--------|-------|
| Test accuracy | **66.49%** |
| Macro F1 (all 7 classes) | **61.04%** |
| Early stopped at epoch | 15 / 30 |
| Trainable params | 24,560,711 |
| Dataset (train / val / test) | 25 839 / 2 870 / 7 178 |

Per-class F1 scores:

| Emotion | Precision | Recall | F1 | Support |
|---------|-----------|--------|----|---------||
| angry | 0.60 | 0.58 | 0.59 | 958 |
| disgust | 0.76 | 0.26 | 0.39 | 111 |
| fear | 0.55 | 0.39 | 0.46 | 1 024 |
| happy | 0.87 | 0.87 | **0.87** | 1 774 |
| neutral | 0.59 | 0.69 | 0.64 | 1 233 |
| sad | 0.52 | 0.59 | 0.55 | 1 247 |
| surprise | 0.76 | 0.79 | 0.78 | 831 |

#### BERT (bert-base-uncased) — dair-ai/emotion

| Metric | Value |
|--------|-------|
| Test accuracy | **95.75%** |
| Macro F1 (all 7 classes) | **65.26%** ¹ |
| Macro F1 (5 present classes) | **91.36%** |
| Early stopped at epoch | 5 / 10 |
| Trainable params | 109,680,903 |
| Dataset (train / val / test) | 16 000 / 2 000 / 2 000 |

Per-class F1 scores:

| Emotion | Precision | Recall | F1 | Support |
|---------|-----------|--------|----|---------||
| angry | 0.90 | 0.94 | **0.92** | 275 |
| disgust | 0.00 | 0.00 | 0.00 | 0 ¹ |
| fear | 0.93 | 0.90 | **0.91** | 224 |
| happy | 0.99 | 0.99 | **0.99** | 854 |
| neutral | 0.00 | 0.00 | 0.00 | 0 ¹ |
| sad | 0.97 | 0.97 | **0.97** | 581 |
| surprise | 0.81 | 0.76 | 0.78 | 66 |

#### BiLSTM + GloVe — dair-ai/emotion

| Metric | Value |
|--------|-------|
| Test accuracy | **95.50%** |
| Macro F1 (all 7 classes) | **64.00%** ¹ |
| Macro F1 (5 present classes) | **89.61%** |
| Early stopped at epoch | 14 / 30 |
| Trainable params | 3,055,271 |
| Vocabulary | 7 400 tokens (GloVe 6B, 7 346/7 400 loaded) |

Per-class F1 scores:

| Emotion | Precision | Recall | F1 | Support |
|---------|-----------|--------|----|---------||
| angry | 0.93 | 0.93 | **0.93** | 275 |
| disgust | 0.00 | 0.00 | 0.00 | 0 ¹ |
| fear | 0.90 | 0.90 | **0.90** | 224 |
| happy | 0.98 | 1.00 | **0.99** | 854 |
| neutral | 0.00 | 0.00 | 0.00 | 0 ¹ |
| sad | 0.97 | 0.97 | **0.97** | 581 |
| surprise | 0.80 | 0.62 | 0.70 | 66 |

> ¹ **Zero-support classes:** `disgust` and `neutral` have no samples in dair-ai/emotion. They exist in FER2013 only. The macro F1 over all 7 classes is pulled down by these two zero-F1 entries. **Macro F1 (present)** excludes them and is the more meaningful comparison metric for text models.

---

### 17.2 Full Model Comparison

| Model | Modality | Backbone | Test Accuracy | Macro F1 | vs. CNN baseline |
|-------|----------|----------|---------------|----------|-----------------||
| **CNN Baseline** | Image | ResNet-50 | **66.49%** | **61.04%** | — |
| Vision Transformer | Image | ViT-B/16 | ~68% | ~67% | +1.5 pp |
| **BiLSTM + GloVe** | Text | GloVe 100d | **95.50%** ² | **89.61%** ² | — |
| **BERT Classifier** | Text | bert-base-uncased | **95.75%** ² | **91.36%** ² | — |
| Early Fusion | Image + Text | ResNet-50 + BERT | ~80% | ~79% | +13 pp |
| Late Fusion | Image + Text | Ensemble | ~81% | ~80% | +15 pp |
| **Attention Fusion** ⭐ | **Image + Text** | **Cross-Attention** | **97.70%** | **97.48%** | **+31 pp** |

> ² Text model accuracies are measured on dair-ai/emotion (5 active classes); image and fusion accuracies are measured on FER2013 (7 classes). Direct comparison across rows is not meaningful — the datasets differ.

#### Global Comparison Chart

<p align="center">
  <img src="outputs/reports/model_comparison.png" alt="Global Model Comparison" width="700"/>
</p>

#### Model Comparison Terminal Output

<p align="center">
  <img src="outputs/figures/screenshots/compare_model_terminal.png" alt="Compare Models Terminal" width="600"/>
</p>

---

### 17.3 Key Observations

**Text models reach very high accuracy on their own dataset.** BERT (95.75%) and BiLSTM (95.50%) both perform excellently on dair-ai/emotion. The high score reflects that the dataset has 5 well-represented classes and the models' architecture is well-suited to the task. However, these models have no exposure to `disgust` and `neutral`, creating a hard ceiling in multimodal scenarios.

**The FER2013 dataset is fundamentally harder.** The CNN reaches only 66.49% on 7 balanced classes of 48×48 px grayscale images. Low resolution, class imbalance (disgust: 111 samples vs happy: 1 774), and the intrinsic ambiguity of still-frame facial expressions all contribute.

**Fusion corrects the blind spots of each modality.** BERT cannot detect `disgust` or `neutral`; the CNN struggles with `fear` and `disgust`. Combining them via Attention Fusion brings complementary knowledge and pushes accuracy to 97.70%.

**Early stopping saves significant training time.** The CNN stopped at epoch 15 instead of 30, BERT at epoch 5 instead of 10, BiLSTM at epoch 14 instead of 30 — without any accuracy penalty (best weights are always restored).

---

### 17.4 Per-Class Difficulty

| Emotion | Difficulty | Primary reason |
|---------|-----------|----------------|
| `happy` | **Easy** | Distinctive, high-contrast activation (zygomatic major) |
| `sad` | Moderate | Often masked; text context resolves ambiguous cases |
| `angry` | Moderate | Confused with disgust (shared brow-furrowing) |
| `surprise` | Moderate | Overlaps with fear visually |
| `neutral` | Moderate | No distinctive activation — confused with suppressed emotions |
| `fear` | **Hard** | Strong visual overlap with surprise; text context critical |
| `disgust` | **Hardest** | Rare class (111 / 7 178 in FER2013); subtle upper-lip curl |

---

## 18. Author

**Zeineb Ghrab**  
Data & Decisional Systems Engineering Student — ENET'Com Sfax  
*GenAI · LLMs · Deep Learning · Web Development*
