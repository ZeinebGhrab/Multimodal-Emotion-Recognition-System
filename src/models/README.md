# src/models/ — Model Architectures

Three model files implement the unimodal classifiers. Each exposes two public methods used by the fusion module:

- `forward(x)` — full classification pass → logits `(B, 7)`
- `extract_features(x)` — feature vector before the classifier head

```
src/models/
├── __init__.py
├── cnn_model.py     ResNet-50  (image → 2 048-d features → 7 classes)
├── vit_model.py     ViT-B/16   (image → 768-d features  → 7 classes)
└── lstm_model.py    BiLSTM + GloVe  AND  BERT classifier
```

---

## cnn_model.py — ResNet-50

### Architecture

```
Input (B, 3, 224, 224)
  → ResNet-50 backbone (ImageNet pretrained)
  → GlobalAvgPool + Flatten
(B, 2048)
  → Dropout(0.50) → Linear(2048 → 512) → ReLU
  → Dropout(0.25) → Linear(512 → 7)
Logits (B, 7)
```

### Key Design Choices

**Residual connections** (`output = F(x) + x`) allow gradients to flow directly to early layers, solving the vanishing-gradient problem in 50-layer networks.

**Two-group optimizer** — pretrained backbone trains at a much lower LR than the new classifier head:

```python
optimizer = AdamW([
    {"params": backbone_params, "lr": lr * 0.1, "weight_decay": 1e-4},  # 1e-5
    {"params": head_params,     "lr": lr,       "weight_decay": 1e-4},  # 1e-4
])
```

This avoids catastrophic forgetting of ImageNet features while still allowing the backbone to adapt to facial expressions.

### Public Interface

```python
from src.models.cnn_model import EmotionCNN

model = EmotionCNN(num_classes=7, pretrained=True)

# Full forward pass
logits = model(images)                  # (B, 7)

# Feature extraction (used by fusion)
features = model.extract_features(images)  # (B, 2048)
```

### Measured Results — FER2013 (7 178 samples)

| Metric | Value |
|--------|-------|
| Test accuracy | **66.49%** |
| Macro F1 | **61.04%** |
| Early stopped at | Epoch 15 / 30 |
| Trainable params | 24,560,711 |

---

## vit_model.py — Vision Transformer (ViT-B/16)

### Architecture

```
Input (B, 3, 224, 224)
  → 196 non-overlapping 16×16 patches, linearly projected → (B, 196, 768)
  → Prepend [CLS] token + sinusoidal position embeddings  → (B, 197, 768)
  → 12 Transformer encoder layers (MHSA + FFN)
  → [CLS] token → (B, 768)
  → Dropout(0.10) → Linear(768 → 256) → GELU
  → Dropout(0.05) → Linear(256 → 7)
Logits (B, 7)
```

### Key Design Choices

**Global self-attention from layer 1** — every patch attends to every other patch at every layer. This makes ViT superior at correlating distant facial regions (e.g. eyebrow position + lip corners) that CNNs can only relate after many layers.

**Why ViT is not the default encoder:** It requires ~8 GB GPU RAM (vs ~4 GB for ResNet-50) and trains slower. ResNet-50 is the default for fusion because the accuracy gap closes in the multimodal setting, where BERT already provides global context.

### Public Interface

```python
from src.models.vit_model import EmotionViT

model = EmotionViT(num_classes=7, pretrained=True)
logits   = model(images)                    # (B, 7)
features = model.extract_features(images)   # (B, 768)
```

---

## lstm_model.py — BiLSTM + GloVe and BERT

This single file contains two classifiers with a shared interface.

### BiLSTMClassifier

```
Input token IDs (B, T)
  → Embedding(vocab_size=7 400, 100d) — GloVe initialised
  → BiLSTM(256 units/direction, 2 layers, inter-layer dropout)  → (B, T, 512)
  → AttentionPooling (learnable 1-layer importance scorer)       → (B, 512)
  → LayerNorm → Dropout(0.4) → Linear(512 → 7)
Logits (B, 7)
```

**AttentionPooling:** Instead of using the final hidden state, a learnable scorer assigns importance weights to each token. Emotionally salient words (`terrible`, `ecstatic`) dominate the sentence vector instead of being diluted by function words.

**GloVe initialisation:** Each embedding starts from a pre-trained 100-d vector — semantic priors (`sad ≈ unhappy ≈ miserable`) dramatically speed up convergence.

```python
from src.models.lstm_model import BiLSTMClassifier

model = BiLSTMClassifier(
    vocab_size=7400,
    embedding_dim=100,
    hidden_dim=256,
    num_layers=2,
    num_classes=7,
    glove_path="data/raw/glove.6B.100d.txt"
)

logits   = model(token_ids)                    # (B, 7)
features = model.extract_features(token_ids)   # (B, 512)
```

### BERTClassifier

```
Input token IDs + attention mask (B, T=128)
  → bert-base-uncased (12 layers, 768 hidden, 12 attention heads)
  → [CLS] token → (B, 768)
  → Dropout(0.30) → Linear(768 → 256) → ReLU
  → Dropout(0.15) → Linear(256 → 7)
Logits (B, 7)
```

Fine-tuned with a differential weight decay schedule — bias terms and LayerNorm weights receive zero L2 penalty:

```python
optimizer_groups = [
    {"params": bert_decay_params,    "weight_decay": 0.01},   # attention, FFN
    {"params": bert_no_decay_params, "weight_decay": 0.0},    # bias, LayerNorm
    {"params": classifier_params,    "weight_decay": 0.01},
]
```

```python
from src.models.lstm_model import BERTClassifier

model = BERTClassifier(model_name="bert-base-uncased", num_classes=7)

logits   = model(input_ids, attention_mask)                   # (B, 7)
features = model.extract_features(input_ids, attention_mask)  # (B, 768)
```

---

## Model Comparison

| Criterion | BiLSTM + GloVe | BERT |
|-----------|----------------|------|
| Embedding type | Static | Contextual |
| Polysemy handling | ❌ One vector/word | ✅ Context-dependent |
| Training speed | ✅ Fast | ⚠ 3–8 min/epoch |
| Parameters | ~3 M | ~110 M |
| Test accuracy | 95.50% | **95.75%** |
| **Used in fusion** | Optional | **✅ Default** |

---

*Last Updated: 23/05/2026 — Status: Active ✓*
