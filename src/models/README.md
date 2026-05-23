🧠 src/models/ — Model Architectures
======================================

## Overview

Three model files implement the unimodal classifiers.
Each exposes two public methods used by the fusion module:

- `forward(x)` — full classification pass → logits
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
  ↓
ResNet-50 backbone (ImageNet pretrained)
  ↓ GlobalAvgPool + Flatten
(B, 2048)
  ↓
Dropout(0.50) → Linear(2048 → 512) → ReLU
  ↓
Dropout(0.25) → Linear(512 → 7)
  ↓
Logits (B, 7)
```

### Why ResNet-50?

Residual connections (`output = F(x) + x`) allow gradients to flow directly
to early layers, solving the vanishing-gradient problem present in plain
50-layer networks. Each block learns **residual corrections** on top of an
identity shortcut, enabling:

- Training 50 layers without degradation
- Strong generalisation from ImageNet → FER2013 (shared low-level features)
- Good accuracy / compute trade-off vs ResNet-101

### Key Design Details

```python
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7, dropout=0.5,
                 pretrained=True, freeze_bn=False):
        ...
        # Remove original FC, keep backbone up to avg-pool
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 2048          # used by fusion module

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes)
        )
```

### Two-Group Optimizer

```python
optimizer = AdamW([
    {"params": backbone_params, "lr": lr * 0.1, "weight_decay": wd},
    {"params": head_params,     "lr": lr,       "weight_decay": wd},
])
```

The backbone trains at `lr × 0.1 = 1e-5` to avoid catastrophic forgetting
of ImageNet features. The new classification head trains at the full `lr = 1e-4`.

### Training Utilities

| Function            | Signature                              | Purpose                              |
|---------------------|----------------------------------------|--------------------------------------|
| `build_optimizer`   | `(model, lr, backbone_lr_factor, wd)` | Two-group AdamW                      |
| `build_scheduler`   | `(optimizer, epochs, warmup_epochs)`  | Cosine annealing with linear warmup  |
| `train_one_epoch`   | `(model, loader, opt, crit, device, scaler)` | Single training epoch + AMP support |
| `evaluate`          | `(model, loader, criterion, device)`  | Val/test evaluation                  |

### Measured Results (FER2013 test set)

| Metric            | Value      |
|-------------------|------------|
| Test accuracy     | **66.49%** |
| Macro F1 (all 7)  | **61.04%** |
| Trainable params  | 24 560 711 |
| Best epoch        | 15 / 30    |

Per-class F1 highlights:
- `happy`: 0.87 (most discriminative class)
- `disgust`: 0.39 (severe class imbalance — only 111 test samples)
- `fear`: 0.46 (confused with surprise — shared brow raise)

---

## vit_model.py — ViT-B/16

### Architecture

```
Input (B, 3, 224, 224)
  ↓
Patch embedding: 196 patches of 16×16 px
  + position embedding + [CLS] token
  → (B, 197, 768)
  ↓
12 Transformer encoder layers
  (Multi-Head Self-Attention + FFN, each with LayerNorm + residual)
  ↓
[CLS] token → (B, 768)
  ↓
Dropout(0.10) → Linear(768 → 256) → GELU
  ↓
Dropout(0.05) → Linear(256 → 7)
  ↓
Logits (B, 7)
```

### Why ViT?

Self-attention is **global by design** — every patch attends to every other
patch at every layer. Layer 1 can already relate the left eyebrow to the
corner of the mouth, which is critical for reading holistic facial
expressions (e.g. Duchenne marker: genuine vs forced smile).

### Why ResNet-50 is the Default

| Criterion         | ResNet-50       | ViT-B/16        |
|-------------------|-----------------|-----------------|
| FER2013 accuracy  | ~65%            | ~68%            |
| GPU memory (B=32) | ~4 GB           | ~8 GB           |
| Inference speed   | Faster          | ~2× slower      |
| Fusion overhead   | Low             | High (768-d → projection) |
| Pretrain data     | ImageNet-1k     | ImageNet-21k needed for best results |

When fused with BERT, ViT doubles GPU memory requirements.
ResNet-50 is the default; ViT is available as a swap-in.

### Key Design Details

```python
class EmotionViT(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224",
                 num_classes=7, dropout=0.1, pretrained=True):
        ...
        self.vit = ViTModel.from_pretrained(model_name)
        self.feature_dim = 768       # used by fusion module

    def extract_features(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0, :]   # CLS token → (B, 768)
```

---

## lstm_model.py — BiLSTM + GloVe AND BERT

This file contains **two** classifiers sharing the same module.

---

### BiLSTMClassifier

#### Architecture

```
Input token IDs (B, T)
  ↓
Embedding(vocab_size, 100) [GloVe-initialised]
  + Dropout(0.4)
  ↓
BiLSTM(256 units/direction, 2 layers, inter-layer dropout)
  → (B, T, 512)
  ↓
AttentionPooling (learnable 1-layer scorer)
  → (B, 512)
  ↓
LayerNorm → Dropout(0.4) → Linear(512 → 7)
  ↓
Logits (B, 7)
```

#### Key Design Choices

**Bidirectional LSTM** — Both forward and backward context are available at
every token position. Negation is correctly resolved:
`"I don't feel happy"` — the forward pass sees `don't` before `happy`.

**GloVe initialisation** — 100-d co-occurrence vectors bring semantic priors
(sad ≈ unhappy ≈ miserable) instead of training from random noise.
Coverage: 7 346 / 7 400 vocabulary words (99.3%).

**AttentionPooling** — Learns a scalar importance weight per token.
Emotionally salient words (`terrible`, `hopeless`) dominate the sentence
vector rather than being diluted by function words (`the`, `and`).

```python
class AttentionPooling(nn.Module):
    def forward(self, lstm_out, mask=None):
        scores = self.attn_weights(lstm_out).squeeze(-1)   # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (weights * lstm_out).sum(dim=1)             # (B, 512)
```

#### Measured Results (dair-ai/emotion)

| Metric                    | Value      |
|---------------------------|------------|
| Test accuracy             | **95.50%** |
| Macro F1 (5 present)      | **89.61%** |
| Macro F1 (all 7 incl. 0-support) | 64.00% |
| Trainable params          | 3 055 271  |
| Best epoch                | 14 / 30    |
| GloVe coverage            | 7 346 / 7 400 |

---

### BERTClassifier

#### Architecture

```
Input token IDs + attention mask (B, T=128)
  ↓
bert-base-uncased
  (12 transformer layers, 768 hidden, 12 attention heads, 110M params)
  ↓
[CLS] token → (B, 768)
  ↓
Dropout(0.30) → Linear(768 → 256) → ReLU
  ↓
Dropout(0.15) → Linear(256 → 7)
  ↓
Logits (B, 7)
```

#### Why BERT over BiLSTM?

| Criterion            | BiLSTM + GloVe | BERT               |
|----------------------|----------------|--------------------|
| Embedding type       | Static         | Contextual         |
| Negation handling    | Partial        | Strong             |
| Polysemy (`sick`)    | One vector     | Context-dependent  |
| Training speed       | < 1 min/epoch  | 3–8 min/epoch (GPU)|
| Parameters           | ~3 M           | ~110 M             |
| Test accuracy        | 95.50%         | 95.75%             |
| Macro F1 (present)   | 89.61%         | 91.36%             |

#### AdamW with No-Decay Exemptions

```python
optimizer_groups = [
    {"params": bert_decay_params,    "weight_decay": 0.01},  # attention + FFN weights
    {"params": bert_no_decay_params, "weight_decay": 0.0},   # bias + LayerNorm.weight
    {"params": classifier_params,    "weight_decay": 0.01},
]
```

Bias terms and LayerNorm weights receive **zero** weight decay.
Applying L2 to these parameters provides no regularisation benefit and
can destabilise BERT fine-tuning (per original Devlin et al. recipe).

#### Measured Results (dair-ai/emotion)

| Metric                    | Value      |
|---------------------------|------------|
| Test accuracy             | **95.75%** |
| Macro F1 (5 present)      | **91.36%** |
| Macro F1 (all 7 incl. 0-support) | 65.26% |
| Trainable params          | 109 680 903 |
| Best epoch                | 5 / 10     |

---

## Feature Dimensions Summary

| Model           | `feature_dim` | Used by fusion as |
|-----------------|--------------|-------------------|
| ResNet-50       | 2048         | `img_feats`       |
| ViT-B/16        | 768          | `img_feats`       |
| BiLSTM + GloVe  | 512          | `txt_feats`       |
| BERT            | 768          | `txt_feats`       |

All models implement `.extract_features()` returning `(B, feature_dim)`.
The fusion module calls this method, making any encoder interchangeable.

---

## Swapping Encoders

To swap ResNet-50 for ViT in the fusion model:

```python
from src.models.vit_model import EmotionViT
from src.models.lstm_model import BERTClassifier
from src.fusion.fusion_models import MultimodalEmotionModel

model = MultimodalEmotionModel(
    image_encoder=EmotionViT(pretrained=True),   # ← swap here
    text_encoder=BERTClassifier(),
    fusion_type="attention"
)
```

The fusion module reads `image_encoder.feature_dim` dynamically,
so no other code needs to change.

---

Last Updated: 23/05/2026
Status: Active ✓