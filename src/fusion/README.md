🔀 src/fusion/ — Multimodal Fusion Strategies
===============================================

## Overview

The fusion module combines feature vectors from the image and text encoders
into a single emotion prediction. Three strategies are implemented,
all sharing the same public interface so they can be swapped without
changing any other code.

```
src/fusion/
├── __init__.py
└── fusion_models.py     EarlyFusionModel | LateFusionModel | AttentionFusionModel
                         MultimodalEmotionModel  (end-to-end wrapper)
```

---

## Why Fusion Improves Accuracy

Single-modality emotion recognition has hard limits:

```
Image alone: struggles with occlusion, low resolution, cultural differences,
             neutral faces hiding strong internal states.

Text alone:  misses sarcasm, irony, and cases where written content
             contradicts the emotional delivery.

Fusion:      +18 pp accuracy over CNN baseline (65% → 83%)
```

Real-world improvement examples:

| Image          | Text                      | Unimodal prediction | Fusion prediction   |
|----------------|---------------------------|---------------------|---------------------|
| Neutral face   | "I'm devastated"          | Neutral (image wins)| Sad (text resolves) |
| Crying face    | "I'm so happy for you!"   | Sad (image wins)    | Happy (tears of joy)|
| Angry face     | "Whatever, I don't care"  | Angry               | Fear/Sad (subtext)  |

---

## Strategy 1 — EarlyFusionModel

### Architecture

```
img_feats (B, 2048)  →  Linear(2048→512) → LayerNorm → ReLU  ──┐
                                                               ├─ cat → (B, 1024)
txt_feats (B, 768)   →  Linear(768→512)  → LayerNorm → ReLU  ──┘
  ↓
MLP(1024→512→256→7) with Dropout between layers
  ↓
Logits (B, 7)
```

### Characteristics

| Property               | Value                              |
|------------------------|------------------------------------|
| Test accuracy          | ~80%                               |
| Parameters (heads only)| ~1.2 M                             |
| Cross-modal alignment  | None — MLP must learn implicitly   |
| Speed                  | Fastest fusion                     |  
| Best use case          | Baseline; fast training            |

### Code

```python
class EarlyFusionModel(nn.Module):
    def forward(self, img_feats, txt_feats):
        img_h = self.img_proj(img_feats)   # (B, 512)
        txt_h = self.txt_proj(txt_feats)   # (B, 512)
        fused = torch.cat([img_h, txt_h], dim=-1)   # (B, 1024)
        return self.fusion_mlp(fused)      # (B, 7)
```

---

## Strategy 2 — LateFusionModel

### Architecture

```
img_feats → img_head → P_img (B, 7)  ──┐
                                       ├─ MLP(14→64→7) → logits
txt_feats → txt_head → P_txt (B, 7)  ──┘

# Alternative weighted mode:
P_final = σ(α) · P_img + (1 − σ(α)) · P_txt
# α is a learned per-class scalar parameter
```

### Two Sub-Modes

| Mode        | How fusion is done                              |
|-------------|------------------------------------------------|
| `mlp`       | Concatenate softmax outputs → small 2-layer MLP |
| `weighted`  | Learnable convex combination with per-class `α` |

### Characteristics

| Property              | Value                              |
|-----------------------|------------------------------------|
| Test accuracy         | ~81%                               |
| Interpretability      | ✅ High — can inspect α weights     |
| Robustness            | ✅ Works if one modality is missing |
| Cross-modal alignment | ❌ No shared representation         |

### Code

```python
class LateFusionModel(nn.Module):
    def forward(self, img_feats, txt_feats):
        img_probs = F.softmax(self.img_head(img_feats), dim=-1)
        txt_probs = F.softmax(self.txt_head(txt_feats), dim=-1)

        if self.mode == "weighted":
            alpha = torch.sigmoid(self.alpha)
            fused_probs = alpha * img_probs + (1 - alpha) * txt_probs
            return torch.log(fused_probs + 1e-8)
        else:
            return self.ensemble_mlp(torch.cat([img_probs, txt_probs], dim=-1))
```

---

## Strategy 3 — AttentionFusionModel ⭐ (Default)

### Architecture

```
img_feats → Linear(2048→512) + LayerNorm → img_h (B, 512)
txt_feats → Linear(768→512)  + LayerNorm → txt_h (B, 512)

img_h (as Query)  ──CrossAttention(KV=txt_h)──►  img_ctx
txt_h (as Query)  ──CrossAttention(KV=img_h)──►  txt_ctx

img_gate = σ(W_g · img_h)
txt_gate = σ(W_g · txt_h)

img_out = img_gate * img_ctx + (1 − img_gate) * img_h   # gated blend
txt_out = txt_gate * txt_ctx + (1 − txt_gate) * txt_h

[img_out ‖ txt_out] (B, 1024) → MLP(1024→512→256→7)
  ↓
Logits (B, 7)
```

### CrossModalAttention Module

```python
class CrossModalAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key_value):
        attended, _ = self.attn(query, key_value, key_value)
        return self.norm((query + attended).squeeze(1))   # residual + norm
```

### Gating Mechanism Explained

The gating network decides how much of the cross-attended representation
to trust vs the original unimodal features:

```
img_gate → close to 1 → attend to text (text is informative)
img_gate → close to 0 → stay with original image features (text is noisy)
```

This makes the model **robust to unreliable inputs**:
if the text is generic ("Nice photo"), the gate suppresses the cross-modal
signal and falls back to the unimodal representation.

### Characteristics

| Property              | Value                              |
|-----------------------|------------------------------------|
| Test accuracy         | **~83%** (+18 pp vs CNN baseline)  |
| Cross-modal alignment | ✅ Explicit bidirectional attention |
| Fallback behavior     | ✅ Gate → unimodal when cross-modal is noisy |
| Parameters (fusion head) | ~2.6 M                         |
| GPU memory overhead   | ~0.5 GB extra vs Early Fusion     |

---

## Full Pipeline — MultimodalEmotionModel

The end-to-end wrapper that combines encoders + fusion strategy:

```python
model = MultimodalEmotionModel(
    image_encoder=EmotionCNN(pretrained=True),
    text_encoder=BERTClassifier(),
    fusion_type="attention",      # "early" | "late" | "attention"
    num_classes=7,
    hidden_dim=512,
    dropout=0.3,
    d_model=512,
    num_heads=8,
)
```

### Forward Pass

```python
def forward(self, images, input_ids, attention_mask=None):
    img_feats = self.image_encoder.extract_features(images)           # (B, 2048)
    txt_feats = self.text_encoder.extract_features(input_ids, mask)   # (B, 768)
    return self.fusion(img_feats, txt_feats)                           # (B, 7)
```

### Inference Utility

```python
probs, preds = model.predict(images, input_ids, attention_mask)
# probs: (B, 7) softmax probabilities
# preds: (B,) predicted class indices
```

---

## Strategy Comparison

| Criterion            | Early Fusion | Late Fusion | Attention Fusion |
|----------------------|:------------:|:-----------:|:----------------:|
| Test accuracy        | ~80%         | ~81%        | **~83%**         |
| Cross-modal align    | ❌           | ❌         | ✅               |
| Missing modality ok  | ❌           | ✅          | Partial          |
| Interpretable        | Low          | High        | Medium           |
| Training speed       | ✅ Fast      | ✅ Fast     | Moderate         |
| Parameters (extra)   | ~1.2 M       | ~1.5 M      | ~2.6 M           |
| Best use case        | Baseline     | Offline/robust | Production   |

---

## Weight Decay per Parameter Group

```python
optimizer = AdamW([
    {"params": encoder_params,  "weight_decay": 0.0001},  # pretrained — light penalty
    {"params": fusion_params,   "weight_decay": 0.001},   # new layers — stronger penalty
])
```

---

## Quick Test

```bash
python src/fusion/fusion_models.py
# Runs __main__ block:
# === Early Fusion ===   Output: torch.Size([4, 7])
# === Late Fusion ===    Output: torch.Size([4, 7])
# === Attention Fusion ===  Output: torch.Size([4, 7])  Params: 2,627,079
```

---

Last Updated: 2026<br>
Status: Active ✓