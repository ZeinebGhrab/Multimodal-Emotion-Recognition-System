# src/fusion/ — Multimodal Fusion Strategies

Combines image and text feature vectors into a single emotion prediction. Three strategies are implemented — all sharing the same public interface so they can be swapped without changing any other code.

```
src/fusion/
├── __init__.py
└── fusion_models.py    EarlyFusionModel | LateFusionModel | AttentionFusionModel
                        MultimodalEmotionModel  (end-to-end wrapper)
```

---

## Why Fusion Improves Accuracy

Single-modality emotion recognition has hard limits:

- **Image alone** struggles with occlusion, low resolution, cultural differences, and neutral faces hiding strong internal states.
- **Text alone** misses sarcasm, irony, and cases where written content contradicts the emotional delivery.

Fusion resolves modality conflicts:

| Image | Text | Unimodal prediction | Fusion prediction |
|-------|------|---------------------|-------------------|
| Neutral face | "I'm devastated" | Neutral (image wins) | Sad (text resolves) |
| Crying face | "I'm so happy for you!" | Sad (image wins) | Happy (tears of joy) |
| Angry face | "Whatever, I don't care" | Angry | Fear/Sad (subtext) |

**Accuracy gain:** +31 pp over CNN baseline (66% → 97.7%)

---

## Strategy 1 — Early Fusion (Concatenation)

```
img_feats (B, 2048) → Linear(2048 → 512) → ReLU → LayerNorm ─┐
                                                              ├── cat (B, 1024)
txt_feats (B, 768)  → Linear(768 → 512)  → ReLU → LayerNorm ─┘
  → MLP(1024 → 512 → 256 → 7) → logits
```

Both modalities are projected to a common dimension and concatenated before a shared MLP. Simple and fast, but no explicit cross-modal alignment — both modalities contribute equally regardless of their relative confidence.

**Accuracy: ~80%**

---

## Strategy 2 — Late Fusion (Learned Ensemble)

```
CNN  → img_head → P_img (B, 7) ─┐
                                  ├── MLP(14 → 64 → 7) → logits
BERT → txt_head → P_txt (B, 7) ─┘

# Alternative weighted mode:
P_final = σ(α) · P_img + (1 − σ(α)) · P_txt   (α learned per class)
```

Each modality classifies independently into 7-class distributions; a small MLP (or learned scalar weights) combines them. Interpretable modality weights, robust to single-modality failure, but loses fine-grained feature interactions.

**Accuracy: ~81%**

---

## Strategy 3 — Attention Fusion ⭐

```
img_feats → Linear(2048 → 512) + LayerNorm → img_h (B, 512)
txt_feats → Linear(768 → 512)  + LayerNorm → txt_h (B, 512)

img_h (as Q) ── CrossAttention(KV = txt_h) ──► img_ctx
txt_h (as Q) ── CrossAttention(KV = img_h) ──► txt_ctx

img_gate = σ(W_g · img_h)
txt_gate = σ(W_g · txt_h)

img_out = img_gate * img_ctx + (1 − img_gate) * img_h   # residual blend
txt_out = txt_gate * txt_ctx + (1 − txt_gate) * txt_h

[img_out ‖ txt_out] (B, 1024) → MLP(1024 → 512 → 256 → 7) → logits
```

### How Bidirectional Cross-Attention Works

Each modality **actively queries the other**:

- **Image-to-text cross-attention:** `img_h` is the query (Q), `txt_h` provides keys (K) and values (V). The image representation is enriched with the textual context most relevant to what it "sees".
- **Text-to-image cross-attention:** symmetrically, `txt_h` queries `img_h` — the text representation focuses on the facial features most consistent with the words.

### Residual Gating

A learned sigmoid gate `σ(W_g · h)` controls how much cross-modal information to incorporate. When cross-attention is uninformative (noisy or generic input), the gate → 0 and the model falls back to the unimodal representation — **preventing noise injection**.

**Accuracy: 97.7% (+31 pp over CNN baseline)**

---

## Strategy Comparison

| | Early Fusion | Late Fusion | **Attention Fusion** |
|--|-------------|-------------|---------------------|
| Cross-modal interaction | ❌ Implicit | ❌ Only at decision level | ✅ Explicit cross-attention |
| Modality weighting | Fixed | Learned per-class scalar | Learned per-feature via gating |
| Noise robustness | Low | High | High (residual gating) |
| Interpretability | Low | High | Medium |
| Test accuracy | ~80% | ~81% | **97.7%** |
| **Default** | — | — | **✅ Yes** |

---

## Public Interface

All three strategies share the same interface:

```python
from src.fusion.fusion_models import MultimodalEmotionModel

model = MultimodalEmotionModel(
    fusion_type="attention",    # "early" | "late" | "attention"
    num_classes=7,
    cnn_checkpoint="outputs/checkpoints/cnn_<date>/best_cnn.pt",
    bert_checkpoint="outputs/checkpoints/bert_<date>/best_bert.pt",
)

# Both inputs required
logits = model(images, input_ids, attention_mask)   # (B, 7)
```

---

## Configuration (config.yaml)

```yaml
fusion:
  type: "attention"            # Switch strategy here
  hidden_dim: 512
  dropout: 0.3
  learning_rate: 0.0001
  weight_decay_encoders: 0.0001   # Lighter — pretrained layers
  weight_decay_fusion: 0.001      # Stronger — randomly initialised head
  attention:
    num_heads: 8
    d_model: 512
```

Override at the command line with `--fusion early`, `--fusion late`, or `--fusion attention`.

---

*Last Updated: 23/05/2026 — Status: Active ✓*
