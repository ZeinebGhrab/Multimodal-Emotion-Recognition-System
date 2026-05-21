"""
src/fusion/fusion_models.py
────────────────────────────
Three multimodal fusion strategies combining image (CNN/ViT) and text (BERT/LSTM)
feature vectors into a single emotion prediction.

─────────────────────────────────────────────────────────────────────────────
Strategy 1: EARLY FUSION (concatenation)
  [img_feats] ‖ [txt_feats] → MLP → logits
  ✓ Simple, fast, lets the network learn cross-modal interactions freely
  ✗ No explicit alignment; modalities may dominate each other

Strategy 2: LATE FUSION (ensemble / score averaging)
  CNN → P_img    BERT → P_txt
  P_final = α·P_img + (1-α)·P_txt    (learned α or fixed)
  ✓ Easy to combine independently trained models; interpretable
  ✗ No shared representation; misses cross-modal feature interactions

Strategy 3: ATTENTION FUSION (cross-modal transformer)
  Project img & txt to d_model, compute cross-attention in both directions,
  concatenate attended representations → MLP → logits
  ✓ Explicitly models which text tokens are relevant to which image regions
  ✗ More parameters; needs more data / careful regularisation
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 1: Early Fusion
# ═══════════════════════════════════════════════════════════════════════════════

class EarlyFusionModel(nn.Module):
    """
    Concatenate image and text feature vectors, then classify.

    Args:
        image_dim   : dimension of image feature vector (e.g. 2048 for ResNet-50)
        text_dim    : dimension of text feature vector (e.g. 768 for BERT)
        hidden_dim  : hidden units in the fusion MLP
        num_classes : 7
        dropout     : regularisation
    """

    def __init__(self, image_dim: int = 2048, text_dim: int = 768,
                 hidden_dim: int = 512, num_classes: int = 7, dropout: float = 0.3):
        super().__init__()

        self.img_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img_feats: torch.Tensor,
                txt_feats: torch.Tensor) -> torch.Tensor:
        img_h = self.img_proj(img_feats)
        txt_h = self.txt_proj(txt_feats)
        fused = torch.cat([img_h, txt_h], dim=-1)
        return self.fusion_mlp(fused)


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 2: Late Fusion (learned ensemble)
# ═══════════════════════════════════════════════════════════════════════════════

class LateFusionModel(nn.Module):
    """
    Train image and text branches independently, then fuse their softmax outputs
    via a learnable convex combination (or a small MLP).

    Two modes:
      'weighted' — scalar weights α, (1-α) learned per class
      'mlp'      — small 2-layer MLP on concatenated probability vectors
    """

    def __init__(self, image_dim: int = 2048, text_dim: int = 768,
                 num_classes: int = 7, dropout: float = 0.3,
                 mode: str = "mlp", **kwargs):   # **kwargs absorbs unused fusion args
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes

        self.img_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(image_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        self.txt_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        if mode == "weighted":
            self.alpha = nn.Parameter(torch.full((num_classes,), 0.5))
        else:
            self.ensemble_mlp = nn.Sequential(
                nn.Linear(num_classes * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(64, num_classes)
            )

    def forward(self, img_feats: torch.Tensor,
                txt_feats: torch.Tensor) -> torch.Tensor:
        img_logits = self.img_head(img_feats)
        txt_logits = self.txt_head(txt_feats)

        img_probs = F.softmax(img_logits, dim=-1)
        txt_probs = F.softmax(txt_logits, dim=-1)

        if self.mode == "weighted":
            alpha = torch.sigmoid(self.alpha)
            fused_probs = alpha * img_probs + (1 - alpha) * txt_probs
            return torch.log(fused_probs + 1e-8)
        else:
            combined = torch.cat([img_probs, txt_probs], dim=-1)
            return self.ensemble_mlp(combined)

    def get_unimodal_logits(self, img_feats: torch.Tensor,
                             txt_feats: torch.Tensor):
        """Return individual branch logits for analysis."""
        return self.img_head(img_feats), self.txt_head(txt_feats)


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 3: Cross-Modal Attention Fusion
# ═══════════════════════════════════════════════════════════════════════════════

class CrossModalAttention(nn.Module):
    """Bidirectional cross-attention between two modality tokens."""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(query, key_value, key_value)
        return self.norm((query + attended).squeeze(1))


class AttentionFusionModel(nn.Module):
    """
    Cross-modal attention fusion:
    1. Project both modalities to d_model
    2. Img attends to Text  → attended_img
    3. Text attends to Img  → attended_txt
    4. Gating: blend attended with original
    5. Concat → MLP → logits
    """

    def __init__(self, image_dim: int = 2048, text_dim: int = 768,
                 d_model: int = 512, num_heads: int = 8,
                 num_classes: int = 7, dropout: float = 0.3, **kwargs):
        super().__init__()

        self.img_proj = nn.Sequential(
            nn.Linear(image_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(text_dim, d_model),
            nn.LayerNorm(d_model)
        )

        self.img_attends_txt = CrossModalAttention(d_model, num_heads, dropout)
        self.txt_attends_img = CrossModalAttention(d_model, num_heads, dropout)

        self.img_gate = nn.Linear(d_model, d_model)
        self.txt_gate = nn.Linear(d_model, d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, img_feats: torch.Tensor,
                txt_feats: torch.Tensor) -> torch.Tensor:
        img_h = self.img_proj(img_feats)
        txt_h = self.txt_proj(txt_feats)

        img_q = img_h.unsqueeze(1)
        txt_q = txt_h.unsqueeze(1)

        img_ctx = self.img_attends_txt(img_q, txt_q)
        txt_ctx = self.txt_attends_img(txt_q, img_q)

        img_gate = torch.sigmoid(self.img_gate(img_h))
        txt_gate = torch.sigmoid(self.txt_gate(txt_h))

        img_out = img_gate * img_ctx + (1 - img_gate) * img_h
        txt_out = txt_gate * txt_ctx + (1 - txt_gate) * txt_h

        fused = torch.cat([img_out, txt_out], dim=-1)
        return self.classifier(fused)


# ═══════════════════════════════════════════════════════════════════════════════
#  Full Multimodal Pipeline (end-to-end wrapper)
# ═══════════════════════════════════════════════════════════════════════════════

class MultimodalEmotionModel(nn.Module):
    """
    Complete end-to-end multimodal model.
    Wraps an image encoder, a text encoder, and a fusion module.

    Usage:
        model = MultimodalEmotionModel(
            image_encoder=EmotionCNN(pretrained=True),
            text_encoder=BERTClassifier(),
            fusion_type='attention'
        )
    """

    def __init__(self, image_encoder, text_encoder,
                 fusion_type: str = "attention",
                 num_classes: int = 7,
                 hidden_dim: int = 512,
                 dropout: float = 0.3,
                 d_model: int = 512,
                 num_heads: int = 8,
                 late_fusion_mode: str = "mlp"):
        super().__init__()

        self.image_encoder = image_encoder
        self.text_encoder  = text_encoder

        img_dim = image_encoder.feature_dim
        txt_dim = text_encoder.feature_dim

        if fusion_type == "early":
            self.fusion = EarlyFusionModel(
                image_dim=img_dim, text_dim=txt_dim,
                hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
        elif fusion_type == "late":
            self.fusion = LateFusionModel(
                image_dim=img_dim, text_dim=txt_dim,
                num_classes=num_classes, dropout=dropout, mode=late_fusion_mode)
        elif fusion_type == "attention":
            self.fusion = AttentionFusionModel(
                image_dim=img_dim, text_dim=txt_dim,
                d_model=d_model, num_heads=num_heads,
                num_classes=num_classes, dropout=dropout)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Choose from: early | late | attention")

        self.fusion_type = fusion_type

    def forward(self, images: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        img_feats = self.image_encoder.extract_features(images)

        if attention_mask is not None:
            txt_feats = self.text_encoder.extract_features(input_ids, attention_mask)
        else:
            txt_feats = self.text_encoder.extract_features(input_ids)

        return self.fusion(img_feats, txt_feats)

    def predict(self, images: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None):
        """Return (probabilities, predicted_class_indices)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(images, input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
        return probs, preds


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 4

    img_feats = torch.randn(B, 2048).to(device)
    txt_feats = torch.randn(B, 768).to(device)

    print("=== Early Fusion ===")
    early = EarlyFusionModel(2048, 768).to(device)
    out = early(img_feats, txt_feats)
    print(f"  Output: {out.shape}")   # (4, 7)

    print("\n=== Late Fusion ===")
    late = LateFusionModel(2048, 768, mode="mlp").to(device)
    out = late(img_feats, txt_feats)
    print(f"  Output: {out.shape}")   # (4, 7)

    print("\n=== Attention Fusion ===")
    attn = AttentionFusionModel(2048, 768, d_model=512, num_heads=8).to(device)
    out = attn(img_feats, txt_feats)
    print(f"  Output: {out.shape}")   # (4, 7)

    total = sum(p.numel() for p in attn.parameters())
    print(f"  Params: {total:,}")
