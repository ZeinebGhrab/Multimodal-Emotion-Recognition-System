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

        fused_dim = image_dim + text_dim

        # Project each modality to the same hidden space before concatenation
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

        # Fusion MLP (on concatenated projections)
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
        """
        Args:
            img_feats : (B, image_dim)
            txt_feats : (B, text_dim)
        Returns:
            logits    : (B, num_classes)
        """
        img_h = self.img_proj(img_feats)          # (B, hidden)
        txt_h = self.txt_proj(txt_feats)          # (B, hidden)
        fused = torch.cat([img_h, txt_h], dim=-1) # (B, 2*hidden)
        return self.fusion_mlp(fused)             # (B, num_classes)


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
                 mode: str = "mlp"):
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes

        # Independent classifiers
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
            # Learnable per-class weight in [0,1]
            self.alpha = nn.Parameter(torch.full((num_classes,), 0.5))
        else:
            # Small MLP on concatenated softmax probabilities
            self.ensemble_mlp = nn.Sequential(
                nn.Linear(num_classes * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(64, num_classes)
            )

    def forward(self, img_feats: torch.Tensor,
                txt_feats: torch.Tensor) -> torch.Tensor:
        img_logits = self.img_head(img_feats)   # (B, C)
        txt_logits = self.txt_head(txt_feats)   # (B, C)

        img_probs = F.softmax(img_logits, dim=-1)
        txt_probs = F.softmax(txt_logits, dim=-1)

        if self.mode == "weighted":
            alpha = torch.sigmoid(self.alpha)   # (C,)
            fused_probs = alpha * img_probs + (1 - alpha) * txt_probs
            return torch.log(fused_probs + 1e-8)   # log-probs for NLL loss
        else:
            combined = torch.cat([img_probs, txt_probs], dim=-1)  # (B, 2C)
            return self.ensemble_mlp(combined)                    # (B, C)

    def get_unimodal_logits(self, img_feats: torch.Tensor,
                             txt_feats: torch.Tensor):
        """Return individual branch logits for analysis."""
        return self.img_head(img_feats), self.txt_head(txt_feats)


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 3: Cross-Modal Attention Fusion
# ═══════════════════════════════════════════════════════════════════════════════

class CrossModalAttention(nn.Module):
    """
    Single-head cross-attention: query from modality A, key/value from modality B.
    Used symmetrically in both directions (img→txt and txt→img).
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query     : (B, 1, d_model) — attending modality
            key_value : (B, 1, d_model) — attended modality
        Returns:
            (B, d_model) — query updated by attending to key_value
        """
        attended, _ = self.attn(query, key_value, key_value)
        return self.norm((query + attended).squeeze(1))   # residual + norm


class AttentionFusionModel(nn.Module):
    """
    Cross-modal attention fusion:

    1. Project both modalities to d_model
    2. Img attends to Text  → attended_img
    3. Text attends to Img  → attended_txt
    4. Concat [attended_img || attended_txt] → MLP → logits

    This captures "which text tokens reinforce what we see in the face" and vice versa.

    Args:
        image_dim  : image feature dim
        text_dim   : text feature dim
        d_model    : internal attention dimension (must be divisible by num_heads)
        num_heads  : multi-head attention heads
        num_classes: 7
        dropout    : regularisation
    """

    def __init__(self, image_dim: int = 2048, text_dim: int = 768,
                 d_model: int = 512, num_heads: int = 8,
                 num_classes: int = 7, dropout: float = 0.3):
        super().__init__()

        # Projection to common space
        self.img_proj = nn.Sequential(
            nn.Linear(image_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(text_dim, d_model),
            nn.LayerNorm(d_model)
        )

        # Cross-attention (bidirectional)
        self.img_attends_txt = CrossModalAttention(d_model, num_heads, dropout)
        self.txt_attends_img = CrossModalAttention(d_model, num_heads, dropout)

        # Gating: learn how much each attended representation to keep
        self.img_gate = nn.Linear(d_model, d_model)
        self.txt_gate = nn.Linear(d_model, d_model)

        # Classifier MLP
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
        """
        Args:
            img_feats : (B, image_dim)
            txt_feats : (B, text_dim)
        Returns:
            logits    : (B, num_classes)
        """
        img_h = self.img_proj(img_feats)       # (B, d_model)
        txt_h = self.txt_proj(txt_feats)       # (B, d_model)

        # Unsqueeze to (B, 1, d_model) for MultiheadAttention
        img_q = img_h.unsqueeze(1)
        txt_q = txt_h.unsqueeze(1)

        # Cross-attention
        img_ctx = self.img_attends_txt(img_q, txt_q)   # (B, d_model)
        txt_ctx = self.txt_attends_img(txt_q, img_q)   # (B, d_model)

        # Gating mechanism (element-wise sigmoid gate)
        img_gate = torch.sigmoid(self.img_gate(img_h))
        txt_gate = torch.sigmoid(self.txt_gate(txt_h))

        img_out = img_gate * img_ctx + (1 - img_gate) * img_h
        txt_out = txt_gate * txt_ctx + (1 - txt_gate) * txt_h

        fused = torch.cat([img_out, txt_out], dim=-1)  # (B, 2*d_model)
        return self.classifier(fused)                  # (B, num_classes)


# ═══════════════════════════════════════════════════════════════════════════════
#  Full Multimodal Pipeline (end-to-end wrapper)
# ═══════════════════════════════════════════════════════════════════════════════

class MultimodalEmotionModel(nn.Module):
    """
    Complete end-to-end multimodal model.

    Wraps an image encoder, a text encoder, and a fusion module.
    Keeps encoders as sub-modules so gradients flow through them.

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
                 **fusion_kwargs):
        super().__init__()

        self.image_encoder = image_encoder
        self.text_encoder  = text_encoder

        img_dim = image_encoder.feature_dim   # e.g. 2048
        txt_dim = text_encoder.feature_dim    # e.g. 768

        if fusion_type == "early":
            self.fusion = EarlyFusionModel(
                img_dim, txt_dim, num_classes=num_classes, **fusion_kwargs)
        elif fusion_type == "late":
            self.fusion = LateFusionModel(
                img_dim, txt_dim, num_classes=num_classes, **fusion_kwargs)
        elif fusion_type == "attention":
            self.fusion = AttentionFusionModel(
                img_dim, txt_dim, num_classes=num_classes, **fusion_kwargs)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.fusion_type = fusion_type

    def forward(self, images: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            images        : (B, 3, H, W)
            input_ids     : (B, seq_len)  token ids
            attention_mask: (B, seq_len)  for BERT; None for LSTM
        Returns:
            logits        : (B, num_classes)
        """
        img_feats = self.image_encoder.extract_features(images)

        # Support both BERT and LSTM text encoders
        if attention_mask is not None:
            txt_feats = self.text_encoder.extract_features(input_ids, attention_mask)
        else:
            txt_feats = self.text_encoder.extract_features(input_ids)

        return self.fusion(img_feats, txt_feats)


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
