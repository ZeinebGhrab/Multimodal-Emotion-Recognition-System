"""
src/models/vit_model.py
────────────────────────
Vision Transformer (ViT-B/16) fine-tuned for facial emotion classification.

Architecture:
  - HuggingFace ViT-base-patch16-224 pretrained on ImageNet-21k
  - Replace CLS classification head with: Dropout → Linear(768→7)
  - Fine-tune with low learning rate (2e-5) for stable convergence

Why ViT over CNN?
  - Self-attention captures global facial structure (e.g. brow + mouth together)
  - Patch embeddings encode spatial relationships without inductive bias
  - Outperforms ResNet-50 on FER2013 when pretrained on large datasets
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class EmotionViT(nn.Module):
    """
    ViT-B/16 backbone with a custom classification head for emotion recognition.

    Args:
        model_name  : HuggingFace model identifier
        num_classes : number of emotion classes (default 7)
        dropout     : dropout before the classification head
        pretrained  : load pretrained ImageNet-21k weights
    """

    def __init__(self,
                 model_name: str = "google/vit-base-patch16-224",
                 num_classes: int = 7,
                 dropout: float = 0.1,
                 pretrained: bool = True):
        super().__init__()

        if pretrained:
            self.vit = ViTModel.from_pretrained(model_name)
        else:
            config = ViTConfig.from_pretrained(model_name)
            self.vit = ViTModel(config)

        hidden_size = self.vit.config.hidden_size   # 768 for ViT-B/16
        self.feature_dim = hidden_size

        # Classification head (replaces ViT's default pooler)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes)
        )

        self._init_head()

    def _init_head(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Return the CLS token embedding (B, 768).
        Used by the multimodal fusion model.

        Args:
            pixel_values : (B, 3, 224, 224) normalized image tensors
        """
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0, :]   # CLS token

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values : (B, 3, 224, 224)
        Returns:
            logits       : (B, num_classes)
        """
        cls_feats = self.extract_features(pixel_values)
        return self.classifier(cls_feats)


# ─── Training utilities ───────────────────────────────────────────────────────

def build_vit_optimizer(model: EmotionViT, lr: float = 2e-5,
                         backbone_lr_factor: float = 1.0):
    """
    Optimizer with optional layer-wise learning rate decay.
    ViT typically works best with uniform (or mildly decayed) LR.
    """
    backbone_params = list(model.vit.parameters())
    head_params     = list(model.classifier.parameters())

    return torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * backbone_lr_factor, "weight_decay": 0.01},
        {"params": head_params,     "lr": lr,                       "weight_decay": 0.01},
    ])


@torch.no_grad()
def evaluate_vit(model: EmotionViT, loader, criterion, device):
    """Evaluation loop for ViT."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load without downloading weights for unit test
    model = EmotionViT(pretrained=False).to(device)
    dummy = torch.randn(2, 3, 224, 224).to(device)

    logits = model(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {logits.shape}")       # (2, 7)

    feats = model.extract_features(dummy)
    print(f"Features: {feats.shape}")       # (2, 768)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")
