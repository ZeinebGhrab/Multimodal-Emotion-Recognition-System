"""
src/models/cnn_model.py
───────────────────────
ResNet-50 fine-tuned for facial emotion classification.

Architecture decision:
  - Use ImageNet-pretrained ResNet-50 as backbone (transfer learning)
  - Replace the final FC layer with: Dropout → Linear → 7 classes
  - Fine-tune the entire network with a low learning rate
  - Alternative: freeze backbone and train only the head (faster, worse accuracy)

Why ResNet-50?
  - Deep residual connections solve vanishing gradients in deep nets
  - Strong pretrained features generalise well to facial textures
  - Good balance of accuracy vs. compute (vs ResNet-101/152)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class EmotionCNN(nn.Module):
    """
    ResNet-50 backbone with a custom classification head.

    Args:
        num_classes : number of emotion classes (default 7)
        dropout     : dropout probability before the final layer
        pretrained  : use ImageNet weights
        freeze_bn   : freeze BatchNorm layers during fine-tuning
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.5,
                 pretrained: bool = True, freeze_bn: bool = False):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # ── Backbone ──────────────────────────────────────────────────────────
        # Remove the original FC layer; keep everything up to the avg pool.
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        # Output: (batch, 2048, 1, 1) → flatten to (batch, 2048)
        self.feature_dim = backbone.fc.in_features   # 2048

        # Optionally freeze batch normalisation layers
        if freeze_bn:
            for module in self.features.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    for p in module.parameters():
                        p.requires_grad = False

        # ── Classification head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes)
        )

        # ── Weight init for new layers ────────────────────────────────────────
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the 2048-d feature vector (before classifier).
        Used by the multimodal fusion model.
        """
        out = self.features(x)           # (B, 2048, 1, 1)
        return out.flatten(1)            # (B, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)  # (B, 2048)
        return self.classifier(features)     # (B, num_classes)


# ─── Training utilities ───────────────────────────────────────────────────────

def build_optimizer(model: EmotionCNN, lr: float = 1e-4,
                    backbone_lr_factor: float = 0.1):
    """
    Two-group optimizer: backbone at lr*0.1, head at lr.
    Standard practice for fine-tuning pretrained models.
    """
    backbone_params = list(model.features.parameters())
    head_params     = list(model.classifier.parameters())

    return torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * backbone_lr_factor},
        {"params": head_params,     "lr": lr},
    ], weight_decay=1e-4)


def build_scheduler(optimizer, epochs: int, warmup_epochs: int = 5):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """Single training epoch with optional AMP (Automatic Mixed Precision)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluation on val/test split. Returns loss and accuracy."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = EmotionCNN(num_classes=7, dropout=0.5, pretrained=False).to(device)
    dummy = torch.randn(4, 3, 224, 224).to(device)

    # Forward pass
    logits = model(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {logits.shape}")   # (4, 7)

    # Feature extraction
    feats = model.extract_features(dummy)
    print(f"Features: {feats.shape}")  # (4, 2048)

    # Param count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")
