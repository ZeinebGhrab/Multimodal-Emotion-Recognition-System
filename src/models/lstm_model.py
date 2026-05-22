"""
src/models/lstm_model.py & bert_model.py
─────────────────────────────────────────
Two text-only emotion classifiers:

1. BiLSTMClassifier  — GloVe embeddings + Bidirectional LSTM + attention pooling
2. BERTClassifier    — bert-base-uncased + linear head (HuggingFace)

Architecture justifications:
─ Bi-LSTM: captures sequential context in both directions; GloVe gives semantic
  priors; self-attention pooling selects emotionally relevant tokens.
─ BERT: contextual embeddings handle polysemy and negation better than static
  word vectors; fine-tuning propagates emotion-specific gradients into all layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, get_linear_schedule_with_warmup


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Bidirectional LSTM Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionPooling(nn.Module):
    """
    Token-level self-attention to produce a single fixed-size vector.
    Learns which positions carry emotional signal.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            lstm_out : (B, T, H)
            mask     : (B, T) — 1 for real tokens, 0 for padding
        Returns:
            context  : (B, H) — weighted sum of token hidden states
        """
        scores = self.attn_weights(lstm_out).squeeze(-1)   # (B, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        return (weights * lstm_out).sum(dim=1)             # (B, H)


class BiLSTMClassifier(nn.Module):
    """
    Bi-directional LSTM emotion classifier.

    Architecture:
      Embedding (GloVe init) → BiLSTM (2 layers) → Attention Pooling
      → LayerNorm → Dropout → FC → 7-class logits

    Args:
        vocab_size    : vocabulary size (including PAD/UNK)
        embed_dim     : embedding dimension (100 for GloVe-100d)
        hidden_size   : LSTM hidden units per direction (256 → 512 total)
        num_layers    : stacked LSTM layers
        num_classes   : 7 emotion classes
        dropout       : applied between layers and before classifier
        pretrained_emb: optional FloatTensor (vocab_size, embed_dim)
        freeze_emb    : whether to freeze embedding weights
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 100,
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 num_classes: int = 7,
                 dropout: float = 0.4,
                 pretrained_emb: torch.Tensor | None = None,
                 freeze_emb: bool = False):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_emb is not None:
            self.embedding.weight = nn.Parameter(pretrained_emb)
            if freeze_emb:
                self.embedding.weight.requires_grad = False

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.feature_dim = hidden_size * 2   # bidirectional

        # Attention pooling
        self.attention = AttentionPooling(self.feature_dim)

        # Classifier head
        self.norm = nn.LayerNorm(self.feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def extract_features(self, input_ids: torch.Tensor,
                         mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return the attention-pooled LSTM representation (B, hidden*2).

        Args:
            input_ids : (B, T) token indices
            mask      : (B, T) optional attention mask — 1 for real tokens,
                        0 for padding. When omitted, derived from input_ids != 0.
        """
        emb = self.dropout(self.embedding(input_ids))        # (B, T, E)
        lstm_out, _ = self.lstm(emb)                         # (B, T, H*2)
        if mask is None:
            mask = (input_ids != 0).long()                   # (B, T)
        pooled = self.attention(lstm_out, mask)               # (B, H*2)
        return self.norm(pooled)

    def forward(self, input_ids: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            input_ids : (B, T) token indices
            mask      : (B, T) optional padding mask from the DataLoader
                        (1 = real token, 0 = pad). Passed through to
                        AttentionPooling so padded positions are excluded
                        from the attention softmax.
        """
        features = self.extract_features(input_ids, mask)    # (B, H*2)
        return self.fc(self.dropout(features))               # (B, num_classes)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. BERT Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class BERTClassifier(nn.Module):
    """
    Fine-tuned BERT for emotion classification.

    Architecture:
      BERT (bert-base-uncased) → [CLS] token → Dropout → Linear → 7 classes

    We use the [CLS] representation which aggregates sentence-level semantics
    through BERT's self-attention across all 12 layers.

    Args:
        model_name  : HuggingFace model identifier
        num_classes : 7 emotion classes
        dropout     : applied before classifier head
        freeze_bert : train only the head (faster, lower accuracy)
    """

    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 num_classes: int = 7,
                 dropout: float = 0.3,
                 freeze_bert: bool = False):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name)
        self.feature_dim = self.bert.config.hidden_size  # 768

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, input_ids: torch.Tensor,
                          attention_mask: torch.Tensor) -> torch.Tensor:
        """Return [CLS] embedding (B, 768)."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]   # [CLS] token

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        cls_emb = self.extract_features(input_ids, attention_mask)  # (B, 768)
        return self.classifier(self.dropout(cls_emb))               # (B, 7)


# ─── Optimizer / scheduler helpers ────────────────────────────────────────────

def build_bert_optimizer(model: BERTClassifier,
                          lr: float = 2e-5,
                          weight_decay: float = 1e-2,
                          no_decay_params: list[str] | None = None):
    """
    AdamW with configurable weight decay (L2 regularisation).

    The standard BERT fine-tuning recipe exempts bias terms and LayerNorm
    weights from weight decay — applying L2 to these provides no benefit
    and can destabilise training.

    Args:
        weight_decay     : L2 penalty for regular parameters (from bert.weight_decay
                           in config.yaml). Typical value: 0.01
        no_decay_params  : param name substrings that should NOT receive weight
                           decay (from bert.no_decay_params in config.yaml).
                           Defaults to ["bias", "LayerNorm.weight"]
    """
    if no_decay_params is None:
        no_decay_params = ["bias", "LayerNorm.weight"]

    bert_decay, bert_no_decay = [], []
    for name, param in model.bert.named_parameters():
        if any(nd in name for nd in no_decay_params):
            bert_no_decay.append(param)
        else:
            bert_decay.append(param)

    optimizer_groups = [
        # BERT params WITH weight decay
        {"params": bert_decay,
         "lr": lr, "weight_decay": weight_decay},
        # BERT params WITHOUT weight decay (bias / LayerNorm)
        {"params": bert_no_decay,
         "lr": lr, "weight_decay": 0.0},
        # Classifier head — always apply weight decay
        {"params": model.classifier.parameters(),
         "lr": lr, "weight_decay": weight_decay},
    ]
    return torch.optim.AdamW(optimizer_groups)


def build_bert_scheduler(optimizer, num_training_steps: int,
                          warmup_steps: int = 500):
    """Linear warmup → linear decay (HuggingFace recommendation for BERT)."""
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )


# ─── Shared training loop (works for both models) ─────────────────────────────

def train_text_epoch(model, loader, optimizer, criterion, device,
                     model_type: str = "bert"):
    """Generic training epoch for LSTM or BERT."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        labels = batch["label"].to(device)
        optimizer.zero_grad()

        if model_type == "bert":
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )
        else:
            logits = model(batch["input_ids"].to(device))

        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_text(model, loader, criterion, device, model_type: str = "bert"):
    """Evaluation loop for text models."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        labels = batch["label"].to(device)

        if model_type == "bert":
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )
        else:
            logits = model(batch["input_ids"].to(device))

        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T = 4, 64

    # BiLSTM test
    lstm_model = BiLSTMClassifier(vocab_size=5000, embed_dim=100,
                                   hidden_size=256, num_classes=7).to(device)
    ids  = torch.randint(0, 5000, (B, T)).to(device)
    mask = (ids != 0).long()
    out  = lstm_model(ids, mask)
    print(f"BiLSTM output (with mask) : {out.shape}")   # (4, 7)
    out2 = lstm_model(ids)
    print(f"BiLSTM output (no mask)   : {out2.shape}")  # (4, 7)
    feats = lstm_model.extract_features(ids, mask)
    print(f"BiLSTM feats              : {feats.shape}")  # (4, 512)

    # BERT test (without downloading weights)
    config = BertConfig(vocab_size=1000, hidden_size=128, num_hidden_layers=2,
                        num_attention_heads=4, intermediate_size=256)
    bert_model = BERTClassifier.__new__(BERTClassifier)
    bert_model.bert = BertModel(config)
    bert_model.feature_dim = 128
    bert_model.dropout = nn.Dropout(0.3)
    bert_model.classifier = nn.Sequential(nn.Linear(128, 7))
    bert_model = bert_model.to(device)

    # Optimizer — now reads weight_decay explicitly
    opt = build_bert_optimizer(bert_model, lr=2e-5, weight_decay=0.01)
    for i, g in enumerate(opt.param_groups):
        print(f"  group {i}: lr={g['lr']}  weight_decay={g['weight_decay']}  "
              f"params={len(g['params'])}")

    inp = torch.randint(0, 1000, (B, T)).to(device)
    mask = torch.ones(B, T, dtype=torch.long).to(device)
    out = bert_model(inp, mask)
    print(f"BERT output  : {out.shape}")  # (4, 7)