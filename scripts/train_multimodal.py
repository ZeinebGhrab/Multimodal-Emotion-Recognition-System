"""
scripts/train_multimodal.py
────────────────────────────
End-to-end training script for the multimodal fusion model.

Usage:
    python scripts/train_multimodal.py --fusion attention --epochs 20
    python scripts/train_multimodal.py --fusion early --epochs 15
    python scripts/train_multimodal.py --fusion late --no_finetune_encoders
    python scripts/train_multimodal.py --cnn_checkpoint outputs/checkpoints/cnn_xxx/best_cnn.pt \
                                        --bert_checkpoint outputs/checkpoints/bert_xxx/best_bert.pt
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn_model import EmotionCNN
from src.models.lstm_model import BERTClassifier
from src.fusion.fusion_models import MultimodalEmotionModel   # ← fixed import
from src.evaluation.metrics import compute_metrics, plot_confusion_matrix, plot_training_curves
from src.genai.report_generator import generate_emotion_report, print_report


# ─── Argument parser ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train multimodal emotion model")
    p.add_argument("--fusion", default="attention",
                   choices=["early", "late", "attention"],
                   help="Fusion strategy")
    p.add_argument("--epochs",              type=int,   default=20)
    p.add_argument("--batch_size",          type=int,   default=32)
    p.add_argument("--lr",                  type=float, default=1e-4)
    p.add_argument("--encoder_lr_factor",   type=float, default=0.1,
                   help="LR multiplier for pretrained encoders (< 1 recommended)")
    p.add_argument("--no_finetune_encoders", action="store_true",
                   help="Freeze encoder weights, train only fusion head")
    p.add_argument("--cnn_checkpoint",      default=None,
                   help="Path to pretrained CNN checkpoint (.pt)")
    p.add_argument("--bert_checkpoint",     default=None,
                   help="Path to pretrained BERT checkpoint (.pt)")
    p.add_argument("--config",              default="configs/config.yaml")
    p.add_argument("--device",              default="auto",
                   choices=["auto", "cuda", "cpu", "mps"])
    return p.parse_args()


# ─── Dummy multimodal dataset ─────────────────────────────────────────────────

class DummyMultimodalDataset(Dataset):
    """
    Placeholder dataset — replace with a real implementation.

    Real multimodal datasets can be constructed by:
      1. FER2013 + BLIP-2/LLaVA generated captions (image → text)
      2. CMU-MOSI / CMU-MOSEI (sentiment + audio + video)
      3. Label-aligned pairing: match FER2013 images to emotion-NLP samples
         by emotion class (same label → positive pair)
    """

    def __init__(self, size: int = 1000, img_size: int = 224,
                 seq_len: int = 128, num_classes: int = 7):
        self.size = size
        self.img_size = img_size
        self.seq_len = seq_len
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "image":          torch.randn(3, self.img_size, self.img_size),
            "input_ids":      torch.randint(0, 30000, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "label":          torch.randint(0, self.num_classes, ()),
        }


# ─── Training / evaluation loops ──────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = total_correct = total = 0

    for batch in loader:
        images = batch["image"].to(device)
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images, ids, mask)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss    += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total         += labels.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total = 0
    all_preds, all_labels = [], []

    for batch in loader:
        images = batch["image"].to(device)
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(images, ids, mask)
        loss   = criterion(logits, labels)

        total_loss    += loss.item() * labels.size(0)
        preds          = logits.argmax(1)
        total_correct += (preds == labels).sum().item()
        total         += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, total_correct / total, all_preds, all_labels


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps"  if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    print(f"[Multimodal Train] Device  : {device}")
    print(f"[Multimodal Train] Fusion  : {args.fusion}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["project"]["seed"])

    run_id  = f"{args.fusion}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(cfg["paths"]["checkpoints"]) / run_id
    fig_dir = Path(cfg["paths"]["figures"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Build encoders ─────────────────────────────────────────────────────
    print("[Multimodal Train] Building image encoder (ResNet-50) …")
    image_encoder = EmotionCNN(
        num_classes=cfg["emotions"]["num_classes"],
        dropout=cfg["cnn"]["dropout"],
        pretrained=cfg["cnn"]["pretrained"]
    )
    if args.cnn_checkpoint and os.path.exists(args.cnn_checkpoint):
        image_encoder.load_state_dict(
            torch.load(args.cnn_checkpoint, map_location="cpu"))
        print(f"  Loaded CNN checkpoint: {args.cnn_checkpoint}")

    print("[Multimodal Train] Building text encoder (BERT) …")
    text_encoder = BERTClassifier(
        model_name=cfg["bert"]["model_name"],
        num_classes=cfg["emotions"]["num_classes"],
        dropout=cfg["bert"]["dropout"]
    )
    if args.bert_checkpoint and os.path.exists(args.bert_checkpoint):
        # BERT checkpoints may be wrapped in a dict
        ckpt = torch.load(args.bert_checkpoint, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        text_encoder.load_state_dict(state)
        print(f"  Loaded BERT checkpoint: {args.bert_checkpoint}")

    # Optionally freeze encoders
    if args.no_finetune_encoders:
        for p in image_encoder.parameters():
            p.requires_grad = False
        for p in text_encoder.parameters():
            p.requires_grad = False
        print("[Multimodal Train] Encoder weights FROZEN — training fusion head only")

    # ── Build fusion model ─────────────────────────────────────────────────
    fuse_cfg = cfg["fusion"]
    model = MultimodalEmotionModel(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        fusion_type=args.fusion,
        num_classes=cfg["emotions"]["num_classes"],
        hidden_dim=fuse_cfg["hidden_dim"],
        dropout=fuse_cfg["dropout"],
        d_model=fuse_cfg["attention"]["d_model"],
        num_heads=fuse_cfg["attention"]["num_heads"]
    ).to(device)

    total    = sum(p.numel() for p in model.parameters())
    trainable= sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Multimodal Train] Total params    : {total:,}")
    print(f"[Multimodal Train] Trainable params: {trainable:,}")

    # ── Dataloaders ────────────────────────────────────────────────────────
    print("[Multimodal Train] Building data loaders (DUMMY — swap in real data) …")
    train_ds = DummyMultimodalDataset(2000, cfg["image"]["resize"], cfg["text"]["max_length"])
    val_ds   = DummyMultimodalDataset(400,  cfg["image"]["resize"], cfg["text"]["max_length"])
    test_ds  = DummyMultimodalDataset(400,  cfg["image"]["resize"], cfg["text"]["max_length"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Loss & optimizer ───────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    encoder_params = (list(model.image_encoder.parameters()) +
                      list(model.text_encoder.parameters()))
    fusion_params  = list(model.fusion.parameters())

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr * args.encoder_lr_factor,
         "weight_decay": 1e-4},
        {"params": fusion_params,  "lr": args.lr,
         "weight_decay": 1e-3},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    print(f"\n[Multimodal Train] Starting {args.fusion} fusion training "
          f"for {args.epochs} epochs\n")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, _, _ = evaluate_epoch(
            model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);    val_accs.append(vl_acc)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f} | "
              f"Val   loss: {vl_loss:.4f}  acc: {vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            ckpt_path = out_dir / "best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ New best model saved (val_acc={best_val_acc:.4f})")

    # ── Final evaluation ───────────────────────────────────────────────────
    print("\n[Eval] Loading best checkpoint for final test evaluation …")
    model.load_state_dict(
        torch.load(out_dir / "best_model.pt", map_location=device))
    _, test_acc, preds, labels = evaluate_epoch(
        model, test_loader, criterion, device)

    emotion_names = cfg["emotions"]["classes"]
    metrics = compute_metrics(labels, preds, emotion_names)

    print(f"\n[Eval] Test Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"[Eval] Macro F1      : {metrics['macro_f1']*100:.2f}%")

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items()
                   if k != "classification_report"}, f, indent=2)

    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        model_name=f"{args.fusion.capitalize()} Fusion",
        save_path=str(fig_dir / f"{run_id}_curves.png")
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"], emotion_names,
        title=f"{args.fusion.capitalize()} Fusion — Confusion Matrix",
        save_path=str(fig_dir / f"{run_id}_cm.png")
    )

    # ── Demo GenAI report ──────────────────────────────────────────────────
    print("\n[GenAI] Generating demo emotional report …")
    import random; random.seed(0)
    pred_emotion = emotion_names[preds[0]] if preds else "neutral"
    scores = {e: random.random() for e in emotion_names}
    scores[pred_emotion] += 1.0
    total_s = sum(scores.values())
    scores = {e: s / total_s for e, s in scores.items()}

    report = generate_emotion_report(
        emotion=pred_emotion,
        scores=scores,
        user_text="I feel quite overwhelmed lately.",
        use_llm=bool(os.getenv("ANTHROPIC_API_KEY")),
        save_path=str(out_dir / "sample_report.json")
    )
    print_report(report)

    print(f"\n[Done] Run artifacts saved to: {out_dir}/")


if __name__ == "__main__":
    main()
