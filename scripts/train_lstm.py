"""
scripts/train_lstm.py
──────────────────────
Training script for the BiLSTM + GloVe text-only emotion classifier.

Usage:
    python scripts/train_lstm.py
    python scripts/train_lstm.py --epochs 30 --lr 1e-3 --batch_size 128
    python scripts/train_lstm.py --no_glove   # random embeddings
"""

import os
import sys
import json
import argparse
import yaml
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lstm_model import BiLSTMClassifier
from src.preprocessing.text_preprocessing import (
    Vocabulary, LSTMEmotionDataset, load_glove_embeddings, load_emotion_csv
)
from src.evaluation.metrics import (
    compute_metrics, plot_confusion_matrix, plot_training_curves
)
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="Train BiLSTM emotion classifier")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--no_glove",   action="store_true",
                   help="Use random embeddings (skip GloVe loading)")
    p.add_argument("--freeze_emb", action="store_true",
                   help="Freeze embedding layer during training")
    p.add_argument("--device",     default="auto",
                   choices=["auto", "cuda", "cpu", "mps"])
    return p.parse_args()


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = total_correct = total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask      = batch["mask"].to(device)
        labels    = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, mask)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

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
        input_ids = batch["input_ids"].to(device)
        mask      = batch["mask"].to(device)
        labels    = batch["label"].to(device)

        logits = model(input_ids, mask)
        loss   = criterion(logits, labels)

        total_loss    += loss.item() * labels.size(0)
        preds          = logits.argmax(1)
        total_correct += (preds == labels).sum().item()
        total         += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, total_correct / total, all_preds, all_labels


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    lstm_cfg   = cfg["lstm"]
    epochs     = args.epochs    or lstm_cfg["epochs"]
    lr         = args.lr        or lstm_cfg["learning_rate"]
    batch_size = args.batch_size or lstm_cfg["batch_size"]
    max_length = cfg["text"]["max_length"]
    embed_dim  = cfg["text"]["embedding_dim"]
    glove_path = cfg["text"]["glove_path"]

    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps"  if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    print(f"[LSTM Train] Device: {device}\n")

    torch.manual_seed(cfg["project"]["seed"])

    run_id  = f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(cfg["paths"]["checkpoints"]) / run_id
    fig_dir = Path(cfg["paths"]["figures"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_csv = os.path.join(cfg["paths"]["data_raw"], "emotion_train.csv")

    if not os.path.exists(train_csv):
        print(f"[WARN] Dataset not found at {train_csv}. Using dummy data.\n")
        vocab_size = cfg["text"]["vocab_size"]
        vocab = None
        pretrained_emb = None

        class DummyLoader:
            def __iter__(self):
                for _ in range(10):
                    yield {
                        "input_ids": torch.randint(0, vocab_size, (batch_size, max_length)),
                        "mask":      torch.ones(batch_size, max_length, dtype=torch.long),
                        "label":     torch.randint(0, 7, (batch_size,))
                    }
            def __len__(self): return 10

        loaders = {"train": DummyLoader(), "val": DummyLoader(), "test": DummyLoader()}
    else:
        # Build vocabulary from training data
        train_df = load_emotion_csv(train_csv)
        vocab = Vocabulary(min_freq=2)
        vocab.build(train_df["text"].tolist())
        print(f"[LSTM Train] Vocabulary size: {len(vocab):,}")

        # Load GloVe if available
        pretrained_emb = None
        if not args.no_glove and os.path.exists(glove_path):
            print(f"[LSTM Train] Loading GloVe from {glove_path} …")
            pretrained_emb = load_glove_embeddings(vocab, glove_path, embed_dim)
        elif not args.no_glove:
            print(f"[WARN] GloVe not found at {glove_path}. Using random embeddings.")

        val_csv  = os.path.join(cfg["paths"]["data_raw"], "emotion_val.csv")
        test_csv = os.path.join(cfg["paths"]["data_raw"], "emotion_test.csv")

        train_ds = LSTMEmotionDataset(train_df, vocab, max_length)
        val_ds   = LSTMEmotionDataset(load_emotion_csv(val_csv),  vocab, max_length)
        test_ds  = LSTMEmotionDataset(load_emotion_csv(test_csv), vocab, max_length)

        loaders = {
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
            "val":   DataLoader(val_ds,   batch_size=batch_size, num_workers=2),
            "test":  DataLoader(test_ds,  batch_size=batch_size, num_workers=2),
        }

    # ── Model ─────────────────────────────────────────────────────────────────
    vocab_size = len(vocab) if vocab else cfg["text"]["vocab_size"]
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=lstm_cfg["hidden_size"],
        num_layers=lstm_cfg["num_layers"],
        num_classes=cfg["emotions"]["num_classes"],
        dropout=lstm_cfg["dropout"],
        pretrained_emb=pretrained_emb,
        freeze_emb=args.freeze_emb
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LSTM Train] Total params    : {total:,}")
    print(f"[LSTM Train] Trainable params: {trainable:,}\n")

    # ── Loss + optimizer ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    print(f"[LSTM Train] Starting training for {epochs} epochs …\n")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, loaders["train"], optimizer, None, criterion, device)
        vl_loss, vl_acc, _, _ = evaluate_epoch(
            model, loaders["val"], criterion, device)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);    val_accs.append(vl_acc)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f} | "
              f"Val   loss: {vl_loss:.4f}  acc: {vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab
            }, out_dir / "best_lstm.pt")
            print(f"  ✓ Best model saved (val_acc={best_val_acc:.4f})")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n[LSTM Eval] Loading best checkpoint …")
    ckpt = torch.load(out_dir / "best_lstm.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _, test_acc, preds, labels = evaluate_epoch(
        model, loaders["test"], criterion, device)

    emotion_names = cfg["emotions"]["classes"]
    metrics = compute_metrics(labels, preds, emotion_names)

    print(f"\n[LSTM Eval] Test Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"[LSTM Eval] Macro F1      : {metrics['macro_f1']*100:.2f}%")

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items()
                   if k != "classification_report"}, f, indent=2)

    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        model_name="BiLSTM + GloVe",
        save_path=str(fig_dir / f"{run_id}_curves.png")
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"], emotion_names,
        title="BiLSTM — Confusion Matrix",
        save_path=str(fig_dir / f"{run_id}_cm.png")
    )

    print(f"\n[Done] Artifacts saved to: {out_dir}/")
    return str(out_dir / "best_lstm.pt")


if __name__ == "__main__":
    main()
