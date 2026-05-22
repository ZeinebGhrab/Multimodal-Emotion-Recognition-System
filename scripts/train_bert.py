"""
scripts/train_bert.py
──────────────────────
Training script for the BERT text-only emotion classifier.

Usage:
    python scripts/train_bert.py
    python scripts/train_bert.py --epochs 5 --lr 2e-5 --batch_size 16
    python scripts/train_bert.py --model_name distilbert-base-uncased
    python scripts/train_bert.py --weight_decay 0.005 --patience 4
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lstm_model import BERTClassifier, build_bert_optimizer
from src.preprocessing.text_preprocessing import (
    BERTEmotionDataset, load_emotion_csv, get_bert_dataloaders
)
from src.evaluation.metrics import (
    compute_metrics, plot_confusion_matrix, plot_training_curves
)
from src.utils.early_stopping import EarlyStopping


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train BERT emotion classifier")
    p.add_argument("--config",       default="configs/config.yaml")
    p.add_argument("--model_name",   default=None,
                   help="HuggingFace model name (overrides config)")
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--max_length",   type=int,   default=None)
    p.add_argument("--warmup_steps", type=int,   default=None)
    p.add_argument("--weight_decay", type=float, default=None,
                   help="L2 weight decay (overrides config)")
    p.add_argument("--patience",     type=int,   default=None,
                   help="Early stopping patience (overrides config)")
    p.add_argument("--device",       default="auto",
                   choices=["auto", "cuda", "cpu", "mps"])
    return p.parse_args()


# ─── Epoch loops ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = total_correct = total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask      = batch["attention_mask"].to(device)
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
        mask      = batch["attention_mask"].to(device)
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    bert_cfg = cfg["bert"]
    es_cfg   = cfg["training"]["early_stopping"]

    model_name   = args.model_name   or bert_cfg["model_name"]
    epochs       = args.epochs       or bert_cfg["epochs"]
    lr           = args.lr           or bert_cfg["learning_rate"]
    batch_size   = args.batch_size   or bert_cfg["batch_size"]
    max_length   = args.max_length   or cfg["text"]["max_length"]
    warmup_steps = args.warmup_steps or bert_cfg["warmup_steps"]
    weight_decay = (args.weight_decay if args.weight_decay is not None
                    else bert_cfg["weight_decay"])
    patience     = (args.patience if args.patience is not None
                    else es_cfg["patience"])

    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps"  if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device

    print(f"[BERT Train] Device       : {device}")
    print(f"[BERT Train] Model        : {model_name}")
    print(f"[BERT Train] weight_decay : {weight_decay}")
    print(f"[BERT Train] ES patience  : {patience}\n")

    torch.manual_seed(cfg["project"]["seed"])

    run_id  = f"bert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(cfg["paths"]["checkpoints"]) / run_id
    fig_dir = Path(cfg["paths"]["figures"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = str(out_dir / "best_bert.pt")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_csv = os.path.join(cfg["paths"]["data_raw"], "emotion_train.csv")
    val_csv   = os.path.join(cfg["paths"]["data_raw"], "emotion_val.csv")
    test_csv  = os.path.join(cfg["paths"]["data_raw"], "emotion_test.csv")

    if not os.path.exists(train_csv):
        print(f"[WARN] Dataset not found at {train_csv} — using dummy data.\n")

        class DummyLoader:
            def __iter__(self):
                for _ in range(5):
                    yield {
                        "input_ids":      torch.randint(0, 30000, (batch_size, max_length)),
                        "attention_mask": torch.ones(batch_size, max_length, dtype=torch.long),
                        "label":          torch.randint(0, 7, (batch_size,)),
                    }
            def __len__(self): return 5

        loaders = {"train": DummyLoader(), "val": DummyLoader(), "test": DummyLoader()}
    else:
        loaders = get_bert_dataloaders(
            train_csv=train_csv, val_csv=val_csv, test_csv=test_csv,
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BERTClassifier(
        model_name=model_name,
        num_classes=cfg["emotions"]["num_classes"],
        dropout=bert_cfg["dropout"]
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[BERT Train] Total params    : {total:,}")
    print(f"[BERT Train] Trainable params: {trainable:,}\n")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    from transformers import get_linear_schedule_with_warmup

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = build_bert_optimizer(
        model, lr=lr,
        weight_decay=weight_decay,
        no_decay_params=bert_cfg.get("no_decay_params", ["bias", "LayerNorm.weight"])
    )
    total_steps = len(loaders["train"]) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # ── Early stopping ────────────────────────────────────────────────────────
    monitor = es_cfg.get("monitor", "val_loss")
    es_mode = "min" if monitor == "val_loss" else "max"
    early_stopper = EarlyStopping(
        patience=patience,
        min_delta=es_cfg["min_delta"],
        mode=es_mode,
        restore_best=es_cfg.get("restore_best", True),
        verbose=True
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    print(f"[BERT Train] Starting training for up to {epochs} epochs …")
    print(f"[BERT Train] Early stopping on '{monitor}' (mode={es_mode})\n")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, loaders["train"], optimizer, scheduler, criterion, device)
        vl_loss, vl_acc, _, _ = evaluate_epoch(
            model, loaders["val"], criterion, device)

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);    val_accs.append(vl_acc)

        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f} | "
              f"Val   loss: {vl_loss:.4f}  acc: {vl_acc:.4f}")

        es_metric = vl_loss if monitor == "val_loss" else vl_acc
        if early_stopper(es_metric, model, ckpt_path):
            print(f"\n[BERT Train] Early stopping triggered at epoch {epoch}.")
            break

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n[BERT Eval] Loading best checkpoint …")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    _, test_acc, preds, labels = evaluate_epoch(
        model, loaders["test"], criterion, device)

    emotion_names = cfg["emotions"]["classes"]
    metrics = compute_metrics(labels, preds, emotion_names)

    print(f"\n[BERT Eval] Test Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"[BERT Eval] Macro F1      : {metrics['macro_f1']*100:.2f}%")

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items()
                   if k != "classification_report"}, f, indent=2)

    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        model_name="BERT Classifier",
        save_path=str(fig_dir / f"{run_id}_curves.png")
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"], emotion_names,
        title="BERT Classifier — Confusion Matrix",
        save_path=str(fig_dir / f"{run_id}_cm.png")
    )

    print(f"\n[Done] Artifacts saved to: {out_dir}/")
    return ckpt_path


if __name__ == "__main__":
    main()