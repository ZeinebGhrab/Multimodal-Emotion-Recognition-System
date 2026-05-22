"""
scripts/train_cnn.py
─────────────────────
Training script for the ResNet-50 image-only emotion model.

Usage:
    python scripts/train_cnn.py
    python scripts/train_cnn.py --epochs 40 --lr 5e-5 --batch_size 128
    python scripts/train_cnn.py --backbone resnet50 --freeze_bn
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
from torch.cuda.amp import GradScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn_model import (
    EmotionCNN, build_optimizer, build_scheduler,
    train_one_epoch, evaluate
)
from src.preprocessing.image_preprocessing import get_dataloaders
from src.evaluation.metrics import (
    compute_metrics, plot_confusion_matrix, plot_training_curves
)
from src.utils.early_stopping import EarlyStopping


def parse_args():
    p = argparse.ArgumentParser(description="Train CNN emotion model")
    p.add_argument("--config",       default="configs/config.yaml")
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--dropout",      type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None,
                   help="L2 weight decay (overrides config)")
    p.add_argument("--patience",     type=int,   default=None,
                   help="Early stopping patience (overrides config)")
    p.add_argument("--freeze_bn",    action="store_true",
                   help="Freeze BatchNorm during fine-tuning")
    p.add_argument("--no_pretrain",  action="store_true",
                   help="Train from scratch (no ImageNet weights)")
    p.add_argument("--amp",          action="store_true",
                   help="Use Automatic Mixed Precision (GPU only)")
    p.add_argument("--device",       default="auto",
                   choices=["auto", "cuda", "cpu", "mps"])
    return p.parse_args()


def main():
    args = parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cnn_cfg = cfg["cnn"]
    es_cfg  = cfg["training"]["early_stopping"]

    epochs       = args.epochs       or cnn_cfg["epochs"]
    lr           = args.lr           or cnn_cfg["learning_rate"]
    batch_size   = args.batch_size   or cnn_cfg["batch_size"]
    dropout      = args.dropout      or cnn_cfg["dropout"]
    weight_decay = args.weight_decay if args.weight_decay is not None \
                   else cnn_cfg["weight_decay"]
    patience     = args.patience     if args.patience is not None \
                   else es_cfg["patience"]

    # ── Device ──────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = ("cuda"  if torch.cuda.is_available() else
                  "mps"   if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    print(f"[CNN Train] Device       : {device}")
    print(f"[CNN Train] weight_decay : {weight_decay}")
    print(f"[CNN Train] ES patience  : {patience}\n")

    torch.manual_seed(cfg["project"]["seed"])

    # ── Output dirs ──────────────────────────────────────────────────────────
    run_id  = f"cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(cfg["paths"]["checkpoints"]) / run_id
    fig_dir = Path(cfg["paths"]["figures"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = str(out_dir / "best_cnn.pt")

    # ── Data ─────────────────────────────────────────────────────────────────
    data_raw = cfg["paths"]["data_raw"]
    try:
        loaders = get_dataloaders(
            data_raw=data_raw,
            img_size=cfg["image"]["resize"],
            batch_size=batch_size,
            num_workers=4
        )
    except FileNotFoundError:
        print("[WARN] FER2013 not found — using dummy data for smoke-test.\n")
        from torch.utils.data import TensorDataset, DataLoader
        dummy_imgs   = torch.randn(200, 3, 224, 224)
        dummy_labels = torch.randint(0, 7, (200,))
        ds = TensorDataset(dummy_imgs, dummy_labels)
        loaders = {
            "train": DataLoader(ds, batch_size=batch_size, shuffle=True),
            "val":   DataLoader(ds, batch_size=batch_size),
            "test":  DataLoader(ds, batch_size=batch_size),
        }

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EmotionCNN(
        num_classes=cfg["emotions"]["num_classes"],
        dropout=dropout,
        pretrained=not args.no_pretrain,
        freeze_bn=args.freeze_bn
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[CNN Train] Total params    : {total:,}")
    print(f"[CNN Train] Trainable params: {trainable:,}\n")

    # ── Loss, optimizer, scheduler ───────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = build_optimizer(
        model, lr=lr,
        backbone_lr_factor=cnn_cfg.get("backbone_lr_factor", 0.1),
        weight_decay=weight_decay          # ← from config, no longer hardcoded
    )
    scheduler = build_scheduler(optimizer, epochs=epochs, warmup_epochs=5)
    scaler    = GradScaler() if (args.amp and device == "cuda") else None

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
    print(f"[CNN Train] Starting training for up to {epochs} epochs …")
    print(f"[CNN Train] Early stopping on '{monitor}' (mode={es_mode})\n")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device, scaler)
        vl_loss, vl_acc, _, _ = evaluate(
            model, loaders["val"], criterion, device)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);    val_accs.append(vl_acc)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f} | "
              f"Val   loss: {vl_loss:.4f}  acc: {vl_acc:.4f}")

        es_metric = vl_loss if monitor == "val_loss" else vl_acc
        if early_stopper(es_metric, model, ckpt_path):
            print(f"\n[CNN Train] Early stopping triggered at epoch {epoch}.")
            break

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n[CNN Eval] Loading best checkpoint …")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, test_acc, preds, labels = evaluate(model, loaders["test"], criterion, device)

    emotion_names = cfg["emotions"]["classes"]
    metrics = compute_metrics(labels, preds, emotion_names)

    print(f"\n[CNN Eval] Test Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"[CNN Eval] Macro F1      : {metrics['macro_f1']*100:.2f}%")

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items()
                   if k != "classification_report"}, f, indent=2)

    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        model_name="CNN (ResNet-50)",
        save_path=str(fig_dir / f"{run_id}_curves.png")
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"], emotion_names,
        title="CNN (ResNet-50) — Confusion Matrix",
        save_path=str(fig_dir / f"{run_id}_cm.png")
    )

    print(f"\n[Done] Artifacts saved to: {out_dir}/")
    return ckpt_path


if __name__ == "__main__":
    main()