"""
scripts/train_multimodal.py  — VERSION CORRIGÉE FINALE
────────────────────────────────────────────────────────

Usage :
    python scripts/train_multimodal.py ^
        --fusion attention ^
        --cnn_checkpoint  outputs/checkpoints/cnn_20260522_171658/best_cnn.pt ^
        --bert_checkpoint outputs/checkpoints/bert_20260522_165038/best_bert.pt ^
        --no_finetune_encoders ^
        --epochs 30
"""

import os
import sys
import random
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn_model import EmotionCNN
from src.models.lstm_model import BERTClassifier
from src.fusion.fusion_models import MultimodalEmotionModel
from src.evaluation.metrics import compute_metrics, plot_confusion_matrix, plot_training_curves
from src.genai.report_generator import generate_emotion_report, print_report
from src.utils.early_stopping import EarlyStopping


# ─── Argument parser ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train multimodal emotion model")
    p.add_argument("--fusion",               default="attention",
                   choices=["early", "late", "attention"])
    p.add_argument("--epochs",               type=int,   default=None)
    p.add_argument("--batch_size",           type=int,   default=None)
    p.add_argument("--lr",                   type=float, default=None)
    p.add_argument("--encoder_lr_factor",    type=float, default=0.1)
    p.add_argument("--weight_decay_enc",     type=float, default=None)
    p.add_argument("--weight_decay_fusion",  type=float, default=None)
    p.add_argument("--patience",             type=int,   default=None)
    p.add_argument("--val_ratio",            type=float, default=0.1,
                   help="Fraction du dossier train/ utilisee comme validation")
    p.add_argument("--no_finetune_encoders", action="store_true",
                   help="Geler les encodeurs — entrainer seulement la tete de fusion")
    p.add_argument("--cnn_checkpoint",       default=None)
    p.add_argument("--bert_checkpoint",      default=None)
    p.add_argument("--config",               default="configs/config.yaml")
    p.add_argument("--device",               default="auto",
                   choices=["auto", "cuda", "cpu", "mps"])
    return p.parse_args()


# ─── Dataset aligne par label — version BERT tokenisee ───────────────────────

class AlignedMultimodalDatasetBERT(Dataset):
    """
    Associe images FER2013 et textes dair-ai/emotion par classe d'emotion.

    Split FER2013 :
        split='train' -> 90% du dossier fer2013/train/  (entrainement)
        split='val'   -> 10% du dossier fer2013/train/  (validation, jamais vu en train)
        split='test'  -> dossier fer2013/test/          (evaluation finale)

    Textes :
        emotion_train.csv -> pour train
        emotion_val.csv   -> pour val
        emotion_test.csv  -> pour test

    Pour chaque image, un texte de la MEME classe est tire aleatoirement.
    Les classes 'disgust' et 'neutral' n'ont pas de textes NLP.
    """

    FOLDER_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    def __init__(self, fer_root, text_csv, split,
                 bert_model_name="bert-base-uncased",
                 img_size=224, max_len=128,
                 val_ratio=0.1, seed=42):

        from torchvision import datasets as tv_datasets
        from transformers import BertTokenizerFast
        from src.preprocessing.image_preprocessing import get_transforms
        from src.preprocessing.text_preprocessing import load_emotion_csv, clean_text

        self.max_len   = max_len
        self.rng       = random.Random(seed)
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)

        # ── Images ────────────────────────────────────────────────────────────
        if split in ("train", "val"):
            # Charger tout le dossier train/ puis decouper reproductiblement
            full_folder = tv_datasets.ImageFolder(
                os.path.join(fer_root, "train"),
                transform=get_transforms(split, img_size)
            )
            n_val   = int(len(full_folder) * val_ratio)
            n_train = len(full_folder) - n_val
            gen = torch.Generator().manual_seed(seed)
            train_sub, val_sub = random_split(
                full_folder, [n_train, n_val], generator=gen
            )
            # Garder la reference au folder complet pour acceder aux samples
            self.img_folder = full_folder
            subset_indices  = (train_sub if split == "train" else val_sub).indices

            # Grouper les indices par label
            self.imgs_by_label = {i: [] for i in range(7)}
            for idx in subset_indices:
                _, lbl = full_folder.samples[idx]
                self.imgs_by_label[lbl].append(idx)

        else:  # test — dossier fer2013/test/ jamais vu pendant l'entrainement
            test_folder = tv_datasets.ImageFolder(
                os.path.join(fer_root, "test"),
                transform=get_transforms("test", img_size)
            )
            self.img_folder = test_folder
            self.imgs_by_label = {i: [] for i in range(7)}
            for idx, (_, lbl) in enumerate(test_folder.samples):
                self.imgs_by_label[lbl].append(idx)

        # ── Textes ────────────────────────────────────────────────────────────
        # Textes par défaut pour les classes absentes du dataset NLP (disgust=1, neutral=4)
        DEFAULT_TEXTS = {
            1: [
                "I feel disgusted", "this is revolting", "I am filled with disgust",
                "that is absolutely disgusting", "I find this repulsive",
                "this makes me feel sick", "how disgusting", "I am repulsed by this",
            ],
            4: [
                "I feel nothing in particular", "feeling calm and neutral",
                "neutral mood today", "I have no strong feelings right now",
                "everything is fine, nothing special", "I feel indifferent",
                "no particular emotion at the moment", "just a normal day",
            ],
        }

        df = load_emotion_csv(text_csv)
        self.texts_by_label = {i: [] for i in range(7)}
        for _, row in df.iterrows():
            lbl = int(row["label"])
            txt = clean_text(str(row["text"]))
            if txt and 0 <= lbl <= 6:
                self.texts_by_label[lbl].append(txt)

        # Ajouter les textes par défaut pour disgust et neutral
        for lbl, texts in DEFAULT_TEXTS.items():
            if not self.texts_by_label[lbl]:
                self.texts_by_label[lbl] = texts
                print(f"  [AlignedDatasetBERT] '{self.FOLDER_CLASSES[lbl]}' : "
                      f"{len(texts)} textes par défaut ajoutés")

        # ── Toutes les classes ayant des images sont valides ──────────────────
        valid_classes = [c for c in range(7) if self.imgs_by_label[c]]

        self.samples = [
            (lbl, img_idx)
            for lbl in valid_classes
            for img_idx in self.imgs_by_label[lbl]
        ]

        n_per_class = {
            self.FOLDER_CLASSES[c]: len(self.imgs_by_label[c])
            for c in valid_classes
        }
        print(f"  [AlignedDatasetBERT] {split}: {len(self.samples):,} paires")
        print(f"  [AlignedDatasetBERT] Distribution : {n_per_class}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, img_idx = self.samples[idx]

        # Image
        image, _ = self.img_folder[img_idx]

        # Texte : tirage aleatoire dans la meme classe
        texts = self.texts_by_label.get(label, [""])
        text  = self.rng.choice(texts) if texts else ""

        # Tokenisation BERT
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "image":          image,
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ─── Fallback Dummy Dataset ───────────────────────────────────────────────────

class DummyMultimodalDataset(Dataset):
    """Utilise uniquement si les donnees reelles sont absentes."""

    def __init__(self, size=1000, img_size=224, seq_len=128, num_classes=7):
        self.size        = size
        self.img_size    = img_size
        self.seq_len     = seq_len
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


# ─── Selection du dataset ─────────────────────────────────────────────────────

def _build_datasets(cfg, bert_model_name, val_ratio):
    """
    Charge AlignedMultimodalDatasetBERT si les donnees sont disponibles.

    Splits :
        train -> 90% de fer2013/train/ + emotion_train.csv
        val   -> 10% de fer2013/train/ + emotion_val.csv
        test  -> fer2013/test/         + emotion_test.csv  (jamais vu)
    """
    data_raw = cfg["paths"]["data_raw"]
    img_size = cfg["image"]["resize"]
    max_len  = cfg["text"]["max_length"]
    seed     = cfg["project"]["seed"]

    # Localiser FER2013
    fer_root = None
    for candidate in [
        os.path.join(data_raw, "fer2013"),
        data_raw,
    ]:
        if (os.path.isdir(os.path.join(candidate, "train")) and
                os.path.isdir(os.path.join(candidate, "test"))):
            fer_root = candidate
            break

    train_csv = os.path.join(data_raw, "emotion_train.csv")
    val_csv   = os.path.join(data_raw, "emotion_val.csv")
    test_csv  = os.path.join(data_raw, "emotion_test.csv")

    has_images = fer_root is not None
    has_texts  = all(os.path.exists(p) for p in [train_csv, val_csv, test_csv])

    if has_images and has_texts:
        print("\n[Dataset] Donnees reelles detectees — AlignedMultimodalDatasetBERT")
        print(f"  FER2013 root : {fer_root}")
        print(f"  Split val    : {int(val_ratio*100)}% de fer2013/train/")
        print(f"  Split test   : fer2013/test/ (jamais vu pendant l'entrainement)")
        print(f"  Textes CSV   : {data_raw}/emotion_{{train,val,test}}.csv\n")

        train_ds = AlignedMultimodalDatasetBERT(
            fer_root, train_csv, "train",
            bert_model_name=bert_model_name,
            img_size=img_size, max_len=max_len,
            val_ratio=val_ratio, seed=seed
        )
        val_ds = AlignedMultimodalDatasetBERT(
            fer_root, val_csv, "val",
            bert_model_name=bert_model_name,
            img_size=img_size, max_len=max_len,
            val_ratio=val_ratio, seed=seed
        )
        test_ds = AlignedMultimodalDatasetBERT(
            fer_root, test_csv, "test",
            bert_model_name=bert_model_name,
            img_size=img_size, max_len=max_len,
            val_ratio=val_ratio, seed=seed + 1
        )
        return train_ds, val_ds, test_ds, True

    else:
        print("\n" + "!" * 70)
        print("[Dataset] DONNEES MANQUANTES — donnees factices utilisees")
        if not has_images:
            print(f"  FER2013 non trouve dans '{data_raw}'")
            print(f"  Attendu : fer2013/train/<classe>/ ET fer2013/test/<classe>/")
        if not has_texts:
            print("  CSVs de textes manquants — lancez : python scripts/preprocess_all.py")
        print("  Accuracy attendue ~14% (chance) — aucune valeur d'entrainement")
        print("!" * 70 + "\n")

        train_ds = DummyMultimodalDataset(2000, img_size, max_len)
        val_ds   = DummyMultimodalDataset(400,  img_size, max_len)
        test_ds  = DummyMultimodalDataset(400,  img_size, max_len)
        return train_ds, val_ds, test_ds, False


# ─── Boucles d'entrainement et d'evaluation ───────────────────────────────────

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

    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps"  if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    fuse_cfg = cfg["fusion"]
    es_cfg   = cfg["training"]["early_stopping"]

    epochs     = args.epochs          or fuse_cfg["epochs"]
    batch_size = args.batch_size      or fuse_cfg["batch_size"]
    lr         = args.lr              or fuse_cfg["learning_rate"]
    wd_enc     = (args.weight_decay_enc    if args.weight_decay_enc    is not None
                  else fuse_cfg["weight_decay_encoders"])
    wd_fusion  = (args.weight_decay_fusion if args.weight_decay_fusion is not None
                  else fuse_cfg["weight_decay_fusion"])
    patience   = (args.patience if args.patience is not None
                  else es_cfg["patience"])
    val_ratio  = args.val_ratio

    bert_model_name = cfg["bert"]["model_name"]

    print(f"[Multimodal Train] Device               : {device}")
    print(f"[Multimodal Train] Fusion               : {args.fusion}")
    print(f"[Multimodal Train] weight_decay (enc)   : {wd_enc}")
    print(f"[Multimodal Train] weight_decay (fusion): {wd_fusion}")
    print(f"[Multimodal Train] ES patience          : {patience}")
    print(f"[Multimodal Train] Val ratio            : {val_ratio} "
          f"({int(val_ratio*100)}% de fer2013/train/)")

    # Verification des checkpoints
    if args.no_finetune_encoders:
        if not args.cnn_checkpoint or not args.bert_checkpoint:
            print("\n" + "!" * 70)
            print("[ERREUR] --no_finetune_encoders necessite --cnn_checkpoint ET --bert_checkpoint")
            print("  Geler des encodeurs a poids aleatoires = extraire du bruit pur (~14% accuracy)")
            print("  Solution : fournissez les deux checkpoints ou retirez --no_finetune_encoders")
            print("!" * 70 + "\n")
        else:
            print(f"[Multimodal Train] CNN checkpoint  : {args.cnn_checkpoint}")
            print(f"[Multimodal Train] BERT checkpoint : {args.bert_checkpoint}")
            print(f"[Multimodal Train] Mode            : encodeurs GELES")

    torch.manual_seed(cfg["project"]["seed"])

    run_id  = f"{args.fusion}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(cfg["paths"]["checkpoints"]) / run_id
    fig_dir = Path(cfg["paths"]["figures"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = str(out_dir / "best_model.pt")

    # Datasets
    train_ds, val_ds, test_ds, using_real_data = _build_datasets(
        cfg, bert_model_name, val_ratio
    )

    num_workers = 2 if using_real_data else 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda")
    )

    # Encodeurs
    print("\n[Multimodal Train] Construction de l'encodeur image (ResNet-50) ...")
    image_encoder = EmotionCNN(
        num_classes=cfg["emotions"]["num_classes"],
        dropout=cfg["cnn"]["dropout"],
        pretrained=cfg["cnn"]["pretrained"]
    )
    if args.cnn_checkpoint and os.path.exists(args.cnn_checkpoint):
        image_encoder.load_state_dict(
            torch.load(args.cnn_checkpoint, map_location="cpu")
        )
        print(f"  Checkpoint CNN charge : {args.cnn_checkpoint}")
    elif args.cnn_checkpoint:
        print(f"  Checkpoint CNN introuvable : {args.cnn_checkpoint}")
    else:
        print(f"  Aucun checkpoint CNN — poids ImageNet pre-entraines")

    print("[Multimodal Train] Construction de l'encodeur texte (BERT) ...")
    text_encoder = BERTClassifier(
        model_name=bert_model_name,
        num_classes=cfg["emotions"]["num_classes"],
        dropout=cfg["bert"]["dropout"]
    )
    if args.bert_checkpoint and os.path.exists(args.bert_checkpoint):
        state = torch.load(args.bert_checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        text_encoder.load_state_dict(state)
        print(f"  Checkpoint BERT charge : {args.bert_checkpoint}")
    elif args.bert_checkpoint:
        print(f"  Checkpoint BERT introuvable : {args.bert_checkpoint}")
    else:
        print(f"  Aucun checkpoint BERT — poids bert-base-uncased")

    # Gel des encodeurs uniquement si les deux checkpoints sont fournis
    if args.no_finetune_encoders:
        if args.cnn_checkpoint and args.bert_checkpoint:
            for p in image_encoder.parameters():
                p.requires_grad = False
            for p in text_encoder.parameters():
                p.requires_grad = False
            print("[Multimodal Train] Encodeurs GELES (checkpoints pre-entraines)")
        else:
            print("[Multimodal Train] Gel ignore — aucun checkpoint fourni")
            print("  -> Fine-tuning complet active")

    # Modele de fusion
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

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Multimodal Train] Total params     : {total:,}")
    print(f"[Multimodal Train] Trainable params : {trainable:,}")

    # ── Class weights (inverse-frequency) to handle imbalance ────────────────
    # Example: disgust=392 vs happy=6484 → happy is weighted down, disgust up
    num_classes = cfg["emotions"]["num_classes"]

    if using_real_data and hasattr(train_ds, "imgs_by_label"):
        counts = torch.zeros(num_classes)
        for lbl, indices in train_ds.imgs_by_label.items():
            counts[lbl] = len(indices)
        # Replace zeros (classes with no images) with 1 to avoid division by zero
        counts = counts.clamp(min=1)
        # weight[i] = total / (num_classes * count[i]) — standard balanced weighting
        class_weights = counts.sum() / (num_classes * counts)
        class_weights = class_weights.to(device)
        print(f"\n[Loss] Class weights (inverse-frequency):")
        for i, (name, w) in enumerate(zip(cfg["emotions"]["classes"], class_weights)):
            print(f"  {name:>10}: count={int(counts[i]):5d}  weight={w:.4f}")
    else:
        class_weights = None
        print("\n[Loss] No class weights (dummy data or weights unavailable)")

    # Loss et Optimizer
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1
    )

    encoder_params = [
        p for p in (list(model.image_encoder.parameters()) +
                    list(model.text_encoder.parameters()))
        if p.requires_grad
    ]
    fusion_params = [p for p in model.fusion.parameters() if p.requires_grad]

    param_groups = []
    if encoder_params:
        param_groups.append({
            "params":       encoder_params,
            "lr":           lr * args.encoder_lr_factor,
            "weight_decay": wd_enc,
        })
    if fusion_params:
        param_groups.append({
            "params":       fusion_params,
            "lr":           lr,
            "weight_decay": wd_fusion,
        })

    if not param_groups:
        print("[ERREUR] Aucun parametre entrainable. Verifiez la configuration.")
        return

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # Early Stopping
    monitor = es_cfg.get("monitor", "val_loss")
    es_mode = "min" if monitor == "val_loss" else "max"
    early_stopper = EarlyStopping(
        patience=patience,
        min_delta=es_cfg["min_delta"],
        mode=es_mode,
        restore_best=es_cfg.get("restore_best", True),
        verbose=True
    )

    # Boucle d'entrainement
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    print(f"\n[Multimodal Train] Debut fusion {args.fusion} — {epochs} epochs max")
    print(f"[Multimodal Train] Early stopping sur '{monitor}' (mode={es_mode})\n")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        vl_loss, vl_acc, _, _ = evaluate_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f} | "
              f"Val   loss: {vl_loss:.4f}  acc: {vl_acc:.4f}")

        es_metric = vl_loss if monitor == "val_loss" else vl_acc
        if early_stopper(es_metric, model, ckpt_path):
            print(f"\n[Multimodal Train] Early stopping a l'epoch {epoch}.")
            break

    # Evaluation finale sur test/ (jamais vu)
    print("\n[Eval] Chargement du meilleur checkpoint ...")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    _, test_acc, preds, labels = evaluate_epoch(
        model, test_loader, criterion, device
    )

    emotion_names = cfg["emotions"]["classes"]
    metrics = compute_metrics(labels, preds, emotion_names)

    present = metrics.get("macro_f1_present", metrics["macro_f1"])
    missing = metrics.get("zero_support_classes", [])

    print(f"\n[Eval] Test Accuracy       : {metrics['accuracy']*100:.2f}%")
    print(f"[Eval] Macro F1 (all 7)    : {metrics['macro_f1']*100:.2f}%"
          + (f"  <- inclut {missing} (support=0)" if missing else ""))
    print(f"[Eval] Macro F1 (presentes): {present*100:.2f}%")

    if not using_real_data:
        print("\n  Metriques calculees sur donnees FACTICES — aucune valeur reelle.")

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(
            {k: v for k, v in metrics.items() if k != "classification_report"},
            f, indent=2
        )

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

    # Rapport GenAI de demonstration
    print("\n[GenAI] Generation du rapport de demonstration ...")
    random.seed(0)
    pred_emotion = emotion_names[preds[0]] if preds else "neutral"
    scores = {e: random.random() for e in emotion_names}
    scores[pred_emotion] += 1.0
    total_s = sum(scores.values())
    scores  = {e: s / total_s for e, s in scores.items()}

    report = generate_emotion_report(
        emotion=pred_emotion,
        scores=scores,
        user_text="Je me sens assez deborde ces derniers temps.",
        use_llm=bool(os.getenv("ANTHROPIC_API_KEY")),
        save_path=str(out_dir / "sample_report.json")
    )
    print_report(report)

    print(f"\n[Done] Artefacts sauvegardes dans : {out_dir}/")


if __name__ == "__main__":
    main()