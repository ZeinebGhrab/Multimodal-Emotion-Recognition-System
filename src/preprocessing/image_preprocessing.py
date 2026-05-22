"""
src/preprocessing/image_preprocessing.py
─────────────────────────────────────────
Image preprocessing pipeline for FER2013 dataset.

Supports TWO dataset formats automatically:

  Format A — Folder-based (msambare/fer2013 on Kaggle, most common):
    data/raw/fer2013/
      train/  angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
      test/   angry/ ...

  Format B — CSV-based (original FER2013):
    data/raw/fer2013.csv
      columns: emotion (int 0-6), pixels (space-separated 48×48), Usage

Use get_dataloaders() for either format — it auto-detects.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
import yaml


# ─── Emotion mapping ──────────────────────────────────────────────────────────
EMOTION_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

# Folder names → label index  (torchvision sorts alphabetically by default)
FOLDER_CLASS_ORDER = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(split: str, img_size: int = 224) -> transforms.Compose:
    """
    Return torchvision transforms for train / val / test splits.

    Training: augmentation (flip, crop, jitter, rotation) + normalize
    Val/Test: only resize + normalize
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),   # handles grayscale PNGs
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomCrop(img_size, padding=8),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize,
        ])


# ─── Format A: Folder-based Dataset ───────────────────────────────────────────

class FER2013FolderDataset(Dataset):
    """
    Wraps torchvision.datasets.ImageFolder for the folder-based FER2013 layout.

    data/raw/fer2013/
      train/  angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
      test/   angry/ ...

    Since the folder version has no validation split, we carve out
    val_ratio of the training set as validation.
    """

    def __init__(self, root: str, split: str,
                 transform=None, val_ratio: float = 0.1, seed: int = 42):
        """
        Args:
            root      : path to fer2013 folder (contains train/ and test/)
            split     : 'train' | 'val' | 'test'
            transform : torchvision transforms
            val_ratio : fraction of training images used for validation
        """
        self.transform = transform

        if split in ("train", "val"):
            full = datasets.ImageFolder(os.path.join(root, "train"))
            n_val   = int(len(full) * val_ratio)
            n_train = len(full) - n_val
            gen = torch.Generator().manual_seed(seed)
            train_sub, val_sub = random_split(full, [n_train, n_val], generator=gen)
            self._subset = train_sub if split == "train" else val_sub
            self._base   = full
        else:
            folder = datasets.ImageFolder(os.path.join(root, "test"))
            self._subset = folder
            self._base   = folder

        n = len(self._subset)
        print(f"[FER2013FolderDataset] {split}: {n} samples")

    def __len__(self):
        return len(self._subset)

    def __getitem__(self, idx):
        img, label = self._subset[idx]
        # img is a PIL Image from ImageFolder; apply our own transform
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# ─── Format B: CSV-based Dataset ──────────────────────────────────────────────

class FER2013CSVDataset(Dataset):
    """
    PyTorch Dataset wrapping the original FER2013 CSV.

    Columns: emotion (int 0-6), pixels (space-separated 48×48), Usage
    Usage values: 'Training' | 'PublicTest' | 'PrivateTest'
    """

    _SPLIT_MAP = {
        "train": "Training",
        "val":   "PublicTest",
        "test":  "PrivateTest",
    }

    def __init__(self, csv_path: str, split: str, transform=None):
        df = pd.read_csv(csv_path)
        fer_split = self._SPLIT_MAP[split]
        self.data = df[df["Usage"] == fer_split].reset_index(drop=True)
        self.transform = transform

        print(f"[FER2013CSVDataset] {split} ({fer_split}): {len(self.data)} samples")
        for idx, cnt in self.data["emotion"].value_counts().sort_index().items():
            print(f"  {EMOTION_MAP[idx]:>10}: {cnt:5d}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row   = self.data.iloc[idx]
        label = int(row["emotion"])
        pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        img    = Image.fromarray(pixels, mode="L").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# ─── Auto-detect format & build DataLoaders ───────────────────────────────────

def _detect_fer2013(data_raw: str):
    """
    Returns ('folder', folder_root) or ('csv', csv_path) depending on
    what is present under data_raw.  Raises FileNotFoundError if neither found.
    """
    # Check folder layout
    folder_root = os.path.join(data_raw, "fer2013")
    if os.path.isdir(os.path.join(folder_root, "train")):
        return "folder", folder_root

    # Also accept train/ directly inside data_raw
    if os.path.isdir(os.path.join(data_raw, "train")):
        return "folder", data_raw

    # Check CSV
    csv_path = os.path.join(data_raw, "fer2013.csv")
    if os.path.isfile(csv_path):
        return "csv", csv_path

    raise FileNotFoundError(
        f"FER2013 dataset not found in '{data_raw}'.\n"
        "Expected one of:\n"
        f"  {data_raw}/fer2013/train/<emotion>/  (folder format)\n"
        f"  {data_raw}/train/<emotion>/           (folder format, flat)\n"
        f"  {data_raw}/fer2013.csv                (CSV format)\n"
        "Download from: https://www.kaggle.com/datasets/msambare/fer2013"
    )


def get_dataloaders(data_raw: str,
                    img_size:    int   = 224,
                    batch_size:  int   = 64,
                    num_workers: int   = 4,
                    val_ratio:   float = 0.1) -> dict:
    """
    Build train / val / test DataLoaders from FER2013.
    Auto-detects folder vs CSV format.

    Returns:
        dict with keys 'train', 'val', 'test'
    """
    fmt, path = _detect_fer2013(data_raw)
    print(f"[FER2013] Detected format: {fmt}  →  {path}")

    loaders = {}
    for split in ("train", "val", "test"):
        if fmt == "folder":
            ds = FER2013FolderDataset(
                root=path, split=split,
                transform=get_transforms(split, img_size),
                val_ratio=val_ratio,
            )
        else:
            ds = FER2013CSVDataset(
                csv_path=path, split=split,
                transform=get_transforms(split, img_size),
            )

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders


# ─── Image-only utility: load a single image ──────────────────────────────────

def load_single_image(img_path: str, img_size: int = 224) -> torch.Tensor:
    """
    Load and preprocess a single image file for inference.
    Returns a (1, 3, H, W) tensor.
    """
    transform = get_transforms("test", img_size)
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)   # add batch dim


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    csv_path = os.path.join(cfg["paths"]["data_raw"], "fer2013.csv")

    if not os.path.exists(csv_path):
        print(f"[WARN] Dataset not found at {csv_path}")
        print("       Download from: https://www.kaggle.com/datasets/msambare/fer2013")
    else:
        loaders = get_dataloaders(
            csv_path=csv_path,
            img_size=cfg["image"]["resize"],
            batch_size=cfg["cnn"]["batch_size"]
        )
        imgs, labels = next(iter(loaders["train"]))
        print(f"\nBatch shape : {imgs.shape}")      # (64, 3, 224, 224)
        print(f"Labels shape: {labels.shape}")      # (64,)
        print(f"Label range : {labels.min()}–{labels.max()}")
