"""
src/preprocessing/image_preprocessing.py
─────────────────────────────────────────
Image preprocessing pipeline for FER2013 dataset.

FER2013 format:
  - CSV with columns: emotion (int 0-6), pixels (space-separated), Usage (train/val/test)
  - 48×48 grayscale images → convert to 3-channel RGB for pretrained models
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
            transforms.ToTensor(),
            normalize,
        ])


# ─── Dataset ──────────────────────────────────────────────────────────────────

class FER2013Dataset(Dataset):
    """
    PyTorch Dataset wrapping the FER2013 CSV.

    Args:
        csv_path  : path to fer2013.csv
        split     : 'Training' | 'PublicTest' | 'PrivateTest'
        transform : torchvision transforms
    """

    def __init__(self, csv_path: str, split: str, transform=None):
        df = pd.read_csv(csv_path)
        self.data = df[df["Usage"] == split].reset_index(drop=True)
        self.transform = transform

        print(f"[FER2013Dataset] {split}: {len(self.data)} samples")
        self._log_class_distribution()

    def _log_class_distribution(self):
        counts = self.data["emotion"].value_counts().sort_index()
        for idx, cnt in counts.items():
            print(f"  {EMOTION_MAP[idx]:>10}: {cnt:5d} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = int(row["emotion"])

        # Parse pixel string → 48×48 uint8 array → PIL Image (RGB)
        pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels, mode="L").convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


# ─── DataLoaders ──────────────────────────────────────────────────────────────

def get_dataloaders(csv_path: str,
                    img_size: int = 224,
                    batch_size: int = 64,
                    num_workers: int = 4) -> dict:
    """
    Build train / val / test DataLoaders from FER2013 CSV.

    Returns:
        dict with keys 'train', 'val', 'test'
    """
    split_map = {
        "train": "Training",
        "val":   "PublicTest",
        "test":  "PrivateTest"
    }

    loaders = {}
    for name, fer_split in split_map.items():
        dataset = FER2013Dataset(
            csv_path=csv_path,
            split=fer_split,
            transform=get_transforms(name, img_size)
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=True
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
