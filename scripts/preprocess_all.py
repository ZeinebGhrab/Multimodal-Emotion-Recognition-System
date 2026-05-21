"""
scripts/preprocess_all.py
──────────────────────────
One-shot data preparation script for both modalities.

Steps:
  1. Validate raw data files are present
  2. Preprocess FER2013 CSV  → verify splits and class distribution
  3. Preprocess Emotion NLP  → clean text, split into train/val/test CSV files
  4. Optionally download GloVe 100d embeddings

Usage:
    python scripts/preprocess_all.py
    python scripts/preprocess_all.py --skip_glove
    python scripts/preprocess_all.py --data_dir /path/to/raw/data
"""

import os
import sys
import argparse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.text_preprocessing import clean_text, FER_EMOTION_MAP


# ─── Emotion label unification ────────────────────────────────────────────────

# dair-ai/emotion dataset labels → unified FER label space
NLP_TO_FER = {
    "joy":      "happy",
    "sadness":  "sad",
    "anger":    "angry",
    "fear":     "fear",
    "love":     "happy",      # closest positive emotion
    "surprise": "surprise",
}

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"

# dair-ai/emotion on HuggingFace (replaces the removed Kaggle mirror)
HF_EMOTION_DATASET = "dair-ai/emotion"
HF_EMOTION_CONFIG  = "split"          # use the pre-split version


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess FER2013 + Emotion NLP")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--data_dir",   default=None,
                   help="Override data_raw path from config")
    p.add_argument("--skip_glove", action="store_true",
                   help="Skip GloVe download")
    p.add_argument("--val_ratio",  type=float, default=0.1,
                   help="Fraction of training data to use as validation")
    return p.parse_args()


# ─── FER2013 ──────────────────────────────────────────────────────────────────

def validate_fer2013(csv_path: str):
    """Load and print summary statistics of FER2013 CSV."""
    print(f"\n{'─'*60}")
    print(f"  Validating FER2013: {csv_path}")
    print(f"{'─'*60}")

    df = pd.read_csv(csv_path)
    print(f"  Columns  : {list(df.columns)}")
    print(f"  Total    : {len(df):,} samples")
    print(f"  Splits   : {dict(df['Usage'].value_counts())}")
    print()

    label_map = {
        0: "angry", 1: "disgust", 2: "fear",
        3: "happy", 4: "neutral", 5: "sad", 6: "surprise"
    }
    for split in ["Training", "PublicTest", "PrivateTest"]:
        sub = df[df["Usage"] == split]
        print(f"  [{split}]  {len(sub):,} samples")
        for idx, label in label_map.items():
            cnt = (sub["emotion"] == idx).sum()
            bar = "█" * int(cnt / sub["emotion"].count() * 30)
            print(f"    {label:>10}: {cnt:5d}  {bar}")
        print()

    # Verify pixel format
    sample = df.iloc[0]["pixels"].split()
    assert len(sample) == 48 * 48, f"Expected 2304 pixels, got {len(sample)}"
    print("  ✓ Pixel format OK (48×48 grayscale)")
    return True


# ─── Emotion NLP dataset ──────────────────────────────────────────────────────

def preprocess_emotion_nlp(raw_path: str, out_dir: str,
                            val_ratio: float = 0.1):
    """
    Load the dair-ai/emotion CSV, clean text, unify labels,
    split into train/val/test, and save.

    Expected input columns: text, label  (or  sentence, emotion)
    """
    print(f"\n{'─'*60}")
    print(f"  Preprocessing Emotion NLP: {raw_path}")
    print(f"{'─'*60}")

    df = pd.read_csv(raw_path)
    print(f"  Columns: {list(df.columns)}")

    # Normalise column names
    if "sentence" in df.columns:
        df = df.rename(columns={"sentence": "text", "emotion": "label_str"})
    elif "label" in df.columns:
        label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        df["label_str"] = df["label"].map(lambda i: label_names[i] if isinstance(i, int) else i)
    else:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")

    # Clean text
    print("  Cleaning text …")
    df["text"] = df["text"].astype(str).apply(clean_text)

    # Unify labels to FER space
    df["emotion"] = df["label_str"].map(NLP_TO_FER)
    df = df.dropna(subset=["emotion"])
    df["label"] = df["emotion"].map(FER_EMOTION_MAP)

    print(f"  After cleaning: {len(df):,} samples")
    print(f"  Label distribution:")
    for emo, cnt in df["emotion"].value_counts().items():
        print(f"    {emo:>10}: {cnt:5d}")

    # Train / val / test split
    # Use existing split if present, else create one
    if "split" in df.columns:
        train_df = df[df["split"] == "train"]
        val_df   = df[df["split"] == "val"]
        test_df  = df[df["split"] == "test"]
    else:
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42,
                                              stratify=df["label"])
        val_df,   test_df = train_test_split(temp_df, test_size=0.5, random_state=42,
                                              stratify=temp_df["label"])

    cols = ["text", "label", "emotion"]
    os.makedirs(out_dir, exist_ok=True)

    train_df[cols].to_csv(os.path.join(out_dir, "emotion_train.csv"), index=False)
    val_df[cols].to_csv(os.path.join(out_dir, "emotion_val.csv"),   index=False)
    test_df[cols].to_csv(os.path.join(out_dir, "emotion_test.csv"),  index=False)

    print(f"\n  ✓ Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    print(f"  ✓ Saved to {out_dir}/")
    return True


# ─── GloVe ───────────────────────────────────────────────────────────────────

def download_glove(out_path: str, embed_dim: int = 100):
    """Download and extract GloVe 6B embeddings."""
    print(f"\n{'─'*60}")
    print(f"  Downloading GloVe 6B (822 MB) …")
    print(f"{'─'*60}")

    zip_path = out_path.replace(f"glove.6B.{embed_dim}d.txt", "glove.6B.zip")
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)

    def progress(block, block_size, total):
        done = block * block_size
        pct  = min(100, int(done * 100 / total))
        if pct % 10 == 0:
            print(f"  … {pct}% ({done // 1_000_000} MB)", end="\r")

    urllib.request.urlretrieve(GLOVE_URL, zip_path, progress)
    print(f"\n  Extracting {embed_dim}d vectors …")

    with zipfile.ZipFile(zip_path, "r") as z:
        target = f"glove.6B.{embed_dim}d.txt"
        z.extract(target, os.path.dirname(out_path))

    os.rename(os.path.join(os.path.dirname(out_path), target), out_path)
    os.remove(zip_path)

    print(f"  ✓ GloVe saved to {out_path}")


# ─── Emotion NLP auto-download ───────────────────────────────────────────────

def download_emotion_nlp(out_csv: str) -> str:
    """
    Download dair-ai/emotion from Hugging Face and save as a single CSV.

    The dataset already provides train / validation / test splits;
    we merge them and add a 'split' column so that
    ``preprocess_emotion_nlp`` can honour the original splits instead
    of re-splitting randomly.

    Returns the path to the merged CSV.
    """
    print(f"\n{'─'*60}")
    print(f"  Downloading dair-ai/emotion from Hugging Face …")
    print(f"{'─'*60}")

    try:
        from datasets import load_dataset  # already in requirements.txt
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for automatic download.\n"
            "  pip install datasets>=2.17.0"
        )

    ds = load_dataset(HF_EMOTION_DATASET, HF_EMOTION_CONFIG)

    label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    frames = []
    for split_name, split_ds in ds.items():
        df = split_ds.to_pandas()
        df["label_str"] = df["label"].map(lambda i: label_names[i])
        df["split"]     = split_name          # 'train' | 'validation' | 'test'
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    # Normalise 'validation' → 'val' so preprocess_emotion_nlp finds it
    merged["split"] = merged["split"].replace("validation", "val")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"  ✓ Saved {len(merged):,} samples → {out_csv}")
    return out_csv


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_raw = args.data_dir or cfg["paths"]["data_raw"]
    os.makedirs(data_raw, exist_ok=True)

    print("=" * 60)
    print("  MULTIMODAL EMOTION RECOGNITION — Data Preparation")
    print("=" * 60)

    errors = []

    # 1. FER2013
    fer_csv = os.path.join(data_raw, "fer2013.csv")
    if os.path.exists(fer_csv):
        try:
            validate_fer2013(fer_csv)
        except Exception as e:
            errors.append(f"FER2013 validation error: {e}")
    else:
        print(f"\n[SKIP] FER2013 not found at {fer_csv}")
        print("       Download from: https://www.kaggle.com/datasets/msambare/fer2013")
        print("       Place fer2013.csv in:", data_raw)

    # 2. Emotion NLP
    nlp_candidates = [
        os.path.join(data_raw, "emotion.csv"),
        os.path.join(data_raw, "emotion_dataset.csv"),
        os.path.join(data_raw, "training.csv"),
    ]
    nlp_csv = next((p for p in nlp_candidates if os.path.exists(p)), None)
    if nlp_csv:
        try:
            preprocess_emotion_nlp(nlp_csv, data_raw, args.val_ratio)
        except Exception as e:
            errors.append(f"Emotion NLP error: {e}")
    else:
        # Auto-download from Hugging Face (former Kaggle mirror no longer exists)
        auto_csv = os.path.join(data_raw, "emotion.csv")
        print(f"\n[INFO] Emotion NLP dataset not found locally.")
        print(f"       Attempting automatic download from Hugging Face ({HF_EMOTION_DATASET}) …")
        try:
            download_emotion_nlp(auto_csv)
            preprocess_emotion_nlp(auto_csv, data_raw, args.val_ratio)
        except Exception as e:
            errors.append(f"Emotion NLP error: {e}")
            print(f"\n[WARN] Auto-download failed: {e}")
            print("       Manual alternative — download dair-ai/emotion from Hugging Face:")
            print("         https://huggingface.co/datasets/dair-ai/emotion")
            print("       Or via Python:  from datasets import load_dataset")
            print(f"                      load_dataset('dair-ai/emotion', 'split')")
            print("       Save the CSV as emotion.csv and place it in:", data_raw)

    # 3. GloVe
    glove_path = cfg["text"]["glove_path"]
    if not args.skip_glove:
        if os.path.exists(glove_path):
            print(f"\n[SKIP] GloVe already at {glove_path}")
        else:
            try:
                download_glove(glove_path, embed_dim=cfg["text"]["embedding_dim"])
            except Exception as e:
                errors.append(f"GloVe download error: {e}")
                print(f"\n[WARN] Could not download GloVe: {e}")
                print("       Download manually from: https://nlp.stanford.edu/data/glove.6B.zip")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print("  COMPLETED WITH WARNINGS:")
        for err in errors:
            print(f"  ✗ {err}")
    else:
        print("  ✓ All preprocessing steps completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
