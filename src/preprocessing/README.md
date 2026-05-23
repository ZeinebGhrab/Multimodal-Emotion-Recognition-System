🔧 src/preprocessing/ — Data Preprocessing Pipeline
=====================================================

## Overview

This module converts raw datasets into tensors ready for model training.
It handles both modalities independently:

- `image_preprocessing.py` — FER2013 image loading, augmentation, normalization
- `text_preprocessing.py`  — tokenization, vocabulary, GloVe embeddings, DataLoaders

---

## Folder Structure

```
src/preprocessing/
├── __init__.py
├── image_preprocessing.py       FER2013 → (B, 3, 224, 224) tensors
└── text_preprocessing.py        Text → token IDs + attention masks
```

---

## image_preprocessing.py

### Responsibilities

```
Raw image (48×48 grayscale PNG)
  ↓
1. Convert grayscale → RGB (3 channels)
2. Resize to 224×224 (ResNet / ViT requirement)
3. Apply augmentation (train only)
4. Normalize with ImageNet statistics
5. Return FloatTensor (3, 224, 224)
```

### Transform Pipelines

#### Training Split (with augmentation)

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),   # grayscale → RGB
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomCrop(224, padding=8),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])
```

#### Validation / Test Split (no augmentation)

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])
```

> **Why ImageNet mean/std?**  ResNet-50 and ViT-B/16 are pretrained on
> ImageNet and expect input statistics matching that distribution.
> Without this normalization, fine-tuning is significantly less stable.

### Dataset Classes

#### FER2013FolderDataset

Used with the folder-based format (Kaggle msambare/fer2013):

```
data/raw/fer2013/
  train/  angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
  test/   angry/ ...
```

Key behavior:
- Wraps `torchvision.datasets.ImageFolder`
- Carves out a validation split from training data (`val_ratio=0.1` → 10%)
- Uses a seeded `random_split` for reproducibility

#### FER2013CSVDataset

Used with the original FER2013 CSV format:

```
data/raw/fer2013.csv
  columns: emotion (int 0–6), pixels (space-separated 48×48), Usage
  Usage: 'Training' | 'PublicTest' | 'PrivateTest'
```

Key behavior:
- Parses pixel strings → 48×48 numpy arrays → PIL Images
- Maps `Usage` column to `train | val | test` splits

### Auto-detection Logic

```python
def _detect_fer2013(data_raw: str):
    # 1. Check for folder format
    if os.path.isdir(os.path.join(data_raw, "fer2013", "train")):
        return "folder", os.path.join(data_raw, "fer2013")

    # 2. Check for flat folder format
    if os.path.isdir(os.path.join(data_raw, "train")):
        return "folder", data_raw

    # 3. Check for CSV format
    if os.path.isfile(os.path.join(data_raw, "fer2013.csv")):
        return "csv", os.path.join(data_raw, "fer2013.csv")

    raise FileNotFoundError(...)
```

### Main Entry Point

```python
from src.preprocessing.image_preprocessing import get_dataloaders

loaders = get_dataloaders(
    data_raw="data/raw",
    img_size=224,
    batch_size=64,
    num_workers=4,
    val_ratio=0.1,
)
# Returns: {"train": DataLoader, "val": DataLoader, "test": DataLoader}
```

### Single-Image Inference Utility

```python
from src.preprocessing.image_preprocessing import load_single_image

tensor = load_single_image("face.jpg", img_size=224)
# Returns shape: (1, 3, 224, 224) — batch dimension included
```

---

## text_preprocessing.py

### Key Constants

```python
FER_EMOTION_MAP = {
    "angry": 0, "disgust": 1, "fear": 2, "happy": 3,
    "neutral": 4, "sad": 5, "surprise": 6,
}

HF_TO_FER_LABEL_MAP = {
    "sadness": 5,    # → sad
    "joy":     3,    # → happy
    "love":    3,    # → happy (closest)
    "anger":   0,    # → angry
    "fear":    2,    # → fear
    "surprise": 6,   # → surprise
}
# disgust (1) and neutral (4) have no NLP equivalent → support=0
```

### Text Cleaning

```python
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)    # remove URLs
    text = re.sub(r"@\w+", "", text)                # remove @mentions
    text = re.sub(r"#(\w+)", r"\1", text)           # #hashtag → word
    text = re.sub(r"[^a-z\s']", " ", text)          # keep letters + apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

### Vocabulary (for BiLSTM)

```python
vocab = Vocabulary(min_freq=2)    # discard words appearing < 2 times
vocab.build(train_texts)          # builds word2idx / idx2word
vocab.encode("I feel sad", max_len=128)   # → padded list of int indices

# Special tokens:
# vocab[0] = <PAD>  (padding)
# vocab[1] = <UNK>  (unknown word)
```

**Measured vocabulary size:** 7 400 tokens on the dair-ai/emotion dataset.

### GloVe Embedding Matrix

```python
from src.preprocessing.text_preprocessing import load_glove_embeddings

emb_matrix = load_glove_embeddings(vocab, "data/raw/glove.6B.100d.txt", embed_dim=100)
# Returns FloatTensor of shape (vocab_size, 100)
# Coverage: 7 346 / 7 400 words (99.3%)
```

Words not found in GloVe → random init `N(0, 0.01)`.
`<PAD>` token → zero vector.

### Dataset Classes

#### BERTEmotionDataset

```python
ds = BERTEmotionDataset(
    df,
    tokenizer_or_name="bert-base-uncased",   # or pass a loaded tokenizer
    label_map=HF_TO_FER_LABEL_MAP,
    max_length=128,
)
# Each item: {"input_ids": (128,), "attention_mask": (128,), "label": scalar}
```

Label resolution order:
1. If `label` is a string key in `label_map` → use mapped integer
2. Else try `int(label)` directly
3. Fallback → `0` (angry)

#### LSTMEmotionDataset

```python
ds = LSTMEmotionDataset(df, vocab, label_map=HF_TO_FER_LABEL_MAP, max_length=128)
# Each item: {"input_ids": (128,), "mask": (128,), "label": scalar}
```

`mask` is `1` for real tokens, `0` for padding.
Used by `AttentionPooling` in `BiLSTMClassifier` to exclude pad positions.

### DataLoader Factories

#### For BERT (pre-split CSVs)

```python
from src.preprocessing.text_preprocessing import get_bert_dataloaders

loaders = get_bert_dataloaders(
    train_csv="data/raw/emotion_train.csv",
    val_csv="data/raw/emotion_val.csv",
    test_csv="data/raw/emotion_test.csv",
    model_name="bert-base-uncased",
    max_length=128,
    batch_size=32,
)
```

#### For LSTM (HuggingFace auto-download)

```python
from src.preprocessing.text_preprocessing import get_text_dataloaders

loaders = get_text_dataloaders(
    model_type="lstm",
    source="huggingface",
    batch_size=128,
    max_length=128,
    vocab=vocab,
)
```

#### From Local CSV

```python
loaders = get_text_dataloaders(
    model_type="bert",
    source="csv",
    csv_path="data/raw/emotion.csv",
    batch_size=32,
)
```

---

## Bug Fixes Applied

Three non-obvious bugs were corrected in `text_preprocessing.py`:

| # | Bug                                               | Impact                                    | Fix                                         |
|---|---------------------------------------------------|-------------------------------------------|---------------------------------------------|
| 1 | `HF_TO_FER_LABEL_MAP` defined inside a function   | Not shared between functions → label mismatch | Moved to module-level constant             |
| 2 | `BERTEmotionDataset` defined twice (conflicting signatures) | Second definition silently overrode first → all labels defaulted to 0 | Merged into one canonical class    |
| 3 | `get_bert_dataloaders` passed tokenizer as `label_map` positional arg | Label lookup always failed → model predicted only class 0 | Switched to `tokenizer_or_name=` keyword |

---

## Emotion Label Index Map

```
Index   FER label    NLP source          Model coverage
  0     angry        dair-ai/emotion     CNN ✓  BERT ✓  Fusion ✓
  1     disgust      FER2013 only        CNN ✓  BERT ✗  Fusion partial
  2     fear         dair-ai/emotion     CNN ✓  BERT ✓  Fusion ✓
  3     happy        dair-ai/emotion     CNN ✓  BERT ✓  Fusion ✓
  4     neutral      FER2013 only        CNN ✓  BERT ✗  Fusion partial
  5     sad          dair-ai/emotion     CNN ✓  BERT ✓  Fusion ✓
  6     surprise     dair-ai/emotion     CNN ✓  BERT ✓  Fusion ✓
```

---

Last Updated: 23/05/2026<br>
Status: Active ✓