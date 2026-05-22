"""
src/preprocessing/text_preprocessing.py
─────────────────────────────────────────
Text preprocessing pipeline for emotion NLP datasets.

Supports:
  - BERT tokenizer (HuggingFace)
  - Custom vocabulary tokenizer for LSTM with GloVe embeddings
  - Dataset loading from CSV: (text, label) pairs
"""

import re
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast


# ─── Emotion mapping (dair-ai/emotion dataset) ────────────────────────────────
EMOTION_MAP = {
    "sadness":  0,
    "joy":      1,
    "love":     2,
    "anger":    3,
    "fear":     4,
    "surprise": 5,
}

# Align with FER2013 if using combined multimodal dataset
FER_EMOTION_MAP = {
    "angry":    0,
    "disgust":  1,
    "fear":     2,
    "happy":    3,
    "neutral":  4,
    "sad":      5,
    "surprise": 6,
}

# ─── Label map for HuggingFace dair-ai/emotion → FER2013 indices ─────────────
# BUG FIX 1: This mapping was defined inside get_text_dataloaders() and was
# NOT used by get_bert_dataloaders(). It is now a module-level constant so
# all functions share the same consistent mapping.
HF_TO_FER_LABEL_MAP = {
    "sadness":  5,   # → FER 'sad'
    "joy":      3,   # → FER 'happy'
    "love":     3,   # → FER 'happy' (closest)
    "anger":    0,   # → FER 'angry'
    "fear":     2,   # → FER 'fear'
    "surprise": 6,   # → FER 'surprise'
    # NOTE: 'disgust' (index 1) and 'neutral' (index 4) have no equivalent
    # in the dair-ai/emotion dataset, so they will correctly appear with
    # support=0 in the classification report — this is expected behaviour.
}


# ─── Text cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Minimal but effective text cleaning:
    - Lowercase
    - Remove URLs, mentions, hashtags
    - Remove special characters (keep apostrophes)
    - Collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)       # URLs
    text = re.sub(r"@\w+", "", text)                    # @mentions
    text = re.sub(r"#(\w+)", r"\1", text)               # #hashtag → word
    text = re.sub(r"[^a-z\s']", " ", text)             # keep letters + apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Vocabulary builder (for LSTM) ────────────────────────────────────────────

class Vocabulary:
    """
    Simple word-level vocabulary with special tokens.
    Used by the LSTM model with GloVe embeddings.
    """

    PAD_TOKEN = "<PAD>"   # index 0
    UNK_TOKEN = "<UNK>"   # index 1

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    def build(self, texts: list):
        """Build vocab from a list of cleaned strings."""
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        print(f"[Vocabulary] Size: {len(self.word2idx)} tokens")

    def encode(self, text: str, max_len: int) -> list:
        """Encode a string to a padded list of indices."""
        tokens = text.split()[:max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]
        pad = [0] * (max_len - len(ids))
        return ids + pad

    def __len__(self):
        return len(self.word2idx)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "rb") as f:
            return pickle.load(f)


# ─── GloVe embedding matrix ───────────────────────────────────────────────────

def load_glove_matrix(glove_path: str, vocab: Vocabulary, embed_dim: int = 100) -> torch.Tensor:
    """
    Build an embedding matrix aligned with `vocab` from a GloVe .txt file.
    Unknown words → random init.  PAD → zeros.

    Returns: FloatTensor of shape (vocab_size, embed_dim)
    """
    print(f"[GloVe] Loading {glove_path} ...")
    glove = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            glove[word] = vec

    matrix = np.zeros((len(vocab), embed_dim), dtype=np.float32)
    hits = 0
    for word, idx in vocab.word2idx.items():
        if word in glove:
            matrix[idx] = glove[word]
            hits += 1
        elif idx > 1:  # not PAD or UNK
            matrix[idx] = np.random.normal(0, 0.01, embed_dim)

    print(f"[GloVe] Coverage: {hits}/{len(vocab)} words ({100*hits/len(vocab):.1f}%)")
    return torch.tensor(matrix)


def load_glove_embeddings(vocab: "Vocabulary", glove_path: str,
                           embed_dim: int = 100) -> "torch.Tensor":
    """
    Load GloVe vectors for words in the vocabulary.
    Words not found in GloVe are initialised to random vectors.

    Returns:
        FloatTensor of shape (vocab_size, embed_dim)
    """
    vectors = np.random.randn(len(vocab), embed_dim).astype(np.float32) * 0.01
    found   = 0

    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word  = parts[0]
            if word in vocab.word2idx:
                idx = vocab.word2idx[word]
                vectors[idx] = np.array(parts[1:], dtype=np.float32)
                found += 1

    print(f"[GloVe] Loaded {found}/{len(vocab)} vectors from {glove_path}")
    return torch.FloatTensor(vectors)


# ─── Datasets ─────────────────────────────────────────────────────────────────

# BUG FIX 2: The file previously defined BERTEmotionDataset and
# LSTMEmotionDataset TWICE (at lines ~146 and ~457).  Python's class
# resolution means the second definition silently overwrites the first.
# The second BERTEmotionDataset accepted (df, tokenizer, max_length) while
# the first accepted (df, label_map, model_name, max_length).
# get_bert_dataloaders() called the second signature but train_bert.py relied
# on the first — causing a silent label-map miss (all labels defaulted to 0).
# Solution: keep ONE canonical definition per class, combining the best of
# both signatures so every caller works correctly.

class BERTEmotionDataset(Dataset):
    """
    Tokenizes text with BERT tokenizer.

    Accepts either a pre-built tokenizer object OR a HuggingFace model name
    string. When a string is passed the tokenizer is loaded once and cached.

    CSV / DataFrame must have columns: text, label.
    'label' may be a string emotion name (looked up in label_map) or already
    an integer index (used as-is when label_map is None or the key is absent).
    """

    def __init__(self, df: "pd.DataFrame",
                 tokenizer_or_name=None,
                 label_map: dict = None,
                 max_length: int = 128):
        self.df         = df.reset_index(drop=True)
        self.label_map  = label_map or {}
        self.max_length = max_length

        # Accept either a tokenizer object or a model-name string
        if tokenizer_or_name is None or isinstance(tokenizer_or_name, str):
            model_name = tokenizer_or_name or "bert-base-uncased"
            self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer_or_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        text  = clean_text(str(row["text"]))
        raw_label = str(row["label"]).strip().lower()
        # Use label_map when the label is a string; fall back to int cast
        if raw_label in self.label_map:
            label = self.label_map[raw_label]
        else:
            try:
                label = int(raw_label)
            except ValueError:
                label = 0  # unknown → default to class 0

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


class LSTMEmotionDataset(Dataset):
    """
    Encodes text using a pre-built Vocabulary.
    CSV / DataFrame must have columns: text, label.
    """

    def __init__(self, df: "pd.DataFrame", vocab: "Vocabulary",
                 label_map: dict = None, max_length: int = 128):
        self.df         = df.reset_index(drop=True)
        self.vocab      = vocab
        self.label_map  = label_map or {}
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = clean_text(str(row["text"]))
        raw_label = str(row["label"]).strip().lower()
        if raw_label in self.label_map:
            label = self.label_map[raw_label]
        else:
            try:
                label = int(raw_label)
            except ValueError:
                label = 0

        tokens  = text.split()[:self.max_length]
        ids     = [self.vocab.word2idx.get(t, 1) for t in tokens]   # 1 = <UNK>
        pad_len = self.max_length - len(ids)
        mask    = [1] * len(ids) + [0] * pad_len
        ids     = ids + [0] * pad_len   # 0 = <PAD>

        return {
            "input_ids": torch.tensor(ids,  dtype=torch.long),
            "mask":      torch.tensor(mask, dtype=torch.long),
            "label":     torch.tensor(label, dtype=torch.long),
        }


# ─── DataLoader factory ───────────────────────────────────────────────────────

def get_text_dataloaders_from_csv(csv_path: str,
                                   model_type: str = "bert",
                                   label_map: dict = None,
                                   val_ratio: float = 0.1,
                                   test_ratio: float = 0.1,
                                   batch_size: int = 32,
                                   max_length: int = 128,
                                   vocab: Vocabulary = None) -> dict:
    """
    Load a CSV (columns: text, label), split into train/val/test, return DataLoaders.
    Compatible with Kaggle datasets (praveengovi/emotions-dataset-for-nlp, etc.)
    """
    if label_map is None:
        label_map = FER_EMOTION_MAP

    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df)
    n_val  = int(n * val_ratio)
    n_test = int(n * test_ratio)

    splits = {
        "train": df.iloc[n_val + n_test:],
        "val":   df.iloc[:n_val],
        "test":  df.iloc[n_val:n_val + n_test],
    }

    loaders = {}
    for name, subset in splits.items():
        if model_type == "bert":
            dataset = BERTEmotionDataset(subset, label_map=label_map, max_length=max_length)
        else:
            assert vocab is not None, "vocab required for LSTM"
            dataset = LSTMEmotionDataset(subset, vocab, label_map=label_map, max_length=max_length)

        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=2,
            pin_memory=True
        )
    return loaders


def get_text_dataloaders(model_type: str = "bert",
                         source: str = "huggingface",
                         csv_path: str = None,
                         batch_size: int = 32,
                         max_length: int = 128,
                         vocab: Vocabulary = None) -> dict:
    """
    Unified DataLoader factory supporting two data sources:

    source='huggingface'  →  loads dair-ai/emotion automatically (recommended)
                              pip install datasets  — no Kaggle account needed
                              ~16 000 train / 2 000 val / 2 000 test
                              Labels: sadness joy love anger fear surprise

    source='csv'          →  loads from a local CSV (columns: text, label)
                              Compatible with:
                              - kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp
                              - kaggle.com/datasets/pashupatigupta/emotion-detection-from-text
                              - Any CSV with 'text' and 'label' columns
    """
    if source == "huggingface":
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets  — required for HuggingFace source")

        print("[TextData] Loading dair-ai/emotion from HuggingFace ...")
        hf_dataset = load_dataset("dair-ai/emotion")

        def hf_split_to_df(split_name):
            split = hf_dataset[split_name]
            label_names = split.features["label"].names
            rows = [{"text": ex["text"],
                     "label": label_names[ex["label"]]}
                    for ex in split]
            return pd.DataFrame(rows)

        splits = {
            "train": hf_split_to_df("train"),
            "val":   hf_split_to_df("validation"),
            "test":  hf_split_to_df("test"),
        }
        label_map = HF_TO_FER_LABEL_MAP

    elif source == "csv":
        assert csv_path, "csv_path required when source='csv'"
        return get_text_dataloaders_from_csv(
            csv_path=csv_path,
            model_type=model_type,
            batch_size=batch_size,
            max_length=max_length,
            vocab=vocab
        )
    else:
        raise ValueError(f"Unknown source: {source}. Use 'huggingface' or 'csv'.")

    loaders = {}
    for name, df in splits.items():
        if model_type == "bert":
            dataset = BERTEmotionDataset(df, label_map=label_map, max_length=max_length)
        else:
            assert vocab is not None, "vocab required for LSTM"
            dataset = LSTMEmotionDataset(df, vocab, label_map=label_map, max_length=max_length)

        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=2,
            pin_memory=True
        )
    return loaders


# ─── Inference utility ────────────────────────────────────────────────────────

def encode_single_text_bert(text: str,
                             model_name: str = "bert-base-uncased",
                             max_length: int = 128) -> dict:
    """
    Tokenize a single string for BERT inference.
    Returns dict with input_ids and attention_mask tensors (batch size 1).
    """
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    enc = tokenizer(clean_text(text), max_length=max_length,
                    padding="max_length", truncation=True, return_tensors="pt")
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


# ─── CSV helpers ──────────────────────────────────────────────────────────────

def load_emotion_csv(path: str) -> "pd.DataFrame":
    """Load a preprocessed emotion CSV with columns: text, label."""
    return pd.read_csv(path)


# BUG FIX 3: get_bert_dataloaders used to call BERTEmotionDataset(df, tokenizer, max_length)
# — i.e. passing the tokenizer as the second positional arg (label_map position).
# This meant label_map was a tokenizer object, so every label lookup failed and
# defaulted to 0 (angry), giving misleading near-perfect training accuracy while
# the model only ever predicted one class.
# Fix: pass tokenizer via keyword arg; also inject HF_TO_FER_LABEL_MAP so
# labels map correctly to the 7 FER classes.
def get_bert_dataloaders(train_csv: str, val_csv: str, test_csv: str,
                          model_name: str = "bert-base-uncased",
                          max_length: int = 128,
                          batch_size: int = 32) -> dict:
    """
    Build train / val / test DataLoaders for BERT training from pre-split CSVs.

    Returns:
        dict with keys 'train', 'val', 'test'
    """
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    loaders = {}
    for name, path in [("train", train_csv), ("val", val_csv), ("test", test_csv)]:
        df = load_emotion_csv(path)
        ds = BERTEmotionDataset(
            df,
            tokenizer_or_name=tokenizer,   # ← correct keyword
            label_map=HF_TO_FER_LABEL_MAP, # ← inject label map
            max_length=max_length
        )
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=2
        )
        print(f"[BERT DataLoader] {name}: {len(ds):,} samples")

    return loaders


# ─── Entry-point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_texts = [
        "I am so happy today! Everything is wonderful!",
        "I feel absolutely terrible and hopeless.",
        "This is making me really angry!!",
        "I'm scared of what might happen next.",
    ]

    print("=== Clean text examples ===")
    for t in sample_texts:
        print(f"  IN : {t}")
        print(f"  OUT: {clean_text(t)}\n")

    # Vocabulary quick test
    vocab = Vocabulary(min_freq=1)
    vocab.build([clean_text(t) for t in sample_texts])
    print(f"Vocab size: {len(vocab)}")
    encoded = vocab.encode(clean_text(sample_texts[0]), max_len=20)
    print(f"Encoded  : {encoded}")
