# data/ — Raw & Processed Datasets

The `data/` folder stores all input data consumed by the pipeline, split into two sub-directories:

- `raw/` — original files downloaded from Kaggle / HuggingFace / Stanford NLP (never modified)
- `processed/` — preprocessed CSVs and pickled objects produced by `scripts/preprocess_all.py`

> **Both sub-directories are in `.gitignore`.** Only the preprocessing code is version-controlled.

---

## Folder Structure

```
data/
│
├── raw/
│   ├── fer2013/                      # Format A — folder-based (most common)
│   │   ├── train/
│   │   │   ├── angry/     *.jpg
│   │   │   ├── disgust/   *.jpg
│   │   │   ├── fear/      *.jpg
│   │   │   ├── happy/     *.jpg
│   │   │   ├── neutral/   *.jpg
│   │   │   ├── sad/       *.jpg
│   │   │   └── surprise/  *.jpg
│   │   └── test/
│   │       └── <same 7 class folders>
│   │
│   ├── fer2013.csv                   # Format B — original CSV (alternative)
│   │
│   ├── emotion.csv                   # dair-ai/emotion (auto-downloaded)
│   ├── emotion_train.csv             # produced by preprocess_all.py
│   ├── emotion_val.csv               # produced by preprocess_all.py
│   ├── emotion_test.csv              # produced by preprocess_all.py
│   │
│   └── glove.6B.100d.txt             # Stanford GloVe 6B 100-d embeddings
│
└── processed/                        # Reserved for future derived artifacts
```

---

## Dataset 1 — FER2013 (Vision)

| Property | Value |
|----------|-------|
| Task | Facial expression recognition |
| Classes | 7: angry, disgust, fear, happy, neutral, sad, surprise |
| Total images | ~35 000 |
| Native resolution | 48 × 48 px, grayscale |
| Resize to | 224 × 224 px (ResNet-50 / ViT requirement) |
| Source | [Kaggle — msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013) |
| Format | Folder-based (auto-detected) |

### Class Distribution (FER2013 test set — 7 178 images)

```
angry   :   958  (13.3%)  ████████
disgust :   111  ( 1.5%)  █
fear    : 1 024  (14.3%)  █████████
happy   : 1 774  (24.7%)  ████████████████
neutral : 1 233  (17.2%)  ███████████
sad     : 1 247  (17.4%)  ███████████
surprise:   831  (11.6%)  ████████
```

`disgust` is severely underrepresented (111 vs 1 774 for `happy`), which explains its lower F1 score (0.39 vs 0.87).

### Supported Formats

The pipeline auto-detects which format is present:

**Format A — Folder-based (Kaggle msambare/fer2013)**
```
data/raw/fer2013/
  train/  angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
  test/   <same 7 folders>
```

**Format B — CSV (original FER2013)**
```
data/raw/fer2013.csv
  columns: emotion (int 0–6), pixels (space-separated 48×48), Usage
  Usage values: 'Training' | 'PublicTest' | 'PrivateTest'
```

### Data Split (Format A)

The folder-based format has no validation split, so 10% of training images are carved out automatically:

```
train/  28 709 images → 25 838 train + 2 871 val
test/    7 178 images → held-out test set
```

### Download Instructions

```bash
# Kaggle CLI
kaggle datasets download -d msambare/fer2013 -p data/raw/
unzip data/raw/fer2013.zip -d data/raw/fer2013/

# Or download manually from:
# https://www.kaggle.com/datasets/msambare/fer2013
```

---

## Dataset 2 — dair-ai/emotion (Text)

| Property | Value |
|----------|-------|
| Task | Text emotion classification |
| Classes | 6 NLP labels → 5 FER labels (remapped) |
| Total | 20 000 sentences |
| Split | 16 000 train / 2 000 val / 2 000 test (pre-defined) |
| Source | [HuggingFace dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) |
| Auto-download | Yes — `preprocess_all.py` fetches it automatically |

### Label Mapping — NLP → FER

| NLP label | → FER label | FER index | Rationale |
|-----------|-------------|-----------|-----------|
| `sadness` | `sad` | 5 | Direct equivalence |
| `joy` | `happy` | 3 | Direct equivalence |
| `love` | `happy` | 3 | Closest positive valence |
| `anger` | `angry` | 0 | Direct equivalence |
| `fear` | `fear` | 2 | Direct equivalence |
| `surprise` | `surprise` | 6 | Direct equivalence |

`disgust` (index 1) and `neutral` (index 4) exist only in FER2013. Text models have **zero support** for these two classes — their F1 is 0.00, which is expected.

### Preprocessed CSVs

```
data/raw/emotion_train.csv    16 000 rows
data/raw/emotion_val.csv       2 000 rows
data/raw/emotion_test.csv      2 000 rows
```

Each row:

| Column | Type | Example |
|--------|------|---------|
| `text` | string | `"i feel absolutely devastated right now"` |
| `label` | int | `5` (→ sad) |
| `emotion` | string | `"sad"` |

---

## Dataset 3 — GloVe 6B 100-d (Word Embeddings)

| Property | Value |
|----------|-------|
| Source | [Stanford NLP](https://nlp.stanford.edu/data/glove.6B.zip) |
| File | `glove.6B.100d.txt` |
| Vectors | 400 000 tokens × 100 dimensions |
| File size | ~822 MB (extracted) |
| Coverage | 7 346 / 7 400 LSTM vocabulary words (99.3%) |

GloVe is used **only** by the BiLSTM model. Skip the download if you only plan to use BERT or fusion models:

```bash
python scripts/preprocess_all.py --skip_glove
```

### Manual Download

```bash
wget https://nlp.stanford.edu/data/glove.6B.zip -P data/raw/
unzip data/raw/glove.6B.zip glove.6B.100d.txt -d data/raw/
rm data/raw/glove.6B.zip
```

---

## Running the Data Pipeline

```bash
# Full pipeline
python scripts/preprocess_all.py

# Skip GloVe (BERT-only training)
python scripts/preprocess_all.py --skip_glove

# Custom raw data directory
python scripts/preprocess_all.py --data_dir /path/to/my/data
```

---

## File Size Overview

```
Large files (not committed):
├── data/raw/fer2013/           ~200 MB   (folder format)
├── data/raw/fer2013.csv        ~350 MB   (CSV format)
├── data/raw/glove.6B.100d.txt  ~822 MB

Medium files:
├── data/raw/emotion.csv         ~1.8 MB
├── data/raw/emotion_train.csv   ~1.4 MB
├── data/raw/emotion_val.csv     ~180 KB
├── data/raw/emotion_test.csv    ~180 KB
```

---

*Last Updated: 23/05/2026 — Status: Active ✓*
