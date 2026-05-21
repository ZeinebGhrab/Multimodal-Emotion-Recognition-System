# Multimodal Emotion Recognition System
### Deep Learning + Generative AI — Academic Project

> Detect human emotions from facial images (FER2013) and text (tweets), fuse both
> modalities, and generate an AI-powered emotional report via the Claude API.

---

## Project Structure

```
multimodal_emotion_recognition/
├── configs/
│   └── config.yaml              # All hyperparameters & paths
├── data/
│   ├── raw/                     # Original datasets (not committed)
│   └── processed/               # Preprocessed tensors / CSVs
├── src/
│   ├── preprocessing/
│   │   ├── image_preprocessing.py   # FER2013 Dataset + transforms
│   │   └── text_preprocessing.py   # BERT/LSTM tokenizers + helpers
│   ├── models/
│   │   ├── cnn_model.py             # ResNet-50 (image only)
│   │   ├── vit_model.py             # Vision Transformer ViT-B/16
│   │   └── lstm_model.py            # Bi-LSTM + BERT classifiers
│   ├── fusion/
│   │   └── fusion_models.py         # Early | Late | Attention fusion
│   ├── evaluation/
│   │   └── metrics.py               # Accuracy, F1, confusion matrix, plots
│   └── genai/
│       └── report_generator.py      # Claude-powered emotion reports
├── api/
│   └── app.py                   # FastAPI inference server
├── scripts/
│   ├── preprocess_all.py        # One-shot data preparation
│   ├── train_cnn.py             # Train ResNet-50
│   ├── train_lstm.py            # Train BiLSTM + GloVe
│   ├── train_bert.py            # Train BERT classifier
│   ├── train_multimodal.py      # Train multimodal fusion
│   └── compare_models.py        # Compare all checkpoints
├── notebooks/
│   └── full_pipeline.ipynb      # End-to-end Jupyter walkthrough
├── outputs/
│   ├── checkpoints/             # Saved model weights
│   ├── figures/                 # Plots & confusion matrices
│   └── reports/                 # Generated JSON reports
└── requirements.txt
```

---

## Datasets

| Dataset | Task | Source |
|---|---|---|
| FER2013 | Facial expression (7 classes) | [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) |
| Emotion NLP | Text emotion (6–7 classes) | [Kaggle — dair-ai/emotion](https://www.kaggle.com/datasets/parulpandey/emotion-dataset-for-nlp) |
| GloVe 6B | Word embeddings (100d) | [Stanford NLP](https://nlp.stanford.edu/data/glove.6B.zip) |

Place downloaded files in `data/raw/`.

---

## Models

| Model | Modality | Backbone | Expected Acc. |
|---|---|---|---|
| CNN Baseline | Image | ResNet-50 | ~65% |
| Vision Transformer | Image | ViT-B/16 | ~68% |
| Bi-LSTM | Text | GloVe 100d | ~70% |
| BERT Classifier | Text | bert-base-uncased | ~78% |
| Early Fusion | Image + Text | CNN + BERT | ~80% |
| Late Fusion | Image + Text | Ensemble | ~81% |
| Attention Fusion | Image + Text | Cross-Attention | ~83% |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download & preprocess data
python scripts/preprocess_all.py

# 3. Train individual models
python scripts/train_cnn.py
python scripts/train_bert.py
python scripts/train_lstm.py

# 4. Train multimodal fusion
python scripts/train_multimodal.py --fusion attention

# 5. Compare all models
python scripts/compare_models.py

# 6. Generate emotion report (GenAI — needs ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python src/genai/report_generator.py --emotion happy --text "I feel great today!"
```

---

## FastAPI Server

```bash
# Start the inference server
uvicorn api.app:app --reload --port 8000

# Predict from text
curl -X POST http://localhost:8000/predict/text \
     -H "Content-Type: application/json" \
     -d '{"text": "I am so excited!"}'

# Predict from image
curl -X POST http://localhost:8000/predict/image \
     -F "file=@face.jpg"

# Multimodal prediction (most accurate)
curl -X POST http://localhost:8000/predict/multimodal \
     -F "file=@face.jpg" \
     -F "text=I feel wonderful today"

# Swagger UI
open http://localhost:8000/docs
```

---

## Fusion Strategies

### Early Fusion (Concatenation)
```
[img_feats] ‖ [txt_feats] → MLP → logits
```
Simple, fast. Lets the network freely learn cross-modal interactions.

### Late Fusion (Ensemble)
```
CNN → P_img    BERT → P_txt
P_final = α·P_img + (1-α)·P_txt   (learned α)
```
Works well when unimodal models are already strong.

### Attention Fusion (Best)
```
img_proj → Cross-Attention(img→txt) → gate
txt_proj → Cross-Attention(txt→img) → gate
[img_out ‖ txt_out] → MLP → logits
```
Bidirectional cross-attention explicitly models which image regions relate to
which text tokens — captures subtle emotional interactions.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Deep Learning | PyTorch 2.x |
| NLP | Hugging Face Transformers |
| Vision | torchvision, timm (ViT), OpenCV |
| Data | Pandas, NumPy, scikit-learn |
| Visualisation | Matplotlib, Seaborn |
| GenAI Reports | Anthropic Claude API |
| Serving | FastAPI + Uvicorn |

---

## Emotion Classes

`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for GenAI reports |
| `CNN_CHECKPOINT` | `outputs/checkpoints/best_cnn.pt` | CNN model path for API |
| `BERT_CHECKPOINT` | `outputs/checkpoints/best_bert.pt` | BERT model path for API |
| `FUSION_CHECKPOINT` | `outputs/checkpoints/best_fusion.pt` | Fusion model for API |
| `FUSION_TYPE` | `attention` | Fusion strategy in API (`early`/`late`/`attention`) |

---

## Authors

Academic project — Deep Learning & Multimodal AI
