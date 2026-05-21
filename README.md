# Multimodal Emotion Recognition System
### Deep Learning + Generative AI — Academic Project

> Detect human emotions from facial images (FER2013) and text (tweets), fuse both modalities, and generate an AI-powered emotional report.

---

## Project Structure

```
multimodal_emotion_recognition/
├── configs/
│   └── config.yaml              # All hyperparameters & paths
├── data/
│   ├── raw/                     # Original datasets
│   └── processed/               # Preprocessed tensors/CSVs
├── src/
│   ├── preprocessing/
│   │   ├── image_preprocessing.py
│   │   └── text_preprocessing.py
│   ├── models/
│   │   ├── cnn_model.py         # ResNet-based CNN (image only)
│   │   ├── vit_model.py         # Vision Transformer (bonus)
│   │   ├── lstm_model.py        # Bi-LSTM (text only)
│   │   └── bert_model.py        # BERT classifier (text only)
│   ├── fusion/
│   │   ├── early_fusion.py      # Concatenation-based fusion
│   │   ├── late_fusion.py       # Ensemble / score fusion
│   │   └── attention_fusion.py  # Cross-attention fusion
│   ├── evaluation/
│   │   └── metrics.py           # Accuracy, F1, confusion matrix
│   └── genai/
│       └── report_generator.py  # LLM-powered emotion report
├── notebooks/
│   └── full_pipeline.ipynb      # End-to-end Jupyter notebook
├── scripts/
│   ├── train_cnn.py
│   ├── train_lstm.py
│   ├── train_bert.py
│   ├── train_multimodal.py
│   └── compare_models.py
└── outputs/
    ├── checkpoints/             # Saved model weights
    ├── figures/                 # Plots & confusion matrices
    └── reports/                 # Generated PDF/JSON reports
```

---

## Datasets

| Dataset | Task | Source |
|---|---|---|
| FER2013 | Facial expression (7 classes) | [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) |
| Emotion NLP | Text emotion (6–7 classes) | [Kaggle — dair-ai/emotion](https://www.kaggle.com/datasets/parulpandey/emotion-dataset-for-nlp) |

---

## Models Comparison

| Model | Modality | Backbone | Expected Acc. |
|---|---|---|---|
| CNN Baseline | Image | ResNet-50 | ~65% |
| Vision Transformer | Image | ViT-B/16 | ~68% |
| Bi-LSTM | Text | GloVe 100d | ~70% |
| BERT Classifier | Text | bert-base | ~78% |
| Early Fusion | Image+Text | CNN+BERT | ~80% |
| Late Fusion | Image+Text | Ensemble | ~81% |
| Attention Fusion | Image+Text | Cross-Attn | ~83% |

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

# 4. Train multimodal fusion
python scripts/train_multimodal.py --fusion attention

# 5. Compare all models
python scripts/compare_models.py

# 6. Generate emotion report (GenAI)
python src/genai/report_generator.py --image path/to/face.jpg --text "I feel great today!"
```

---

## Tech Stack

- **Deep Learning**: PyTorch 2.x
- **NLP**: Hugging Face Transformers
- **Vision**: torchvision, OpenCV
- **Data**: Pandas, NumPy
- **Viz**: Matplotlib, Seaborn
- **GenAI**: Anthropic Claude API / HuggingFace Inference
- **Serving** (optional): FastAPI + React

---

## Emotion Classes

`happy` · `sad` · `angry` · `fear` · `surprise` · `neutral` · `disgust`

---

## Authors

Academic project — Deep Learning & Multimodal AI
