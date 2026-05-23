"""
api/app.py
───────────
FastAPI inference server for the Multimodal Emotion Recognition system.

Endpoints:
  POST /predict/image       — emotion from a face image
  POST /predict/text        — emotion from a text string
  POST /predict/multimodal  — emotion from image + text (fusion)
  GET  /health              — liveness check

Usage:
    pip install fastapi uvicorn python-multipart
    uvicorn api.app:app --reload --port 8000

    # Or from project root:
    python -m uvicorn api.app:app --reload

Example curl:
    curl -X POST http://localhost:8000/predict/text \
         -H "Content-Type: application/json" \
         -d '{"text": "I am so happy today!"}'

    curl -X POST http://localhost:8000/predict/image \
         -F "file=@face.jpg"

    curl -X POST http://localhost:8000/predict/multimodal \
         -F "file=@face.jpg" \
         -F "text=I feel wonderful"
"""

import os
import sys
import io
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Form
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError as e:
    raise ImportError(
        "FastAPI dependencies not installed.\n"
        "Run: pip install fastapi uvicorn python-multipart"
    ) from e

from src.models.cnn_model import EmotionCNN
from src.models.lstm_model import BERTClassifier
from src.fusion.fusion_models import MultimodalEmotionModel
from src.preprocessing.image_preprocessing import get_transforms
from src.genai.report_generator import generate_emotion_report


# ─── Config ───────────────────────────────────────────────────────────────────

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMG_SIZE        = 224
MAX_TEXT_LEN    = 128
BERT_MODEL      = os.getenv("BERT_MODEL", "bert-base-uncased")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

CNN_CHECKPOINT    = os.getenv("CNN_CHECKPOINT",
    "outputs/checkpoints/cnn_20260522_171658/best_cnn.pt")

BERT_CHECKPOINT   = os.getenv("BERT_CHECKPOINT",
    "outputs/checkpoints/bert_20260522_165038/best_bert.pt")

FUSION_CHECKPOINT = os.getenv("FUSION_CHECKPOINT",
    "outputs/checkpoints/attention_20260523_160317/best_model.pt")

FUSION_TYPE       = os.getenv("FUSION_TYPE", "attention")

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multimodal Emotion Recognition API",
    description="Detect emotions from faces, text, or both (multimodal fusion).",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Model registry ───────────────────────────────────────────────────────────

_models: dict = {}
_tokenizer = None


def _load_models():
    """Lazy-load models on first request."""
    global _models, _tokenizer

    if _models:
        return

    print(f"[API] Loading models on {DEVICE} …")
    
    print(f"[API] Working directory : {os.getcwd()}")
    print(f"[API] BERT_CHECKPOINT   : {BERT_CHECKPOINT}")
    print(f"[API] Exists?           : {os.path.exists(BERT_CHECKPOINT)}")

    # CNN
    cnn = EmotionCNN(num_classes=7, dropout=0.5, pretrained=False).to(DEVICE)
    if os.path.exists(CNN_CHECKPOINT):
        cnn.load_state_dict(torch.load(CNN_CHECKPOINT, map_location=DEVICE))
        print(f"[API] ✅ CNN loaded from {CNN_CHECKPOINT}")
    else:
        print(f"[API] ⚠️  CNN checkpoint NOT FOUND at: {CNN_CHECKPOINT}")
        print(f"[API] ⚠️  CNN running with random weights — predictions will be unreliable!")
        print(f"[API] ⚠️  Working directory: {os.getcwd()}")
    cnn.eval()
    _models["cnn"] = cnn

    # BERT
    bert = BERTClassifier(model_name=BERT_MODEL, num_classes=7, dropout=0.3).to(DEVICE)
    if os.path.exists(BERT_CHECKPOINT):
        bert.load_state_dict(torch.load(BERT_CHECKPOINT, map_location=DEVICE))
        print(f"[API] ✅ BERT loaded from {BERT_CHECKPOINT}")
    else:
        print(f"[API] ⚠️  BERT checkpoint NOT FOUND at: {BERT_CHECKPOINT}")
        print(f"[API] ⚠️  BERT running with random weights — predictions will be unreliable!")
        print(f"[API] ⚠️  Working directory: {os.getcwd()}")
    bert.eval()
    _models["bert"] = bert

    # Fusion
    fusion = MultimodalEmotionModel(
        image_encoder=cnn,
        text_encoder=bert,
        fusion_type=FUSION_TYPE,
        num_classes=7
    ).to(DEVICE)
    if os.path.exists(FUSION_CHECKPOINT):
        fusion.load_state_dict(torch.load(FUSION_CHECKPOINT, map_location=DEVICE))
        print(f"[API] ✅ Fusion model loaded from {FUSION_CHECKPOINT}")
    else:
        print(f"[API] ⚠️  FUSION checkpoint NOT FOUND at: {FUSION_CHECKPOINT}")
        print(f"[API] ⚠️  Fusion running with random weights — will always predict the same class!")
        print(f"[API] ⚠️  Working directory: {os.getcwd()}")
        print(f"[API] ⚠️  Fix: run uvicorn from the project root, or set FUSION_CHECKPOINT env var.")
    fusion.eval()
    _models["fusion"] = fusion

    # Tokenizer
    from transformers import BertTokenizerFast
    _tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)
    print("[API] Models ready.\n")


# ─── Preprocessing helpers ────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Decode image bytes → (1, 3, 224, 224) tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = get_transforms("test", IMG_SIZE)
    return transform(img).unsqueeze(0).to(DEVICE)


def preprocess_text(text: str):
    """Tokenize text → (input_ids, attention_mask) tensors."""
    enc = _tokenizer(
        text,
        max_length=MAX_TEXT_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


def logits_to_response(logits: torch.Tensor, include_report: bool = False,
                        text: str = "") -> dict:
    """Convert raw logits → structured API response."""
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    scores = {cls: round(p, 4) for cls, p in zip(EMOTION_CLASSES, probs)}
    predicted = max(scores, key=scores.get)
    confidence = scores[predicted]

    result = {
        "emotion":    predicted,
        "confidence": confidence,
        "scores":     scores,
    }

    if include_report:
        use_llm = bool(os.getenv("ANTHROPIC_API_KEY"))
        result["report"] = generate_emotion_report(
            emotion=predicted,
            scores=scores,
            user_text=text,
            use_llm=use_llm
        )

    return result


# ─── Schemas ──────────────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str
    include_report: bool = False


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: bool
    fusion_type: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return {
        "status":        "ok",
        "device":        DEVICE,
        "models_loaded": bool(_models),
        "fusion_type":   FUSION_TYPE,
    }


@app.post("/predict/text", tags=["Inference"])
def predict_text(req: TextRequest):
    """
    Predict emotion from a text string.

    Body: { "text": "I feel great today!", "include_report": false }
    """
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text cannot be empty")

    _load_models()

    input_ids, mask = preprocess_text(req.text)
    with torch.no_grad():
        logits = _models["bert"](input_ids, mask)

    return logits_to_response(logits, req.include_report, req.text)


@app.post("/predict/image", tags=["Inference"])
async def predict_image(
    file: UploadFile = File(..., description="Face image (JPEG/PNG)"),
    include_report: bool = Form(False)
):
    """Predict emotion from a face image."""
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=422,
                            detail="File must be an image (JPEG, PNG, or WebP)")

    _load_models()

    img_bytes = await file.read()
    try:
        img_tensor = preprocess_image(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {e}")

    with torch.no_grad():
        logits = _models["cnn"](img_tensor)

    return logits_to_response(logits, include_report)


@app.post("/predict/multimodal", tags=["Inference"])
async def predict_multimodal(
    file: UploadFile = File(..., description="Face image (JPEG/PNG)"),
    text: str = Form(..., description="Accompanying text / caption"),
    include_report: bool = Form(False)
):
    """
    Predict emotion from image + text.

    Strategy: run image and text independently, then combine their probability
    distributions via weighted average (image weight=0.6, text weight=0.4).

    This avoids the train/inference mismatch of the fusion model, which was
    trained on aligned (image, same-class text) pairs and produces unreliable
    results when the user's free-form text conflicts with the facial expression.
    """
    _load_models()

    img_bytes = await file.read()
    try:
        img_tensor = preprocess_image(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {e}")

    input_ids, mask = preprocess_text(text)

    with torch.no_grad():
        img_logits  = _models["cnn"](img_tensor)
        txt_logits  = _models["bert"](input_ids, mask)

        img_probs = torch.softmax(img_logits, dim=-1)
        txt_probs = torch.softmax(txt_logits, dim=-1)

        # Weighted combination: image is primary signal for facial emotion
        combined_probs = 0.6 * img_probs + 0.4 * txt_probs
        # Convert back to logits for logits_to_response
        logits = torch.log(combined_probs + 1e-8)

    result = logits_to_response(logits, include_report, text)

    # Add per-modality detail for transparency
    img_scores = {cls: round(p, 4) for cls, p in zip(EMOTION_CLASSES, img_probs.squeeze(0).cpu().tolist())}
    txt_scores = {cls: round(p, 4) for cls, p in zip(EMOTION_CLASSES, txt_probs.squeeze(0).cpu().tolist())}
    result["image_scores"] = img_scores
    result["text_scores"]  = txt_scores
    result["fusion_method"] = "weighted_average_0.6_img_0.4_txt"

    return result


@app.get("/classes", tags=["System"])
def list_classes():
    """Return the list of recognisable emotion classes."""
    return {"classes": EMOTION_CLASSES, "num_classes": len(EMOTION_CLASSES)}


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)