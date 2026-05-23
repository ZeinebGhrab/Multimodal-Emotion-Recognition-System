🌐 api/ — FastAPI Inference Server
====================================

## Overview

The API module exposes the trained models as REST endpoints.
It is the backend that the Streamlit UI and the Ollama agent call
during inference. Models are loaded lazily on the first request.

```
api/
└── app.py     FastAPI application (5 endpoints, lazy model loading)
```

---

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn python-multipart

# Start the server
uvicorn api.app:app --reload --port 8000

# Interactive docs
open http://localhost:8000/docs
```

---

## Endpoints

| Method | Path                 | Input                    | Model used        | Description                   |
|--------|----------------------|--------------------------|-------------------|-------------------------------|
| `GET`  | `/health`            | —                        | —                 | Liveness check + model status |
| `GET`  | `/classes`           | —                        | —                 | List of 7 emotion class names |
| `POST` | `/predict/text`      | JSON body                | BERT              | Emotion from text             |
| `POST` | `/predict/image`     | Form file upload         | ResNet-50         | Emotion from face image       |
| `POST` | `/predict/multimodal`| File + form text         | Attention Fusion  | Emotion from image + text     |

---

## Endpoint Details

### GET /health

```json
{
  "status": "ok",
  "device": "cuda",
  "models_loaded": true,
  "fusion_type": "attention"
}
```

`models_loaded` is `false` until the first prediction request arrives
(lazy loading).

---

### POST /predict/text

**Request:**
```bash
curl -X POST http://localhost:8000/predict/text \
     -H "Content-Type: application/json" \
     -d '{"text": "I feel wonderful today!", "include_report": false}'
```

**Body schema:**

| Field            | Type    | Required | Default | Description                         |
|------------------|---------|----------|---------|-------------------------------------|
| `text`           | str     | ✅       | —       | Text to classify (min 1 char)       |
| `include_report` | bool    | ❌       | false   | Append GenAI psychological report   |

---

### POST /predict/image

**Request:**
```bash
curl -X POST http://localhost:8000/predict/image \
     -F "file=@face.jpg"
```

Accepted MIME types: `image/jpeg`, `image/png`, `image/webp`.
The file is decoded in memory — not saved to disk.

---

### POST /predict/multimodal

**Request:**
```bash
curl -X POST http://localhost:8000/predict/multimodal \
     -F "file=@face.jpg" \
     -F "text=I feel wonderful today" \
     -F "include_report=true"
```

This is the **most accurate endpoint** — uses the Attention Fusion model
which combines both modalities for ~83% accuracy.

---

## Response Schema (all prediction endpoints)

```json
{
  "emotion":    "happy",
  "confidence": 0.8732,
  "scores": {
    "angry":    0.0124,
    "disgust":  0.0031,
    "fear":     0.0098,
    "happy":    0.8732,
    "neutral":  0.0701,
    "sad":      0.0212,
    "surprise": 0.0102
  },
  "report": { ... }    // only present when include_report=true
}
```

---

## Lazy Model Loading

Models are **not** loaded at server startup. They load on the first
prediction request and stay in memory for all subsequent requests.

```python
_models: dict = {}
_tokenizer = None

def _load_models():
    global _models, _tokenizer
    if _models:
        return          # already loaded → skip

    # Load CNN, BERT, Fusion model, Tokenizer
    ...
    print("[API] Models ready.")
```

**Reason:** Loading all three models (~2 GB combined) takes ~20 seconds.
Lazy loading allows the server health check to respond immediately, and
the first prediction request pays the loading cost.

---

## Preprocessing Inside the API

### Image preprocessing

```python
def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = get_transforms("test", IMG_SIZE)    # test → no augmentation
    return transform(img).unsqueeze(0).to(DEVICE)   # (1, 3, 224, 224)
```

### Text preprocessing

```python
def preprocess_text(text: str):
    enc = _tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
```

### logits_to_response()

```python
def logits_to_response(logits, include_report=False, text=""):
    probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    scores = {cls: round(p, 4) for cls, p in zip(EMOTION_CLASSES, probs)}
    predicted  = max(scores, key=scores.get)
    confidence = scores[predicted]
    result = {"emotion": predicted, "confidence": confidence, "scores": scores}
    if include_report:
        result["report"] = generate_emotion_report(...)
    return result
```

---

## Environment Variables

| Variable           | Default                               | Description                    |
|--------------------|---------------------------------------|--------------------------------|
| `CNN_CHECKPOINT`   | `outputs/checkpoints/best_cnn.pt`    | Path to trained ResNet-50 weights |
| `BERT_CHECKPOINT`  | `outputs/checkpoints/best_bert.pt`   | Path to trained BERT weights   |
| `FUSION_CHECKPOINT`| `outputs/checkpoints/best_fusion.pt` | Path to trained Fusion weights |
| `FUSION_TYPE`      | `attention`                           | Fusion strategy used           |
| `BERT_MODEL`       | `bert-base-uncased`                  | HuggingFace model identifier   |
| `ANTHROPIC_API_KEY`| —                                     | Enables LLM reports            |

```bash
# Run with custom checkpoint paths
CNN_CHECKPOINT=outputs/checkpoints/cnn_20260521/best_cnn.pt \
uvicorn api.app:app --port 8000
```

---

## CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # all origins — restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)
```

For production, replace `"*"` with the Streamlit app URL:

```python
allow_origins=["http://localhost:8501", "https://your-domain.com"]
```

---

## Error Handling

| Condition                      | HTTP Status | Response                                   |
|--------------------------------|-------------|---------------------------------------------|
| Empty text body                | 422         | `{"detail": "text cannot be empty"}`       |
| Non-image file uploaded        | 422         | `{"detail": "File must be an image..."}`   |
| Corrupt / undecodable image    | 422         | `{"detail": "Could not decode image: ..."}`|
| Checkpoint file missing        | 200 (warn)  | Model uses random weights — logs warning   |

---

## Running in Production

```bash
# Multi-worker production server
uvicorn api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --no-access-log
```

> **Note:** With multiple workers, each worker loads its own copy of the
> models. On a machine with 8 GB GPU RAM and 3 models, use `--workers 1`
> or switch to a model server (TorchServe, Triton Inference Server).

---

Last Updated: 23/05/2026<br>
Status: Active ✓