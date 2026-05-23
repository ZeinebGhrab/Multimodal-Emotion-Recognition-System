# src/genai/ — GenAI Emotion Report Generator

Produces a structured, human-readable psychological report from the model's prediction. Supports two generation modes:

- **LLM mode** — calls the Claude API for rich, empathetic, clinically-informed reports
- **Rule-based mode** — generates reports from predefined psychological profiles (fast, offline, deterministic)

```
src/genai/
├── __init__.py
└── report_generator.py    generate_emotion_report() | print_report()
```

---

## Generation Modes

| Mode | Trigger | Output style |
|------|---------|-------------|
| **LLM** | `ANTHROPIC_API_KEY` is set | Rich, empathetic, personalised, JSON |
| **Rule-based** | No API key present | Template-based, fast, offline |

Both modes produce the same JSON schema.

---

## Report Schema

```json
{
  "timestamp": "2026-05-21T14:32:00",
  "emotion": "sad",
  "confidence": 0.812,
  "top_emotions": [
    {"emotion": "sad",     "score": 0.812},
    {"emotion": "neutral", "score": 0.103},
    {"emotion": "fear",    "score": 0.051}
  ],
  "description":            "The detected emotion is sad (confidence: 81%), characterised by...",
  "psychological_insight":  "From a psychological perspective, sadness...",
  "physical_signals":       "drooped corners of mouth, lowered gaze, slumped posture",
  "recommendations": [
    "Allow yourself to feel — suppressing sadness prolongs it",
    "Reach out to a trusted person",
    "Gentle activity (walking, journaling) can help shift mood"
  ],
  "wellbeing_tip": "...",
  "source": "claude:claude-sonnet-4-20250514"
}
```

`source` is `"rule-based"` when the fallback is used.

---

## Usage

### From the command line

```bash
export ANTHROPIC_API_KEY=sk-ant-...

python src/genai/report_generator.py \
  --emotion happy \
  --text "I feel great today!"
```

### From Python

```python
from src.genai.report_generator import generate_emotion_report, print_report

scores = {
    "happy": 0.87, "neutral": 0.07, "sad": 0.02,
    "angry": 0.01, "fear": 0.01, "surprise": 0.01, "disgust": 0.01
}

report = generate_emotion_report(
    emotion="happy",
    scores=scores,
    user_text="I feel wonderful today!",   # optional — enriches LLM prompt
)

print_report(report)
```

### Via the FastAPI server

```bash
curl -X POST http://localhost:8000/predict/text \
     -H "Content-Type: application/json" \
     -d '{"text": "I feel wonderful today!", "include_report": true}'
```

---

## Rule-Based Fallback

Each emotion has a predefined psychological profile in `EMOTION_PROFILES`:

```python
EMOTION_PROFILES = {
    "happy": {
        "description":            "positive affect and elevated mood",
        "psychological_insight":  "Happiness activates reward circuits...",
        "physical_signals":       "raised cheeks, lip corners pulled up, bright eyes",
        "recommendations": [
            "Share your positive mood with others",
            "Use this energy for creative tasks or physical activity",
            "Note what triggered this feeling for future reference"
        ],
        "wellbeing_tip": "Savouring positive moments extends their duration."
    },
    "sad": { ... },
    # ... all 7 emotions
}
```

The fallback is triggered automatically when `ANTHROPIC_API_KEY` is not set, so the system always produces a report regardless of API availability.

---

## Configuration (config.yaml)

```yaml
genai:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"
  max_tokens: 600
  temperature: 0.7
```

Set `temperature: 0.0` for deterministic, reproducible reports in testing.

---

## Environment Variable

| Variable | Effect |
|----------|--------|
| `ANTHROPIC_API_KEY` | If set → LLM mode. If absent → rule-based mode. |

---

*Last Updated: 23/05/2026 — Status: Active ✓*
