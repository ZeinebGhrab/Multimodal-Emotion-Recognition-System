✨ src/genai/ — GenAI Emotion Report Generator
================================================

## Overview

The GenAI module produces a structured, human-readable emotional analysis
report from the model's prediction. It supports two generation modes:

- **LLM mode** (default): calls the Claude API for rich, empathetic,
  clinically-informed reports in JSON format
- **Rule-based mode** (fallback): generates reports from predefined
  psychological profiles — fast, offline, deterministic

```
src/genai/
├── __init__.py
└── report_generator.py    generate_emotion_report() | print_report()
```

---

## Report Structure

Both modes produce the same JSON schema:

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
  "description": "The detected emotion is sad (confidence: 81%), characterised by...",
  "psychological_insight": "From a psychological perspective, sadness...",
  "physical_signals": "drooped corners of mouth, lowered gaze, slumped posture",
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

## EMOTION_PROFILES — Rule-Based Templates

Each emotion has a predefined psychological profile:

```python
EMOTION_PROFILES = {
    "happy": {
        "description": "positive affect and elevated mood",
        "body_signals": "raised cheeks, relaxed brow, open posture",
        "psychology":   "associated with dopamine/serotonin release; linked to social connection",
        "suggestions":  [
            "Savour the moment through mindful appreciation",
            "Share the positive energy — it is contagious",
            "Use this high-energy state for creative or demanding tasks"
        ]
    },
    "sad": { ... },
    "angry": { ... },
    "fear": { ... },
    "surprise": { ... },
    "neutral": { ... },
    "disgust": { ... },
}
```

---

## Generation Modes

### Mode 1 — LLM (Claude API)

Triggered when `ANTHROPIC_API_KEY` is set in the environment.

**Prompt design:**
```
System: You are an expert clinical psychologist.
        Respond ONLY with a valid JSON object.
        Keys: description, psychological_insight, physical_signals,
              recommendations (list of 3 strings), wellbeing_tip

User:   Detected emotion: sad
        Confidence scores: sad: 81%, neutral: 10%, fear: 5%, ...
        Physical signals: drooped corners of mouth, lowered gaze
        User text: "I feel awful today"
        Generate a warm, empathetic report. Return ONLY the JSON.
```

**Post-processing:**

```python
raw = message.content[0].text.strip()

# Strip accidental markdown fences
if raw.startswith("```"):
    raw = raw.split("```")[1]
    if raw.startswith("json"):
        raw = raw[4:]

llm_data = json.loads(raw)
```

**On any API/parse error → automatic fallback to rule-based.**

### Mode 2 — Rule-Based (Offline)

Used when `ANTHROPIC_API_KEY` is not set or LLM call fails.

```python
report = generate_rule_based_report(
    emotion="sad",
    scores={"sad": 0.812, "neutral": 0.103, ...},
    user_text="I feel awful today"
)
```

Constructs the report from `EMOTION_PROFILES[emotion]` + top-2 scores.

---

## Main Entry Point

```python
from src.genai.report_generator import generate_emotion_report

report = generate_emotion_report(
    emotion="sad",
    scores={"angry": 0.01, "disgust": 0.00, "fear": 0.05,
            "happy": 0.02, "neutral": 0.10, "sad": 0.81, "surprise": 0.01},
    user_text="I feel terrible today.",
    use_llm=True,                       # False → always rule-based
    save_path="outputs/reports/sad.json"  # optional
)
```

| Parameter   | Type   | Description                                  |
|-------------|--------|----------------------------------------------|
| `emotion`   | str    | Predicted emotion string                     |
| `scores`    | dict   | Probabilities for all 7 classes (sum to 1)   |
| `user_text` | str    | Raw text input for context (optional)        |
| `use_llm`   | bool   | True = try Claude API first                  |
| `save_path` | str    | If provided, saves JSON to this path         |

---

## Mode Selection Logic

```
ANTHROPIC_API_KEY set?
    │
    ├─ YES → use_llm=True → generate_llm_report()
    │            │
    │            ├─ API call succeeds → rich Claude report
    │            └─ API error / parse error → fallback rule-based
    │
    └─ NO  → generate_rule_based_report()
```

---

## Configuration (config.yaml)

```yaml
genai:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"
  max_tokens: 600
  temperature: 0.7
```

`temperature: 0.7` produces varied, natural-sounding descriptions.
Set to `0.0` for fully deterministic reports (useful in CI tests).

---

## CLI Usage

```bash
# Generate a happy report (LLM mode)
export ANTHROPIC_API_KEY=sk-ant-...
python src/genai/report_generator.py --emotion happy --text "I feel wonderful!"

# Generate without LLM (rule-based)
python src/genai/report_generator.py --emotion sad --text "I feel bad" --no-llm

# Save report to file
python src/genai/report_generator.py --emotion angry --save outputs/reports/angry.json
```

---

## print_report() — Console Display

```python
from src.genai.report_generator import print_report
print_report(report)
```

Output:

```
────────────────────────────────────────────────────────────
  EMOTIONAL ANALYSIS REPORT
  2026-05-21T14:32:00
────────────────────────────────────────────────────────────
  Detected Emotion : SAD  (confidence: 81%)

  Top emotions:
           sad: ████████████████     81%
       neutral: ██                   10%
          fear: █                     5%

  Description:
  The detected emotion is sad (confidence: 81%)...

  Psychological Insight:
  Sadness often signals loss or unmet needs...

  Recommendations:
    1. Allow yourself to feel — suppressing sadness prolongs it
    2. Reach out to a trusted person
    3. Gentle activity (walking, journaling) can help shift mood

  Wellbeing Tip:
  Even a 10-minute walk can measurably improve mood...

  Source: claude:claude-sonnet-4-20250514
────────────────────────────────────────────────────────────
```

---

## Integration with the Agent

The Ollama agent calls `generate_report` as a tool:

```python
def tool_generate_report(emotion, scores, user_text=""):
    # No API call — uses rule-based generator for speed
    return {
        "emotion": emotion,
        "scores":  scores,
        "top3":    sorted(scores.items(), ...)[:3],
        "report":  f"Émotion dominante : {emotion} ({scores[emotion]:.2%}). ..."
    }
```

The full Claude-powered report is available via `generate_emotion_report(use_llm=True)`
in the FastAPI endpoint (`/predict/multimodal?include_report=true`).

---

## Extending with New Profiles

```python
# Add a new emotion profile
EMOTION_PROFILES["contempt"] = {
    "description": "moral superiority and dismissal",
    "body_signals": "asymmetric lip corner raised, narrowed gaze",
    "psychology":   "associated with social comparison...",
    "suggestions":  [
        "Examine the assumptions behind the contempt",
        "Practice perspective-taking",
        "Consider whether the feeling is proportionate"
    ]
}
```

---

Last Updated: 23/05/2026
Status: Active ✓