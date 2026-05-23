# src/agent/ — Ollama ReAct Agent

The agent module implements an autonomous **ReAct loop** (Reason → Act → Observe) that orchestrates emotion inference by selecting and calling the right tool based on available inputs. The LLM handles reasoning; the tools handle actual inference.

```
src/agent/
└── emotion_agent.py    Agent loop, tool definitions, tool dispatcher
```

---

## ReAct Loop Architecture

```
User input (text? image? both?)
              │
    ┌─────────▼──────────────┐
    │  Ollama LLM            │  ◄── System prompt + routing rules
    │  (llama3.2 etc.)       │
    └─────────┬──────────────┘
              │  tool_calls in response?
    ┌─────────▼───────────────────────────────────┐
    │  Tool Dispatcher (dispatch_tool)            │
    │                                             │
    │  analyze_text       → FastAPI /predict/text │
    │  analyze_image      → FastAPI /predict/image│
    │  analyze_multimodal → FastAPI /predict/mm   │
    │  generate_report    → local rule-based      │
    └─────────┬───────────────────────────────────┘
              │  JSON observation string
              ▼
    Next iteration  ──► Final answer when no tool called
```

---

## Tool Routing Logic

| Available inputs | Tool called | Reason |
|-----------------|-------------|--------|
| Text only | `analyze_text` | No image context available |
| Image only | `analyze_image` | No text context available |
| Text + Image | `analyze_multimodal` | Highest accuracy (~97.7%) |
| After any analysis | `generate_report` | Always appended |

---

## Tool Definitions

Each tool is declared as an OpenAI-compatible function spec passed to `ollama.chat(tools=[...])`. The LLM decides which to call and with what arguments.

### analyze_text

```python
{
    "name": "analyze_text",
    "description": (
        "Analyse a short text to detect the expressed emotion. "
        "Uses a BERT model fine-tuned on dair-ai/emotion. "
        "Call this tool when the user provides text but no image."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to analyse"}
        },
        "required": ["text"]
    }
}
```

**Backend:** `POST /predict/text` → BERT classifier

---

### analyze_image

```python
{
    "name": "analyze_image",
    "description": (
        "Analyse a face image to detect the expressed emotion. "
        "Uses a ResNet-50 model trained on FER2013. "
        "Call this tool when the user provides an image but no text."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {"type": "string", "description": "Path to the image file"}
        },
        "required": ["image_path"]
    }
}
```

**Backend:** `POST /predict/image` → ResNet-50

---

### analyze_multimodal

```python
{
    "name": "analyze_multimodal",
    "description": (
        "Analyse both a face image and text together for maximum accuracy (~97.7%). "
        "Uses Attention Fusion (ResNet-50 + BERT). "
        "Call this tool when BOTH image and text are available."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {"type": "string"},
            "text":       {"type": "string"}
        },
        "required": ["image_path", "text"]
    }
}
```

**Backend:** `POST /predict/multimodal` → Attention Fusion

---

### generate_report

```python
{
    "name": "generate_report",
    "description": (
        "Generate a detailed psychological emotion report from an analysis result. "
        "Always call this tool after any analysis to produce the final output."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "emotion": {"type": "string"},
            "scores":  {"type": "object"},
            "user_text": {"type": "string"}
        },
        "required": ["emotion", "scores"]
    }
}
```

**Backend:** Local rule-based report generator (no API call)

---

## Configurable Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| Model | `llama3.2` | Any Ollama model with tool-calling | LLM used for reasoning |
| Temperature | `0.3` | 0.0 – 1.0 | Lower = more deterministic routing |
| Max iterations | `6` | 2 – 12 | Max tool calls before forced stop |
| System prompt | Multi-rule (French) | Editable in Streamlit Tab 3 | Controls agent behaviour |

### Supported Ollama Models

Any model that supports tool-calling:

```
llama3.2      llama3.2:1b    llama3.2:3b
llama3.1      llama3.1:8b
qwen2.5       qwen2.5:7b
mistral       mistral-nemo
command-r     gemma3
```

**Recommended:** `llama3.2` (default). Pull with `ollama pull llama3.2`.

---

## Configuration (config.yaml)

```yaml
agent:
  model: "llama3.2"
  ollama_host: "http://localhost:11434"
  temperature: 0.3
  max_iterations: 6
```

---

## Prerequisites

```bash
# Install Ollama
# → https://ollama.com

ollama pull llama3.2
ollama serve

# FastAPI server must also be running
uvicorn api.app:app --port 8000
```

---

*Last Updated: 23/05/2026 — Status: Active ✓*
