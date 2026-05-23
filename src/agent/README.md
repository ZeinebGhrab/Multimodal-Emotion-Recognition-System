🤖 src/agent/ — Ollama ReAct Agent
====================================

## Overview

The agent module implements an autonomous **ReAct** loop
(Reason → Act → Observe) that orchestrates emotion inference by
selecting and calling the right tool based on the available inputs.
The LLM does the reasoning; the tools do the actual inference.

```
src/agent/
└── emotion_agent.py     Agent loop, tool definitions, dispatcher
```

---

## ReAct Loop Architecture

```
User input (text? image? both?)
              │
              ▼
    ┌─────────────────────┐
    │  Ollama LLM         │  ◄── System prompt (role + routing rules)
    │  (llama3.2 etc.)    │
    └──────────┬──────────┘
               │  tool_calls in response?
    ┌──────────▼────────────────────────────────┐
    │  Tool Dispatcher (dispatch_tool)           │
    │                                            │
    │  analyze_text       → FastAPI /predict/text│
    │  analyze_image      → FastAPI /predict/image│
    │  analyze_multimodal → FastAPI /predict/mm  │
    │  generate_report    → local rule-based     │
    └──────────┬─────────────────────────────────┘
               │  Observation (JSON result string)
               ▼
    Next iteration  ──► Final answer when no tool called
```

---

## Tool Definitions (OpenAI-compatible format)

Each tool is declared as a function spec passed to `ollama.chat(tools=[...])`.
The LLM decides which tool to call and with what arguments based on the spec.

### analyze_text

```python
{
    "name": "analyze_text",
    "description": (
        "Analyse un texte court pour détecter l'émotion. "
        "Utilise BERT fine-tuné sur dair-ai/emotion. "
        "Appelle cet outil quand l'utilisateur fournit du texte sans image."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Texte à analyser"}
        },
        "required": ["text"]
    }
}
```

Calls: `POST /predict/text` → BERT classifier

### analyze_image

```python
{
    "name": "analyze_image",
    "description": "... Appelle cet outil quand l'utilisateur fournit une image.",
    "parameters": {
        "properties": {
            "image_path": {"type": "string", "description": "Chemin local vers l'image"}
        },
        "required": ["image_path"]
    }
}
```

Calls: `POST /predict/image` → ResNet-50 classifier

### analyze_multimodal

```python
{
    "name": "analyze_multimodal",
    "description": "... C'est l'outil le plus précis (~83% accuracy).",
    "parameters": {
        "properties": {
            "image_path": {"type": "string"},
            "text":       {"type": "string"}
        },
        "required": ["image_path", "text"]
    }
}
```

Calls: `POST /predict/multimodal` → Attention Fusion model

### generate_report

```python
{
    "name": "generate_report",
    "description": "Génère un rapport psychologique. Appelle APRÈS détection.",
    "parameters": {
        "properties": {
            "emotion":   {"type": "string", "enum": EMOTION_CLASSES},
            "scores":    {"type": "object"},
            "user_text": {"type": "string"}
        },
        "required": ["emotion", "scores"]
    }
}
```

Returns a local rule-based report (no API call needed).

---

## Agent Decision Logic

The system prompt encodes routing rules the LLM follows:

| Available inputs    | Tool selected          | Model behind it    |
|---------------------|------------------------|--------------------|
| Text only           | `analyze_text`         | BERT               |
| Image only          | `analyze_image`        | ResNet-50          |
| Text + Image        | `analyze_multimodal`   | Attention Fusion   |
| After any detection | `generate_report`      | Rule-based         |

The system prompt also adds an explicit guard for the no-image case:

```
[IMPORTANT : Aucune image fournie. Tu DOIS appeler uniquement analyze_text —
NE PAS appeler analyze_image ni analyze_multimodal.]
```

This prevents the LLM from hallucinating an image path.

---

## Agent Loop Implementation

```python
def run_agent(user_message: str, image_path: str = None) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": content},
    ]

    for iteration in range(max_iterations):
        response = client.chat(
            model=model_name,
            messages=messages,
            tools=TOOLS,
            options={"temperature": temperature},
        )

        msg = response.message
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": msg.tool_calls or []})

        if not msg.tool_calls:
            return msg.content          # ← final answer, no more tool calls

        for tc in msg.tool_calls:
            result_str = dispatch_tool(tc.function.name, tc.function.arguments)
            messages.append({"role": "tool", "content": result_str})

    return "⚠️ Max iterations reached."
```

---

## Tool Dispatcher

```python
def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "analyze_image":
        result = tool_analyze_image(tool_input["image_path"])
    elif tool_name == "analyze_text":
        result = tool_analyze_text(tool_input["text"])
    elif tool_name == "analyze_multimodal":
        result = tool_analyze_multimodal(
            tool_input["image_path"], tool_input["text"])
    elif tool_name == "generate_report":
        result = tool_generate_report(
            emotion=tool_input["emotion"],
            scores=tool_input["scores"],
            user_text=tool_input.get("user_text", ""),
        )

    # Side effect: update last_result for visualization
    if "scores" in result and "emotion" in result:
        st.session_state.last_result = result

    return json.dumps(result, ensure_ascii=False)
```

### Argument Normalization

Ollama may return tool arguments as a `dict`, a JSON string, or a custom
object. The dispatcher normalizes all forms:

```python
raw_args = tc.function.arguments
if isinstance(raw_args, dict):
    tool_input = raw_args
elif isinstance(raw_args, str):
    tool_input = json.loads(raw_args)
else:
    tool_input = dict(raw_args)   # custom Ollama type
```

---

## Supported Ollama Models

Any model that supports **tool-calling / function-calling**:

```
llama3.2          llama3.2:1b     llama3.2:3b
llama3.1          llama3.1:8b
qwen2.5           qwen2.5:7b      qwen2.5:14b
mistral           mistral-nemo
command-r         gemma3
```

**Recommended:** `llama3.2` — best balance of speed, accuracy, and
reliable tool-calling at the 3B parameter scale.

```bash
ollama pull llama3.2
ollama serve
```

---

## Configurable Parameters

| Parameter        | Default       | Range     | Effect                                      |
|------------------|---------------|-----------|---------------------------------------------|
| `model`          | `llama3.2`    | Any Ollama| LLM used for reasoning                      |
| `temperature`    | `0.3`         | 0.0 – 1.0 | Lower = more deterministic tool selection   |
| `max_iterations` | `6`           | 2 – 12    | Max tool calls before forcing a stop        |
| `system_prompt`  | French, multi-rule | Editable | Controls routing and response language |

---

## Error Handling

| Error condition          | Agent behavior                                      |
|--------------------------|-----------------------------------------------------|
| Model not found          | Returns `❌ Model '...' not found. Run ollama pull` |
| Ollama unreachable       | Returns `❌ Cannot reach Ollama at <host>`           |
| FastAPI unreachable      | Tool returns `{"error": "Cannot reach FastAPI..."}`; agent explains to user |
| Image path missing       | Tool returns `{"error": "Image not found"}`         |
| Max iterations reached   | Returns `⚠️ Max iterations reached`                |
| Tool returns `None`      | Dispatcher replaces with `{"error": "Tool returned None"}` |

---

## Adding a Custom Tool

### Step 1 — Declare the spec

```python
TOOLS.append({
    "type": "function",
    "function": {
        "name": "analyze_audio",
        "description": "Analyse la prosodie vocale pour détecter l'émotion.",
        "parameters": {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"}
            },
            "required": ["audio_path"]
        }
    }
})
```

### Step 2 — Implement the handler in dispatch_tool

```python
elif tool_name == "analyze_audio":
    result = tool_analyze_audio(tool_input.get("audio_path", ""))
```

### Step 3 — Implement the tool function

```python
def tool_analyze_audio(audio_path: str) -> dict:
    ...
    return _api_call("predict/audio", method="post", files={"file": ...})
```

### Step 4 — Update the system prompt

```
- Si l'utilisateur fournit un fichier audio → utilise analyze_audio
```

---

Last Updated: 23/05/026<br>
Status: Active ✓