"""
src/agent/emotion_agent.py
───────────────────────────
AI Agent for multimodal emotion recognition.

The agent uses Claude's tool_use (function calling) to:
  1. Receive a user request (text, image path, or both)
  2. Decide which analysis tools to call (image / text / multimodal)
  3. Reason over the results (ReAct loop)
  4. Generate a structured emotional response

Tools available to the agent:
  - analyze_image(image_path)        → calls ResNet-50 via FastAPI
  - analyze_text(text)               → calls BERT via FastAPI
  - analyze_multimodal(image, text)  → calls Attention Fusion via FastAPI
  - generate_report(emotion, scores, text) → calls Claude GenAI module

Usage:
    python src/agent/emotion_agent.py \
        --text "Je me sens vraiment débordé aujourd'hui" \
        --image path/to/face.jpg

    python src/agent/emotion_agent.py \
        --text "I feel great!"
"""

import os
import sys
import json
import base64
import argparse
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ─── Config ───────────────────────────────────────────────────────────────────

API_BASE_URL   = os.getenv("API_BASE_URL",   "http://localhost:8000")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
AGENT_MODEL    = "claude-sonnet-4-20250514"
MAX_ITERATIONS = 6   # max tool-call rounds before forcing a final answer

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ─── Tool implementations ─────────────────────────────────────────────────────

def _api_call(endpoint: str, method: str = "get", **kwargs) -> dict:
    """Generic helper to call the FastAPI inference server."""
    url = f"{API_BASE_URL}/{endpoint}"
    try:
        if method == "get":
            resp = requests.get(url, timeout=30)
        else:
            resp = requests.post(url, timeout=60, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {
            "error": (
                f"Cannot reach inference server at {API_BASE_URL}. "
                "Start it with: uvicorn api.app:app --port 8000"
            )
        }
    except Exception as e:
        return {"error": str(e)}


def tool_analyze_image(image_path: str) -> dict:
    """
    Analyse une image faciale et retourne l'émotion prédite.
    Appelle le endpoint POST /predict/image de FastAPI.
    """
    path = Path(image_path)
    if not path.exists():
        return {"error": f"Image not found: {image_path}"}

    with open(path, "rb") as f:
        files = {"file": (path.name, f, "image/jpeg")}
        result = _api_call("predict/image", method="post", files=files)

    return result


def tool_analyze_text(text: str) -> dict:
    """
    Analyse un texte et retourne l'émotion prédite.
    Appelle le endpoint POST /predict/text de FastAPI.
    """
    if not text or not text.strip():
        return {"error": "text cannot be empty"}

    result = _api_call(
        "predict/text",
        method="post",
        json={"text": text, "include_report": False}
    )
    return result


def tool_analyze_multimodal(image_path: str, text: str) -> dict:
    """
    Analyse combinée image + texte via le modèle Attention Fusion.
    Appelle le endpoint POST /predict/multimodal de FastAPI.
    """
    path = Path(image_path)
    if not path.exists():
        return {"error": f"Image not found: {image_path}"}

    with open(path, "rb") as f:
        files = {"file": (path.name, f, "image/jpeg")}
        data  = {"text": text, "include_report": "false"}
        result = _api_call("predict/multimodal", method="post", files=files, data=data)

    return result


def tool_generate_report(emotion: str, scores: dict, user_text: str = "") -> dict:
    """
    Génère un rapport émotionnel structuré via le module GenAI (Claude API).
    Utilise generate_emotion_report de src/genai/report_generator.py.
    """
    try:
        from src.genai.report_generator import generate_emotion_report
        report = generate_emotion_report(
            emotion=emotion,
            scores=scores,
            user_text=user_text,
            use_llm=bool(ANTHROPIC_KEY),
        )
        return report
    except Exception as e:
        return {"error": str(e)}


# ─── Tool schema (Anthropic tool_use format) ─────────────────────────────────

TOOLS = [
    {
        "name": "analyze_image",
        "description": (
            "Analyse une image faciale pour détecter l'émotion. "
            "Utilise le modèle ResNet-50 fine-tuné sur FER2013. "
            "Appelle cet outil quand l'utilisateur fournit une image."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Chemin local vers l'image (JPEG ou PNG)"
                }
            },
            "required": ["image_path"]
        }
    },
    {
        "name": "analyze_text",
        "description": (
            "Analyse un texte court pour détecter l'émotion exprimée. "
            "Utilise BERT fine-tuné sur dair-ai/emotion. "
            "Appelle cet outil quand l'utilisateur fournit du texte sans image."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Texte à analyser (phrase, tweet, caption...)"
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "analyze_multimodal",
        "description": (
            "Analyse combinée image + texte pour une détection plus précise. "
            "Utilise le modèle Attention Fusion (ResNet-50 + BERT + Cross-Attention). "
            "Appelle cet outil quand l'utilisateur fournit BOTH une image ET un texte. "
            "C'est l'outil le plus précis (~83% accuracy)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Chemin local vers l'image"
                },
                "text": {
                    "type": "string",
                    "description": "Texte accompagnant l'image"
                }
            },
            "required": ["image_path", "text"]
        }
    },
    {
        "name": "generate_report",
        "description": (
            "Génère un rapport psychologique structuré à partir du résultat de détection. "
            "Utilise Claude API pour produire une analyse empathique, des insights "
            "psychologiques et des recommandations personnalisées. "
            "Appelle cet outil APRÈS avoir obtenu un résultat de détection."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "emotion": {
                    "type": "string",
                    "description": "Émotion principale détectée",
                    "enum": EMOTION_CLASSES
                },
                "scores": {
                    "type": "object",
                    "description": "Dict {emotion: probability} pour les 7 classes"
                },
                "user_text": {
                    "type": "string",
                    "description": "Texte original de l'utilisateur (contexte pour le rapport)"
                }
            },
            "required": ["emotion", "scores"]
        }
    }
]


# ─── Tool dispatcher ──────────────────────────────────────────────────────────

def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """Execute the requested tool and return the result as a JSON string."""
    print(f"\n  🔧 Tool called: {tool_name}")
    print(f"     Input: {json.dumps(tool_input, ensure_ascii=False)[:120]}...")

    if tool_name == "analyze_image":
        result = tool_analyze_image(tool_input["image_path"])
    elif tool_name == "analyze_text":
        result = tool_analyze_text(tool_input["text"])
    elif tool_name == "analyze_multimodal":
        result = tool_analyze_multimodal(
            tool_input["image_path"], tool_input["text"]
        )
    elif tool_name == "generate_report":
        result = tool_generate_report(
            emotion=tool_input["emotion"],
            scores=tool_input["scores"],
            user_text=tool_input.get("user_text", "")
        )
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    print(f"     Result: {json.dumps(result, ensure_ascii=False)[:120]}...")
    return json.dumps(result, ensure_ascii=False)


# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un agent expert en reconnaissance des émotions multimodale.
Tu as accès à 4 outils :
  1. analyze_image  — analyse une image faciale (ResNet-50)
  2. analyze_text   — analyse un texte (BERT)
  3. analyze_multimodal — combine image + texte pour plus de précision (Attention Fusion, 83% accuracy)
  4. generate_report — génère un rapport psychologique structuré après détection

Règles de raisonnement :
- Si l'utilisateur fournit image ET texte → utilise analyze_multimodal (plus précis)
- Si l'utilisateur fournit uniquement une image → utilise analyze_image
- Si l'utilisateur fournit uniquement du texte → utilise analyze_text
- Après toute analyse → appelle toujours generate_report avec les scores obtenus
- Si un outil retourne une erreur → explique le problème clairement
- Réponds toujours en français sauf si l'utilisateur écrit en anglais

Ta réponse finale doit inclure :
  • L'émotion détectée et le score de confiance
  • Le top-3 des émotions avec leurs probabilités
  • Les insights psychologiques du rapport
  • Les recommandations concrètes
  • Quelle modalité / quel modèle a été utilisé et pourquoi"""


# ─── Agent loop (ReAct pattern) ───────────────────────────────────────────────

def run_agent(user_message: str, image_path: str = None) -> str:
    """
    ReAct agent loop using Claude tool_use.

    Pattern:
      User message → Claude thinks → calls tool → observes result
      → Claude thinks again → calls next tool → ...
      → Claude produces final text response

    Args:
        user_message : text from the user
        image_path   : optional path to a face image

    Returns:
        Final assistant response as a string
    """
    try:
        import anthropic
    except ImportError:
        return "❌ Package 'anthropic' manquant. Installez-le : pip install anthropic"

    if not ANTHROPIC_KEY:
        return "❌ ANTHROPIC_API_KEY non définie. Exportez-la avant de lancer l'agent."

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    # Build initial user message
    if image_path:
        content = (
            f"{user_message}\n\n"
            f"[Image fournie : {image_path}]"
        )
    else:
        content = user_message

    messages = [{"role": "user", "content": content}]

    print(f"\n{'═'*60}")
    print(f"  AGENT DÉMARRÉ")
    print(f"  Message : {user_message[:80]}")
    if image_path:
        print(f"  Image   : {image_path}")
    print(f"{'═'*60}")

    # ── ReAct loop ────────────────────────────────────────────────────────────
    for iteration in range(MAX_ITERATIONS):
        print(f"\n[Iteration {iteration+1}/{MAX_ITERATIONS}] Appel Claude...")

        response = client.messages.create(
            model=AGENT_MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        # Append assistant response to history
        messages.append({"role": "assistant", "content": response.content})

        # ── If no tool calls → final answer ───────────────────────────────────
        if response.stop_reason == "end_turn":
            final = " ".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )
            print(f"\n[Agent] Réponse finale générée ({len(final)} chars)")
            return final

        # ── Process tool calls ─────────────────────────────────────────────────
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result_str = dispatch_tool(block.name, block.input)
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result_str
                })

        if not tool_results:
            break

        # Feed tool results back to the agent
        messages.append({"role": "user", "content": tool_results})

    return "⚠️ L'agent a atteint le nombre maximum d'itérations sans réponse finale."


# ─── Pretty printer ───────────────────────────────────────────────────────────

def print_agent_response(response: str):
    sep = "─" * 60
    print(f"\n{sep}")
    print("  RÉPONSE DE L'AGENT")
    print(sep)
    print(response)
    print(sep + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AI Agent for multimodal emotion recognition"
    )
    p.add_argument("--text",  default="",
                   help="Text to analyse (tweet, caption, sentence...)")
    p.add_argument("--image", default=None,
                   help="Path to face image (JPEG/PNG)")
    p.add_argument("--api",   default="http://localhost:8000",
                   help="FastAPI server URL (default: http://localhost:8000)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.text and not args.image:
        print("Usage: python emotion_agent.py --text 'I feel...' [--image face.jpg]")
        print("       At least --text or --image is required.")
        sys.exit(1)

    global API_BASE_URL
    API_BASE_URL = args.api

    response = run_agent(
        user_message=args.text or "Analyse l'émotion dans cette image.",
        image_path=args.image
    )
    print_agent_response(response)