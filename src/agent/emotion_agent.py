"""
src/agent/emotion_agent.py
───────────────────────────
AI Agent for multimodal emotion recognition.

The agent uses Ollama's tool-calling (function calling) to:
  1. Receive a user request (text, image path, or both)
  2. Decide which analysis tools to call (image / text / multimodal)
  3. Reason over the results (ReAct loop)
  4. Generate a structured emotional response

Tools available to the agent:
  - analyze_image(image_path)        → calls ResNet-50 via FastAPI
  - analyze_text(text)               → calls BERT via FastAPI
  - analyze_multimodal(image, text)  → calls Attention Fusion via FastAPI
  - generate_report(emotion, scores, text) → rule-based + Ollama GenAI

Usage:
    python src/agent/emotion_agent.py \
        --text "Je me sens vraiment débordé aujourd'hui" \
        --image path/to/face.jpg

    python src/agent/emotion_agent.py \
        --text "I feel great!" \
        --model qwen2.5:7b \
        --ollama http://localhost:11434
"""

import sys
import json
import argparse
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import ollama
except ImportError as exc:
    raise ImportError(
        "Package 'ollama' manquant. Installez-le : pip install ollama"
    ) from exc


# ─── Config (mutable dict — évite tout `global`) ──────────────────────────────

_config: dict = {
    "api_base_url": "http://localhost:8000",
    "ollama_host":  "http://localhost:11434",
    "agent_model":  "llama3.2",
    "max_iterations": 6,
    "temperature":    0.3,
}

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOTION_PROFILES: dict[str, dict] = {
    "happy":    {"label": "Joie",       "emoji": "😊",
                 "psychology": "Associée à la dopamine/sérotonine ; liée à la connexion sociale et à la récompense.",
                 "suggestions": ["Savourez le moment par la pleine conscience.", "Partagez cette énergie positive.", "Profitez de cet état pour des tâches créatives ou exigeantes."]},
    "sad":      {"label": "Tristesse",  "emoji": "😢",
                 "psychology": "Signale souvent une perte ou un besoin non satisfait ; favorise la réflexion.",
                 "suggestions": ["Accordez-vous le droit de ressentir.", "Parlez à quelqu'un de confiance.", "Une activité douce (marche, journal) peut aider."]},
    "angry":    {"label": "Colère",     "emoji": "😠",
                 "psychology": "Signal de menace ou d'injustice perçue ; l'adrénaline prépare à l'action.",
                 "suggestions": ["Faites une pause — 10 respirations lentes abaissent le cortisol.", "Identifiez le besoin non satisfait derrière la colère.", "L'exercice physique évacue efficacement cette énergie."]},
    "fear":     {"label": "Peur",       "emoji": "😨",
                 "psychology": "Réponse de survie pilotée par l'amygdale ; aiguise l'attention sensorielle.",
                 "suggestions": ["Ancrez-vous : nommez 5 choses visibles maintenant.", "Réglez votre respiration (technique 4-7-8).", "Distinguez menace réelle et anticipée."]},
    "surprise": {"label": "Surprise",   "emoji": "😲",
                 "psychology": "Réponse d'orientation brève à un stimulus inattendu ; valence neutre.",
                 "suggestions": ["Évaluez la situation avant de réagir.", "La curiosité est la sœur saine de la surprise.", "Profitez de ce moment d'ouverture pour absorber de nouvelles informations."]},
    "neutral":  {"label": "Neutre",     "emoji": "😐",
                 "psychology": "État affectif de base ; indique l'équilibre émotionnel ou la régulation.",
                 "suggestions": ["Bon état pour la pensée analytique et la prise de décision.", "Vérifiez : refoulez-vous quelque chose, ou êtes-vous vraiment en paix ?", "Profitez du calme pour planifier ou réfléchir."]},
    "disgust":  {"label": "Dégoût",     "emoji": "🤢",
                 "psychology": "Évitement de contaminants ou de violations morales perçus.",
                 "suggestions": ["Nommez ce qui a violé vos valeurs — la clarté réduit l'intensité.", "Distinguez dégoût physique et moral.", "Un dialogue ouvert aide souvent à résoudre le dégoût moral."]},
}


# ─── Tool implementations (call FastAPI) ─────────────────────────────────────

def _api_call(endpoint: str, method: str = "get", **kwargs) -> dict:
    """Generic helper to call the FastAPI inference server."""
    url = f"{_config['api_base_url']}/{endpoint}"
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
                f"Impossible de joindre le serveur FastAPI à {_config['api_base_url']}. "
                "Démarrez-le avec : uvicorn api.app:app --port 8000"
            )
        }
    except Exception as exc:
        return {"error": str(exc)}


def tool_analyze_image(image_path: str) -> dict:
    """Analyse une image faciale et retourne l'émotion prédite (ResNet-50)."""
    path = Path(image_path)
    if not path.exists():
        return {"error": f"Image introuvable : {image_path}"}
    with open(path, "rb") as f:
        return _api_call("predict/image", method="post",
                         files={"file": (path.name, f, "image/jpeg")})


def tool_analyze_text(text: str) -> dict:
    """Analyse un texte et retourne l'émotion prédite (BERT)."""
    if not text or not text.strip():
        return {"error": "Le texte ne peut pas être vide."}
    return _api_call("predict/text", method="post",
                     json={"text": text, "include_report": False})


def tool_analyze_multimodal(image_path: str, text: str) -> dict:
    """Analyse combinée image + texte via Attention Fusion."""
    path = Path(image_path)
    if not path.exists():
        return {"error": f"Image introuvable : {image_path}"}
    with open(path, "rb") as f:
        return _api_call(
            "predict/multimodal", method="post",
            files={"file": (path.name, f, "image/jpeg")},
            data={"text": text, "include_report": "false"},
        )


def tool_generate_report(emotion: str, scores: dict, user_text: str = "") -> dict:
    """Génère un rapport psychologique structuré (rule-based + profils intégrés)."""
    profile = EMOTION_PROFILES.get(emotion, {})
    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    return {
        "emotion":     emotion,
        "label_fr":    profile.get("label", emotion),
        "emoji":       profile.get("emoji", "❓"),
        "confidence":  scores.get(emotion, 0.0),
        "top_3":       [{"emotion": e, "score": round(s, 4)} for e, s in top3],
        "psychology":  profile.get("psychology", ""),
        "suggestions": profile.get("suggestions", []),
        "user_text_context": user_text[:300] if user_text else None,
    }


# ─── Tool schema (Ollama / OpenAI format) ────────────────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": (
                "Analyse une image faciale pour détecter l'émotion. "
                "Utilise le modèle ResNet-50 fine-tuné sur FER2013. "
                "Appelle cet outil quand l'utilisateur fournit une image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Chemin local vers l'image (JPEG ou PNG)",
                    }
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_text",
            "description": (
                "Analyse un texte court pour détecter l'émotion exprimée. "
                "Utilise BERT fine-tuné sur dair-ai/emotion. "
                "Appelle cet outil quand l'utilisateur fournit du texte sans image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Texte à analyser (phrase, tweet, caption...)",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_multimodal",
            "description": (
                "Analyse combinée image + texte pour une détection plus précise. "
                "Utilise le modèle Attention Fusion (ResNet-50 + BERT + Cross-Attention). "
                "Appelle cet outil quand l'utilisateur fournit BOTH une image ET un texte. "
                "C'est l'outil le plus précis (~83 % accuracy)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Chemin local vers l'image",
                    },
                    "text": {
                        "type": "string",
                        "description": "Texte accompagnant l'image",
                    },
                },
                "required": ["image_path", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": (
                "Génère un rapport psychologique structuré à partir du résultat de détection. "
                "Appelle cet outil APRÈS avoir obtenu un résultat de détection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "enum": EMOTION_CLASSES,
                        "description": "Émotion principale détectée",
                    },
                    "scores": {
                        "type": "object",
                        "description": "Dict {emotion: probability} pour les 7 classes",
                    },
                    "user_text": {
                        "type": "string",
                        "description": "Texte original de l'utilisateur (contexte pour le rapport)",
                    },
                },
                "required": ["emotion", "scores"],
            },
        },
    },
]


# ─── Tool dispatcher ──────────────────────────────────────────────────────────

def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """Execute the requested tool and return the result as a JSON string."""
    print(f"\n  🔧 Tool called : {tool_name}")
    print(f"     Input       : {json.dumps(tool_input, ensure_ascii=False)[:120]}...")

    if tool_name == "analyze_image":
        result = tool_analyze_image(tool_input["image_path"])
    elif tool_name == "analyze_text":
        result = tool_analyze_text(tool_input["text"])
    elif tool_name == "analyze_multimodal":
        result = tool_analyze_multimodal(tool_input["image_path"], tool_input["text"])
    elif tool_name == "generate_report":
        result = tool_generate_report(
            emotion=tool_input["emotion"],
            scores=tool_input["scores"],
            user_text=tool_input.get("user_text", ""),
        )
    else:
        result = {"error": f"Outil inconnu : {tool_name}"}

    print(f"     Result      : {json.dumps(result, ensure_ascii=False)[:120]}...")
    return json.dumps(result, ensure_ascii=False)


# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un agent expert en reconnaissance des émotions multimodale.
Tu as accès à 4 outils :
  1. analyze_image       — analyse une image faciale (ResNet-50)
  2. analyze_text        — analyse un texte (BERT)
  3. analyze_multimodal  — combine image + texte (Attention Fusion, ~83 % accuracy)
  4. generate_report     — génère un rapport psychologique structuré après détection

Règles de raisonnement :
- Image ET texte fournis  → utilise analyze_multimodal (plus précis)
- Image seule             → utilise analyze_image
- Texte seul              → utilise analyze_text
- Après toute analyse     → appelle toujours generate_report avec les scores obtenus
- Si un outil retourne une erreur → explique le problème clairement à l'utilisateur
- Réponds en français sauf si l'utilisateur écrit en anglais

Ta réponse finale doit inclure :
  • L'émotion détectée et le score de confiance
  • Le top-3 des émotions avec leurs probabilités
  • Les insights psychologiques du rapport
  • Les recommandations concrètes
  • Quelle modalité / quel modèle a été utilisé et pourquoi"""


# ─── Agent loop (ReAct pattern) ───────────────────────────────────────────────

def run_agent(user_message: str, image_path: str | None = None) -> str:
    """
    ReAct agent loop using Ollama tool-calling.

    Pattern:
      User message → Ollama thinks → calls tool → observes result
      → Ollama thinks again → calls next tool → ...
      → Ollama produces final text response

    Args:
        user_message : text from the user
        image_path   : optional path to a face image

    Returns:
        Final assistant response as a string
    """
    client = ollama.Client(host=_config["ollama_host"])

    # Build initial conversation
    content = user_message
    if image_path:
        content += f"\n\n[Image fournie : {image_path}]"

    messages: list[dict] = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": content},
    ]

    print(f"\n{'═' * 60}")
    print("  AGENT DÉMARRÉ  (Ollama · model: " + _config["agent_model"] + ")")
    print(f"  Message : {user_message[:80]}")
    if image_path:
        print(f"  Image   : {image_path}")
    print(f"{'═' * 60}")

    # ── ReAct loop ────────────────────────────────────────────────────────────
    for iteration in range(_config["max_iterations"]):
        print(f"\n[Itération {iteration + 1}/{_config['max_iterations']}] Appel Ollama...")

        try:
            response = client.chat(
                model=_config["agent_model"],
                messages=messages,
                tools=TOOLS,
                options={"temperature": _config["temperature"]},
            )
        except ollama.ResponseError as exc:
            if "model not found" in str(exc).lower():
                return (
                    f"❌ Modèle '{_config['agent_model']}' introuvable dans Ollama. "
                    f"Lancez : ollama pull {_config['agent_model']}"
                )
            return f"❌ Erreur Ollama : {exc}"
        except Exception as exc:
            return f"❌ Impossible de joindre Ollama ({_config['ollama_host']}) : {exc}"

        msg = response.message

        # Append assistant turn (keep tool_calls only when present)
        assistant_entry: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = msg.tool_calls
        messages.append(assistant_entry)

        # ── No tool calls → final answer ──────────────────────────────────────
        if not msg.tool_calls:
            final = msg.content or "⚠️ L'agent n'a pas retourné de réponse."
            print(f"\n[Agent] Réponse finale générée ({len(final)} chars)")
            return final

        # ── Process tool calls ────────────────────────────────────────────────
        for tc in msg.tool_calls:
            fn         = tc.function
            tool_name  = fn.name
            tool_input = fn.arguments if isinstance(fn.arguments, dict) else json.loads(fn.arguments)

            result_str = dispatch_tool(tool_name, tool_input)

            messages.append({"role": "tool", "content": result_str})

    return "⚠️ L'agent a atteint le nombre maximum d'itérations sans réponse finale."


# ─── Pretty printer ───────────────────────────────────────────────────────────

def print_agent_response(response: str) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  RÉPONSE DE L'AGENT")
    print(sep)
    print(response)
    print(sep + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI Agent — Multimodal Emotion Recognition (Ollama backend)"
    )
    p.add_argument("--text",   default="",
                   help="Texte à analyser (phrase, tweet, caption...)")
    p.add_argument("--image",  default=None,
                   help="Chemin vers l'image faciale (JPEG/PNG)")
    p.add_argument("--api",    default="http://localhost:8000",
                   help="URL du serveur FastAPI  (défaut : http://localhost:8000)")
    p.add_argument("--ollama", default="http://localhost:11434",
                   help="URL du serveur Ollama   (défaut : http://localhost:11434)")
    p.add_argument("--model",  default="llama3.2",
                   help="Modèle Ollama à utiliser (défaut : llama3.2)")
    p.add_argument("--max-iter", type=int, default=6,
                   help="Nombre max d'itérations de l'agent (défaut : 6)")
    p.add_argument("--temperature", type=float, default=0.3,
                   help="Température de génération (défaut : 0.3)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.text and not args.image:
        print("Usage : python emotion_agent.py --text 'Je ressens...' [--image face.jpg]")
        print("        Au moins --text ou --image est requis.")
        sys.exit(1)

    # Mise à jour de la config centralisée — aucun `global` nécessaire
    _config["api_base_url"]   = args.api
    _config["ollama_host"]    = args.ollama
    _config["agent_model"]    = args.model
    _config["max_iterations"] = args.max_iter
    _config["temperature"]    = args.temperature

    result = run_agent(
        user_message=args.text or "Analyse l'émotion dans cette image.",
        image_path=args.image,
    )
    print_agent_response(result)