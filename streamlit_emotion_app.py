"""
streamlit_emotion_app.py
─────────────────────────
Interface Streamlit pour l'agent AI de reconnaissance d'émotions multimodal.

Fonctionnalités :
  1. Analyse de texte, image, ou combiné (multimodal)
  2. Visualisation du raisonnement de l'agent (ReAct loop)
  3. Panneau d'amélioration de l'agent (system prompt, modèle, paramètres)
  4. Historique des sessions
  5. Visualisation radar des scores d'émotions

Usage :
    streamlit run streamlit_emotion_app.py
    # Avec serveur FastAPI démarré :
    # uvicorn api.app:app --port 8000
"""

import os
import sys
import json
import time
import base64
import tempfile
import requests
import ollama as ollama_lib
from pathlib import Path
from datetime import datetime
from io import BytesIO

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ─── Page config (doit être le premier appel Streamlit) ───────────────────────

st.set_page_config(
    page_title="EmotionSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Import Fonts */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default header/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* App background */
.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111118;
    border-right: 1px solid #1e1e2e;
}

/* Title Banner */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #7c6af7 0%, #a855f7 40%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #6b6b8a;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Cards */
.card {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.card-accent {
    border-left: 3px solid #7c6af7;
}

/* Emotion badge */
.emotion-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.02em;
}

/* Agent step */
.agent-step {
    background: #0f0f1a;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
    font-family: 'DM Mono', monospace;
}

.step-thinking { border-left: 3px solid #7c6af7; }
.step-tool     { border-left: 3px solid #f59e0b; }
.step-result   { border-left: 3px solid #10b981; }
.step-final    { border-left: 3px solid #ec4899; }

/* Metric cards */
.metric-card {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #7c6af7;
}

.metric-label {
    font-size: 0.78rem;
    color: #6b6b8a;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c6af7, #a855f7);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    letter-spacing: 0.02em;
    padding: 0.65rem 1.5rem;
    width: 100%;
    transition: opacity 0.2s;
}

.stButton > button:hover {
    opacity: 0.85;
}

/* Input styling */
.stTextArea textarea, .stTextInput input {
    background: #0f0f1a !important;
    border: 1px solid #1e1e30 !important;
    border-radius: 10px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #111118;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b6b8a;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
}

.stTabs [aria-selected="true"] {
    background: #1e1e30 !important;
    color: #e8e8f0 !important;
}

/* Divider */
.emotion-divider {
    border: none;
    border-top: 1px solid #1e1e30;
    margin: 1.5rem 0;
}

/* History item */
.history-item {
    background: #111118;
    border: 1px solid #1e1e30;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: border-color 0.2s;
}
.history-item:hover { border-color: #7c6af7; }

/* Scrollable container */
.scroll-box {
    max-height: 420px;
    overflow-y: auto;
    padding-right: 6px;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #0f0f1a !important;
    border: 1px solid #1e1e30 !important;
    color: #e8e8f0 !important;
}

/* Slider */
.stSlider > div { color: #e8e8f0; }

/* Info / warning boxes */
.stAlert {
    background: #13131f !important;
    border: 1px solid #1e1e30 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOTION_META = {
    "angry":    {"emoji": "😠", "color": "#ef4444", "label": "Colère"},
    "disgust":  {"emoji": "🤢", "color": "#84cc16", "label": "Dégoût"},
    "fear":     {"emoji": "😨", "color": "#f97316", "label": "Peur"},
    "happy":    {"emoji": "😊", "color": "#22c55e", "label": "Joie"},
    "neutral":  {"emoji": "😐", "color": "#94a3b8", "label": "Neutre"},
    "sad":      {"emoji": "😢", "color": "#3b82f6", "label": "Tristesse"},
    "surprise": {"emoji": "😲", "color": "#a855f7", "label": "Surprise"},
}

DEFAULT_SYSTEM_PROMPT = """Tu es un agent expert en reconnaissance des émotions multimodale.
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

# Tools in Ollama/OpenAI format (type: function)
TOOLS = [
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
                        "description": "Chemin local vers l'image (JPEG ou PNG)"
                    }
                },
                "required": ["image_path"]
            }
        }
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
                        "description": "Texte à analyser"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_multimodal",
            "description": (
                "Analyse combinée image + texte pour une détection plus précise. "
                "Utilise le modèle Attention Fusion (ResNet-50 + BERT + Cross-Attention). "
                "Appelle cet outil quand l'utilisateur fournit BOTH une image ET un texte. "
                "C'est l'outil le plus précis (~83% accuracy)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Chemin local vers l'image"},
                    "text": {"type": "string", "description": "Texte accompagnant l'image"}
                },
                "required": ["image_path", "text"]
            }
        }
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
                        "description": "Émotion principale détectée"
                    },
                    "scores": {
                        "type": "object",
                        "description": "Dict {emotion: probability} pour les 7 classes"
                    },
                    "user_text": {
                        "type": "string",
                        "description": "Texte original de l'utilisateur (contexte)"
                    }
                },
                "required": ["emotion", "scores"]
            }
        }
    }
]


# ─── Session state init ───────────────────────────────────────────────────────

def init_state():
    defaults = {
        "ollama_host": "http://localhost:11434",
        "api_base_url": "http://localhost:8000",
        "history": [],
        "agent_steps": [],
        "last_result": None,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "model": "llama3.2",
        "max_iterations": 6,
        "temperature": 0.3,
        "running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Tool implementations (call FastAPI) ─────────────────────────────────────

def _api_call(endpoint: str, method: str = "get", **kwargs) -> dict:
    url = f"{st.session_state.api_base_url}/{endpoint}"
    try:
        if method == "get":
            resp = requests.get(url, timeout=30)
        else:
            resp = requests.post(url, timeout=90, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Impossible de joindre le serveur FastAPI à {st.session_state.api_base_url}. Démarrez-le avec : uvicorn api.app:app --port 8000"}
    except Exception as e:
        return {"error": str(e)}


def tool_analyze_image(image_path: str) -> dict:
    path = Path(image_path)
    if not path.exists():
        return {"error": f"Image introuvable : {image_path}"}
    with open(path, "rb") as f:
        files = {"file": (path.name, f, "image/jpeg")}
        return _api_call("predict/image", method="post", files=files)


def tool_analyze_text(text: str) -> dict:
    if not text.strip():
        return {"error": "Le texte ne peut pas être vide"}
    return _api_call("predict/text", method="post", json={"text": text, "include_report": False})


def tool_analyze_multimodal(image_path: str, text: str) -> dict:
    path = Path(image_path)
    if not path.exists():
        return {"error": f"Image introuvable : {image_path}"}
    with open(path, "rb") as f:
        files = {"file": (path.name, f, "image/jpeg")}
        data = {"text": text, "include_report": "false"}
        return _api_call("predict/multimodal", method="post", files=files, data=data)


def tool_generate_report(emotion: str, scores: dict, user_text: str = "") -> dict:
    """Fallback rule-based report (no API call needed for report)."""
    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    meta = EMOTION_META.get(emotion, {})
    return {
        "emotion": emotion,
        "label_fr": meta.get("label", emotion),
        "confidence": scores.get(emotion, 0),
        "top_3": [{"emotion": e, "score": s} for e, s in top3],
        "summary": f"L'émotion dominante détectée est '{meta.get('label', emotion)}' avec une confiance de {scores.get(emotion, 0)*100:.1f}%.",
        "user_text_context": user_text[:200] if user_text else None,
    }


def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    step = {"type": "tool", "name": tool_name, "input": tool_input, "time": datetime.now().strftime("%H:%M:%S")}
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
            user_text=tool_input.get("user_text", "")
        )
    else:
        result = {"error": f"Outil inconnu : {tool_name}"}

    step["result"] = result
    st.session_state.agent_steps.append(step)

    # Extract scores for visualization
    if "scores" in result and "emotion" in result:
        st.session_state.last_result = result

    return json.dumps(result, ensure_ascii=False)


# ─── Ollama helpers ───────────────────────────────────────────────────────────

def get_ollama_client():
    """Return an Ollama client pointed at the configured host."""
    return ollama_lib.Client(host=st.session_state.ollama_host)


def list_ollama_models() -> list[str]:
    """Return list of locally available Ollama models."""
    try:
        client = get_ollama_client()
        models = client.list()
        return [m.model for m in models.models]
    except Exception:
        return []


def check_ollama_health() -> tuple[bool, str]:
    """Ping Ollama host."""
    try:
        client = get_ollama_client()
        models = client.list()
        names = [m.model for m in models.models]
        return True, f"{len(names)} modèle(s) disponible(s)"
    except Exception as e:
        return False, str(e)


# ─── Agent loop ───────────────────────────────────────────────────────────────

def run_agent(user_message: str, image_path: str = None) -> str:
    st.session_state.agent_steps = []
    st.session_state.last_result = None

    client = get_ollama_client()

    # Build conversation
    content = user_message
    if image_path:
        content += f"\n\n[Image fournie : {image_path}]"

    messages = [
        {"role": "system", "content": st.session_state.system_prompt},
        {"role": "user",   "content": content},
    ]

    st.session_state.agent_steps.append({
        "type": "thinking",
        "text": f"Analyse de la requête : '{user_message[:80]}'" + (" + image" if image_path else ""),
        "time": datetime.now().strftime("%H:%M:%S")
    })

    for iteration in range(st.session_state.max_iterations):
        try:
            response = client.chat(
                model=st.session_state.model,
                messages=messages,
                tools=TOOLS,
                options={"temperature": st.session_state.temperature},
            )
        except ollama_lib.ResponseError as e:
            if "model not found" in str(e).lower():
                return (f"❌ Modèle '{st.session_state.model}' introuvable. "
                        f"Lancez : `ollama pull {st.session_state.model}`")
            return f"❌ Erreur Ollama : {e}"
        except Exception as e:
            return f"❌ Impossible de joindre Ollama ({st.session_state.ollama_host}) : {e}"

        msg = response.message

        # Append assistant turn to history
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls or []})

        # ── No tool calls → final answer ──────────────────────────────────────
        if not msg.tool_calls:
            final = msg.content or "⚠️ L'agent n'a pas retourné de réponse."
            st.session_state.agent_steps.append({
                "type": "final",
                "text": final,
                "time": datetime.now().strftime("%H:%M:%S")
            })
            return final

        # ── Process tool calls ────────────────────────────────────────────────
        for tc in msg.tool_calls:
            fn = tc.function
            tool_name  = fn.name
            tool_input = fn.arguments if isinstance(fn.arguments, dict) else json.loads(fn.arguments)

            result_str = dispatch_tool(tool_name, tool_input)

            # Feed tool result back as a tool message
            messages.append({
                "role":    "tool",
                "content": result_str,
            })

    return "⚠️ L'agent a atteint le nombre maximum d'itérations sans réponse finale."


# ─── Visualization helpers ────────────────────────────────────────────────────

def render_radar_chart(scores: dict) -> go.Figure:
    emotions = list(scores.keys())
    values = list(scores.values())
    values_pct = [v * 100 for v in values]

    colors = [EMOTION_META[e]["color"] for e in emotions]
    dominant = max(scores, key=scores.get)
    dominant_color = EMOTION_META[dominant]["color"]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_pct + [values_pct[0]],
        theta=emotions + [emotions[0]],
        fill='toself',
        fillcolor=f"rgba{tuple(int(dominant_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.15,)}",
        line=dict(color=dominant_color, width=2),
        name="Scores",
        hovertemplate="%{theta}: %{r:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(
                tickfont=dict(size=11, color="#94a3b8", family="DM Sans"),
                linecolor="#1e1e30",
                gridcolor="#1e1e30",
            ),
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=9, color="#6b6b8a"),
                gridcolor="#1e1e30",
                linecolor="#1e1e30",
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20, l=30, r=30),
        height=280,
        showlegend=False,
    )
    return fig


def render_bar_chart(scores: dict) -> go.Figure:
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    emotions = [e for e, _ in sorted_items]
    values = [v * 100 for _, v in sorted_items]
    colors = [EMOTION_META[e]["color"] for e in emotions]
    emojis = [EMOTION_META[e]["emoji"] for e in emotions]
    labels = [f"{EMOTION_META[e]['emoji']} {EMOTION_META[e]['label']}" for e in emotions]

    fig = go.Figure(go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker=dict(
            color=colors,
            opacity=[1.0 if i == 0 else 0.45 for i in range(len(emotions))],
            line=dict(width=0),
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=11, color="#94a3b8", family="DM Sans"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 115],
            showgrid=True,
            gridcolor="#1e1e30",
            tickfont=dict(color="#6b6b8a", size=10),
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(color="#e8e8f0", size=11, family="DM Sans"),
            categoryorder="total ascending",
        ),
        margin=dict(t=10, b=10, l=10, r=60),
        height=240,
        bargap=0.3,
    )
    return fig


# ─── Check API health ─────────────────────────────────────────────────────────

def check_api_health():
    try:
        r = requests.get(f"{st.session_state.api_base_url}/health", timeout=3)
        return r.status_code == 200, r.json() if r.status_code == 200 else {}
    except:
        return False, {}


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="hero-title">🧠 EmotionSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Agent Ollama · Multimodal</div>', unsafe_allow_html=True)

    # Ollama host
    st.markdown("#### 🦙 Serveur Ollama")
    ollama_host = st.text_input(
        "Ollama host",
        value=st.session_state.ollama_host,
        label_visibility="collapsed",
        placeholder="http://localhost:11434"
    )
    if ollama_host != st.session_state.ollama_host:
        st.session_state.ollama_host = ollama_host

    col_h1, col_h2 = st.columns([2, 1])
    with col_h1:
        if st.button("🔍 Vérifier Ollama"):
            ok, msg = check_ollama_health()
            if ok:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")

    # FastAPI URL
    st.markdown("#### 🌐 Serveur FastAPI")
    api_url = st.text_input(
        "FastAPI URL",
        value=st.session_state.api_base_url,
        label_visibility="collapsed",
    )
    st.session_state.api_base_url = api_url

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        if st.button("🔍 Vérifier FastAPI"):
            ok, health = check_api_health()
            if ok:
                st.success(f"✅ Serveur OK · {health.get('device', '?')}")
            else:
                st.error("❌ Serveur inaccessible")

    st.divider()

    # ── Agent Config ──
    st.markdown("#### ⚙️ Configuration Agent")

    # Fetch available models dynamically
    available_models = list_ollama_models()
    TOOL_CAPABLE_MODELS = [
        "llama3.2", "llama3.2:1b", "llama3.2:3b",
        "llama3.1", "llama3.1:8b", "llama3.1:70b",
        "qwen2.5", "qwen2.5:7b", "qwen2.5:14b",
        "mistral", "mistral-nemo",
        "command-r", "gemma3",
    ]
    # Merge: local models first, then suggestions not yet pulled
    model_options = available_models if available_models else TOOL_CAPABLE_MODELS
    # Ensure current model is in list
    if st.session_state.model not in model_options:
        model_options = [st.session_state.model] + model_options

    model_choice = st.selectbox(
        "Modèle Ollama",
        model_options,
        index=model_options.index(st.session_state.model) if st.session_state.model in model_options else 0,
        help="Seuls les modèles supportant le tool-calling fonctionnent (llama3.2, qwen2.5, mistral...)"
    )
    st.session_state.model = model_choice

    if available_models:
        st.caption(f"💾 {len(available_models)} modèle(s) local (locaux)")
    else:
        st.caption("⚠️ Ollama inaccessible — modèles par défaut affichés")
        st.code(f"ollama pull {model_choice}", language="bash")

    max_iter = st.slider("Max itérations", 2, 12, st.session_state.max_iterations)
    st.session_state.max_iterations = max_iter

    temperature = st.slider("Température", 0.0, 1.0, st.session_state.temperature, 0.05)
    st.session_state.temperature = temperature

    st.divider()

    # ── History ──
    if st.session_state.history:
        st.markdown("#### 🕓 Historique")
        for i, session in enumerate(reversed(st.session_state.history[-8:])):
            emotion = session.get("emotion", "?")
            meta = EMOTION_META.get(emotion, {})
            emoji = meta.get("emoji", "❓")
            label = meta.get("label", emotion)
            ts = session.get("timestamp", "")
            modality = session.get("modality", "?")
            st.markdown(
                f'<div class="history-item"><b>{emoji} {label}</b><br>'
                f'<span style="color:#6b6b8a;font-size:0.78rem;">{ts} · {modality}</span></div>',
                unsafe_allow_html=True
            )


# ─── Main layout ──────────────────────────────────────────────────────────────

tab_analyse, tab_agent, tab_improve = st.tabs(["🎯 Analyse", "🤖 Agent & Raisonnement", "🔧 Améliorer l'Agent"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Analyse
# ══════════════════════════════════════════════════════════════════════════════

with tab_analyse:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
        st.markdown("### 📝 Entrée")

        user_text = st.text_area(
            "Texte à analyser",
            placeholder="Écrivez quelque chose... ex: 'Je me sens vraiment débordé aujourd'hui'",
            height=100,
            label_visibility="collapsed",
        )

        uploaded_image = st.file_uploader(
            "Image faciale (optionnel)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Téléchargez une image de visage pour l'analyse faciale"
        )

        if uploaded_image:
            st.image(uploaded_image, caption="Image chargée", use_container_width=True)

        # Modality indicator
        if user_text and uploaded_image:
            st.info("🔮 Mode **Multimodal** — Fusion Attention (précision max ~83%)")
        elif uploaded_image:
            st.info("🖼️ Mode **Image** — ResNet-50")
        elif user_text:
            st.info("📝 Mode **Texte** — BERT")
        else:
            st.warning("Entrez du texte et/ou une image pour commencer.")

        st.markdown('</div>', unsafe_allow_html=True)

        run_btn = st.button("🚀 Analyser avec l'agent AI", use_container_width=True)

    with col_result:
        if run_btn:
            if not user_text and not uploaded_image:
                st.error("Fournissez du texte ou une image.")
            elif not st.session_state.ollama_host:
                st.error("URL Ollama manquante dans la sidebar.")
            else:
                # Save image to temp file
                tmp_path = None
                if uploaded_image:
                    suffix = Path(uploaded_image.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_image.getvalue())
                        tmp_path = tmp.name

                # Build message
                if not user_text:
                    msg = "Analyse l'émotion dans cette image."
                else:
                    msg = user_text

                with st.spinner("L'agent réfléchit… (Ollama)"):
                    t0 = time.time()
                    final_response = run_agent(msg, tmp_path)
                    elapsed = time.time() - t0

                # Display results
                result = st.session_state.last_result

                if result and "emotion" in result:
                    emotion = result["emotion"]
                    meta = EMOTION_META.get(emotion, {})
                    confidence = result.get("confidence", 0)

                    # Emotion badge
                    badge_color = meta.get("color", "#7c6af7")
                    st.markdown(
                        f'<div style="text-align:center;margin-bottom:1rem;">'
                        f'<span class="emotion-badge" style="background:{badge_color}22;color:{badge_color};border:1px solid {badge_color}44;">'
                        f'{meta.get("emoji","❓")} {meta.get("label", emotion).upper()} — {confidence*100:.1f}%'
                        f'</span></div>',
                        unsafe_allow_html=True
                    )

                    # Charts
                    scores = result.get("scores", {})
                    if scores:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.plotly_chart(render_radar_chart(scores), use_container_width=True)
                        with c2:
                            st.plotly_chart(render_bar_chart(scores), use_container_width=True)

                    # Modality
                    steps = st.session_state.agent_steps
                    tool_used = next((s["name"] for s in steps if s.get("type") == "tool" and s.get("name") in ["analyze_image", "analyze_text", "analyze_multimodal"]), "?")
                    modality_map = {"analyze_image": "Image", "analyze_text": "Texte", "analyze_multimodal": "Multimodal"}
                    modality = modality_map.get(tool_used, "?")

                    # Save to history
                    st.session_state.history.append({
                        "emotion": emotion,
                        "confidence": confidence,
                        "scores": scores,
                        "response": final_response,
                        "modality": modality,
                        "elapsed": round(elapsed, 1),
                        "timestamp": datetime.now().strftime("%H:%M"),
                    })

                else:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### Réponse de l'agent")
                    st.markdown(final_response)
                    st.markdown('</div>', unsafe_allow_html=True)

        # Show last response if exists
        if st.session_state.history:
            last = st.session_state.history[-1]
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 💬 Réponse détaillée")
            st.markdown(last.get("response", ""))
            st.markdown(
                f'<div style="color:#6b6b8a;font-size:0.8rem;margin-top:0.5rem;">'
                f'⏱ {last.get("elapsed","?")}s · 🧠 {last.get("modality","?")} · {last.get("timestamp","")}</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Agent & Raisonnement
# ══════════════════════════════════════════════════════════════════════════════

with tab_agent:
    st.markdown("### 🤖 Trace de raisonnement de l'agent (ReAct)")
    st.caption("Chaque itération du loop Reasoning → Action → Observation est affichée ici.")

    if not st.session_state.agent_steps:
        st.markdown(
            '<div style="text-align:center;color:#6b6b8a;padding:3rem;">'
            '⚡ Lancez une analyse dans l\'onglet "Analyse" pour voir le raisonnement ici.'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        step_icons = {
            "thinking": ("💭", "step-thinking", "RAISONNEMENT"),
            "tool":     ("🔧", "step-tool",     "APPEL OUTIL"),
            "final":    ("✅", "step-final",    "RÉPONSE FINALE"),
        }

        for i, step in enumerate(st.session_state.agent_steps):
            stype = step.get("type", "thinking")
            icon, css_class, label = step_icons.get(stype, ("❓", "", "?"))
            ts = step.get("time", "")

            if stype == "thinking":
                st.markdown(
                    f'<div class="agent-step {css_class}">'
                    f'<span style="color:#6b6b8a;font-size:0.75rem;">[{ts}] {label}</span><br>'
                    f'{icon} {step.get("text","")}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            elif stype == "tool":
                tool_name = step.get("name", "?")
                inp = step.get("input", {})
                result = step.get("result", {})
                emotion = result.get("emotion")
                conf = result.get("confidence", 0)
                err = result.get("error")

                with st.expander(f"🔧 {tool_name}  [{ts}]", expanded=(i == len(st.session_state.agent_steps) - 1)):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Input**")
                        st.code(json.dumps(inp, ensure_ascii=False, indent=2), language="json")
                    with c2:
                        st.markdown("**Résultat**")
                        if err:
                            st.error(err)
                        elif emotion:
                            meta = EMOTION_META.get(emotion, {})
                            st.success(f"{meta.get('emoji','?')} {meta.get('label', emotion)} · {conf*100:.1f}%")
                            if "scores" in result:
                                top3 = sorted(result["scores"].items(), key=lambda x: x[1], reverse=True)[:3]
                                for e, s in top3:
                                    m = EMOTION_META.get(e, {})
                                    st.progress(s, text=f"{m.get('emoji','')} {m.get('label',e)}: {s*100:.1f}%")
                        else:
                            st.code(json.dumps(result, ensure_ascii=False, indent=2)[:400], language="json")

            elif stype == "final":
                st.markdown(
                    f'<div class="agent-step {css_class}">'
                    f'<span style="color:#6b6b8a;font-size:0.75rem;">[{ts}] {label}</span><br>'
                    f'{icon} Réponse finale générée ({len(step.get("text",""))} caractères)'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # Summary metrics
        st.markdown("---")
        n_tools = sum(1 for s in st.session_state.agent_steps if s.get("type") == "tool")
        n_iter = n_tools
        st.markdown("#### 📊 Résumé de la session")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Étapes agent", len(st.session_state.agent_steps))
        with c2:
            st.metric("Outils appelés", n_tools)
        with c3:
            model_short = st.session_state.model.split(":")[0].capitalize()
            st.metric("Modèle", model_short)
        with c4:
            st.metric("Max itérations", st.session_state.max_iterations)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Améliorer l'Agent
# ══════════════════════════════════════════════════════════════════════════════

with tab_improve:
    st.markdown("### 🔧 Améliorer l'Agent AI")
    st.caption("Modifiez le system prompt, les outils, les paramètres, et testez en direct.")

    col_prompt, col_tools = st.columns([1, 1], gap="large")

    with col_prompt:
        # ── System Prompt Editor ──
        st.markdown("#### 📋 System Prompt")
        st.caption("Modifiez les instructions de l'agent. Cela change son comportement de raisonnement.")

        new_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.system_prompt,
            height=380,
            label_visibility="collapsed",
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("💾 Appliquer", use_container_width=True):
                st.session_state.system_prompt = new_prompt
                st.success("✅ System prompt mis à jour !")
        with col_btn2:
            if st.button("🔄 Réinitialiser", use_container_width=True):
                st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
                st.rerun()

        # ── Prompt suggestions ──
        st.markdown("#### 💡 Suggestions d'amélioration")

        improvements = [
            {
                "title": "🌍 Multilingue",
                "desc": "Répondre en arabe si l'utilisateur écrit en arabe",
                "patch": "\n- Si l'utilisateur écrit en arabe → réponds en arabe (العربية)\n- Si l'utilisateur écrit en anglais → réponds en anglais"
            },
            {
                "title": "🎯 Précision++",
                "desc": "Toujours mentionner l'intervalle de confiance",
                "patch": "\n- Mentionne toujours l'intervalle de confiance (±5% pour BERT, ±3% pour Fusion)\n- Indique le niveau de certitude : faible (<40%), moyen (40-70%), élevé (>70%)"
            },
            {
                "title": "📊 Top-5 émotions",
                "desc": "Afficher les 5 meilleures émotions au lieu de 3",
                "patch": "\n- Ta réponse finale doit inclure le TOP-5 des émotions avec leurs probabilités (pas seulement top-3)"
            },
            {
                "title": "🔁 Auto-retry",
                "desc": "Si erreur serveur, suggérer une alternative",
                "patch": "\n- Si analyze_multimodal échoue → essaie analyze_image puis analyze_text séparément\n- En cas d'erreur serveur FastAPI, propose à l'utilisateur de vérifier le serveur"
            },
        ]

        for imp in improvements:
            with st.expander(f"{imp['title']} — {imp['desc']}"):
                st.code(imp['patch'], language="text")
                if st.button(f"➕ Ajouter au prompt", key=f"add_{imp['title']}"):
                    st.session_state.system_prompt += imp['patch']
                    st.success(f"Ajouté : {imp['title']}")

    with col_tools:
        # ── Tool editor ──
        st.markdown("#### 🛠️ Outils disponibles")
        st.caption("Visualisez et comprenez les outils de l'agent. Les descriptions influencent quand Claude choisit chaque outil.")

        for tool in TOOLS:
            fn = tool["function"]
            with st.expander(f"🔧 `{fn['name']}`"):
                st.markdown("**Description :**")
                st.markdown(f"> {fn['description']}")
                st.markdown(f"**Paramètres requis :** `{', '.join(fn['parameters'].get('required', []))}`")

                new_desc = st.text_area(
                    "Modifier la description",
                    value=fn['description'],
                    height=80,
                    key=f"tool_desc_{fn['name']}"
                )
                if st.button("Mettre à jour", key=f"update_{fn['name']}"):
                    for t in TOOLS:
                        if t["function"]["name"] == fn["name"]:
                            t["function"]["description"] = new_desc
                    st.success(f"✅ Description de `{fn['name']}` mise à jour !")

        st.markdown("---")

        # ── Add custom tool ──
        st.markdown("#### ➕ Ajouter un outil personnalisé")
        with st.expander("Nouvel outil (simulation)"):
            new_tool_name = st.text_input("Nom de l'outil", placeholder="analyze_audio")
            new_tool_desc = st.text_area("Description", placeholder="Analyse la prosodie vocale pour détecter les émotions...", height=80)
            new_tool_params = st.text_input("Paramètre requis", placeholder="audio_path")
            if st.button("➕ Ajouter l'outil"):
                if new_tool_name and new_tool_desc:
                    TOOLS.append({
                        "type": "function",
                        "function": {
                            "name": new_tool_name,
                            "description": new_tool_desc,
                            "parameters": {
                                "type": "object",
                                "properties": {new_tool_params: {"type": "string"}} if new_tool_params else {},
                                "required": [new_tool_params] if new_tool_params else []
                            }
                        }
                    })
                    st.success(f"✅ Outil `{new_tool_name}` ajouté ! (implémentez le handler dans `dispatch_tool`)")
        st.markdown("---")

        # ── Test prompt inline ──
        st.markdown("#### 🧪 Test rapide du prompt")
        test_text = st.text_input("Phrase de test", placeholder="Je suis tellement en colère !")
        if st.button("🔬 Tester (texte seul)", use_container_width=True):
            if test_text:
                with st.spinner("Test en cours…"):
                    resp = run_agent(test_text)
                result = st.session_state.last_result
                if result:
                    emotion = result.get("emotion", "?")
                    meta = EMOTION_META.get(emotion, {})
                    st.success(f"{meta.get('emoji','?')} {meta.get('label', emotion)} · {result.get('confidence',0)*100:.1f}%")
                st.markdown(resp[:500] + "…" if len(resp) > 500 else resp)
            else:
                st.warning("Entrez une phrase de test.")

        # ── Prompt stats ──
        st.markdown("---")
        st.markdown("#### 📈 Stats du prompt actuel")
        prompt = st.session_state.system_prompt
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Tokens ~", len(prompt.split()))
        with c2:
            st.metric("Règles détectées", prompt.count("-"))
        with c3:
            rules_lang = "FR" if "français" in prompt else "EN" if "english" in prompt.lower() else "?"
            st.metric("Langue principale", rules_lang)