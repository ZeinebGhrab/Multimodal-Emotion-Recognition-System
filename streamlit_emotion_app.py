"""
streamlit_emotion_app.py
─────────────────────────
Streamlit interface for the multimodal emotion recognition AI agent.
Design: clean professional theme — imports emotion_styles.py for CSS and charts.

Features:
  1. Text, image, or multimodal analysis
  2. Radar and bar chart visualization of scores
  3. Ollama agent ReAct trace
  4. System prompt and tools editor
  5. Session history

Usage:
    streamlit run streamlit_emotion_app.py
    # Prerequisites: uvicorn api.app:app --port 8000 + ollama serve
"""

import json, time, tempfile, requests
import ollama as ollama_lib
from pathlib import Path
from datetime import datetime

import streamlit as st

# ── Page config (must be FIRST Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="EmotionSense AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system (CSS + chart helpers) ──────────────────────────────────────
from emotion_styles import (
    inject_css,
    EMOTION_META,
    render_radar_chart,
    render_bar_chart,
)
inject_css()

# ─── Constants ────────────────────────────────────────────────────────────────

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

DEFAULT_SYSTEM_PROMPT = """\
Tu es un agent expert en reconnaissance des émotions multimodale.

⚠️ RÈGLE CRITIQUE — NE JAMAIS traduire le texte utilisateur :
  Le paramètre 'text' passé à analyze_text ou analyze_multimodal doit être
  le texte EXACT de l'utilisateur, mot pour mot, sans aucune modification.
  BERT est entraîné sur l'anglais — toute traduction détruit la prédiction.
  ✅ Correct  : text="I feel so glad"
  ❌ Interdit : text="Je me sens tellement heureux"

Outils disponibles :
  1. analyze_image      — image faciale (ResNet-50)
  2. analyze_text       — texte EXACT tel que saisi (BERT anglais)
  3. analyze_multimodal — image + texte exact (Attention Fusion ~83%)
  4. generate_report    — rapport psychologique (appelle TOUJOURS après analyse)

Routing :
  Image ET texte → analyze_multimodal
  Image seule    → analyze_image
  Texte seul     → analyze_text

Émotions valides : angry, disgust, fear, happy, neutral, sad, surprise
NE PAS inventer d'autres labels.
Réponse finale en français.\
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": (
                "Analyse une image faciale pour détecter l'émotion. "
                "Utilise ResNet-50 fine-tuné sur FER2013. "
                "Appelle cet outil quand l'utilisateur fournit une image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Chemin local vers l'image (JPEG/PNG)"}
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
                "Analyse un texte court pour détecter l'émotion. "
                "Utilise BERT fine-tuné sur dair-ai/emotion. "
                "Appelle cet outil quand l'utilisateur fournit du texte sans image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Texte à analyser"}
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
                "Analyse combinée image + texte. Utilise Attention Fusion (ResNet-50 + BERT). "
                "Appelle cet outil quand l'utilisateur fournit UNE IMAGE ET UN TEXTE. "
                "C'est l'outil le plus précis (~83% accuracy)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Chemin local vers l'image"},
                    "text": {"type": "string", "description": "Texte accompagnant l'image"},
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
                "Génère un rapport psychologique structuré. "
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
                        "description": "Texte original de l'utilisateur (contexte)",
                    },
                },
                "required": ["emotion", "scores"],
            },
        },
    },
]


# ─── Session state ────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "ollama_host":   "http://localhost:11434",
        "api_base_url":  "http://localhost:8000",
        "history":       [],
        "agent_steps":   [],
        "last_result":   None,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "model":         "llama3.2",
        "max_iterations": 6,
        "temperature":   0.3,
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
        return {"error": f"Impossible de joindre FastAPI à {st.session_state.api_base_url}. Lancez : uvicorn api.app:app --port 8000"}
    except Exception as e:
        return {"error": str(e)}


def tool_analyze_image(image_path: str) -> dict:
    path = Path(image_path)
    if not path.exists():
        return {"error": f"Image introuvable : {image_path}"}
    with open(path, "rb") as f:
        return _api_call("predict/image", method="post", files={"file": (path.name, f, "image/jpeg")})


def tool_analyze_text(text: str) -> dict:
    # Garde contre les types inattendus
    if not isinstance(text, str):
        text = str(text)
    if not text.strip():
        return {"error": "Le texte ne peut pas être vide"}
    return _api_call("predict/text", method="post",
                     json={"text": text, "include_report": False})


def tool_analyze_multimodal(image_path: str, text: str) -> dict:
    path = Path(image_path)
    if not path.exists():
        return {"error": f"Image introuvable : {image_path}"}
    with open(path, "rb") as f:
        return _api_call(
            "predict/multimodal",
            method="post",
            files={"file": (path.name, f, "image/jpeg")},
            data={"text": text, "include_report": "false"},
        )


def tool_generate_report(emotion: str, scores, user_text: str = "") -> dict:
    """Rapport règle-based (pas d'appel API)."""
    if isinstance(scores, str):
        try:
            scores = json.loads(scores)
        except (json.JSONDecodeError, ValueError):
            scores = {}
    if not isinstance(scores, dict) or not scores:
        scores = {emotion: 1.0}

    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

    # FIX: retourner le résultat (était manquant → retournait None)
    return {
        "emotion":   emotion,
        "scores":    scores,
        "top3":      [{"emotion": e, "score": round(s, 4)} for e, s in top3],
        "user_text": user_text,
        "report": (
            f"Émotion dominante détectée : **{emotion}** "
            f"(score : {scores.get(emotion, 1.0):.2%}). "
            + (f"Top 3 : {', '.join(f'{e} ({s:.0%})' for e, s in top3)}. " if len(top3) > 1 else "")
            + (f'Texte analysé : « {user_text[:120]}{"…" if len(user_text) > 120 else ""} ».' if user_text else "")
        ),
    }


def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except (json.JSONDecodeError, ValueError):
            tool_input = {}

    step = {
        "type": "tool",
        "name": tool_name,
        "input": tool_input,
        "time": datetime.now().strftime("%H:%M:%S"),
    }

    if tool_name == "analyze_image":
        path = tool_input.get("image_path", "")
        if not path or not Path(str(path)).exists():
            result = {"error": "Aucune image disponible. Utilise analyze_text."}
        else:
            result = tool_analyze_image(str(path))

    elif tool_name == "analyze_text":
        text = tool_input.get("text", "")
        # Garde contre dict imbriqué
        if isinstance(text, dict):
            text = str(list(text.values())[0]) if text else ""
        text = str(text).strip()
        # Fallback : si Ollama a traduit, utiliser le texte original
        original = st.session_state.get("_last_user_text", "")
        if original and text != original:
            print(f"[dispatch] Texte modifié détecté → forcé au texte original")
            text = original
        result = tool_analyze_text(text) if text else {"error": "Texte vide"}

    elif tool_name == "analyze_multimodal":
        path = tool_input.get("image_path", "")
        text = tool_input.get("text", "")
        if isinstance(text, dict):
            text = str(list(text.values())[0]) if text else ""
        text = str(text).strip()
        # Fallback texte original
        original = st.session_state.get("_last_user_text", "")
        if original and text != original:
            text = original
        if not path or not Path(str(path)).exists():
            result = {"error": "Aucune image disponible. Utilise analyze_text."}
        else:
            result = tool_analyze_multimodal(str(path), text)

    elif tool_name == "generate_report":
        emotion   = tool_input.get("emotion", "neutral")
        scores    = tool_input.get("scores", {})
        user_text = tool_input.get("user_text", "")
        if isinstance(user_text, dict):
            user_text = ""
        # Valider l'émotion
        VALID = {"angry","disgust","fear","happy","neutral","sad","surprise"}
        if str(emotion) not in VALID:
            emotion = "neutral"
        result = tool_generate_report(str(emotion), scores, str(user_text))
    else:
        result = {"error": f"Outil inconnu : {tool_name}"}

    if result is None:
        result = {"error": f"L'outil '{tool_name}' n'a retourné aucun résultat."}

    step["result"] = result
    st.session_state.agent_steps.append(step)

    if isinstance(result, dict) and "scores" in result and "emotion" in result:
        st.session_state.last_result = result

    return json.dumps(result, ensure_ascii=False)


# ─── Ollama helpers ───────────────────────────────────────────────────────────

def get_ollama_client():
    return ollama_lib.Client(host=st.session_state.ollama_host)

def list_ollama_models() -> list[str]:
    try:
        return [m.model for m in get_ollama_client().list().models]
    except Exception:
        return []

def check_ollama_health() -> tuple[bool, str]:
    try:
        names = [m.model for m in get_ollama_client().list().models]
        return True, f"{len(names)} modèle(s) disponible(s)"
    except Exception as e:
        return False, str(e)

def check_api_health():
    try:
        r = requests.get(f"{st.session_state.api_base_url}/health", timeout=3)
        return r.status_code == 200, r.json() if r.status_code == 200 else {}
    except Exception:
        return False, {}


# ─── ReAct agent loop ─────────────────────────────────────────────────────────

def run_agent(user_message: str, image_path: str = None) -> str:
    st.session_state.agent_steps = []
    st.session_state.last_result = None
    client = get_ollama_client()

    has_image = image_path is not None and Path(image_path).exists()
    has_text  = bool(user_message.strip())

    # ── Sélection des outils selon les inputs disponibles ────────────────────
    TOOL_MAP = {t["function"]["name"]: t for t in TOOLS}
    if has_image and has_text:
        active_tools = [TOOL_MAP["analyze_multimodal"], TOOL_MAP["generate_report"]]
        instruction  = (
            f"Appelle analyze_multimodal avec :\n"
            f'  image_path="{image_path}"\n'
            f'  text="{user_message}"  ← texte EXACT, ne pas modifier\n'
            f"Puis appelle generate_report."
        )
    elif has_image:
        active_tools = [TOOL_MAP["analyze_image"], TOOL_MAP["generate_report"]]
        instruction  = (
            f"Appelle analyze_image avec image_path=\"{image_path}\"\n"
            f"Puis appelle generate_report."
        )
    else:
        active_tools = [TOOL_MAP["analyze_text"], TOOL_MAP["generate_report"]]
        instruction  = (
            f'Appelle analyze_text avec text="{user_message}"\n'
            f"← Texte EXACT, NE PAS traduire, NE PAS modifier.\n"
            f"Puis appelle generate_report."
        )

    content = f"{user_message}\n\n[INSTRUCTION AGENT : {instruction}]"

    messages = [
        {"role": "system", "content": st.session_state.system_prompt},
        {"role": "user",   "content": content},
    ]
    st.session_state.agent_steps.append({
        "type": "thinking",
        "text": f"Requête : '{user_message[:80]}'" + (" + image" if has_image else ""),
        "time": datetime.now().strftime("%H:%M:%S"),
    })

    for _ in range(st.session_state.max_iterations):
        try:
            response = client.chat(
                model=st.session_state.model,
                messages=messages,
                tools=active_tools,      # ← outils filtrés
                options={"temperature": st.session_state.temperature},
            )
        except ollama_lib.ResponseError as e:
            if "model not found" in str(e).lower():
                return f"❌ Modèle '{st.session_state.model}' introuvable."
            return f"❌ Erreur Ollama : {e}"
        except Exception as e:
            return f"❌ Impossible de joindre Ollama : {e}"

        msg = response.message
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": msg.tool_calls or []
        })

        if not msg.tool_calls:
            final = msg.content or "⚠️ Pas de réponse."
            st.session_state.agent_steps.append({
                "type": "final", "text": final,
                "time": datetime.now().strftime("%H:%M:%S"),
            })
            return final

        for tc in msg.tool_calls:
            fn = tc.function
            raw = fn.arguments
            if isinstance(raw, dict):
                tool_input = raw
            elif isinstance(raw, str):
                try:
                    tool_input = json.loads(raw)
                except Exception:
                    tool_input = {}
            else:
                try:
                    tool_input = dict(raw)
                except Exception:
                    tool_input = vars(raw) if hasattr(raw, "__dict__") else {}

            result_str = dispatch_tool(fn.name, tool_input)
            messages.append({"role": "tool", "content": result_str})

    return "⚠️ Nombre maximum d'itérations atteint."


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Brand
    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="sidebar-logo">🎭 EmotionSense</div>'
        '<div class="sidebar-tagline">Multimodal AI · Ollama Agent</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Ollama ──
    st.markdown('<div class="sidebar-section">🦙 Serveur Ollama</div>', unsafe_allow_html=True)
    ollama_host = st.text_input(
        "ollama_host", value=st.session_state.ollama_host,
        label_visibility="collapsed", placeholder="http://localhost:11434"
    )
    if ollama_host != st.session_state.ollama_host:
        st.session_state.ollama_host = ollama_host

    if st.button("Vérifier Ollama", use_container_width=True):
        ok, msg_txt = check_ollama_health()
        (st.success if ok else st.error)(("✅ " if ok else "❌ ") + msg_txt)

    # ── FastAPI ──
    st.markdown('<div class="sidebar-section">🌐 Serveur FastAPI</div>', unsafe_allow_html=True)
    api_url = st.text_input(
        "api_url", value=st.session_state.api_base_url,
        label_visibility="collapsed"
    )
    st.session_state.api_base_url = api_url

    if st.button("Vérifier FastAPI", use_container_width=True):
        ok, health = check_api_health()
        if ok:
            st.success(f"✅ Serveur OK · {health.get('device','?')}")
        else:
            st.error("❌ Serveur inaccessible")

    # ── Agent config ──
    st.markdown('<div class="sidebar-section">⚙️ Configuration Agent</div>', unsafe_allow_html=True)

    TOOL_CAPABLE_MODELS = [
        "llama3.2", "llama3.2:1b", "llama3.2:3b",
        "llama3.1", "llama3.1:8b", "qwen2.5", "qwen2.5:7b",
        "mistral", "mistral-nemo", "command-r", "gemma3",
    ]
    available_models = list_ollama_models()
    model_options = available_models if available_models else TOOL_CAPABLE_MODELS
    if st.session_state.model not in model_options:
        model_options = [st.session_state.model] + model_options

    model_choice = st.selectbox(
        "Modèle Ollama", model_options,
        index=model_options.index(st.session_state.model) if st.session_state.model in model_options else 0,
        help="Modèles supportant le tool-calling : llama3.2, qwen2.5, mistral…"
    )
    st.session_state.model = model_choice

    if not available_models:
        st.caption("⚠️ Ollama inaccessible")
        st.code(f"ollama pull {model_choice}", language="bash")
    else:
        st.caption(f"💾 {len(available_models)} modèle(s) local (locaux)")

    st.session_state.max_iterations = st.slider("Max itérations", 2, 12, st.session_state.max_iterations)
    st.session_state.temperature    = st.slider("Température",    0.0, 1.0, st.session_state.temperature, 0.05)

    # ── History ──
    if st.session_state.history:
        st.markdown('<div class="sidebar-section">🕓 Historique</div>', unsafe_allow_html=True)
        for session in reversed(st.session_state.history[-8:]):
            emotion = session.get("emotion", "?")
            meta    = EMOTION_META.get(emotion, {})
            st.markdown(
                f'<div class="hist-item">'
                f'<div class="hist-emotion">{meta.get("emoji","❓")} {meta.get("label", emotion)}</div>'
                f'<div class="hist-meta">{session.get("timestamp","")} · {session.get("modality","?")} · {session.get("confidence",0)*100:.0f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="page-header">'
    '<div class="page-title">🎭 EmotionSense AI</div>'
    '<div class="page-subtitle">Reconnaissance d\'émotions multimodale — Deep Learning + Agent Ollama</div>'
    '<div class="page-badges">'
    '<span class="page-badge">ResNet-50</span>'
    '<span class="page-badge">BERT</span>'
    '<span class="page-badge">Attention Fusion ~83%</span>'
    '<span class="page-badge">ReAct Agent</span>'
    '<span class="page-badge">FER2013</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

tab_analyse, tab_agent, tab_improve = st.tabs(
    ["🎯  Analyse", "🤖  Agent & Raisonnement", "🔧  Améliorer l'Agent"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Analyse
# ══════════════════════════════════════════════════════════════════════════════

with tab_analyse:
    col_in, col_out = st.columns([1, 1], gap="large")

    # ── Input panel ──────────────────────────────────────────────────────────
    with col_in:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📝 Entrée</div>', unsafe_allow_html=True)

        user_text = st.text_area(
            "texte",
            placeholder="Écrivez quelque chose… ex : « Je me sens vraiment débordé aujourd'hui »",
            height=110,
            label_visibility="collapsed",
        )

        st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)

        uploaded_image = st.file_uploader(
            "Image faciale (optionnel)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Téléchargez une photo de visage pour l'analyse faciale",
        )

        if uploaded_image:
            st.image(uploaded_image, use_column_width=True)

        # Mode indicator
        st.markdown('<div style="margin-top:0.8rem"></div>', unsafe_allow_html=True)
        if user_text and uploaded_image:
            st.markdown(
                '<div class="mode-pill multi">🔮 Mode Multimodal — Attention Fusion (~83%)</div>',
                unsafe_allow_html=True,
            )
        elif uploaded_image:
            st.markdown(
                '<div class="mode-pill image">🖼️ Mode Image — ResNet-50</div>',
                unsafe_allow_html=True,
            )
        elif user_text:
            st.markdown(
                '<div class="mode-pill text">📝 Mode Texte — BERT</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="mode-pill empty">💡 Entrez du texte et/ou une image</div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)
        run_btn = st.button("🚀  Analyser avec l'Agent AI", use_container_width=True)

    # ── Result panel ─────────────────────────────────────────────────────────
    with col_out:
        if run_btn:
            if not user_text and not uploaded_image:
                st.error("Fournissez du texte ou une image.")
            elif not st.session_state.ollama_host:
                st.error("URL Ollama manquante dans la sidebar.")
            else:
                tmp_path = None
                if uploaded_image:
                    suffix = Path(uploaded_image.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_image.getvalue())
                        tmp_path = tmp.name

                msg = user_text if user_text else "Analyse l'émotion dans cette image."
                
                with st.spinner("L'agent réfléchit… (Ollama)"):
                    t0 = time.time()
                    final_response = run_agent(msg, tmp_path)
                    elapsed = time.time() - t0

                result = st.session_state.last_result

                if result and "emotion" in result:
                    emotion    = result["emotion"]
                    meta       = EMOTION_META.get(emotion, {})
                    confidence = result.get("confidence", 0)
                    color      = meta.get("color", "#3B4EE8")
                    bg         = meta.get("bg",    "#EEF0FD")

                    # ── Emotion result card ──
                    st.markdown(
                        f'<div class="result-card" style="border-top:4px solid {color};">'
                        f'<div style="display:flex;align-items:center;gap:1.2rem;margin-bottom:1.2rem;">'
                        f'<div class="result-emoji">{meta.get("emoji","❓")}</div>'
                        f'<div>'
                        f'<div class="result-emotion-name" style="color:{color};">{meta.get("label", emotion).upper()}</div>'
                        f'<div class="result-confidence">Confiance : {confidence*100:.1f}%</div>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Charts side-by-side
                    scores = result.get("scores", {})
                    if scores:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.plotly_chart(render_radar_chart(scores), use_container_width=True)
                        with c2:
                            st.plotly_chart(render_bar_chart(scores), use_container_width=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                    # Determine modality used
                    steps = st.session_state.agent_steps
                    tool_used = next(
                        (s["name"] for s in steps if s.get("type") == "tool"
                         and s.get("name") in ["analyze_image", "analyze_text", "analyze_multimodal"]),
                        "?"
                    )
                    modality_map = {"analyze_image": "Image", "analyze_text": "Texte", "analyze_multimodal": "Multimodal"}
                    modality = modality_map.get(tool_used, "?")

                    # Save to history
                    st.session_state.history.append({
                        "emotion":    emotion,
                        "confidence": confidence,
                        "scores":     scores,
                        "response":   final_response,
                        "modality":   modality,
                        "elapsed":    round(elapsed, 1),
                        "timestamp":  datetime.now().strftime("%H:%M"),
                    })
                else:
                    st.markdown(
                        f'<div class="card"><div class="card-title">💬 Réponse de l\'agent</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(final_response)
                    st.markdown('</div>', unsafe_allow_html=True)

        # Show last detailed response
        if st.session_state.history:
            last = st.session_state.history[-1]
            st.markdown(
                '<div class="card" style="margin-top:1.5rem;">'
                '<div class="card-title">💬 Rapport détaillé</div>',
                unsafe_allow_html=True,
            )
            st.markdown(last.get("response", ""))
            st.markdown(
                f'<div style="margin-top:0.8rem;padding-top:0.8rem;border-top:1px solid #EEF0F8;'
                f'font-size:0.78rem;color:#8F95B2;">'
                f'⏱ {last.get("elapsed","?")}s &nbsp;·&nbsp; '
                f'🧠 {last.get("modality","?")} &nbsp;·&nbsp; '
                f'{last.get("timestamp","")}'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Agent & Raisonnement
# ══════════════════════════════════════════════════════════════════════════════

with tab_agent:
    st.markdown(
        '<div class="card-title" style="font-size:0.95rem;color:#1A1F36;text-transform:none;letter-spacing:0;">'
        '🤖 Trace de raisonnement — Boucle ReAct (Reasoning → Action → Observation)'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("Chaque itération du loop agent est affichée ici après une analyse.")

    if not st.session_state.agent_steps:
        st.markdown(
            '<div style="text-align:center;color:#8F95B2;padding:4rem 2rem;'
            'background:#FFFFFF;border:1px dashed #C7CBFA;border-radius:16px;margin-top:1rem;">'
            '<div style="font-size:2.5rem;margin-bottom:0.8rem;">⚡</div>'
            '<div style="font-weight:600;color:#3B4EE8;">Aucune analyse lancée</div>'
            '<div style="font-size:0.85rem;margin-top:0.3rem;">Lancez une analyse dans l\'onglet <b>Analyse</b></div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        for i, step in enumerate(st.session_state.agent_steps):
            stype = step.get("type", "thinking")
            ts    = step.get("time", "")

            if stype == "thinking":
                st.markdown(
                    f'<div class="trace-step thinking">'
                    f'<div class="trace-label thinking">💭 [{ts}] Raisonnement</div>'
                    f'{step.get("text","")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            elif stype == "tool":
                tool_name = step.get("name", "?")
                inp       = step.get("input", {})
                result    = step.get("result", {})
                emotion   = result.get("emotion") if isinstance(result, dict) else None
                conf      = result.get("confidence", 0) if isinstance(result, dict) else 0
                err       = result.get("error") if isinstance(result, dict) else None

                with st.expander(f"🔧  {tool_name}  [{ts}]", expanded=(i == len(st.session_state.agent_steps) - 1)):
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
                            color = meta.get("color", "#3B4EE8")
                            st.markdown(
                                f'<div style="background:{meta.get("bg","#EEF0FD")};border:1px solid {color}33;'
                                f'border-radius:10px;padding:0.8rem 1rem;">'
                                f'<b style="color:{color};font-size:1.1rem;">{meta.get("emoji","?")} {meta.get("label",emotion)}</b>'
                                f'<div style="font-size:0.82rem;color:#8F95B2;margin-top:0.2rem;">{conf*100:.1f}% confiance</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            if "scores" in result:
                                top3 = sorted(result["scores"].items(), key=lambda x: x[1], reverse=True)[:3]
                                for e, s in top3:
                                    m = EMOTION_META.get(e, {})
                                    st.progress(s, text=f"{m.get('emoji','')} {m.get('label',e)}: {s*100:.1f}%")
                        else:
                            st.code(json.dumps(result, ensure_ascii=False, indent=2)[:500], language="json")

            elif stype == "final":
                st.markdown(
                    f'<div class="trace-step final">'
                    f'<div class="trace-label final"> [{ts}] Réponse finale</div>'
                    f'Réponse générée — {len(step.get("text",""))} caractères'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # KPIs
        st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
        n_tools = sum(1 for s in st.session_state.agent_steps if s.get("type") == "tool")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Étapes agent",   len(st.session_state.agent_steps))
        with c2:
            st.metric("Outils appelés", n_tools)
        with c3:
            st.metric("Modèle",         st.session_state.model.split(":")[0].capitalize())
        with c4:
            st.metric("Max itérations", st.session_state.max_iterations)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Améliorer l'Agent
# ══════════════════════════════════════════════════════════════════════════════

with tab_improve:
    st.markdown(
        '<div class="card-title" style="font-size:0.95rem;color:#1A1F36;text-transform:none;letter-spacing:0;">'
        '🔧 Personnaliser le comportement de l\'Agent'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("Modifiez le system prompt, les descriptions d'outils, et testez en direct.")

    col_p, col_t = st.columns([1, 1], gap="large")

    # ── System Prompt Editor ─────────────────────────────────────────────────
    with col_p:
        st.markdown('<div class="card-title">📋 System Prompt</div>', unsafe_allow_html=True)

        new_prompt = st.text_area(
            "system_prompt",
            value=st.session_state.system_prompt,
            height=340,
            label_visibility="collapsed",
        )

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("💾  Appliquer", use_container_width=True):
                st.session_state.system_prompt = new_prompt
                st.success("✅ System prompt mis à jour !")
        with col_b2:
            if st.button("🔄  Réinitialiser", use_container_width=True):
                st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
                st.rerun()

        # Suggestions
        st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💡 Suggestions rapides</div>', unsafe_allow_html=True)

        IMPROVEMENTS = [
            {
                "title": "🌍 Multilingue",
                "desc":  "Répondre en arabe si l'utilisateur écrit en arabe",
                "patch": "\n- Si l'utilisateur écrit en arabe → réponds en arabe\n- Si l'utilisateur écrit en anglais → réponds en anglais",
            },
            {
                "title": "🎯 Intervalle de confiance",
                "desc":  "Mentionner le niveau de certitude",
                "patch": "\n- Mentionne le niveau de certitude : faible (<40%), moyen (40-70%), élevé (>70%)\n- Indique l'intervalle ±5% pour BERT, ±3% pour Fusion",
            },
            {
                "title": "📊 Top-5 émotions",
                "desc":  "Afficher 5 émotions au lieu de 3",
                "patch": "\n- Ta réponse finale doit toujours inclure le TOP-5 des émotions avec leurs probabilités",
            },
            {
                "title": "🔁 Auto-retry",
                "desc":  "Si erreur serveur, essayer les outils séparément",
                "patch": "\n- Si analyze_multimodal échoue → essaie analyze_image puis analyze_text séparément\n- En cas d'erreur FastAPI, propose de vérifier le serveur",
            },
        ]

        for imp in IMPROVEMENTS:
            with st.expander(f"{imp['title']} — {imp['desc']}"):
                st.code(imp["patch"], language="text")
                if st.button("➕ Ajouter au prompt", key=f"add_{imp['title']}"):
                    st.session_state.system_prompt += imp["patch"]
                    st.success(f"Ajouté : {imp['title']}")

        # Prompt stats
        st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 Stats du prompt</div>', unsafe_allow_html=True)
        prompt = st.session_state.system_prompt
        cs1, cs2, cs3 = st.columns(3)
        with cs1:
            st.metric("Tokens ~",          len(prompt.split()))
        with cs2:
            st.metric("Règles détectées",  prompt.count("-"))
        with cs3:
            lang = "FR" if "français" in prompt else "EN" if "english" in prompt.lower() else "?"
            st.metric("Langue",            lang)

    # ── Tool Editor ──────────────────────────────────────────────────────────
    with col_t:
        st.markdown('<div class="card-title">🛠️ Outils de l\'Agent</div>', unsafe_allow_html=True)
        st.caption("La description de chaque outil influence quand l'agent décide de l'appeler.")

        for tool in TOOLS:
            fn = tool["function"]
            with st.expander(f"🔧 `{fn['name']}`"):
                st.markdown("**Description actuelle :**")
                st.info(fn["description"])
                st.markdown(f"**Paramètres requis :** `{', '.join(fn['parameters'].get('required', []))}`")
                new_desc = st.text_area(
                    "Nouvelle description",
                    value=fn["description"],
                    height=80,
                    key=f"tool_desc_{fn['name']}",
                )
                if st.button("Mettre à jour", key=f"update_{fn['name']}"):
                    for t in TOOLS:
                        if t["function"]["name"] == fn["name"]:
                            t["function"]["description"] = new_desc
                    st.success(f"✅ `{fn['name']}` mis à jour !")

        # Add custom tool
        st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">➕ Ajouter un outil personnalisé</div>', unsafe_allow_html=True)
        with st.expander("Nouvel outil (simulation)"):
            new_name   = st.text_input("Nom",        placeholder="analyze_audio")
            new_desc2  = st.text_area("Description", placeholder="Analyse la prosodie vocale…", height=70)
            new_param  = st.text_input("Paramètre",  placeholder="audio_path")
            if st.button("➕ Ajouter l'outil"):
                if new_name and new_desc2:
                    TOOLS.append({
                        "type": "function",
                        "function": {
                            "name": new_name,
                            "description": new_desc2,
                            "parameters": {
                                "type": "object",
                                "properties": {new_param: {"type": "string"}} if new_param else {},
                                "required": [new_param] if new_param else [],
                            },
                        },
                    })
                    st.success(f"✅ `{new_name}` ajouté ! Implémentez le handler dans `dispatch_tool`.")

        # Quick test
        st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🧪 Test rapide</div>', unsafe_allow_html=True)
        test_text = st.text_input("Phrase de test", placeholder="Je suis tellement en colère !")
        if st.button("🔬  Tester (texte seul)", use_container_width=True):
            if test_text:
                with st.spinner("Test en cours…"):
                    resp = run_agent(test_text)
                result = st.session_state.last_result
                if result:
                    emotion = result.get("emotion", "?")
                    meta    = EMOTION_META.get(emotion, {})
                    color   = meta.get("color", "#3B4EE8")
                    st.markdown(
                        f'<div style="background:{meta.get("bg","#EEF0FD")};border:1px solid {color}33;'
                        f'border-radius:10px;padding:0.8rem 1rem;margin:0.5rem 0;">'
                        f'<b style="color:{color};">{meta.get("emoji","?")} {meta.get("label",emotion)}</b> '
                        f'— {result.get("confidence",0)*100:.1f}%'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(resp[:500] + "…" if len(resp) > 500 else resp)
            else:
                st.warning("Entrez une phrase de test.")