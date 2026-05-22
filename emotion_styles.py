"""
emotion_styles.py
─────────────────
Visual theme and chart configuration for EmotionSense AI.
"""

import plotly.graph_objects as go

# ─── Emotion metadata ─────────────────────────────────────────────────────────

EMOTION_META = {
    "angry":    {"emoji": "😠", "color": "#E5534B", "bg": "#FEF2F2", "label": "Colère"},
    "disgust":  {"emoji": "🤢", "color": "#5B9A4A", "bg": "#F0FDF4", "label": "Dégoût"},
    "fear":     {"emoji": "😨", "color": "#D97706", "bg": "#FFFBEB", "label": "Peur"},
    "happy":    {"emoji": "😊", "color": "#059669", "bg": "#ECFDF5", "label": "Joie"},
    "neutral":  {"emoji": "😐", "color": "#64748B", "bg": "#F8FAFC", "label": "Neutre"},
    "sad":      {"emoji": "😢", "color": "#3B82F6", "bg": "#EFF6FF", "label": "Tristesse"},
    "surprise": {"emoji": "😲", "color": "#7C3AED", "bg": "#F5F3FF", "label": "Surprise"},
}


# ─── CSS Injection ────────────────────────────────────────────────────────────

def inject_css():
    import streamlit as st
    st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Geist+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
}

/* ── Page background ── */
.stApp {
    background: #F4F6FB;
}

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ════════════════════════════════════════
   SIDEBAR
   ════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E5E9F2;
}
section[data-testid="stSidebar"] * {
    color: #1A1F36 !important;
}

/* Sidebar brand area */
.sidebar-brand {
    padding: 0.5rem 0 1.5rem;
    border-bottom: 1px solid #E5E9F2;
    margin-bottom: 1.5rem;
}
.sidebar-logo {
    font-size: 1.65rem;
    font-weight: 800;
    color: #3B4EE8 !important;
    letter-spacing: -0.04em;
    line-height: 1;
}
.sidebar-tagline {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #8F95B2 !important;
    margin-top: 0.35rem;
}

/* Sidebar section headers */
.sidebar-section {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #8F95B2 !important;
    margin: 1.25rem 0 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #EEF0F8;
}

/* History items */
.hist-item {
    background: #F8F9FC;
    border: 1px solid #E5E9F2;
    border-radius: 10px;
    padding: 0.65rem 0.9rem;
    margin-bottom: 0.45rem;
    transition: border-color 0.15s, background 0.15s;
}
.hist-item:hover {
    border-color: #3B4EE8;
    background: #F0F2FE;
}
.hist-emotion {
    font-weight: 700;
    font-size: 0.88rem;
    color: #1A1F36 !important;
}
.hist-meta {
    font-size: 0.72rem;
    color: #8F95B2 !important;
    margin-top: 0.1rem;
}

/* ════════════════════════════════════════
   MAIN CONTENT
   ════════════════════════════════════════ */

/* Page header */
.page-header {
    background: linear-gradient(135deg, #3B4EE8 0%, #6366F1 50%, #8B5CF6 100%);
    border-radius: 20px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(255,255,255,0.07);
}
.page-header::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 20%;
    width: 240px; height: 240px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
}
.page-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin: 0;
}
.page-subtitle {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.75);
    margin-top: 0.4rem;
    font-weight: 500;
}
.page-badges {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.page-badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 100px;
    padding: 0.25rem 0.75rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: rgba(255,255,255,0.9);
    letter-spacing: 0.04em;
    backdrop-filter: blur(4px);
}

/* ── Cards ── */
.card {
    background: #FFFFFF;
    border: 1px solid #E5E9F2;
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 4px rgba(26,31,54,0.04);
}
.card-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8F95B2;
    margin-bottom: 1rem;
}

/* ── Emotion result card ── */
.result-card {
    background: #FFFFFF;
    border-radius: 16px;
    border: 1px solid #E5E9F2;
    padding: 1.75rem;
    box-shadow: 0 2px 12px rgba(59,78,232,0.06);
}
.result-emotion-name {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #1A1F36;
    line-height: 1;
}
.result-confidence {
    font-size: 0.85rem;
    font-weight: 600;
    color: #8F95B2;
    margin-top: 0.3rem;
}
.result-emoji {
    font-size: 3.5rem;
    line-height: 1;
}

/* ── Emotion Badge ── */
.emotion-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 1rem;
    border-radius: 100px;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.02em;
}

/* ── Input area ── */
.input-card {
    background: #FFFFFF;
    border: 1px solid #E5E9F2;
    border-radius: 16px;
    padding: 1.6rem;
    box-shadow: 0 1px 4px rgba(26,31,54,0.04);
}
.input-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8F95B2;
    margin-bottom: 0.6rem;
}

/* ── Mode indicator ── */
.mode-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: #EEF0FD;
    border: 1px solid #C7CBFA;
    border-radius: 100px;
    padding: 0.35rem 0.9rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: #3B4EE8;
    margin-bottom: 1rem;
}
.mode-pill.image { background:#FFF8E6; border-color:#FDE68A; color:#D97706; }
.mode-pill.text  { background:#F0FDF4; border-color:#A7F3D0; color:#059669; }
.mode-pill.multi { background:#EEF0FD; border-color:#C7CBFA; color:#3B4EE8; }
.mode-pill.empty { background:#F8F9FC; border-color:#E5E9F2; color:#8F95B2; }

/* ── Agent step trace ── */
.trace-step {
    background: #F8F9FC;
    border: 1px solid #E5E9F2;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0;
    font-family: 'Geist Mono', monospace;
    font-size: 0.82rem;
    color: #3A4060;
}
.trace-step.thinking { border-left: 3px solid #6366F1; }
.trace-step.tool     { border-left: 3px solid #F59E0B; }
.trace-step.final    { border-left: 3px solid #10B981; }

.trace-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
}
.trace-label.thinking { color: #6366F1; }
.trace-label.tool     { color: #D97706; }
.trace-label.final    { color: #059669; }

/* ── Metric cards ── */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E5E9F2;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(26,31,54,0.04);
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #3B4EE8;
    letter-spacing: -0.04em;
    line-height: 1;
}
.kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8F95B2;
    margin-top: 0.35rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3B4EE8 0%, #6366F1 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 11px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.01em !important;
    padding: 0.62rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(59,78,232,0.28) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(59,78,232,0.38) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Text inputs ── */
.stTextArea textarea,
.stTextInput input {
    background: #F8F9FC !important;
    border: 1.5px solid #E5E9F2 !important;
    border-radius: 11px !important;
    color: #1A1F36 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.15s !important;
}
.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: #3B4EE8 !important;
    background: #FFFFFF !important;
    box-shadow: 0 0 0 3px rgba(59,78,232,0.08) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF;
    border: 1px solid #E5E9F2;
    border-radius: 12px;
    padding: 4px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748B;
    border-radius: 9px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.5rem 1.1rem;
    transition: all 0.15s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3B4EE8, #6366F1) !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(59,78,232,0.25);
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #F8F9FC !important;
    border: 1.5px solid #E5E9F2 !important;
    border-radius: 11px !important;
    color: #1A1F36 !important;
}

/* ── Sliders ── */
.stSlider > div { color: #1A1F36; }
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #3B4EE8 !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: #F8F9FC !important;
    border: 1px solid #E5E9F2 !important;
    border-radius: 11px !important;
    font-weight: 600 !important;
    color: #1A1F36 !important;
}
.streamlit-expanderContent {
    background: #FFFFFF !important;
    border: 1px solid #E5E9F2 !important;
    border-top: none !important;
    border-radius: 0 0 11px 11px !important;
}

/* ── Alerts ── */
.stAlert {
    background: #F8F9FC !important;
    border: 1px solid #E5E9F2 !important;
    border-radius: 11px !important;
    color: #1A1F36 !important;
}
[data-testid="stNotification"] {
    border-radius: 11px !important;
}

/* ── File uploader ── */
.stFileUploader {
    background: #F8F9FC !important;
    border: 1.5px dashed #C7CBFA !important;
    border-radius: 14px !important;
}

/* ── Dividers ── */
hr { border-color: #E5E9F2 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #3B4EE8 !important; }

/* ── Code blocks ── */
code, pre {
    background: #F8F9FC !important;
    color: #3B4EE8 !important;
    border: 1px solid #E5E9F2 !important;
    border-radius: 8px !important;
    font-family: 'Geist Mono', monospace !important;
}

/* ── Progress bars ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #3B4EE8, #6366F1) !important;
    border-radius: 100px !important;
}

/* ── Captions ── */
.stCaption, small, .st-caption {
    color: #8F95B2 !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E5E9F2;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 4px rgba(26,31,54,0.04);
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #3B4EE8 !important;
    font-weight: 800 !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #8F95B2 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F4F6FB; }
::-webkit-scrollbar-thumb { background: #C7CBFA; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3B4EE8; }

/* ── Image preview ── */
.stImage img {
    border-radius: 12px !important;
    border: 1px solid #E5E9F2 !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Plotly chart themes ───────────────────────────────────────────────────────

CHART_FONT = dict(family="Plus Jakarta Sans, sans-serif")
CHART_BG   = "rgba(0,0,0,0)"
GRID_COLOR = "#EEF0F8"
TICK_COLOR = "#8F95B2"


def render_radar_chart(scores: dict) -> go.Figure:
    emotions   = list(scores.keys())
    values_pct = [v * 100 for v in scores.values()]
    dominant   = max(scores, key=scores.get)
    dom_color  = EMOTION_META[dominant]["color"]

    # Convert hex to rgba
    r, g, b = int(dom_color[1:3], 16), int(dom_color[3:5], 16), int(dom_color[5:7], 16)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_pct + [values_pct[0]],
        theta=emotions + [emotions[0]],
        fill="toself",
        fillcolor=f"rgba({r},{g},{b},0.12)",
        line=dict(color=dom_color, width=2.5),
        name="Scores",
        hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        polar=dict(
            bgcolor=CHART_BG,
            angularaxis=dict(
                tickfont=dict(size=11, color=TICK_COLOR, family="Plus Jakarta Sans"),
                linecolor=GRID_COLOR,
                gridcolor=GRID_COLOR,
            ),
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=9, color=TICK_COLOR),
                gridcolor=GRID_COLOR,
                linecolor=GRID_COLOR,
            ),
        ),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        margin=dict(t=20, b=20, l=30, r=30),
        height=300,
        showlegend=False,
        font=CHART_FONT,
    )
    return fig


def render_bar_chart(scores: dict) -> go.Figure:
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    emotions  = [e for e, _ in sorted_items]
    values    = [v * 100 for _, v in sorted_items]
    colors    = [EMOTION_META[e]["color"] for e in emotions]
    labels    = [f"{EMOTION_META[e]['emoji']}  {EMOTION_META[e]['label']}" for e in emotions]
    opacities = [1.0 if i == 0 else 0.35 for i in range(len(emotions))]

    fig = go.Figure(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker=dict(
            color=colors,
            opacity=opacities,
            line=dict(width=0),
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=11, color=TICK_COLOR, family="Plus Jakarta Sans"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        xaxis=dict(
            range=[0, 118],
            showgrid=True,
            gridcolor=GRID_COLOR,
            tickfont=dict(color=TICK_COLOR, size=10),
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(color="#1A1F36", size=11, family="Plus Jakarta Sans"),
            categoryorder="total ascending",
        ),
        margin=dict(t=10, b=10, l=10, r=65),
        height=260,
        bargap=0.32,
        font=CHART_FONT,
    )
    return fig