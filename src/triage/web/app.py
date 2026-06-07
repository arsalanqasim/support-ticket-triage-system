from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from triage.config import API_BASE_URL, APP_NAME, __version__
from triage.inference.predictor import TriagePredictor, validate_prediction_frame
from triage.models.artifacts import read_metadata

# Try importing the Transformer predictor (needs transformers + trained models)
try:
    from triage.inference.predictor_transformer import TransformerPredictor
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_NAME} · AI Ticket Classifier",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---- globals ---- */
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%); }
header[data-testid="stHeader"] { background: transparent !important; }
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.92) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.15);
}
section[data-testid="stSidebar"] * { color: #c7d2fe !important; }

/* ---- hero ---- */
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(124,58,237,0.2));
    border: 1px solid rgba(129,140,248,0.4);
    border-radius: 99px;
    padding: 4px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #a5b4fc;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-top: 6px;
    margin-bottom: 28px;
}

/* ---- glass card ---- */
.glass-card {
    background: rgba(30, 27, 75, 0.55);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(99, 102, 241, 0.18);
    border-radius: 16px;
    padding: 28px 26px;
    margin-bottom: 16px;
    transition: box-shadow 0.3s ease, border-color 0.3s ease;
}
.glass-card:hover {
    box-shadow: 0 0 30px rgba(99, 102, 241, 0.12);
    border-color: rgba(129, 140, 248, 0.35);
}

/* ---- prediction result cards ---- */
.pred-card {
    background: rgba(30, 27, 75, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 14px;
    padding: 22px 20px;
    text-align: center;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.pred-card:hover { transform: translateY(-3px); box-shadow: 0 8px 32px rgba(99,102,241,0.18); }
.pred-card .card-icon { font-size: 2rem; margin-bottom: 6px; }
.pred-card .card-title {
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1.5px; color: #818cf8; margin-bottom: 8px;
}
.pred-card .card-label { font-size: 1.35rem; font-weight: 700; color: #e0e7ff; margin-bottom: 10px; }

/* ---- confidence bar ---- */
.confidence-track {
    width: 100%; height: 8px;
    background: rgba(99, 102, 241, 0.12);
    border-radius: 99px; overflow: hidden; margin-top: 6px;
}
.confidence-fill { height: 100%; border-radius: 99px; transition: width 0.8s cubic-bezier(.4,0,.2,1); }
.confidence-text { font-size: 0.82rem; font-weight: 500; color: #a5b4fc; margin-top: 4px; }

/* gradient helpers */
.grad-indigo { background: linear-gradient(90deg, #6366f1, #818cf8); }
.grad-violet { background: linear-gradient(90deg, #7c3aed, #a78bfa); }
.grad-cyan   { background: linear-gradient(90deg, #06b6d4, #67e8f9); }
.grad-rose   { background: linear-gradient(90deg, #f43f5e, #fb7185); }
.grad-amber  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.grad-green  { background: linear-gradient(90deg, #10b981, #34d399); }

/* ---- category chips ---- */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 4px; }
.chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px; border-radius: 99px; font-size: 0.8rem; font-weight: 600;
    background: rgba(99,102,241,0.15); border: 1px solid rgba(129,140,248,0.3); color: #c7d2fe;
}

/* ---- tab styling ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: rgba(30,27,75,0.4); border-radius: 12px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; color: #94a3b8; font-weight: 500; padding: 8px 24px;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important; color: #a5b4fc !important;
}

/* ---- inputs ---- */
.stTextInput input, .stTextArea textarea {
    background: rgba(30,27,75,0.5) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important; color: #e0e7ff !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 2px rgba(129,140,248,0.2) !important;
}
label, .stTextInput label, .stTextArea label { color: #c7d2fe !important; font-weight: 500 !important; }

/* ---- buttons ---- */
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 10px 32px !important; font-weight: 600 !important; font-size: 0.95rem !important;
    letter-spacing: 0.3px; transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.35) !important;
}
button[data-testid="stBaseButton-secondary"] {
    background: rgba(99,102,241,0.12) !important; color: #a5b4fc !important;
    border: 1px solid rgba(129,140,248,0.3) !important; border-radius: 10px !important;
    font-weight: 500 !important;
}

/* ---- dataframe ---- */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ---- model info sidebar ---- */
.model-badge {
    display: inline-block; padding: 4px 12px; border-radius: 99px;
    font-size: 0.72rem; font-weight: 600;
    background: rgba(99,102,241,0.18); border: 1px solid rgba(129,140,248,0.3);
    color: #a5b4fc; margin-bottom: 6px;
}
.sidebar-metric { font-size: 1.6rem; font-weight: 700; color: #818cf8; }
.sidebar-label {
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;
    color: #64748b; margin-bottom: 2px;
}

/* ---- analytics stat cards ---- */
.stat-card {
    background: rgba(30,27,75,0.5);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px; padding: 18px 20px; text-align: center;
}
.stat-value { font-size: 2rem; font-weight: 800; color: #a78bfa; }
.stat-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: #64748b; }

/* ---- API code block ---- */
.api-snippet {
    background: rgba(15,12,41,0.8);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px; padding: 16px 20px;
    font-family: 'Courier New', monospace; font-size: 0.82rem;
    color: #c7d2fe; white-space: pre-wrap; overflow-x: auto;
}

/* ---- JSON raw output ---- */
.stJson { background: rgba(15,12,41,0.7) !important; border-radius: 10px !important; }

/* ---- file uploader ---- */
[data-testid="stFileUploader"] {
    background: rgba(30,27,75,0.3);
    border: 2px dashed rgba(99,102,241,0.25);
    border-radius: 14px; padding: 16px;
}

/* ---- divider ---- */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.25), transparent);
    margin: 20px 0;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PRIORITY_CONFIG = {
    "critical": {"icon": "🔴", "gradient": "grad-rose"},
    "high":     {"icon": "🟠", "gradient": "grad-amber"},
    "medium":   {"icon": "🟡", "gradient": "grad-violet"},
    "low":      {"icon": "🟢", "gradient": "grad-cyan"},
}

INTENT_ICONS = {
    "technical_issue": "🛠️",
    "billing_inquiry": "💳",
    "cancellation_request": "❌",
    "product_inquiry": "📦",
    "refund_request": "💰",
}


def confidence_bar(value: float, gradient: str = "grad-indigo") -> str:
    pct = round(value * 100, 1)
    return (
        f'<div class="confidence-track">'
        f'<div class="confidence-fill {gradient}" style="width:{pct}%"></div>'
        f'</div>'
        f'<div class="confidence-text">{pct}% confidence</div>'
    )


def render_category_chips(categories: list[dict]) -> str:
    chips = []
    for item in categories:
        pct = round(item["confidence"] * 100)
        chips.append(f'<span class="chip">🏷️ {item["label"]}  <b>{pct}%</b></span>')
    return f'<div class="chip-row">{"".join(chips)}</div>'


def format_categories_text(categories: list[dict]) -> str:
    return ", ".join(f"{item['label']} ({item['confidence']:.2f})" for item in categories)


def predictions_to_frame(source: pd.DataFrame, predictions: list[dict]) -> pd.DataFrame:
    rows = []
    for input_row, prediction in zip(source.to_dict("records"), predictions):
        category = prediction["category"]
        priority = prediction["priority"]
        intent = prediction["intent"]
        rows.append({
            **input_row,
            "category": format_categories_text(category),
            "priority": priority["label"],
            "priority_confidence": priority["confidence"],
            "intent": intent["label"],
            "intent_confidence": intent["confidence"],
            "prediction_json": json.dumps(prediction),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-badge">🎯 AI-Powered Classifier</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-title">{APP_NAME}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    "Classify support tickets by <b>Category</b>, <b>Priority</b>, and <b>Intent</b> — "
    "instantly, with confidence scores."
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load predictor
# ---------------------------------------------------------------------------
@st.cache_resource
def _load_baseline() -> TriagePredictor:
    return TriagePredictor()


@st.cache_resource
def _load_transformer() -> "TransformerPredictor":
    return TransformerPredictor()


# ---------------------------------------------------------------------------
# Sidebar — backend selector
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 🎯 {APP_NAME}")
    st.markdown(f'<span class="model-badge">v{__version__} · Production</span>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### ⚙️ Backend")
    backend_options = ["🤖 DistilBERT (Transformer)", "📊 TF-IDF + LogReg (Baseline)"]
    if not TRANSFORMER_AVAILABLE:
        st.warning("`transformers` not installed. Using baseline only.")
        _backend_choice = backend_options[1]
    else:
        _backend_choice = st.radio(
            "Inference Engine",
            options=backend_options,
            index=0,
            help="Switch between the fine-tuned Transformer and the classical baseline.",
        )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

use_transformer = _backend_choice == backend_options[0]

if use_transformer:
    try:
        predictor = _load_transformer()
        active_backend_label = "DistilBERT fine-tuned (HuggingFace Transformers)"
    except FileNotFoundError:
        st.error(
            "⚠️ Transformer models not found.\n\n"
            "Train them first:\n"
            "```\npython scripts/train_transformer.py --target all\n```"
        )
        st.stop()
else:
    try:
        predictor = _load_baseline()
        active_backend_label = "TF-IDF + Logistic Regression (Baseline)"
    except FileNotFoundError:
        st.error("⚠️ Baseline model artifacts not found. Train them first:")
        st.code("python scripts/train_all.py", language="powershell")
        st.stop()

metadata = read_metadata()

# ---------------------------------------------------------------------------
# Sidebar — model info panel
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 Active Model")
    st.markdown(
        f'<span class="model-badge">🔷 {active_backend_label}</span>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    prefix = "transformer_" if use_transformer else ""
    for target in ["category", "priority", "intent"]:
        info = metadata.get(f"{prefix}{target}", metadata.get(target, {}))
        if not info:
            continue
        icon = {"category": "🏷️", "priority": "⚡", "intent": "🎯"}.get(target, "📊")
        st.markdown(f"**{icon} {target.title()}**")
        st.markdown(
            f'<span class="model-badge">{info.get("model_type", "Unknown")}</span>',
            unsafe_allow_html=True,
        )
        rows = info.get("rows", "?")
        st.caption(f"Trained on **{rows:,}** rows" if isinstance(rows, int) else f"Rows: {rows}")
        metrics = info.get("metrics", {})
        if metrics:
            cols = st.columns(len(metrics))
            for col, (k, v) in zip(cols, metrics.items()):
                with col:
                    st.markdown(
                        f'<div class="sidebar-label">{k.replace("_", " ")}</div>'
                        f'<div class="sidebar-metric">{v:.2f}</div>',
                        unsafe_allow_html=True,
                    )
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if metadata.get("updated_at"):
        st.caption(f"Last trained: {metadata['updated_at'][:10]}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🌐 REST API")
    st.markdown(
        f'<span class="model-badge">FastAPI · {API_BASE_URL}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"[📖 Swagger UI]({API_BASE_URL}/docs)  ·  [📄 ReDoc]({API_BASE_URL}/redoc)",
    )

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_single, tab_batch, tab_analytics, tab_api = st.tabs([
    "🔍  Single Ticket",
    "📁  CSV Upload",
    "📊  Analytics",
    "🌐  REST API",
])

# ── Single ticket ─────────────────────────────────────────────────────────────
with tab_single:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    title = st.text_input(
        "Issue title",
        placeholder="e.g. Login page returns 500 after deploy",
        key="single_title",
    )
    body = st.text_area(
        "Issue body",
        height=160,
        placeholder="Describe the issue, feature request, or question in detail…",
        key="single_body",
    )
    predict_clicked = st.button("🚀  Predict Ticket", type="primary", key="btn_predict")
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked:
        if not title.strip() and not body.strip():
            st.warning("Please enter a title or body before predicting.")
        else:
            t0 = time.perf_counter()
            with st.spinner("Classifying…"):
                prediction = predictor.predict_ticket(title, body)
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)

            cat = prediction["category"]
            pri = prediction["priority"]
            intent = prediction["intent"]

            # Store in session history
            st.session_state.prediction_history.append({
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "title": title[:60] + ("…" if len(title) > 60 else ""),
                "priority": pri["label"],
                "priority_conf": pri["confidence"],
                "intent": intent["label"],
                "intent_conf": intent["confidence"],
                "top_category": cat[0]["label"] if cat else "—",
                "latency_ms": latency_ms,
            })

            pri_cfg = PRIORITY_CONFIG.get(pri["label"], {"icon": "🔵", "gradient": "grad-indigo"})
            int_icon = INTENT_ICONS.get(intent["label"], "🎯")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    f'<div class="pred-card">'
                    f'<div class="card-icon">🏷️</div>'
                    f'<div class="card-title">Category</div>'
                    f'{render_category_chips(cat)}'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f'<div class="pred-card">'
                    f'<div class="card-icon">{pri_cfg["icon"]}</div>'
                    f'<div class="card-title">Priority</div>'
                    f'<div class="card-label">{pri["label"].title()}</div>'
                    f'{confidence_bar(pri["confidence"], pri_cfg["gradient"])}'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f'<div class="pred-card">'
                    f'<div class="card-icon">{int_icon}</div>'
                    f'<div class="card-title">Intent</div>'
                    f'<div class="card-label">{intent["label"].replace("_", " ").title()}</div>'
                    f'{confidence_bar(intent["confidence"], "grad-violet")}'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.caption(f"⚡ Predicted in {latency_ms} ms · backend: {active_backend_label}")

            with st.expander("📋 Raw JSON response", expanded=False):
                st.json(prediction)

# ── CSV upload ────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        "Upload a CSV file with **`title`** and **`body`** columns. "
        "Each row will be classified and you can download the results.",
    )
    uploaded = st.file_uploader("Choose CSV file", type=["csv"], key="csv_uploader")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            validate_prediction_frame(df)

            with st.spinner(f"Classifying {len(df)} tickets…"):
                predictions = predictor.predict_batch(df)
                result_df = predictions_to_frame(df, predictions)

            st.success(f"✅ Classified **{len(df)}** tickets successfully!")
            st.dataframe(result_df, use_container_width=True, height=420)

            st.download_button(
                "⬇️  Download Predictions CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="triageiq_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"❌ {exc}")

# ── Analytics Dashboard ───────────────────────────────────────────────────────
with tab_analytics:
    st.markdown("### 📊 Session Analytics")
    history = st.session_state.prediction_history

    # Top stats row
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-value">{len(history)}</div>'
            f'<div class="stat-label">Predictions Made</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    if history:
        avg_conf_pri = sum(h["priority_conf"] for h in history) / len(history)
        avg_conf_int = sum(h["intent_conf"] for h in history) / len(history)
        avg_latency = sum(h["latency_ms"] for h in history) / len(history)
        with col_b:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value">{avg_conf_pri:.0%}</div>'
                f'<div class="stat-label">Avg Priority Conf.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_c:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value">{avg_conf_int:.0%}</div>'
                f'<div class="stat-label">Avg Intent Conf.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_d:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value">{avg_latency:.0f}ms</div>'
                f'<div class="stat-label">Avg Latency</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if history:
        hist_df = pd.DataFrame(history)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Priority Distribution")
            priority_counts = hist_df["priority"].value_counts()
            st.bar_chart(priority_counts, color="#818cf8")

        with col_right:
            st.markdown("#### Intent Distribution")
            intent_counts = hist_df["intent"].value_counts()
            intent_counts.index = [i.replace("_", " ").title() for i in intent_counts.index]
            st.bar_chart(intent_counts, color="#a78bfa")

        st.markdown("#### 🕐 Recent Predictions")
        display_df = hist_df[["timestamp", "title", "priority", "intent", "top_category", "latency_ms"]].copy()
        display_df.columns = ["Time", "Title", "Priority", "Intent", "Top Category", "Latency (ms)"]
        st.dataframe(display_df, use_container_width=True, height=300)

        if st.button("🗑️ Clear Session History", key="clear_history"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("💡 Make some predictions in the **Single Ticket** tab to see analytics here.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Model performance panel
    st.markdown("### 🧠 Trained Model Performance")
    perf_rows = []
    for target in ["category", "priority", "intent"]:
        info = metadata.get(target, {})
        if info:
            m = info.get("metrics", {})
            perf_rows.append({
                "Target": target.title(),
                "Model": info.get("model_type", "—"),
                "Training Rows": f"{info.get('rows', '?'):,}" if isinstance(info.get("rows"), int) else "?",
                **{k.replace("_", " ").title(): f"{v:.3f}" for k, v in m.items()},
            })
    if perf_rows:
        st.dataframe(pd.DataFrame(perf_rows), use_container_width=True)
    else:
        st.caption("No metadata found. Train models first.")

# ── REST API Guide ────────────────────────────────────────────────────────────
with tab_api:
    st.markdown("### 🌐 REST API Reference")
    st.markdown(
        f"The **TriageIQ FastAPI** server exposes a full REST API at "
        f"[`{API_BASE_URL}`]({API_BASE_URL}) with interactive Swagger UI at "
        f"[`{API_BASE_URL}/docs`]({API_BASE_URL}/docs)."
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 🚀 Start the API server")
    st.code("python scripts/run_api.py", language="bash")

    st.markdown("#### 🔑 Authentication")
    st.markdown(
        "All `POST` endpoints require an `X-API-Key` header matching your `TRIAGEIQ_API_KEY` "
        "from `.env`."
    )

    st.markdown("#### `POST /predict` — Single ticket")
    st.code(
        f"""curl -X POST {API_BASE_URL}/predict \\
  -H "X-API-Key: your-api-key" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "title": "Login page returns 500 after deploy",
    "body": "Users cannot log in since the 2pm deployment."
  }}'""",
        language="bash",
    )

    st.markdown("#### `POST /predict/batch` — Batch tickets")
    st.code(
        f"""curl -X POST {API_BASE_URL}/predict/batch \\
  -H "X-API-Key: your-api-key" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "tickets": [
      {{"title": "Payment failed", "body": "Card was declined"}},
      {{"title": "Add dark mode", "body": "UI feature request"}}
    ]
  }}'""",
        language="bash",
    )

    st.markdown("#### Python SDK example")
    st.code(
        f"""import httpx

API_URL = "{API_BASE_URL}"
HEADERS = {{"X-API-Key": "your-api-key"}}

response = httpx.post(
    f"{{API_URL}}/predict",
    headers=HEADERS,
    json={{"title": "App crashes on save", "body": "Reproducible on every save action"}},
)
prediction = response.json()
print(prediction["priority"]["label"])   # e.g. "high"
print(prediction["intent"]["label"])     # e.g. "technical_issue"
""",
        language="python",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.link_button("📖 Swagger UI (Interactive)", f"{API_BASE_URL}/docs", use_container_width=True)
    with col2:
        st.link_button("📄 ReDoc (Documentation)", f"{API_BASE_URL}/redoc", use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    f'<div style="text-align:center; color:#64748b; font-size:0.78rem; padding:8px 0 24px;">'
    f"🎯 {APP_NAME} · v{__version__} · {active_backend_label} · "
    f"Built with Streamlit + FastAPI + scikit-learn"
    f"</div>",
    unsafe_allow_html=True,
)
