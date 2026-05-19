from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

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
    page_title="Ticket Triage · AI Classifier",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---- globals ---- */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
}
header[data-testid="stHeader"] {
    background: transparent !important;
}
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.92) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.15);
}
section[data-testid="stSidebar"] * {
    color: #c7d2fe !important;
}

/* ---- hero ---- */
.hero-title {
    font-size: 2.6rem;
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
    margin-top: 4px;
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
.pred-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.18);
}
.pred-card .card-icon { font-size: 2rem; margin-bottom: 6px; }
.pred-card .card-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #818cf8;
    margin-bottom: 8px;
}
.pred-card .card-label {
    font-size: 1.35rem;
    font-weight: 700;
    color: #e0e7ff;
    margin-bottom: 10px;
}

/* ---- confidence bar ---- */
.confidence-track {
    width: 100%;
    height: 8px;
    background: rgba(99, 102, 241, 0.12);
    border-radius: 99px;
    overflow: hidden;
    margin-top: 6px;
}
.confidence-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}
.confidence-text {
    font-size: 0.82rem;
    font-weight: 500;
    color: #a5b4fc;
    margin-top: 4px;
}

/* gradient helpers */
.grad-indigo { background: linear-gradient(90deg, #6366f1, #818cf8); }
.grad-violet { background: linear-gradient(90deg, #7c3aed, #a78bfa); }
.grad-cyan   { background: linear-gradient(90deg, #06b6d4, #67e8f9); }
.grad-rose   { background: linear-gradient(90deg, #f43f5e, #fb7185); }
.grad-amber  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }

/* ---- category chips ---- */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 4px; }
.chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 99px;
    font-size: 0.8rem;
    font-weight: 600;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(129, 140, 248, 0.3);
    color: #c7d2fe;
}

/* ---- tab styling ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(30, 27, 75, 0.4);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 500;
    padding: 8px 24px;
}
.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.2) !important;
    color: #a5b4fc !important;
}

/* ---- inputs ---- */
.stTextInput input, .stTextArea textarea {
    background: rgba(30, 27, 75, 0.5) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 10px !important;
    color: #e0e7ff !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.2) !important;
}
label, .stTextInput label, .stTextArea label {
    color: #c7d2fe !important;
    font-weight: 500 !important;
}

/* ---- buttons ---- */
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 32px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35) !important;
}
button[data-testid="stBaseButton-secondary"] {
    background: rgba(99, 102, 241, 0.12) !important;
    color: #a5b4fc !important;
    border: 1px solid rgba(129, 140, 248, 0.3) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
}

/* ---- dataframe ---- */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ---- model info sidebar ---- */
.model-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    background: rgba(99, 102, 241, 0.18);
    border: 1px solid rgba(129, 140, 248, 0.3);
    color: #a5b4fc;
    margin-bottom: 6px;
}
.sidebar-metric {
    font-size: 1.6rem;
    font-weight: 700;
    color: #818cf8;
}
.sidebar-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #64748b;
    margin-bottom: 2px;
}

/* ---- JSON raw output ---- */
.stJson {
    background: rgba(15, 12, 41, 0.7) !important;
    border-radius: 10px !important;
}

/* ---- file uploader ---- */
[data-testid="stFileUploader"] {
    background: rgba(30, 27, 75, 0.3);
    border: 2px dashed rgba(99, 102, 241, 0.25);
    border-radius: 14px;
    padding: 16px;
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
    return ", ".join(
        f"{item['label']} ({item['confidence']:.2f})" for item in categories
    )


def predictions_to_frame(source: pd.DataFrame, predictions: list[dict]) -> pd.DataFrame:
    rows = []
    for input_row, prediction in zip(source.to_dict("records"), predictions):
        category = prediction["category"]
        priority = prediction["priority"]
        intent = prediction["intent"]
        rows.append(
            {
                **input_row,
                "category": format_categories_text(category),
                "priority": priority["label"],
                "priority_confidence": priority["confidence"],
                "intent": intent["label"],
                "intent_confidence": intent["confidence"],
                "prediction_json": json.dumps(prediction),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">🎯 Ticket Triage</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    "AI-powered classification — predict <b>category</b>, <b>priority</b>, "
    "and <b>intent</b> for support tickets instantly."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load predictor (cached — one instance per backend choice)
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
    except FileNotFoundError as _e:
        st.error(
            f"⚠️ Transformer models not found.\n\n"
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
    st.markdown("### 🧠 Active Model Info")
    st.markdown(
        f'<span class="model-badge">🔷 {active_backend_label}</span>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Choose the right metadata prefix
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

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_single, tab_batch = st.tabs(["🔍  Single Ticket", "📁  CSV Upload"])

# ---- Single ticket --------------------------------------------------------
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
            with st.spinner("Classifying…"):
                prediction = predictor.predict_ticket(title, body)

            cat = prediction["category"]
            pri = prediction["priority"]
            intent = prediction["intent"]

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

            with st.expander("📋 Raw JSON response", expanded=False):
                st.json(prediction)

# ---- CSV upload -----------------------------------------------------------
with tab_batch:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        "Upload a CSV file with **`title`** and **`body`** columns. "
        "Each row will be classified and you can download the results.",
    )
    uploaded = st.file_uploader(
        "Choose CSV file",
        type=["csv"],
        key="csv_uploader",
    )
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
                file_name="ticket_triage_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"❌ {exc}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    f'<div style="text-align:center; color:#64748b; font-size:0.78rem; padding:8px 0 24px;">'
    f"Support Ticket Triage System · v1.0.0 · {active_backend_label}"
    f"</div>",
    unsafe_allow_html=True,
)
