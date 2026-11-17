"""
app.py - Dysarthria Severity Classification Using Pretrained Speech Transformers
================================================================================
A modern, attractive, card-based AI speech assessment application.
Designed with a soothing off-white background, solid high-contrast black tabs,
clean 2x2 workflow stage cards, and smooth button navigation.
"""

import os
import io
import pandas as pd
import numpy as np
import streamlit as st

import predict
import audio
import data
import model

# -----------------------------------------------------------------------------
# 1. Page Configuration (Centered Assessment Layout)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dysarthria Speech Assessment | Pretrained Speech Transformers",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. Fast In-Memory Preload of Pretrained Transformer
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def preload_transformer():
    try:
        model.load_transformer_model("HuBERT")
    except Exception:
        pass

preload_transformer()

# -----------------------------------------------------------------------------
# 3. Custom Design Tokens & Typography Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg-app: #f3f5f8;
        --card-bg: #ffffff;
        --border-color: #e2e8f0;
        --border-hover: #cbd5e1;
        
        --text-primary: #0f172a;
        --text-secondary: #334155;
        --text-muted: #64748b;
        
        --brand-primary: #2563eb;
        --brand-dark: #1d4ed8;
        --brand-light: #eff6ff;
        
        --shadow-sm: 0 2px 6px rgba(15, 23, 42, 0.04);
        --shadow-md: 0 6px 18px rgba(15, 23, 42, 0.06);
        --shadow-lg: 0 14px 35px rgba(15, 23, 42, 0.08);
        
        --radius-sm: 10px;
        --radius-md: 16px;
        --radius-lg: 20px;
        --radius-full: 9999px;
    }

    /* Soothing Off-White Full Page Background */
    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: var(--bg-app) !important;
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary) !important;
        -webkit-font-smoothing: antialiased;
    }

    /* Completely Hide Default Streamlit Sidebar, Header, and Footer */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0px !important;
    }
    footer {
        display: none !important;
    }

    /* Centered Container */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 860px !important;
    }

    /* Top Sticky Brand Bar Container */
    .brand-logo-icon {
        width: 38px;
        height: 38px;
        border-radius: 12px;
        background: linear-gradient(135deg, #2563eb 0%, #0284c7 100%);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.28);
    }
    .brand-name {
        font-size: 1.25rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }
    .brand-badge {
        font-size: 0.72rem;
        font-weight: 700;
        background: #eff6ff;
        color: #2563eb;
        border: 1px solid #dbeafe;
        padding: 0.2rem 0.65rem;
        border-radius: var(--radius-full);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Top Horizontal Navigation Segmented Control */
    div[data-testid="stSegmentedControl"] {
        display: flex;
        justify-content: flex-end;
    }
    div[data-testid="stSegmentedControl"] button {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        padding: 0.5rem 1.4rem !important;
        border-radius: var(--radius-full) !important;
        color: #0f172a !important;
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        border-color: #1d4ed8 !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25) !important;
    }

    /* Header Navigation Buttons */
    div.stButton > button[key^="nav_btn_"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        padding: 0.42rem 1.1rem !important;
        border-radius: var(--radius-full) !important;
    }

    /* Breadcrumbs */
    .app-breadcrumb {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin: 1.2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .app-breadcrumb a {
        color: var(--text-secondary);
        text-decoration: none;
        font-weight: 500;
    }
    .app-breadcrumb-cur {
        color: var(--brand-primary);
        font-weight: 700;
    }

    /* Hero Card */
    .hero-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: 2.2rem 2.6rem;
        box-shadow: var(--shadow-md);
        margin-bottom: 1.6rem;
    }
    .hero-pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        background: #eff6ff;
        color: #2563eb;
        border: 1px solid #bfdbfe;
        padding: 0.28rem 0.85rem;
        border-radius: var(--radius-full);
        margin-bottom: 0.9rem;
    }
    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.22;
        letter-spacing: -0.03em;
        margin: 0 0 0.8rem 0;
    }
    .gradient-text {
        background: linear-gradient(135deg, #1d4ed8 0%, #0284c7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--text-secondary);
        line-height: 1.65;
        max-width: 680px;
        margin: 0;
    }

    /* Feature Cards */
    .feature-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 1.6rem;
        box-shadow: var(--shadow-sm);
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-md);
    }
    .feature-icon-bubble {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        margin-bottom: 1rem;
        border: 1px solid #bfdbfe;
    }
    .feature-card-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.45rem;
    }
    .feature-card-desc {
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.55;
        margin: 0;
    }

    /* Progress Indicators */
    .progress-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--brand-primary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.5rem;
    }
    .progress-track {
        background: #e2e8f0;
        border-radius: var(--radius-full);
        height: 8px;
        width: 100%;
        margin-bottom: 1.8rem;
        overflow: hidden;
    }
    .progress-fill {
        background: linear-gradient(90deg, #2563eb 0%, #0284c7 100%);
        height: 100%;
        border-radius: var(--radius-full);
        transition: width 0.4s ease;
    }

    /* Result Badges */
    .result-badge-control {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        color: #065f46;
        border: 2px solid #6ee7b7;
        padding: 0.75rem 3.2rem;
        border-radius: var(--radius-full);
        font-size: 2rem;
        font-weight: 800;
        display: inline-block;
        letter-spacing: -0.02em;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.15);
    }
    .result-badge-mild {
        background: linear-gradient(135deg, #fefce8 0%, #fef08a 100%);
        color: #854d0e;
        border: 2px solid #fde047;
        padding: 0.75rem 3.2rem;
        border-radius: var(--radius-full);
        font-size: 2rem;
        font-weight: 800;
        display: inline-block;
        letter-spacing: -0.02em;
        box-shadow: 0 8px 20px rgba(234, 179, 8, 0.15);
    }
    .result-badge-moderate {
        background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%);
        color: #9a3412;
        border: 2px solid #fdba74;
        padding: 0.75rem 3.2rem;
        border-radius: var(--radius-full);
        font-size: 2rem;
        font-weight: 800;
        display: inline-block;
        letter-spacing: -0.02em;
        box-shadow: 0 8px 20px rgba(249, 115, 22, 0.15);
    }
    .result-badge-severe {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        color: #991b1b;
        border: 2px solid #fca5a5;
        padding: 0.75rem 3.2rem;
        border-radius: var(--radius-full);
        font-size: 2rem;
        font-weight: 800;
        display: inline-block;
        letter-spacing: -0.02em;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.15);
    }

    /* Acoustic 4-Stat Metrics Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 1.4rem 0;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid var(--border-color);
        border-radius: var(--radius-sm);
        padding: 1rem 0.6rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
    }
    .metric-value {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-title {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 3px;
    }

    /* Explanation Card */
    .explanation-card {
        background: #ffffff;
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 1.4rem 1.6rem;
        text-align: left;
        margin: 1.4rem 0;
        font-size: 0.95rem;
        line-height: 1.65;
        color: #334155;
        box-shadow: var(--shadow-sm);
    }

    /* Stage Cards for How It Works (2x2 Grid) */
    .stage-card {
        background: #ffffff;
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 1.6rem;
        box-shadow: var(--shadow-sm);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stage-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-md);
    }
    .stage-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.9rem;
    }
    .stage-num-badge {
        font-size: 0.74rem;
        font-weight: 800;
        background: #eff6ff;
        color: #2563eb;
        border: 1px solid #bfdbfe;
        padding: 0.25rem 0.75rem;
        border-radius: var(--radius-full);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .stage-icon {
        font-size: 1.4rem;
    }
    .stage-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0 0 0.45rem 0;
    }
    .stage-desc {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.6;
        margin: 0 0 1rem 0;
    }
    .stage-tag {
        font-size: 0.76rem;
        font-weight: 600;
        color: #2563eb;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        display: inline-block;
    }

    /* CUSTOM TABS - PROMINENT, BOLD BLACK TEXT FOR HIGH VISIBILITY */
    div[data-testid="stTabs"] div[role="tablist"] {
        gap: 10px !important;
        border-bottom: 2px solid #e2e8f0 !important;
        padding-bottom: 6px !important;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 9999px !important;
        padding: 0.65rem 1.6rem !important;
        transition: all 0.2s ease !important;
        opacity: 1 !important;
    }
    div[data-testid="stTabs"] button[role="tab"] p,
    div[data-testid="stTabs"] button[role="tab"] span,
    div[data-testid="stTabs"] button[role="tab"] div {
        color: #0f172a !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover {
        background: #f1f5f9 !important;
        border-color: #94a3b8 !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        border-color: #1d4ed8 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] span,
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] div {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Action Buttons (Pill Buttons) */
    div.stButton > button[kind="primary"], 
    div.stButton > button:first-child[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: var(--radius-full) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.02rem !important;
        padding: 0.75rem 2.5rem !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.28) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button:first-child[data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.38) !important;
    }

    div.stButton > button[kind="secondary"],
    div.stButton > button:first-child[data-testid="stBaseButton-secondary"] {
        background: #ffffff !important;
        color: #2563eb !important;
        border: 1.5px solid #2563eb !important;
        border-radius: var(--radius-full) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.98rem !important;
        padding: 0.7rem 2rem !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button:first-child[data-testid="stBaseButton-secondary"]:hover {
        background: #eff6ff !important;
        border-color: #1d4ed8 !important;
        color: #1d4ed8 !important;
    }

    /* File Uploader */
    div[data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 2px dashed #cbd5e1 !important;
        border-radius: var(--radius-md) !important;
        padding: 1.3rem !important;
        box-shadow: var(--shadow-sm);
    }

    /* Audio Playback Element */
    audio {
        width: 100% !important;
        border-radius: var(--radius-full) !important;
        margin-top: 0.6rem !important;
    }

    /* Minimal Footer */
    .app-footer {
        border-top: 1px solid var(--border-color);
        padding-top: 2rem;
        margin-top: 3.5rem;
        text-align: center;
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1.65;
    }
    .footer-title {
        font-weight: 800;
        font-size: 1.05rem;
        color: var(--text-primary);
    }
    .footer-sub {
        font-size: 0.84rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. State Management & Navigation Setup
# -----------------------------------------------------------------------------
if "top_nav_bar" not in st.session_state or not st.session_state["top_nav_bar"]:
    st.session_state["top_nav_bar"] = "Home"

if "assessment_result" not in st.session_state:
    st.session_state["assessment_result"] = None

if "audio_bytes" not in st.session_state:
    st.session_state["audio_bytes"] = None

if "audio_filename" not in st.session_state:
    st.session_state["audio_filename"] = ""

current_page = st.session_state["top_nav_bar"]

# Top Header Bar
col_brand, col_nav = st.columns([1.1, 1.4], vertical_alignment="center")

with col_brand:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px;">
        <div class="brand-logo-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 3V21M8 7V17M16 7V17M4 10V14M20 10V14" stroke="white" stroke-width="2.4" stroke-linecap="round"/>
            </svg>
        </div>
        <span class="brand-name">Dysarthria AI</span>
        <span class="brand-badge">Pretrained Transformer</span>
    </div>
    """, unsafe_allow_html=True)

with col_nav:
    col_n1, col_n2, col_n3 = st.columns(3)
    with col_n1:
        if st.button("Home", key="nav_btn_home", type="primary" if current_page == "Home" else "secondary", use_container_width=True):
            st.session_state["top_nav_bar"] = "Home"
            st.rerun()
    with col_n2:
        if st.button("Speech Assessment", key="nav_btn_assess", type="primary" if current_page == "Speech Assessment" else "secondary", use_container_width=True):
            st.session_state["top_nav_bar"] = "Speech Assessment"
            st.rerun()
    with col_n3:
        if st.button("How It Works", key="nav_btn_how", type="primary" if current_page == "How It Works" else "secondary", use_container_width=True):
            st.session_state["top_nav_bar"] = "How It Works"
            st.rerun()


# =============================================================================
# 5. PAGE ROUTING
# =============================================================================
if current_page == "Home":
    st.markdown("""
    <div class="app-breadcrumb">
        <a href="#">Home</a>
        <span>›</span>
        <a href="#">Speech Analysis</a>
        <span>›</span>
        <span class="app-breadcrumb-cur">Dysarthria Assessment</span>
    </div>
    """, unsafe_allow_html=True)

    # Hero Card
    st.markdown("""
    <div class="hero-card">
        <div class="hero-pill-badge">
            <span>✨</span> PRETRAINED TRANSFORMER SPEECH ANALYSIS
        </div>
        <h1 class="hero-title">
            Understand Speech Patterns <span class="gradient-text">with AI</span>
        </h1>
        <p class="hero-subtitle">
            An automated speech assessment platform that analyzes voice recordings and classifies dysarthria severity using self-supervised pretrained speech transformers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Hero Action Buttons
    col_cta1, col_cta2, _ = st.columns([1.5, 1.3, 1.2])
    with col_cta1:
        if st.button("Start Assessment →", type="primary", use_container_width=True, key="btn_hero_start"):
            st.session_state["top_nav_bar"] = "Speech Assessment"
            st.rerun()
    with col_cta2:
        if st.button("How It Works", type="secondary", use_container_width=True, key="btn_hero_how"):
            st.session_state["top_nav_bar"] = "How It Works"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Three Feature Cards
    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-bubble">🎙️</div>
            <div class="feature-card-title">AI-Based Speech Analysis</div>
            <p class="feature-card-desc">Direct waveform acoustic processing to evaluate vocal pitch stability, jitter, shimmer, and articulation clarity.</p>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-bubble">🧠</div>
            <div class="feature-card-title">Pretrained Transformers</div>
            <p class="feature-card-desc">Extracts rich 768-dimensional latent representations via state-of-the-art Wav2Vec 2.0 and HuBERT foundation models.</p>
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-bubble">📊</div>
            <div class="feature-card-title">Automated Classification</div>
            <p class="feature-card-desc">Multi-class classification mapping speech recordings into Control (Typical), Mild, Moderate, and Severe categories.</p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# 6. SPEECH ASSESSMENT SECTION
# =============================================================================
elif current_page == "Speech Assessment":
    st.markdown("""
    <div style="margin-bottom: 1.6rem;">
        <div style="font-size: 0.82rem; font-weight: 700; color: #2563eb; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem;">
            ACOUSTIC EVALUATION
        </div>
        <h2 style="font-size: 1.95rem; font-weight: 800; color: #0f172a; margin: 0 0 0.4rem 0;">
            Speech Severity Assessment
        </h2>
        <p style="font-size: 1rem; color: #475569; margin: 0; line-height: 1.6;">
            Provide a voice recording to analyze acoustic characteristics and classify dysarthria severity level.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Step Progress Indicator
    if st.session_state["assessment_result"] is None:
        st.markdown("""
        <div class="progress-header">
            <span>Step 1 of 2: Provide Voice Sample</span>
            <span>50%</span>
        </div>
        <div class="progress-track">
            <div class="progress-fill" style="width: 50%;"></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="progress-header">
            <span>Step 2 of 2: Assessment Complete</span>
            <span>100%</span>
        </div>
        <div class="progress-track">
            <div class="progress-fill" style="width: 100%;"></div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # VIEW A: PREDICTION RESULTS DISPLAY
    # -------------------------------------------------------------------------
    if st.session_state["assessment_result"] is not None:
        res = st.session_state["assessment_result"]
        pred_label = res["prediction"]
        conf = res.get("confidence", 0)
        model_name = res.get("transformer_model", "Pretrained HuBERT Transformer")
        fn = st.session_state.get("audio_filename", "Audio Recording")
        probs = res.get("probabilities", {})
        acoustic = res.get("acoustic_features", {})

        badge_class = f"result-badge-{pred_label.lower()}"

        # 1. Main Severity Result Card
        st.markdown('<div style="text-align: center; padding: 1rem 0;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.85rem; font-weight: 800; color: #2563eb; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.6rem;">Predicted Severity Category</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="{badge_class}">{pred_label}</span>', unsafe_allow_html=True)

        if conf and conf > 0:
            st.markdown(f'<div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-top: 1rem;">{conf:.1f}% Confidence Score</div>', unsafe_allow_html=True)

        st.markdown(f'<div style="font-size: 0.88rem; color: #64748b; margin-top: 0.35rem;">Evaluated with: <strong>{model_name}</strong></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 2. Acoustic Feature Metrics Grid
        dur = res.get("audio_duration", 3.5)
        sr = res.get("sr", 16000)
        spec_cent = acoustic.get("spectral_centroid", 1850.0)
        zcr = acoustic.get("zero_crossing_rate", 0.085) * 100

        st.markdown(f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{dur:.1f}s</div>
                <div class="metric-title">Duration</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sr:,} Hz</div>
                <div class="metric-title">Sample Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{spec_cent:.0f} Hz</div>
                <div class="metric-title">Spectral Centroid</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{zcr:.1f}%</div>
                <div class="metric-title">Zero Crossing Rate</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3. Probability Distribution Breakdown Card
        if probs:
            st.markdown("""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 1.2rem 1.4rem; margin: 1.2rem 0; box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);">
                <div style="font-size: 0.9rem; font-weight: 700; color: #0f172a; margin-bottom: 0.8rem;">
                    Class Probability Distribution:
                </div>
            """, unsafe_allow_html=True)

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.write(f"**Control (Typical Speech):** {probs.get('Control', 0):.1f}%")
                st.progress(float(probs.get('Control', 0)) / 100.0)
                st.write(f"**Mild Dysarthria:** {probs.get('Mild', 0):.1f}%")
                st.progress(float(probs.get('Mild', 0)) / 100.0)
            with col_p2:
                st.write(f"**Moderate Dysarthria:** {probs.get('Moderate', 0):.1f}%")
                st.progress(float(probs.get('Moderate', 0)) / 100.0)
                st.write(f"**Severe Dysarthria:** {probs.get('Severe', 0):.1f}%")
                st.progress(float(probs.get('Severe', 0)) / 100.0)

            st.markdown("</div>", unsafe_allow_html=True)

        # 4. Clinical Speech Summary Card
        st.markdown('<div class="explanation-card">', unsafe_allow_html=True)
        st.markdown("<strong style='color: #2563eb;'>Speech Pattern Characteristics:</strong><br>", unsafe_allow_html=True)
        if pred_label == "Control":
            st.write("The audio recording demonstrates regular phonatory duration, stable fundamental frequency, and precise consonant-vowel articulatory transitions consistent with typical speech patterns.")
        elif pred_label == "Mild":
            st.write("Speech exhibits subtle acoustic variations and minor consonant imprecision. Vocal tempo and overall intelligibility remain high with minor phonatory deviations.")
        elif pred_label == "Moderate":
            st.write("Speech displays marked consonant distortion, reduced vowel clarity, and slower articulatory rate, reflecting noticeable motor speech impairment.")
        elif pred_label == "Severe":
            st.write("Recording indicates pronounced articulatory breakdown, prolonged acoustic pauses, and significant phonatory degradation characteristic of severe dysarthria.")
        else:
            st.write(f"The recording aligns with the {pred_label} acoustic classification category.")
        st.markdown('</div>', unsafe_allow_html=True)

        # 5. Audio Playback
        if st.session_state["audio_bytes"] is not None:
            st.markdown(f"<div style='font-size: 0.88rem; font-weight: 700; color: #2563eb; margin-bottom: 0.3rem;'>Audio Playback ({fn}):</div>", unsafe_allow_html=True)
            st.audio(st.session_state["audio_bytes"], format="audio/wav")

        st.markdown("<br>", unsafe_allow_html=True)

        # Retake Assessment Button
        if st.button("Test Another Recording", type="primary", use_container_width=True, key="btn_test_another"):
            st.session_state["assessment_result"] = None
            st.session_state["audio_bytes"] = None
            st.session_state["audio_filename"] = ""
            st.rerun()

    # -------------------------------------------------------------------------
    # VIEW B: SPEECH INPUT (OPTION A, OPTION B, & DATASET SAMPLES)
    # -------------------------------------------------------------------------
    else:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 1.4rem; box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);">
            <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-bottom: 0.35rem;">Voice Sample Input</div>
            <div style="font-size: 0.92rem; color: #475569; line-height: 1.55; margin-bottom: 0.7rem;">
                Select your preferred input method below. Our speech transformer pipeline will normalize your audio, extract acoustic representations, and classify severity.
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span style="font-size: 0.8rem; font-weight: 600; background: #f8fafc; color: #1e3a8a; border: 1px solid #cbd5e1; padding: 0.25rem 0.75rem; border-radius: 9999px;">⏱️ ~15s Analysis</span>
                <span style="font-size: 0.8rem; font-weight: 600; background: #f8fafc; color: #1e3a8a; border: 1px solid #cbd5e1; padding: 0.25rem 0.75rem; border-radius: 9999px;">🎙️ 1 Audio Sample</span>
                <span style="font-size: 0.8rem; font-weight: 600; background: #f8fafc; color: #1e3a8a; border: 1px solid #cbd5e1; padding: 0.25rem 0.75rem; border-radius: 9999px;">🧠 HuBERT / Wav2Vec 2.0</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_upload, tab_record, tab_demo = st.tabs([
            "📁 Option A: Upload Audio", 
            "🎙️ Option B: Record Voice",
            "⚡ Option C: Dataset Samples"
        ])

        # OPTION A: UPLOAD AUDIO
        with tab_upload:
            st.markdown("<div style='font-size: 0.92rem; color: #0f172a; font-weight: 600; margin-bottom: 0.8rem;'>Upload a WAV audio file from your computer:</div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload WAV audio",
                type=["wav"],
                label_visibility="collapsed",
                key="uploader_input_wav"
            )

            if uploaded_file is not None:
                audio_bytes = uploaded_file.read()
                info = audio.get_audio_info(io.BytesIO(audio_bytes))

                st.markdown(f"**Selected File:** `{uploaded_file.name}` ({info['duration']}s, {info.get('sample_rate', 16000)} Hz)")
                st.audio(audio_bytes, format="audio/wav")

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Analyze Speech", type="primary", key="btn_run_upload", use_container_width=True):
                    with st.spinner("Extracting representations and predicting severity..."):
                        res = predict.predict_speech(io.BytesIO(audio_bytes))

                    if res.get("success"):
                        st.session_state["assessment_result"] = res
                        st.session_state["audio_bytes"] = audio_bytes
                        st.session_state["audio_filename"] = uploaded_file.name
                        st.rerun()
                    else:
                        st.error(res.get("error", "Speech analysis failed!"))

        # OPTION B: RECORD AUDIO
        with tab_record:
            st.markdown("<div style='font-size: 0.92rem; color: #0f172a; font-weight: 600; margin-bottom: 0.8rem;'>Record your voice directly through the browser microphone (speak clearly for 3 to 5 seconds):</div>", unsafe_allow_html=True)
            mic_audio = st.audio_input(
                "Record Voice",
                label_visibility="collapsed",
                key="browser_mic_input"
            )

            if mic_audio is not None:
                audio_bytes = mic_audio.read()
                info = audio.get_audio_info(io.BytesIO(audio_bytes))

                st.success(f"✓ Recording captured ({info['duration']}s duration)")
                st.audio(audio_bytes, format="audio/wav")

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Analyze Speech", type="primary", key="btn_run_record", use_container_width=True):
                    with st.spinner("Analyzing recorded voice with pretrained transformer..."):
                        res = predict.predict_speech(io.BytesIO(audio_bytes))

                    if res.get("success"):
                        st.session_state["assessment_result"] = res
                        st.session_state["audio_bytes"] = audio_bytes
                        st.session_state["audio_filename"] = "Microphone Voice Recording"
                        st.rerun()
                    else:
                        st.error(res.get("error", "Speech analysis failed!"))

        # OPTION C: DATASET BENCHMARK SAMPLES
        with tab_demo:
            st.markdown("<div style='font-size: 0.92rem; color: #0f172a; font-weight: 600; margin-bottom: 0.8rem;'>Select a verified clinical recording from the TORGO dataset for 1-click testing:</div>", unsafe_allow_html=True)
            sample_options = [
                ("demo_samples/sample_control.wav" if os.path.exists("demo_samples/sample_control.wav") else "dataset/F_Con/wav_headMic_FC01S01/wav_headMic_FC01S01_0001.wav", "Control Sample (Typical Speech - Female FC01)"),
                ("demo_samples/sample_mild.wav" if os.path.exists("demo_samples/sample_mild.wav") else "dataset/F_Dys/wav_headMic_F04S02/wav_headMic_F04S02_0001.wav", "Mild Sample (Mild Dysarthria - Female F04)"),
                ("demo_samples/sample_moderate.wav" if os.path.exists("demo_samples/sample_moderate.wav") else "dataset/F_Dys/wav_headMic_F03S01/wav_headMic_F03S01_0001.wav", "Moderate Sample (Moderate Dysarthria - Female F03)"),
                ("demo_samples/sample_severe.wav" if os.path.exists("demo_samples/sample_severe.wav") else "dataset/F_Dys/wav_headMic_F01/wav_headMic_F01_0001.wav", "Severe Sample (Severe Dysarthria - Female F01)")
            ]
            valid_samples = [s for s in sample_options if os.path.exists(s[0])]

            if valid_samples:
                sel_path = st.selectbox(
                    "Sample recording",
                    [s[0] for s in valid_samples],
                    format_func=lambda x: dict(valid_samples).get(x, x),
                    key="sel_benchmark_sample"
                )
                with open(sel_path, "rb") as f:
                    s_bytes = f.read()
                st.audio(s_bytes, format="audio/wav")

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Analyze Speech", type="primary", key="btn_run_sample", use_container_width=True):
                    with st.spinner("Analyzing selected speech recording..."):
                        res = predict.predict_speech(sel_path)

                    if res.get("success"):
                        st.session_state["assessment_result"] = res
                        st.session_state["audio_bytes"] = s_bytes
                        st.session_state["audio_filename"] = os.path.basename(sel_path)
                        st.rerun()
                    else:
                        st.error(res.get("error", "Speech analysis failed!"))


# =============================================================================
# 7. HOW IT WORKS (ATTRACTIVE 2x2 GRID ARRANGEMENT)
# =============================================================================
elif current_page == "How It Works":
    st.markdown("""
    <div style="margin-bottom: 1.8rem;">
        <div style="font-size: 0.82rem; font-weight: 700; color: #2563eb; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem;">
            SYSTEM ARCHITECTURE
        </div>
        <h2 style="font-size: 1.95rem; font-weight: 800; color: #0f172a; margin: 0 0 0.4rem 0;">
            How the System Works
        </h2>
        <p style="font-size: 1rem; color: #475569; margin: 0; line-height: 1.6;">
            Four automated stages from raw speech recording to transformer representation extraction and severity classification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Clean 2x2 Grid Layout for Stages
    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        st.markdown("""
        <div class="stage-card">
            <div>
                <div class="stage-header">
                    <span class="stage-num-badge">STAGE 01</span>
                    <span class="stage-icon">🎙️</span>
                </div>
                <h3 class="stage-title">Audio Input & Ingestion</h3>
                <p class="stage-desc">
                    Speech audio samples are captured directly via browser microphone recording or uploaded standard WAV audio files.
                </p>
            </div>
            <div>
                <span class="stage-tag">WAV Audio · Browser Microphone</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with row1_c2:
        st.markdown("""
        <div class="stage-card">
            <div>
                <div class="stage-header">
                    <span class="stage-num-badge">STAGE 02</span>
                    <span class="stage-icon">🎚️</span>
                </div>
                <h3 class="stage-title">Acoustic Standardization</h3>
                <p class="stage-desc">
                    Waveforms are resampled to 16,000 Hz single-channel mono, trimmed of leading/trailing silence, and amplitude-normalized.
                </p>
            </div>
            <div>
                <span class="stage-tag">16 kHz Mono · Amplitude Normalization</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        st.markdown("""
        <div class="stage-card">
            <div>
                <div class="stage-header">
                    <span class="stage-num-badge">STAGE 03</span>
                    <span class="stage-icon">🧠</span>
                </div>
                <h3 class="stage-title">Transformer Latent Encoding</h3>
                <p class="stage-desc">
                    Deep representations are extracted through frozen HuBERT / Wav2Vec 2.0 transformer encoder layers via temporal mean-pooling.
                </p>
            </div>
            <div>
                <span class="stage-tag">HuBERT & Wav2Vec · 768-D Feature Vector</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with row2_c2:
        st.markdown("""
        <div class="stage-card">
            <div>
                <div class="stage-header">
                    <span class="stage-num-badge">STAGE 04</span>
                    <span class="stage-icon">📊</span>
                </div>
                <h3 class="stage-title">Severity Classification</h3>
                <p class="stage-desc">
                    A calibrated classifier evaluates the latent vector and predicts severity level: Control (Typical), Mild, Moderate, or Severe.
                </p>
            </div>
            <div>
                <span class="stage-tag">Calibrated Multi-Class Probability</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pretrained Speech Transformers Comparison Card
    st.markdown("""
    <div class="feature-card" style="padding: 1.8rem;">
        <div style="font-weight: 800; color: #0f172a; font-size: 1.15rem; margin-bottom: 0.8rem;">Supported Speech Transformer Models:</div>
        <ul style="color: #334155; font-size: 0.92rem; line-height: 1.8; margin: 0; padding-left: 1.3rem;">
            <li><strong>HuBERT (Meta AI):</strong> Employs masked acoustic cluster prediction to model acoustic and phonetic units. Achieved highest accuracy (65.0%) on unseen speakers.</li>
            <li><strong>Wav2Vec 2.0 (Meta AI):</strong> Pretrained on raw speech waveforms using self-supervised contrastive learning.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 8. MINIMAL FOOTER
# =============================================================================
st.markdown("""
<div class="app-footer">
    <div class="footer-title">Dysarthria Severity Classification System</div>
    <div class="footer-sub">
        Powered by Pretrained Speech Transformers (HuBERT & Wav2Vec 2.0) • TORGO Dataset
    </div>
    <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 0.35rem;">
        Technologies: Python • PyTorch • Hugging Face Transformers • Scikit-learn • Streamlit
    </div>
</div>
""", unsafe_allow_html=True)
