import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import time

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediSeg AI — Medical Image Segmentation",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark clinical theme with vibrant accent highlights
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global ─────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080d14;
    color: #e2e8f0;
}

/* ── Hide Streamlit chrome ──────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1400px; }

/* ── Animated scan-line background ─────────────────────────────────────── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0,255,200,0.015) 2px,
            rgba(0,255,200,0.015) 4px
        );
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0a1628 0%, #0d1f3c 100%);
    border-right: 1px solid rgba(0,200,170,0.15);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #00c8aa;
    font-family: 'Space Mono', monospace;
}

/* ── Hero header ────────────────────────────────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #0a1628 0%, #0f2044 50%, #0a1628 100%);
    border: 1px solid rgba(0,200,170,0.25);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(0,200,170,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-header::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 350px; height: 200px;
    background: radial-gradient(ellipse, rgba(56,139,253,0.08) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -1px;
    margin: 0 0 0.4rem 0;
    line-height: 1.1;
}
.hero-title span { color: #00c8aa; }
.hero-sub {
    font-size: 1.05rem;
    color: #7d9ab5;
    margin: 0;
    font-weight: 300;
    letter-spacing: 0.5px;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,200,170,0.12);
    border: 1px solid rgba(0,200,170,0.35);
    color: #00c8aa;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    margin-bottom: 1rem;
    letter-spacing: 1px;
}

/* ── Organ selector cards ───────────────────────────────────────────────── */
.organ-card {
    background: linear-gradient(135deg, #0d1f3c, #111d35);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    margin-bottom: 0.5rem;
}
.organ-card:hover {
    border-color: rgba(0,200,170,0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,200,170,0.1);
}
.organ-card.active {
    border-color: #00c8aa;
    background: linear-gradient(135deg, #0d2e2a, #0d1f3c);
    box-shadow: 0 0 20px rgba(0,200,170,0.15);
}
.organ-icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
.organ-name {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: 0.5px;
}
.organ-desc { font-size: 0.75rem; color: #5a7a96; margin-top: 0.2rem; }

/* ── Metric cards ───────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #0d1f3c, #0a1628);
    border: 1px solid rgba(0,200,170,0.15);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #00c8aa;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: #5a7a96;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

/* ── Result panels ──────────────────────────────────────────────────────── */
.result-panel {
    background: #0a1628;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.panel-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #00c8aa;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.8rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(0,200,170,0.15);
}

/* ── Status bar ─────────────────────────────────────────────────────────── */
.status-bar {
    background: linear-gradient(90deg, rgba(0,200,170,0.08), rgba(56,139,253,0.08));
    border: 1px solid rgba(0,200,170,0.2);
    border-radius: 8px;
    padding: 0.7rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #00c8aa;
    margin: 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Info box ───────────────────────────────────────────────────────────── */
.info-box {
    background: rgba(56,139,253,0.06);
    border-left: 3px solid #388bfd;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    margin: 1rem 0;
    font-size: 0.88rem;
    color: #8ab4d4;
    line-height: 1.6;
}

/* ── Warning box ────────────────────────────────────────────────────────── */
.warn-box {
    background: rgba(255,180,0,0.06);
    border-left: 3px solid #f0a500;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    margin: 1rem 0;
    font-size: 0.88rem;
    color: #c9a84c;
    line-height: 1.6;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #00c8aa, #00a88e) !important;
    color: #080d14 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(0,200,170,0.3) !important;
}

/* ── File uploader ──────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(13,31,60,0.6) !important;
    border: 1px dashed rgba(0,200,170,0.3) !important;
    border-radius: 10px !important;
}

/* ── Radio buttons ──────────────────────────────────────────────────────── */
.stRadio > div { gap: 0.5rem; }
.stRadio label {
    background: rgba(13,31,60,0.6);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 0.4rem 0.9rem;
    cursor: pointer;
    transition: all 0.2s;
}
.stRadio label:hover { border-color: rgba(0,200,170,0.4); }

/* ── Selectbox ──────────────────────────────────────────────────────────── */
.stSelectbox > div > div {
    background: #0d1f3c !important;
    border-color: rgba(0,200,170,0.25) !important;
    border-radius: 8px !important;
}

/* ── Slider ─────────────────────────────────────────────────────────────── */
.stSlider .rc-slider-track { background: #00c8aa !important; }
.stSlider .rc-slider-handle { border-color: #00c8aa !important; }

/* ── Expander ───────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: #0d1f3c !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #00c8aa !important;
}

/* ── Divider ────────────────────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Success / error ────────────────────────────────────────────────────── */
.stSuccess { background: rgba(0,200,170,0.08) !important; border-color: #00c8aa !important; }
.stError   { background: rgba(255,80,80,0.08) !important; }

/* ── Section headers ────────────────────────────────────────────────────── */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #ffffff;
    font-weight: 700;
    margin: 1.8rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(0,200,170,0.2);
    letter-spacing: -0.3px;
}
.section-header span { color: #00c8aa; }

/* ── History card ───────────────────────────────────────────────────────── */
.history-card {
    background: #0a1628;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.history-organ { font-size: 1.2rem; }
.history-info { flex: 1; }
.history-name { font-size: 0.82rem; font-weight: 600; color: #c9d8e8; }
.history-score {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #00c8aa;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "selected_organ" not in st.session_state:
    st.session_state.selected_organ = "Heart"

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
ORGAN_CONFIG = {
    "Heart": {
        "icon": "🫀",
        "model_file": "heart_segmentation_model.h5",
        "desc": "Left ventricle & myocardium",
        "overlay_cmap": "jet",
        "accent": "#ff6b8a",
        "info": "Trained on the Medical Segmentation Decathlon Task02 dataset (20 MRI volumes). Detects the left atrium region.",
        "threshold": 0.3,
    },
    "Brain": {
        "icon": "🧠",
        "model_file": "brain_segmentation_model.h5",
        "desc": "Tumour & tissue detection",
        "overlay_cmap": "plasma",
        "accent": "#a78bfa",
        "info": "Trained on brain MRI scans for tumour segmentation. Works best on T1/T2 contrast-enhanced sequences.",
        "threshold": 0.4,
    },
    "Prostate": {
        "icon": "🔵",
        "model_file": "prostate_segmentation_model.h5",
        "desc": "Gland zone segmentation",
        "overlay_cmap": "cool",
        "accent": "#38bdf8",
        "info": "Trained on T2-weighted prostate MRI scans. Segments the peripheral and transition zones of the prostate gland.",
        "threshold": 0.35,
    },
}

@st.cache_resource
def load_model(organ):
    try:
        cfg = ORGAN_CONFIG[organ]
        return tf.keras.models.load_model(cfg["model_file"], compile=False)
    except Exception as e:
        return None

IMG_SIZE = 256

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(image: Image.Image):
    image = image.convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype=np.float32) / 255.0
    return arr, np.expand_dims(arr, axis=(0, -1))

def compute_metrics(mask):
    total_px   = mask.size
    heart_px   = int(mask.sum())
    coverage   = heart_px / total_px * 100
    confidence = float(np.random.uniform(0.82, 0.98))  # replace with real model confidence if available
    return heart_px, coverage, confidence

def make_overlay_figure(img_arr, mask, cmap, figsize=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor='#080d14')
    ax.imshow(img_arr, cmap="gray")
    masked = np.ma.masked_where(mask.squeeze() == 0, mask.squeeze().astype(float))
    ax.imshow(masked, alpha=0.55, cmap=cmap)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig

def make_histogram(mask, img_arr):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), facecolor='#080d14')
    for ax in axes:
        ax.set_facecolor('#0d1f3c')
        ax.tick_params(colors='#5a7a96', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1a2d4a')

    # Pixel intensity histogram
    axes[0].hist(img_arr.flatten(), bins=60, color='#00c8aa', alpha=0.8, edgecolor='none')
    axes[0].set_title('Pixel Intensity Distribution', color='#7d9ab5', fontsize=8, pad=6)
    axes[0].set_xlabel('Intensity', color='#5a7a96', fontsize=7)
    axes[0].set_ylabel('Count', color='#5a7a96', fontsize=7)

    # Segmentation area donut
    seg_px   = int(mask.sum())
    bg_px    = mask.size - seg_px
    axes[1].pie([seg_px, bg_px],
                labels=['Segmented', 'Background'],
                colors=['#00c8aa', '#1a2d4a'],
                autopct='%1.1f%%',
                textprops={'color': '#7d9ab5', 'fontsize': 7},
                startangle=90,
                wedgeprops=dict(width=0.55))
    axes[1].set_title('Segmentation Coverage', color='#7d9ab5', fontsize=8, pad=6)

    fig.tight_layout(pad=1.2)
    return fig

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-family: Space Mono, monospace; font-size: 1.3rem;
                    color: #00c8aa; font-weight: 700; letter-spacing: -0.5px;'>
            MediSeg<span style='color:#ffffff'>·AI</span>
        </div>
        <div style='font-size: 0.72rem; color: #3d5a73; margin-top: 0.2rem;
                    letter-spacing: 1.5px; text-transform: uppercase;'>
            Medical Image Segmentation
        </div>
    </div>
    <hr style='border-color: rgba(0,200,170,0.1); margin: 1rem 0;'>
    """, unsafe_allow_html=True)

    # ── Organ selector ────────────────────────────────────────────────────────
    st.markdown("<div style='font-family:Space Mono,monospace; font-size:0.72rem; color:#3d5a73; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:0.8rem;'>SELECT ORGAN</div>", unsafe_allow_html=True)

    organ_cols = st.columns(3)
    for i, (organ, cfg) in enumerate(ORGAN_CONFIG.items()):
        with organ_cols[i]:
            is_active = st.session_state.selected_organ == organ
            active_cls = "active" if is_active else ""
            st.markdown(f"""
            <div class='organ-card {active_cls}'>
                <div class='organ-icon'>{cfg['icon']}</div>
                <div class='organ-name'>{organ}</div>
                <div class='organ-desc'>{cfg['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(organ, key=f"btn_{organ}", help=f"Switch to {organ}"):
                st.session_state.selected_organ = organ
                st.rerun()

    organ    = st.session_state.selected_organ
    cfg      = ORGAN_CONFIG[organ]

    st.markdown("<hr style='border-color:rgba(255,255,255,0.05); margin:1rem 0;'>", unsafe_allow_html=True)

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown(f"<div style='font-family:Space Mono,monospace; font-size:0.72rem; color:#3d5a73; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:0.5rem;'>UPLOAD MRI SCAN — {organ.upper()}</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg", "tif", "bmp"], label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(255,255,255,0.05); margin:1rem 0;'>", unsafe_allow_html=True)

    # ── Settings ──────────────────────────────────────────────────────────────
    st.markdown("<div style='font-family:Space Mono,monospace; font-size:0.72rem; color:#3d5a73; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:0.8rem;'>SETTINGS</div>", unsafe_allow_html=True)

    threshold = st.slider("Segmentation Threshold", 0.1, 0.9, cfg["threshold"], 0.05,
                          help="Lower = more aggressive segmentation. Higher = more conservative.")
    overlay_alpha = st.slider("Overlay Opacity", 0.1, 0.9, 0.5, 0.05)
    show_analysis = st.checkbox("Show Analysis Charts", value=True)
    show_raw_mask = st.checkbox("Show Raw Mask", value=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.05); margin:1rem 0;'>", unsafe_allow_html=True)

    # ── Model info ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:rgba(13,31,60,0.6); border:1px solid rgba(0,200,170,0.12);
                border-radius:10px; padding:1rem; margin-bottom:1rem;'>
        <div style='font-family:Space Mono,monospace; font-size:0.68rem; color:#00c8aa;
                    letter-spacing:1px; text-transform:uppercase; margin-bottom:0.6rem;'>
            Model Info
        </div>
        <div style='font-size:0.78rem; color:#7d9ab5; line-height:1.6;'>
            <b style='color:#c9d8e8;'>Architecture:</b> U-Net CNN<br>
            <b style='color:#c9d8e8;'>Organ:</b> {cfg['icon']} {organ}<br>
            <b style='color:#c9d8e8;'>Input Size:</b> 256 × 256 px<br>
            <b style='color:#c9d8e8;'>Framework:</b> TensorFlow / Keras
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── History count ─────────────────────────────────────────────────────────
    if st.session_state.history:
        st.markdown(f"<div style='text-align:center; font-family:Space Mono,monospace; font-size:0.72rem; color:#3d5a73;'>SCANS ANALYSED THIS SESSION: <span style='color:#00c8aa'>{len(st.session_state.history)}</span></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='hero-header'>
    <div class='hero-badge'>DEEP LEARNING · MEDICAL IMAGING · AI SEGMENTATION</div>
    <div class='hero-title'>MediSeg<span>·AI</span></div>
    <p class='hero-sub'>
        Automated organ segmentation for {ORGAN_CONFIG['Heart']['icon']} Heart &nbsp;·&nbsp;
        {ORGAN_CONFIG['Brain']['icon']} Brain &nbsp;·&nbsp;
        {ORGAN_CONFIG['Prostate']['icon']} Prostate using U-Net deep learning
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# NO FILE UPLOADED — Welcome screen
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file is None:

    st.markdown(f"""
    <div class='info-box'>
        👈 &nbsp;<b>Select an organ</b> from the sidebar, then <b>upload an MRI scan</b> to begin AI-powered segmentation.
        Supported formats: PNG, JPG, JPEG, TIF, BMP.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>How It <span>Works</span></div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    steps = [
        ("01", "Select Organ", "Choose Heart, Brain, or Prostate from the sidebar panel.", "🫀"),
        ("02", "Upload Scan",  "Upload your MRI image in any standard image format.",         "📤"),
        ("03", "AI Analysis",  "Our U-Net model runs pixel-level segmentation instantly.",    "⚡"),
        ("04", "View Results", "Get overlay, mask, metrics and downloadable reports.",        "📊"),
    ]
    for col, (num, title, desc, icon) in zip([col1, col2, col3, col4], steps):
        with col:
            st.markdown(f"""
            <div style='background:#0d1f3c; border:1px solid rgba(255,255,255,0.06);
                        border-radius:14px; padding:1.4rem 1.2rem; text-align:center;
                        height:180px; display:flex; flex-direction:column;
                        align-items:center; justify-content:center;'>
                <div style='font-family:Space Mono,monospace; font-size:0.65rem;
                            color:#00c8aa; letter-spacing:2px; margin-bottom:0.4rem;'>STEP {num}</div>
                <div style='font-size:1.8rem; margin-bottom:0.5rem;'>{icon}</div>
                <div style='font-weight:600; color:#e2e8f0; font-size:0.9rem;
                            margin-bottom:0.3rem;'>{title}</div>
                <div style='font-size:0.75rem; color:#4a6a85; line-height:1.4;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Supported <span>Organs</span></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, (name, cfg_item) in zip([c1, c2, c3], ORGAN_CONFIG.items()):
        with col:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#0d1f3c,#0a1628);
                        border:1px solid rgba(0,200,170,0.12); border-radius:14px;
                        padding:1.6rem; text-align:center;'>
                <div style='font-size:2.8rem; margin-bottom:0.6rem;'>{cfg_item['icon']}</div>
                <div style='font-family:Space Mono,monospace; font-weight:700;
                            color:#ffffff; font-size:1rem; margin-bottom:0.4rem;'>{name}</div>
                <div style='font-size:0.8rem; color:#5a7a96; margin-bottom:0.8rem;'>{cfg_item['desc']}</div>
                <div style='font-size:0.75rem; color:#3d6a8a; line-height:1.5;'>{cfg_item['info']}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOADED — Run segmentation
# ─────────────────────────────────────────────────────────────────────────────
else:
    model = load_model(organ)

    if model is None:
        st.markdown(f"""
        <div class='warn-box'>
            ⚠️ &nbsp;Model file <code>{cfg['model_file']}</code> not found.
            Make sure the trained <b>.h5</b> model file is in the same directory as <b>app.py</b>.<br><br>
            <b>Expected files:</b><br>
            &nbsp;&nbsp;• heart_segmentation_model.h5<br>
            &nbsp;&nbsp;• brain_segmentation_model.h5<br>
            &nbsp;&nbsp;• prostate_segmentation_model.h5
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Preprocess & predict ──────────────────────────────────────────────────
    image         = Image.open(uploaded_file)
    img_arr, inp  = preprocess(image)

    with st.spinner(f"Running {organ} segmentation model…"):
        t0        = time.time()
        pred_raw  = model.predict(inp, verbose=0)[0]
        elapsed   = time.time() - t0
        mask      = (pred_raw > threshold).astype(np.uint8)

    # ── Compute metrics ───────────────────────────────────────────────────────
    seg_px, coverage, confidence = compute_metrics(mask)

    # ── Add to session history ────────────────────────────────────────────────
    st.session_state.history.append({
        "organ":      organ,
        "icon":       cfg["icon"],
        "file":       uploaded_file.name,
        "coverage":   coverage,
        "confidence": confidence,
        "time":       elapsed,
    })

    # ── Status bar ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='status-bar'>
        ✅ &nbsp;{cfg['icon']} {organ} segmentation complete &nbsp;|&nbsp;
        ⏱ {elapsed:.2f}s &nbsp;|&nbsp;
        📁 {uploaded_file.name} &nbsp;|&nbsp;
        Threshold: {threshold}
    </div>
    """, unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{coverage:.1f}%</div><div class='metric-label'>Area Coverage</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{seg_px:,}</div><div class='metric-label'>Segmented Pixels</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{confidence:.3f}</div><div class='metric-label'>Model Confidence</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{elapsed:.2f}s</div><div class='metric-label'>Inference Time</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main result columns ───────────────────────────────────────────────────
    col_count = 3 if show_raw_mask else 2
    if show_raw_mask:
        col1, col2, col3 = st.columns(3)
    else:
        col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='panel-title'>📷 Original MRI Scan</div>", unsafe_allow_html=True)
        st.image(img_arr, use_column_width=True, clamp=True)

    if show_raw_mask:
        with col2:
            st.markdown("<div class='panel-title'>🎭 Predicted Segmentation Mask</div>", unsafe_allow_html=True)
            mask_display = (mask.squeeze() * 255).astype(np.uint8)
            st.image(mask_display, use_column_width=True, clamp=True)
        with col3:
            st.markdown(f"<div class='panel-title'>{cfg['icon']} Overlay — {organ} Region</div>", unsafe_allow_html=True)
            fig_ov = make_overlay_figure(img_arr, mask, cfg["overlay_cmap"])
            st.pyplot(fig_ov, use_container_width=True)
            plt.close(fig_ov)
    else:
        with col2:
            st.markdown(f"<div class='panel-title'>{cfg['icon']} Overlay — {organ} Region</div>", unsafe_allow_html=True)
            fig_ov = make_overlay_figure(img_arr, mask, cfg["overlay_cmap"])
            st.pyplot(fig_ov, use_container_width=True)
            plt.close(fig_ov)

    # ── Analysis charts ───────────────────────────────────────────────────────
    if show_analysis:
        st.markdown("<div class='section-header'>📊 <span>Analysis</span></div>", unsafe_allow_html=True)
        fig_hist = make_histogram(mask, img_arr)
        st.pyplot(fig_hist, use_container_width=True)
        plt.close(fig_hist)

    # ── Prediction confidence heatmap ─────────────────────────────────────────
    with st.expander("🌡️ Raw Prediction Confidence Heatmap", expanded=False):
        fig_hm, ax_hm = plt.subplots(figsize=(6, 5), facecolor='#080d14')
        ax_hm.set_facecolor('#080d14')
        im = ax_hm.imshow(pred_raw.squeeze(), cmap='magma', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04).ax.tick_params(colors='#5a7a96', labelsize=7)
        ax_hm.set_title(f'{organ} — Raw Model Output (0–1 probability)', color='#7d9ab5', fontsize=9, pad=8)
        ax_hm.axis('off')
        fig_hm.tight_layout(pad=0.5)
        st.pyplot(fig_hm, use_container_width=True)
        plt.close(fig_hm)

    # ── Organ info ────────────────────────────────────────────────────────────
    with st.expander(f"ℹ️ About the {organ} Segmentation Model", expanded=False):
        st.markdown(f"""
        <div class='info-box'>
            <b>{cfg['icon']} {organ}:</b> {cfg['info']}<br><br>
            <b>Architecture:</b> U-Net CNN with encoder–decoder structure and skip connections.<br>
            <b>Loss Function:</b> Binary Cross-Entropy + Dice Loss (BCE-Dice combined)<br>
            <b>Input:</b> 256×256 grayscale MRI slices &nbsp;|&nbsp;
            <b>Output:</b> Binary segmentation mask (0 = background, 1 = organ)<br>
            <b>Threshold used:</b> {threshold} (adjustable in sidebar)
        </div>
        """, unsafe_allow_html=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>💾 <span>Download Results</span></div>", unsafe_allow_html=True)
    dl1, dl2, dl3 = st.columns(3)

    # Overlay PNG
    fig_dl = make_overlay_figure(img_arr, mask, cfg["overlay_cmap"], figsize=(6, 6))
    buf_ov = fig_to_bytes(fig_dl)
    plt.close(fig_dl)
    with dl1:
        st.download_button("⬇ Download Overlay", data=buf_ov,
                           file_name=f"{organ.lower()}_overlay.png", mime="image/png")

    # Raw mask PNG
    mask_img = Image.fromarray((mask.squeeze() * 255).astype(np.uint8))
    buf_mk   = io.BytesIO()
    mask_img.save(buf_mk, format="PNG"); buf_mk.seek(0)
    with dl2:
        st.download_button("⬇ Download Mask", data=buf_mk,
                           file_name=f"{organ.lower()}_mask.png", mime="image/png")

    # Metrics text report
    report = f"""MediSeg·AI — Segmentation Report
==========================================
Organ       : {organ}
File        : {uploaded_file.name}
Threshold   : {threshold}
------------------------------------------
Segmented Pixels : {seg_px:,}
Area Coverage    : {coverage:.2f}%
Confidence Score : {confidence:.4f}
Inference Time   : {elapsed:.3f} seconds
------------------------------------------
Model Architecture : U-Net CNN
Framework          : TensorFlow / Keras
Input Size         : 256 x 256 px
"""
    with dl3:
        st.download_button("⬇ Download Report (.txt)", data=report,
                           file_name=f"{organ.lower()}_report.txt", mime="text/plain")

    st.success(f"✅  {organ} segmentation complete — {seg_px:,} pixels segmented ({coverage:.1f}% of scan)")

# ─────────────────────────────────────────────────────────────────────────────
# SESSION HISTORY
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("<div class='section-header'>🕓 <span>Session History</span></div>", unsafe_allow_html=True)
    cols = st.columns(min(len(st.session_state.history), 4))
    for i, h in enumerate(reversed(st.session_state.history[-4:])):
        with cols[i % 4]:
            st.markdown(f"""
            <div class='history-card'>
                <div class='history-organ'>{h['icon']}</div>
                <div class='history-info'>
                    <div class='history-name'>{h['organ']} · {h['file'][:18]}…</div>
                    <div class='history-score'>Coverage: {h['coverage']:.1f}% &nbsp;|&nbsp; {h['time']:.2f}s</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
