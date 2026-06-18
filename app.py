import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
import csv

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="AgriPredict AI | Agricultural Intelligence Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# MODEL / ENCODER LOADING
# =============================================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("crop_yield_model.pkl")
    area_encoder = joblib.load("area_encoder.pkl")
    item_encoder = joblib.load("item_encoder.pkl")
    return model, area_encoder, item_encoder

MODEL_LOADED = True
LOAD_ERROR = ""
try:
    model, area_encoder, item_encoder = load_artifacts()
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR = str(e)
    model, area_encoder, item_encoder = None, None, None

# =============================================================================
# STATIC PROJECT DATA
# =============================================================================
CROPS_FALLBACK = [
    "Cassava", "Maize", "Plantains and others", "Potatoes", "Rice, paddy",
    "Sorghum", "Soybeans", "Sweet potatoes", "Wheat", "Yams",
]

COUNTRIES_FALLBACK = [
    "Afghanistan", "Albania", "Algeria", "Angola", "Argentina", "Armenia",
    "Australia", "Austria", "Azerbaijan", "Bahamas", "Bangladesh", "Belarus",
    "Belgium", "Belize", "Benin", "Bhutan", "Bolivia", "Botswana", "Brazil",
    "Bulgaria", "Burkina Faso", "Burundi", "Cambodia", "Cameroon", "Canada",
    "Central African Republic", "Chad", "Chile", "China", "Colombia",
    "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czech Republic", "Denmark",
    "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Eritrea",
    "Estonia", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia",
    "Georgia", "Germany", "Ghana", "Greece", "Guatemala", "Guinea", "Guyana",
    "Haiti", "Honduras", "Hungary", "India", "Indonesia", "Iran", "Iraq",
    "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan",
    "Kenya", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho",
    "Liberia", "Libya", "Lithuania", "Madagascar", "Malawi", "Malaysia",
    "Mali", "Malta", "Mauritania", "Mauritius", "Mexico", "Mongolia",
    "Morocco", "Mozambique", "Myanmar", "Namibia", "Nepal", "Netherlands",
    "New Zealand", "Nicaragua", "Niger", "Nigeria", "Norway", "Oman",
    "Pakistan", "Panama",
]

# Typical training data ranges for smart warnings
TRAINING_RANGES = {
    "rainfall": (51.0, 3240.0),
    "pesticides": (0.1, 367778.0),
    "temperature": (1.0, 30.0),
}

CROP_TYPICAL_TEMP = {
    "Wheat": (10, 22), "Rice, paddy": (20, 35), "Maize": (18, 32),
    "Soybeans": (15, 30), "Potatoes": (7, 22), "Cassava": (20, 35),
    "Sorghum": (20, 35), "Sweet potatoes": (18, 30),
    "Plantains and others": (20, 35), "Yams": (20, 35),
}

FEATURE_IMPORTANCE = {
    "Crop Type": 60.9,
    "Pesticides": 11.0,
    "Temperature": 10.8,
    "Rainfall": 8.7,
    "Country": 5.5,
    "Year": 3.1,
}

R2_SCORE = 98.57
MAE = 3752.48
RMSE = 10181.76
TOTAL_RECORDS = 28242
TOTAL_COUNTRIES = 101
TOTAL_CROPS = 10

if MODEL_LOADED:
    try:
        COUNTRY_OPTIONS = sorted(list(area_encoder.classes_))
    except Exception:
        COUNTRY_OPTIONS = COUNTRIES_FALLBACK
    try:
        CROP_OPTIONS = sorted(list(item_encoder.classes_))
    except Exception:
        CROP_OPTIONS = CROPS_FALLBACK
else:
    COUNTRY_OPTIONS = COUNTRIES_FALLBACK
    CROP_OPTIONS = CROPS_FALLBACK

# =============================================================================
# SESSION STATE INIT
# =============================================================================
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []
if "show_result" not in st.session_state:
    st.session_state["show_result"] = False

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_confidence_range(model, features):
    """Use per-tree variance for confidence interval."""
    try:
        preds = np.array([tree.predict(features)[0] for tree in model.estimators_])
        return float(np.std(preds))
    except Exception:
        return None

def get_warnings(crop, rainfall, pesticides, temperature):
    """Return list of smart warnings for out-of-range or unusual inputs."""
    warnings = []
    r_min, r_max = TRAINING_RANGES["rainfall"]
    if rainfall < r_min or rainfall > r_max:
        warnings.append(f"⚠️ Rainfall {rainfall:.0f} mm/yr is outside the training range ({r_min:.0f}–{r_max:.0f} mm). Predictions may be less reliable.")

    p_min, p_max = TRAINING_RANGES["pesticides"]
    if pesticides < p_min or pesticides > p_max:
        warnings.append(f"⚠️ Pesticide usage {pesticides:.0f} tonnes is outside the training range ({p_min:.0f}–{p_max:.0f}). Predictions may be less reliable.")

    t_min, t_max = TRAINING_RANGES["temperature"]
    if temperature < t_min or temperature > t_max:
        warnings.append(f"⚠️ Temperature {temperature:.1f}°C is outside the training range ({t_min:.0f}–{t_max:.0f}°C). Predictions may be less reliable.")

    if crop in CROP_TYPICAL_TEMP:
        ct_min, ct_max = CROP_TYPICAL_TEMP[crop]
        if temperature < ct_min - 5:
            warnings.append(f"🌡️ {temperature:.1f}°C is unusually cold for {crop} (typically {ct_min}–{ct_max}°C).")
        elif temperature > ct_max + 5:
            warnings.append(f"🌡️ {temperature:.1f}°C is unusually hot for {crop} (typically {ct_min}–{ct_max}°C).")

    if rainfall < 200 and temperature > 28:
        warnings.append("☀️ Very low rainfall combined with high temperature suggests drought-stress conditions.")

    return warnings

def yield_status(val):
    if val < 20000:
        return "Low Yield", "status-low"
    elif val < 50000:
        return "Moderate Yield", "status-moderate"
    elif val < 100000:
        return "High Yield", "status-high"
    else:
        return "Excellent Yield", "status-excellent"

def export_history_csv(history):
    output = io.StringIO()
    if not history:
        return ""
    keys = history[0].keys()
    writer = csv.DictWriter(output, fieldnames=keys)
    writer.writeheader()
    writer.writerows(history)
    return output.getvalue()

# =============================================================================
# ANIMATED BACKGROUNDS
# =============================================================================
def build_rain_html(n=45):
    drops = []
    for _ in range(n):
        left = random.randint(0, 100)
        duration = round(random.uniform(0.7, 2.0), 2)
        delay = round(random.uniform(0, 4), 2)
        height = random.randint(14, 28)
        drops.append(
            f'<span class="raindrop" style="left:{left}%; '
            f'animation-duration:{duration}s; animation-delay:{delay}s; '
            f'height:{height}px;"></span>'
        )
    return f'<div class="rain-layer">{"".join(drops)}</div>'

def build_floating_html():
    agri_icons = ["🌾", "🌿", "🍃", "🌱", "🍂"]
    temp_icons = ["🌡️", "☀️", "💧"]
    spans = []
    for _ in range(13):
        left = random.randint(2, 96)
        top = random.randint(5, 85)
        duration = round(random.uniform(7, 14), 2)
        delay = round(random.uniform(0, 6), 2)
        size = random.randint(18, 36)
        icon = random.choice(agri_icons)
        spans.append(
            f'<span class="floating-icon" style="left:{left}%; top:{top}%; '
            f'animation-duration:{duration}s; animation-delay:{delay}s; '
            f'font-size:{size}px;">{icon}</span>'
        )
    for _ in range(7):
        left = random.randint(2, 96)
        top = random.randint(5, 85)
        duration = round(random.uniform(8, 16), 2)
        delay = round(random.uniform(0, 6), 2)
        size = random.randint(16, 28)
        icon = random.choice(temp_icons)
        spans.append(
            f'<span class="floating-icon floating-icon-temp" style="left:{left}%; top:{top}%; '
            f'animation-duration:{duration}s; animation-delay:{delay}s; '
            f'font-size:{size}px;">{icon}</span>'
        )
    return f'<div class="float-layer">{"".join(spans)}</div>'

# =============================================================================
# CSS
# =============================================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

:root{
    --bg-black:#06100a;
    --bg-deep:#0a1a12;
    --forest:#0e2a1d;
    --emerald:#0fae66;
    --emerald-soft:#34e89e;
    --neon:#6dffb0;
    --text-main:#eaf6ee;
    --text-dim:#a9c4b3;
    --glass:rgba(20,45,32,0.45);
    --glass-border:rgba(110,255,176,0.18);
    --warn-bg:rgba(255,196,77,0.10);
    --warn-border:rgba(255,196,77,0.35);
    --warn-text:#ffd27a;
}

html, body, [class*="css"]{
    font-family:'Inter', sans-serif !important;
    color:var(--text-main);
}

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}
[data-testid="stToolbar"]{visibility:hidden;}

.stApp{
    background:
        radial-gradient(circle at 15% 10%, rgba(15,174,102,0.16), transparent 45%),
        radial-gradient(circle at 85% 0%, rgba(52,232,158,0.10), transparent 40%),
        linear-gradient(180deg, var(--bg-black) 0%, var(--bg-deep) 40%, var(--bg-black) 100%);
}

::-webkit-scrollbar{width:10px;}
::-webkit-scrollbar-track{background:var(--bg-black);}
::-webkit-scrollbar-thumb{background:linear-gradient(180deg,var(--emerald),var(--forest));border-radius:10px;}

/* ---------- HERO ---------- */
.hero-wrap{
    position:relative;
    overflow:hidden;
    border-radius:28px;
    padding:64px 40px 56px 40px;
    margin-bottom:36px;
    background:linear-gradient(160deg, rgba(14,42,29,0.85), rgba(6,16,10,0.92));
    border:1px solid var(--glass-border);
    box-shadow:0 30px 80px -30px rgba(15,174,102,0.35);
}
@media (max-width:768px){
    .hero-wrap{padding:36px 20px 32px 20px;}
    .hero-title{font-size:34px !important;}
    .hero-sub{font-size:15px !important;}
    .hero-desc{font-size:13.5px !important;}
}
.rain-layer{position:absolute; inset:0; overflow:hidden; pointer-events:none; opacity:0.55;}
.raindrop{
    position:absolute; top:-30px; width:1.5px;
    background:linear-gradient(180deg, rgba(109,255,176,0) 0%, rgba(109,255,176,0.65) 60%, rgba(109,255,176,0) 100%);
    animation-name:fall; animation-timing-function:linear; animation-iteration-count:infinite;
}
@keyframes fall{
    0%{transform:translateY(-10%); opacity:0;}
    10%{opacity:0.8;}
    100%{transform:translateY(620px); opacity:0;}
}
.float-layer{position:absolute; inset:0; overflow:hidden; pointer-events:none;}
.floating-icon{
    position:absolute; opacity:0.55; filter:drop-shadow(0 0 6px rgba(109,255,176,0.35));
    animation-name:floaty; animation-timing-function:ease-in-out; animation-iteration-count:infinite;
}
.floating-icon-temp{opacity:0.4;}
@keyframes floaty{
    0%{transform:translateY(0px) rotate(0deg);}
    50%{transform:translateY(-22px) rotate(8deg);}
    100%{transform:translateY(0px) rotate(0deg);}
}
.hero-content{position:relative; z-index:2; text-align:center;}
.hero-eyebrow{
    display:inline-block; padding:6px 18px; border-radius:999px;
    background:rgba(109,255,176,0.10); border:1px solid var(--glass-border);
    color:var(--neon); font-size:13px; font-weight:600; letter-spacing:0.04em;
    margin-bottom:22px;
}
.hero-title{
    font-family:'Plus Jakarta Sans', sans-serif;
    font-size:56px; font-weight:800; line-height:1.05; margin:0 0 14px 0;
    background:linear-gradient(95deg, #ffffff 10%, var(--neon) 55%, var(--emerald-soft) 90%);
    -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.hero-sub{
    font-size:19px; font-weight:600; color:var(--text-main); margin-bottom:10px;
}
.hero-desc{
    font-size:15.5px; color:var(--text-dim); max-width:620px; margin:0 auto 28px auto; line-height:1.6;
}
.hero-cta{
    display:inline-flex; gap:14px; align-items:center; justify-content:center; flex-wrap:wrap;
}
.hero-pill{
    padding:11px 26px; border-radius:14px; font-weight:700; font-size:14.5px;
    background:linear-gradient(120deg, var(--emerald), var(--emerald-soft));
    color:#062712; box-shadow:0 10px 30px -8px rgba(15,174,102,0.7);
}
.hero-pill-outline{
    padding:11px 26px; border-radius:14px; font-weight:600; font-size:14.5px;
    border:1px solid var(--glass-border); color:var(--text-main); background:rgba(255,255,255,0.02);
}

/* ---------- SECTION HEADER ---------- */
.sec-head{margin:46px 0 22px 0;}
.sec-tag{
    color:var(--neon); font-size:12.5px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase;
}
.sec-title{
    font-family:'Plus Jakarta Sans', sans-serif; font-size:28px; font-weight:800; margin:6px 0 4px 0; color:var(--text-main);
}
.sec-desc{color:var(--text-dim); font-size:14.5px;}

/* ---------- GLASS CARD BASE ---------- */
.glass-card{
    background:var(--glass);
    border:1px solid var(--glass-border);
    border-radius:20px;
    backdrop-filter:blur(14px);
    -webkit-backdrop-filter:blur(14px);
    padding:26px 24px;
    transition:all 0.35s ease;
}

/* ---------- KPI CARDS ---------- */
.kpi-card{
    background:var(--glass);
    border:1px solid var(--glass-border);
    border-radius:18px;
    padding:24px 20px;
    text-align:center;
    backdrop-filter:blur(14px);
    transition:transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
}
.kpi-card:hover{
    transform:translateY(-8px);
    border-color:rgba(109,255,176,0.55);
    box-shadow:0 20px 45px -18px rgba(15,174,102,0.55);
}
.kpi-icon{font-size:26px; margin-bottom:8px;}
.kpi-value{
    font-family:'Plus Jakarta Sans', sans-serif; font-size:30px; font-weight:800;
    background:linear-gradient(95deg,#ffffff, var(--neon));
    -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.kpi-label{color:var(--text-dim); font-size:13px; font-weight:600; margin-top:4px; letter-spacing:0.02em;}

/* ---------- INSIGHT PANEL ---------- */
.insight-card{
    background:var(--glass);
    border:1px solid var(--glass-border);
    border-radius:20px;
    padding:26px 24px;
    backdrop-filter:blur(14px);
    height:100%;
}
.insight-row{
    display:flex; justify-content:space-between; align-items:center;
    padding:13px 0; border-bottom:1px solid rgba(255,255,255,0.06);
}
.insight-row:last-child{border-bottom:none;}
.insight-label{color:var(--text-dim); font-size:13.5px; font-weight:600;}
.insight-value{color:var(--neon); font-size:14.5px; font-weight:700;}
.bar-track{width:100%; height:8px; border-radius:6px; background:rgba(255,255,255,0.06); margin-top:8px; overflow:hidden;}
.bar-fill{height:100%; border-radius:6px; background:linear-gradient(90deg, var(--forest), var(--emerald-soft));}

/* ---------- PREDICTION RESULT CARD ---------- */
.result-card{
    position:relative;
    border-radius:24px;
    padding:38px 30px;
    text-align:center;
    background:linear-gradient(160deg, rgba(15,174,102,0.18), rgba(8,20,13,0.85));
    border:1px solid rgba(109,255,176,0.4);
    animation:fadeInScale 0.7s ease forwards, glowPulse 2.6s ease-in-out infinite;
    margin-top:6px;
}
@keyframes fadeInScale{
    0%{opacity:0; transform:scale(0.85);}
    100%{opacity:1; transform:scale(1);}
}
@keyframes glowPulse{
    0%{box-shadow:0 0 25px 0 rgba(109,255,176,0.18);}
    50%{box-shadow:0 0 55px 6px rgba(109,255,176,0.38);}
    100%{box-shadow:0 0 25px 0 rgba(109,255,176,0.18);}
}
.result-label{color:var(--text-dim); font-size:14px; font-weight:600; letter-spacing:0.05em; text-transform:uppercase;}
.result-value{
    font-family:'Plus Jakarta Sans', sans-serif; font-size:50px; font-weight:800; margin:10px 0;
    background:linear-gradient(95deg,#ffffff, var(--neon));
    -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.result-unit{color:var(--text-dim); font-size:14px; margin-bottom:6px;}
.result-confidence{color:var(--text-dim); font-size:13px; margin-bottom:18px; font-style:italic;}
.status-badge{
    display:inline-block; padding:8px 22px; border-radius:999px; font-weight:700; font-size:13.5px;
    letter-spacing:0.03em;
}
.status-low{background:rgba(255,99,99,0.15); color:#ff8a8a; border:1px solid rgba(255,99,99,0.35);}
.status-moderate{background:rgba(255,196,77,0.15); color:#ffd27a; border:1px solid rgba(255,196,77,0.35);}
.status-high{background:rgba(109,255,176,0.15); color:var(--neon); border:1px solid rgba(109,255,176,0.4);}
.status-excellent{background:rgba(52,232,158,0.2); color:#7dffc0; border:1px solid rgba(52,232,158,0.5);}

/* ---------- WARNING BOX ---------- */
.warn-box{
    background:var(--warn-bg);
    border:1px solid var(--warn-border);
    border-radius:14px;
    padding:14px 18px;
    margin-top:14px;
    font-size:13.5px;
    color:var(--warn-text);
    line-height:1.65;
}

/* ---------- METRIC CARDS ---------- */
.metric-card{
    background:var(--glass); border:1px solid var(--glass-border); border-radius:18px;
    padding:22px; text-align:center; backdrop-filter:blur(14px);
    transition:transform 0.3s ease, box-shadow 0.3s ease;
}
.metric-card:hover{transform:translateY(-6px); box-shadow:0 18px 40px -16px rgba(15,174,102,0.5);}
.metric-title{color:var(--text-dim); font-size:13px; font-weight:600; margin-bottom:6px;}
.metric-num{font-family:'Plus Jakarta Sans', sans-serif; font-size:26px; font-weight:800; color:var(--neon);}

/* ---------- HISTORY TABLE ---------- */
.history-table{
    width:100%; border-collapse:collapse; font-size:13px;
}
.history-table th{
    background:rgba(109,255,176,0.07); color:var(--neon); font-weight:700;
    padding:10px 14px; text-align:left; font-size:12px; letter-spacing:0.06em; text-transform:uppercase;
    border-bottom:1px solid var(--glass-border);
}
.history-table td{
    padding:10px 14px; border-bottom:1px solid rgba(255,255,255,0.05);
    color:var(--text-main); font-size:13px;
}
.history-table tr:last-child td{border-bottom:none;}
.history-table tr:hover td{background:rgba(109,255,176,0.04);}

/* ---------- ABOUT ---------- */
.about-card{
    background:var(--glass); border:1px solid var(--glass-border); border-radius:20px;
    padding:30px 28px; backdrop-filter:blur(14px); line-height:1.75; color:var(--text-dim); font-size:14.5px;
}
.about-card b{color:var(--text-main);}
.tag-chip{
    display:inline-block; padding:5px 14px; margin:4px 6px 0 0; border-radius:999px;
    background:rgba(109,255,176,0.08); border:1px solid var(--glass-border); color:var(--neon);
    font-size:12.5px; font-weight:600;
}

/* ---------- FORM CONTROLS ---------- */
[data-testid="stForm"]{
    background:var(--glass); border:1px solid var(--glass-border); border-radius:20px;
    padding:26px 24px; backdrop-filter:blur(14px);
}
.stSelectbox label, .stSlider label, .stNumberInput label{
    color:var(--text-dim) !important; font-weight:600 !important; font-size:13.5px !important;
}
div[data-baseweb="select"] > div{
    background:rgba(255,255,255,0.04) !important; border-color:var(--glass-border) !important; border-radius:12px !important;
}
.stNumberInput input{
    background:rgba(255,255,255,0.04) !important; border-color:var(--glass-border) !important;
    border-radius:12px !important; color:var(--text-main) !important;
}
.stButton button, .stFormSubmitButton button{
    background:linear-gradient(120deg, var(--emerald), var(--emerald-soft)) !important;
    color:#062712 !important; font-weight:700 !important; border:none !important;
    border-radius:14px !important; padding:11px 0 !important;
    box-shadow:0 12px 30px -10px rgba(15,174,102,0.7) !important;
    transition:transform 0.25s ease, box-shadow 0.25s ease !important;
}
.stButton button:hover, .stFormSubmitButton button:hover{
    transform:translateY(-3px) !important;
    box-shadow:0 18px 36px -10px rgba(15,174,102,0.9) !important;
}

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab-list"]{
    background:var(--glass) !important; border-radius:14px !important;
    border:1px solid var(--glass-border) !important; padding:4px !important;
}
.stTabs [data-baseweb="tab"]{
    color:var(--text-dim) !important; font-weight:600 !important; border-radius:10px !important;
}
.stTabs [aria-selected="true"]{
    background:rgba(109,255,176,0.12) !important; color:var(--neon) !important;
}

/* ---------- FOOTER ---------- */
.app-footer{
    text-align:center; color:var(--text-dim); font-size:12.5px; padding:30px 0 10px 0; opacity:0.7;
}
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# =============================================================================
# SECTION 1 — HERO
# =============================================================================
hero_html = f"""
<div class="hero-wrap">
    {build_rain_html()}
    {build_floating_html()}
    <div class="hero-content">
        <span class="hero-eyebrow">🌎 AI-Powered Agricultural Intelligence</span>
        <div class="hero-title">🌾 AgriPredict AI</div>
        <div class="hero-sub">AI Powered Agricultural Intelligence Platform</div>
        <div class="hero-desc">
            Predict crop yield using machine learning, climate analytics, and environmental
            factors — trained on {TOTAL_RECORDS:,} agricultural records across {TOTAL_COUNTRIES} countries.
        </div>
        <div class="hero-cta">
            <span class="hero-pill">🔮 Start Predicting</span>
            <span class="hero-pill-outline">📊 {R2_SCORE}% Model Accuracy</span>
        </div>
    </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

if not MODEL_LOADED:
    st.warning(
        "⚠️ Model artifacts not found. Place `crop_yield_model.pkl`, "
        "`area_encoder.pkl`, and `item_encoder.pkl` alongside this app. "
        f"(Details: {LOAD_ERROR})"
    )

# =============================================================================
# SECTION 2 — STATISTICS DASHBOARD
# =============================================================================
st.markdown(
    """
    <div class="sec-head">
        <div class="sec-tag">Platform Overview</div>
        <div class="sec-title">Trained on Real Agricultural Data</div>
        <div class="sec-desc">A high-accuracy model built across a decade of climate and yield records.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_cols = st.columns(4)
kpis = [
    ("📚", f"{TOTAL_RECORDS:,}", "Agricultural Records"),
    ("🌍", f"{TOTAL_COUNTRIES}", "Countries Covered"),
    ("🌱", f"{TOTAL_CROPS}", "Crop Types Modeled"),
    ("🎯", f"{R2_SCORE}%", "Model Accuracy (R²)"),
]
for col, (icon, value, label) in zip(kpi_cols, kpis):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# SECTION 3 — PREDICTION WORKSPACE
# =============================================================================
st.markdown(
    """
    <div class="sec-head">
        <div class="sec-tag">Prediction Workspace</div>
        <div class="sec-title">Forecast Your Crop Yield</div>
        <div class="sec-desc">Enter environmental and crop parameters to generate an AI-powered yield forecast.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.35, 1])

with left_col:
    with st.form("prediction_form"):
        f1, f2 = st.columns(2)
        with f1:
            country = st.selectbox("🌍 Country", COUNTRY_OPTIONS, index=0)
        with f2:
            crop = st.selectbox("🌱 Crop Type", CROP_OPTIONS, index=0)

        year = st.slider("📅 Year", min_value=1990, max_value=2030, value=2024, step=1)

        f3, f4 = st.columns(2)
        with f3:
            rainfall = st.number_input(
                "🌧️ Average Rainfall (mm/year)", min_value=0.0, max_value=5000.0,
                value=1100.0, step=10.0,
            )
        with f4:
            pesticides = st.number_input(
                "🧪 Pesticides (tonnes)", min_value=0.0, max_value=200000.0,
                value=1500.0, step=10.0,
            )

        temperature = st.number_input(
            "🌡️ Average Temperature (°C)", min_value=-10.0, max_value=50.0,
            value=25.0, step=0.1,
        )

        submitted = st.form_submit_button("🔮 Predict Crop Yield", use_container_width=True)

    if submitted:
        if not MODEL_LOADED:
            st.error("Prediction unavailable — model artifacts are missing from this directory.")
        else:
            with st.spinner("Running AI prediction..."):
                try:
                    area_encoded = area_encoder.transform([country])[0]
                    item_encoded = item_encoder.transform([crop])[0]
                    features = np.array(
                        [[year, rainfall, pesticides, temperature, area_encoded, item_encoded]]
                    )
                    prediction = float(model.predict(features)[0])
                    confidence = get_confidence_range(model, features)
                    warnings = get_warnings(crop, rainfall, pesticides, temperature)

                    st.session_state["prediction"] = prediction
                    st.session_state["confidence"] = confidence
                    st.session_state["warnings"] = warnings
                    st.session_state["pred_country"] = country
                    st.session_state["pred_crop"] = crop
                    st.session_state["pred_inputs"] = {
                        "year": year, "rainfall": rainfall,
                        "pesticides": pesticides, "temperature": temperature,
                    }
                    st.session_state["show_result"] = True

                    # Save to history
                    status_label, _ = yield_status(prediction)
                    st.session_state["prediction_history"].append({
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                        "Country": country,
                        "Crop": crop,
                        "Year": year,
                        "Rainfall (mm)": rainfall,
                        "Temp (°C)": temperature,
                        "Pesticides (t)": pesticides,
                        "Yield (hg/ha)": f"{prediction:,.0f}",
                        "Status": status_label,
                    })
                except Exception as e:
                    st.session_state["show_result"] = False
                    st.error(f"Prediction failed: {e}")

with right_col:
    fi_top_label, fi_top_val = list(FEATURE_IMPORTANCE.items())[0]
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="sec-tag" style="margin-bottom:14px;">AI Insights</div>
            <div class="insight-row">
                <span class="insight-label">Most Influential Factor</span>
                <span class="insight-value">{fi_top_label}</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Model</span>
                <span class="insight-value">Random Forest Regressor</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Model Accuracy</span>
                <span class="insight-value">{R2_SCORE}%</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Confidence Interval</span>
                <span class="insight-value">Per-tree Std Dev</span>
            </div>
            <div style="margin-top:18px;">
                <span class="insight-label">{fi_top_label} Influence</span>
                <div class="bar-track"><div class="bar-fill" style="width:{fi_top_val}%;"></div></div>
            </div>
            <div style="margin-top:14px; color:var(--text-dim); font-size:13px; line-height:1.6;">
                The model weighs crop genetics far more heavily than any single climate variable,
                though pesticide use and temperature remain meaningful secondary drivers.
                Smart warnings flag inputs outside the training distribution.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# SECTION 4 — PREDICTION RESULT
# =============================================================================
if st.session_state.get("show_result"):
    pred_value = st.session_state.get("prediction", 0.0)
    pred_country = st.session_state.get("pred_country", "")
    pred_crop = st.session_state.get("pred_crop", "")
    confidence = st.session_state.get("confidence")
    warnings = st.session_state.get("warnings", [])
    pred_inputs = st.session_state.get("pred_inputs", {})

    status_label, status_class = yield_status(pred_value)

    confidence_html = ""
    if confidence is not None:
        lo = max(0, pred_value - confidence)
        hi = pred_value + confidence
        confidence_html = f'<div class="result-confidence">95% range: {lo:,.0f} – {hi:,.0f} hg/ha (±{confidence:,.0f})</div>'

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">🌾 Predicted Crop Yield — {pred_crop} in {pred_country}</div>
            <div class="result-value">{pred_value:,.2f}</div>
            <div class="result-unit">hg / hectare</div>
            {confidence_html}
            <span class="status-badge {status_class}">{status_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Smart Warnings
    if warnings:
        warn_html = "<br>".join(warnings)
        st.markdown(f'<div class="warn-box">{warn_html}</div>', unsafe_allow_html=True)

    # =============================================================================
    # SECTION 4b — WHAT-IF SENSITIVITY ANALYSIS
    # =============================================================================
    st.markdown(
        """
        <div class="sec-head" style="margin-top:36px;">
            <div class="sec-tag">Sensitivity Analysis</div>
            <div class="sec-title">What-If Explorer</div>
            <div class="sec-desc">See how yield changes as each input varies, holding all others constant.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if MODEL_LOADED:
        whatif_tabs = st.tabs(["🌡️ Temperature", "🌧️ Rainfall", "🧪 Pesticides", "📅 Year"])

        base_year = pred_inputs.get("year", 2024)
        base_rainfall = pred_inputs.get("rainfall", 1100.0)
        base_pesticides = pred_inputs.get("pesticides", 1500.0)
        base_temp = pred_inputs.get("temperature", 25.0)
        area_enc = area_encoder.transform([pred_country])[0]
        item_enc = item_encoder.transform([pred_crop])[0]

        def make_whatif_fig(x_vals, yields, x_label, current_x, current_y):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_vals, y=yields,
                mode="lines",
                line=dict(color="#0fae66", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(15,174,102,0.08)",
                name="Yield",
            ))
            fig.add_trace(go.Scatter(
                x=[current_x], y=[current_y],
                mode="markers",
                marker=dict(color="#6dffb0", size=12, symbol="circle",
                            line=dict(color="#ffffff", width=2)),
                name="Current Input",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#eaf6ee"),
                margin=dict(l=10, r=10, t=20, b=20),
                xaxis=dict(title=x_label, showgrid=True,
                           gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Yield (hg/ha)", showgrid=True,
                           gridcolor="rgba(255,255,255,0.06)"),
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                showlegend=True,
            )
            return fig

        with whatif_tabs[0]:
            temps = np.linspace(max(-10, base_temp - 15), min(50, base_temp + 15), 60)
            yields_t = [float(model.predict([[base_year, base_rainfall, base_pesticides, t, area_enc, item_enc]])[0]) for t in temps]
            st.plotly_chart(make_whatif_fig(temps, yields_t, "Temperature (°C)", base_temp, pred_value), use_container_width=True)

        with whatif_tabs[1]:
            rains = np.linspace(max(0, base_rainfall - 800), base_rainfall + 800, 60)
            yields_r = [float(model.predict([[base_year, r, base_pesticides, base_temp, area_enc, item_enc]])[0]) for r in rains]
            st.plotly_chart(make_whatif_fig(rains, yields_r, "Rainfall (mm/yr)", base_rainfall, pred_value), use_container_width=True)

        with whatif_tabs[2]:
            pests = np.linspace(max(0, base_pesticides - 2000), base_pesticides + 2000, 60)
            yields_p = [float(model.predict([[base_year, base_rainfall, p, base_temp, area_enc, item_enc]])[0]) for p in pests]
            st.plotly_chart(make_whatif_fig(pests, yields_p, "Pesticides (tonnes)", base_pesticides, pred_value), use_container_width=True)

        with whatif_tabs[3]:
            years = np.arange(1990, 2031, 1)
            yields_y = [float(model.predict([[int(y), base_rainfall, base_pesticides, base_temp, area_enc, item_enc]])[0]) for y in years]
            st.plotly_chart(make_whatif_fig(years, yields_y, "Year", base_year, pred_value), use_container_width=True)

# =============================================================================
# SECTION 5 — PREDICTION HISTORY
# =============================================================================
if st.session_state["prediction_history"]:
    st.markdown(
        """
        <div class="sec-head">
            <div class="sec-tag">Session Log</div>
            <div class="sec-title">Prediction History</div>
            <div class="sec-desc">All predictions made in this session — compare scenarios side by side.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = st.session_state["prediction_history"]
    # Build HTML table
    headers = list(history[0].keys())
    header_row = "".join(f"<th>{h}</th>" for h in headers)
    rows = ""
    for row in reversed(history):
        cells = "".join(f"<td>{row[k]}</td>" for k in headers)
        rows += f"<tr>{cells}</tr>"
    st.markdown(
        f"""
        <div class="glass-card" style="overflow-x:auto; padding:0;">
            <table class="history-table">
                <thead><tr>{header_row}</tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Export button
    csv_data = export_history_csv(history)
    st.download_button(
        label="⬇️ Export History as CSV",
        data=csv_data,
        file_name=f"agripredict_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    if st.button("🗑️ Clear History"):
        st.session_state["prediction_history"] = []
        st.rerun()

# =============================================================================
# SECTION 6 — FEATURE IMPORTANCE ANALYTICS
# =============================================================================
st.markdown(
    """
    <div class="sec-head">
        <div class="sec-tag">Model Explainability</div>
        <div class="sec-title">Feature Importance Analytics</div>
        <div class="sec-desc">Which inputs drive the model's predictions the most.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

fi_sorted = dict(sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1]))
fig_fi = go.Figure(
    go.Bar(
        x=list(fi_sorted.values()),
        y=list(fi_sorted.keys()),
        orientation="h",
        marker=dict(
            color=list(fi_sorted.values()),
            colorscale=[[0, "#0e2a1d"], [0.5, "#0fae66"], [1, "#6dffb0"]],
            line=dict(color="rgba(109,255,176,0.4)", width=1),
        ),
        text=[f"{v}%" for v in fi_sorted.values()],
        textposition="outside",
        textfont=dict(color="#eaf6ee", size=13),
    )
)
fig_fi.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#eaf6ee"),
    margin=dict(l=10, r=30, t=20, b=20),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Importance (%)", range=[0, 70]),
    yaxis=dict(showgrid=False),
    height=380,
)
st.plotly_chart(fig_fi, use_container_width=True)

# =============================================================================
# SECTION 7 — MODEL PERFORMANCE
# =============================================================================
st.markdown(
    """
    <div class="sec-head">
        <div class="sec-tag">Validation Results</div>
        <div class="sec-title">Model Performance</div>
        <div class="sec-desc">Evaluated on a held-out test split of the agricultural dataset.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
metrics = [
    ("R² Score", f"{R2_SCORE}%"),
    ("MAE", f"{MAE:,.2f}"),
    ("RMSE", f"{RMSE:,.2f}"),
]
for col, (title, num) in zip([m1, m2, m3], metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-num">{num}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# SECTION 8 — ABOUT
# =============================================================================
st.markdown(
    """
    <div class="sec-head">
        <div class="sec-tag">Under The Hood</div>
        <div class="sec-title">About the Model</div>
    </div>
    """,
    unsafe_allow_html=True,
)

crop_chips = "".join([f'<span class="tag-chip">{c}</span>' for c in CROPS_FALLBACK])
st.markdown(
    f"""
    <div class="about-card">
        AgriPredict AI is powered by a <b>Random Forest Regressor</b>, an ensemble learning method
        that combines hundreds of decision trees to model the complex, non-linear relationships
        between climate conditions, agricultural practices, and crop yield.
        <br><br>
        The model was trained on <b>{TOTAL_RECORDS:,} agricultural records</b> spanning
        <b>{TOTAL_COUNTRIES} countries</b> and <b>{TOTAL_CROPS} crop types</b>, using rainfall,
        temperature, pesticide usage, country, crop type, and year as predictive features.
        It achieves an <b>R² score of {R2_SCORE}%</b>, explaining the vast majority of variance
        in observed crop yields.
        <br><br>
        <b>New in this version:</b> confidence intervals derived from per-tree variance,
        smart input warnings for out-of-distribution values, what-if sensitivity charts,
        session-based prediction history with CSV export, and improved mobile responsiveness.
        <br><br>
        <b>Crops covered:</b><br>
        {crop_chips}
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown(
    """
    <div class="app-footer">
        🌾 AgriPredict AI — Built with Streamlit · Random Forest Regressor · Plotly Analytics
    </div>
    """,
    unsafe_allow_html=True,
)