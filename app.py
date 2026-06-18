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
# MODEL / ENCODER LOADING  ← UNCHANGED
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
# STATIC PROJECT DATA  ← UNCHANGED
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
# HELPER FUNCTIONS  ← UNCHANGED LOGIC
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
        return "Low Yield", "status-low", "#ff6b6b", 20
    elif val < 50000:
        return "Moderate Yield", "status-moderate", "#ffd27a", 50
    elif val < 100000:
        return "High Yield", "status-high", "#6dffb0", 80
    else:
        return "Excellent Yield", "status-excellent", "#34e89e", 98

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
# WEATHER THEME HELPERS
# =============================================================================
def get_weather_theme(rainfall, temperature):
    """Determine weather theme from inputs for dynamic visual effects."""
    if rainfall < 200 and temperature > 28:
        return "drought"
    elif rainfall > 2000:
        return "tropical"
    elif temperature < 5:
        return "cold"
    elif rainfall > 800 and temperature > 18:
        return "lush"
    else:
        return "temperate"

def get_weather_description(theme):
    desc = {
        "drought": ("☀️ Drought Conditions", "Low rainfall + high heat. Stress-tolerant crops recommended."),
        "tropical": ("🌧️ Tropical Climate", "Heavy rainfall supports water-intensive crops like rice & cassava."),
        "cold": ("❄️ Cold Climate", "Low temperatures suit cold-season crops like wheat and potatoes."),
        "lush": ("🌿 Lush Growing Conditions", "Warm and wet — near-ideal for high-biomass crops."),
        "temperate": ("🌤️ Temperate Climate", "Balanced conditions support a wide variety of crops."),
    }
    return desc.get(theme, ("🌍 Variable Climate", "Mixed conditions detected."))

def get_ai_recommendations(crop, rainfall, temperature, pesticides, pred_value):
    """Generate contextual AI insight cards based on inputs."""
    insights = []

    # Yield tier insight
    if pred_value >= 100000:
        insights.append(("🏆", "Excellent Yield Potential", "Predicted yield is in the top tier. Current conditions align well with this crop's optimal growth profile."))
    elif pred_value >= 50000:
        insights.append(("📈", "Strong Yield Forecast", "Conditions support above-average production. Minor climate optimizations could push yield further."))
    elif pred_value >= 20000:
        insights.append(("⚙️", "Moderate Yield Expected", "Yield is within normal range. Review pesticide levels and irrigation strategy for improvement."))
    else:
        insights.append(("⚠️", "Below-Average Yield Risk", "Consider switching to a drought-tolerant variety or adjusting the growing season."))

    # Rainfall insight
    if rainfall < 400:
        insights.append(("💧", "Irrigation Advisory", f"At {rainfall:.0f} mm/yr, supplemental irrigation is likely needed. Consider drip systems for efficiency."))
    elif rainfall > 2500:
        insights.append(("🌊", "Drainage Management", f"High rainfall ({rainfall:.0f} mm/yr) may cause waterlogging. Ensure raised beds and adequate drainage channels."))

    # Temperature insight
    if crop in CROP_TYPICAL_TEMP:
        ct_min, ct_max = CROP_TYPICAL_TEMP[crop]
        if ct_min <= temperature <= ct_max:
            insights.append(("🌡️", "Optimal Temperature Window", f"{temperature:.1f}°C is within the ideal range for {crop} ({ct_min}–{ct_max}°C). Thermal conditions are favorable."))
        elif temperature > ct_max:
            insights.append(("🔥", "Heat Stress Risk", f"At {temperature:.1f}°C, {crop} may experience heat stress. Consider shade netting or shifting harvest calendar."))

    # Pesticide insight
    if pesticides < 100:
        insights.append(("🌿", "Low Pesticide Input", "Minimal chemical inputs detected. Monitor for pest pressure — biological controls may supplement protection."))
    elif pesticides > 50000:
        insights.append(("🧪", "High Chemical Input", "Elevated pesticide usage. Review application efficiency and consider integrated pest management (IPM)."))

    return insights[:4]  # Cap at 4 cards

# =============================================================================
# ANIMATED HERO BACKGROUND BUILDERS
# =============================================================================
def build_rain_html(n=50):
    drops = []
    for _ in range(n):
        left = random.randint(0, 100)
        duration = round(random.uniform(0.6, 2.2), 2)
        delay = round(random.uniform(0, 5), 2)
        height = random.randint(12, 30)
        opacity = round(random.uniform(0.3, 0.8), 2)
        drops.append(
            f'<span class="raindrop" style="left:{left}%;'
            f'animation-duration:{duration}s;animation-delay:{delay}s;'
            f'height:{height}px;opacity:{opacity};"></span>'
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
            f'<span class="floating-icon" style="left:{left}%;top:{top}%;'
            f'animation-duration:{duration}s;animation-delay:{delay}s;'
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
            f'<span class="floating-icon floating-icon-temp" style="left:{left}%;top:{top}%;'
            f'animation-duration:{duration}s;animation-delay:{delay}s;'
            f'font-size:{size}px;">{icon}</span>'
        )
    return f'<div class="float-layer">{"".join(spans)}</div>'

def build_weather_particles(theme, rainfall=0, temperature=25):
    """Render dynamic weather-themed particles based on climate inputs."""
    if theme == "drought":
        # Heat shimmer particles — amber/orange dots
        particles = []
        for _ in range(20):
            left = random.randint(5, 95)
            top = random.randint(10, 90)
            dur = round(random.uniform(3, 8), 2)
            delay = round(random.uniform(0, 4), 2)
            size = random.randint(3, 8)
            particles.append(
                f'<span class="weather-particle drought-particle" style="left:{left}%;top:{top}%;'
                f'width:{size}px;height:{size}px;animation-duration:{dur}s;animation-delay:{delay}s;"></span>'
            )
        return f'<div class="weather-layer">{"".join(particles)}</div>'

    elif theme == "tropical":
        # Heavy rain effect
        drops = []
        for _ in range(70):
            left = random.randint(0, 100)
            duration = round(random.uniform(0.4, 1.2), 2)
            delay = round(random.uniform(0, 3), 2)
            height = random.randint(18, 40)
            drops.append(
                f'<span class="raindrop tropical-drop" style="left:{left}%;'
                f'animation-duration:{duration}s;animation-delay:{delay}s;height:{height}px;"></span>'
            )
        return f'<div class="weather-layer">{"".join(drops)}</div>'

    elif theme == "cold":
        # Snowflake-like dots
        flakes = []
        for _ in range(30):
            left = random.randint(5, 95)
            dur = round(random.uniform(5, 12), 2)
            delay = round(random.uniform(0, 6), 2)
            size = random.randint(4, 10)
            flakes.append(
                f'<span class="weather-particle snow-particle" style="left:{left}%;'
                f'width:{size}px;height:{size}px;animation-duration:{dur}s;animation-delay:{delay}s;"></span>'
            )
        return f'<div class="weather-layer">{"".join(flakes)}</div>'

    elif theme == "lush":
        # Floating green sparkles
        sparks = []
        for _ in range(25):
            left = random.randint(5, 95)
            top = random.randint(10, 90)
            dur = round(random.uniform(4, 10), 2)
            delay = round(random.uniform(0, 5), 2)
            size = random.randint(4, 9)
            sparks.append(
                f'<span class="weather-particle lush-particle" style="left:{left}%;top:{top}%;'
                f'width:{size}px;height:{size}px;animation-duration:{dur}s;animation-delay:{delay}s;"></span>'
            )
        return f'<div class="weather-layer">{"".join(sparks)}</div>'

    return ""

# =============================================================================
# PLOTLY GAUGE CHART
# =============================================================================
def make_gauge_chart(value, max_val=200000):
    """Premium gauge chart for yield display with animated needle feel."""
    pct = min(value / max_val, 1.0)
    if pct < 0.2:
        gauge_color = "#ff6b6b"
        bar_color = "rgba(255,107,107,0.85)"
    elif pct < 0.5:
        gauge_color = "#ffd27a"
        bar_color = "rgba(255,210,122,0.85)"
    elif pct < 0.8:
        gauge_color = "#6dffb0"
        bar_color = "rgba(109,255,176,0.85)"
    else:
        gauge_color = "#34e89e"
        bar_color = "rgba(52,232,158,0.9)"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number={
            "suffix": " hg/ha",
            "font": {"size": 22, "color": "#eaf6ee", "family": "Plus Jakarta Sans"},
            "valueformat": ",.0f",
        },
        delta={
            "reference": 50000,
            "increasing": {"color": "#6dffb0"},
            "decreasing": {"color": "#ff6b6b"},
            "font": {"size": 13},
        },
        gauge={
            "axis": {
                "range": [0, max_val],
                "tickwidth": 1,
                "tickcolor": "rgba(255,255,255,0.2)",
                "tickfont": {"color": "#a9c4b3", "size": 10},
                "nticks": 6,
            },
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20000], "color": "rgba(255,107,107,0.10)"},
                {"range": [20000, 50000], "color": "rgba(255,210,122,0.10)"},
                {"range": [50000, 100000], "color": "rgba(109,255,176,0.08)"},
                {"range": [100000, max_val], "color": "rgba(52,232,158,0.12)"},
            ],
            "threshold": {
                "line": {"color": gauge_color, "width": 3},
                "thickness": 0.82,
                "value": value,
            },
        },
        title={
            "text": "Yield Score",
            "font": {"size": 13, "color": "#a9c4b3", "family": "Inter"},
        },
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "#eaf6ee"},
        margin=dict(l=20, r=20, t=30, b=10),
        height=230,
    )
    return fig

# =============================================================================
# CUSTOM CSS — PREMIUM DARK GLASSMORPHISM
# =============================================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800;900&display=swap');

/* ── DESIGN TOKENS ─────────────────────────────────────────────────────────── */
:root {
    --bg-void:       #050d09;
    --bg-deep:       #080f0b;
    --forest:        #0c2016;
    --forest-mid:    #0e2a1d;
    --emerald:       #0fae66;
    --emerald-soft:  #34e89e;
    --neon:          #6dffb0;
    --neon-dim:      rgba(109,255,176,0.55);
    --text-main:     #eaf6ee;
    --text-dim:      #8eb09e;
    --text-ghost:    rgba(234,246,238,0.35);
    --glass:         rgba(12,32,22,0.55);
    --glass-hover:   rgba(14,38,26,0.72);
    --glass-border:  rgba(109,255,176,0.14);
    --glass-border-h:rgba(109,255,176,0.45);
    --warn-bg:       rgba(255,196,77,0.08);
    --warn-border:   rgba(255,196,77,0.30);
    --warn-text:     #ffd27a;
    --shadow-glow:   0 0 40px -10px rgba(15,174,102,0.40);
    --shadow-card:   0 24px 60px -20px rgba(0,0,0,0.6);
}

/* ── GLOBAL RESETS ──────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-main);
}
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden; }

/* Background */
.stApp {
    background:
        radial-gradient(ellipse 60% 40% at 10% 5%,  rgba(15,174,102,0.14), transparent),
        radial-gradient(ellipse 50% 35% at 90% 0%,  rgba(52,232,158,0.08), transparent),
        radial-gradient(ellipse 80% 50% at 50% 100%, rgba(6,30,18,0.80),    transparent),
        linear-gradient(180deg, var(--bg-void) 0%, var(--bg-deep) 100%);
    min-height: 100vh;
}

/* Scrollbar */
::-webkit-scrollbar       { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, var(--emerald), var(--forest)); border-radius: 8px; }

/* ── HERO ───────────────────────────────────────────────────────────────────── */
.hero-wrap {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    padding: 72px 48px 64px;
    margin-bottom: 40px;
    background: linear-gradient(160deg, rgba(12,32,22,0.90) 0%, rgba(5,13,9,0.95) 100%);
    border: 1px solid var(--glass-border);
    box-shadow: var(--shadow-card), var(--shadow-glow);
}
@media (max-width: 768px) {
    .hero-wrap { padding: 40px 20px 36px; }
    .hero-title { font-size: 36px !important; }
    .hero-sub   { font-size: 15px !important; }
    .hero-desc  { font-size: 13.5px !important; }
}

/* Rain */
.rain-layer { position: absolute; inset: 0; overflow: hidden; pointer-events: none; }
.raindrop {
    position: absolute; top: -30px; width: 1.5px;
    background: linear-gradient(180deg, rgba(109,255,176,0) 0%, rgba(109,255,176,0.60) 60%, rgba(109,255,176,0) 100%);
    animation-name: fall; animation-timing-function: linear; animation-iteration-count: infinite;
}
@keyframes fall {
    0%   { transform: translateY(-5%);   opacity: 0; }
    8%   { opacity: 0.9; }
    100% { transform: translateY(640px); opacity: 0; }
}

/* Floating icons */
.float-layer    { position: absolute; inset: 0; overflow: hidden; pointer-events: none; }
.floating-icon  {
    position: absolute; filter: drop-shadow(0 0 7px rgba(109,255,176,0.30));
    animation-name: floaty; animation-timing-function: ease-in-out; animation-iteration-count: infinite;
}
.floating-icon-temp { opacity: 0.35; }
@keyframes floaty {
    0%   { transform: translateY(0px) rotate(0deg); }
    50%  { transform: translateY(-24px) rotate(9deg); }
    100% { transform: translateY(0px) rotate(0deg); }
}

/* Weather particles */
.weather-layer { position: absolute; inset: 0; overflow: hidden; pointer-events: none; z-index: 1; }
.weather-particle { position: absolute; border-radius: 50%; animation-iteration-count: infinite; }
.drought-particle {
    background: radial-gradient(circle, rgba(255,165,50,0.7), rgba(255,120,20,0));
    animation-name: heatShimmer; animation-timing-function: ease-in-out;
}
@keyframes heatShimmer {
    0%,100% { transform: translateY(0) scale(1);   opacity: 0.5; }
    50%     { transform: translateY(-18px) scale(1.4); opacity: 0; }
}
.snow-particle {
    background: rgba(180,220,255,0.65);
    animation-name: snowFall; animation-timing-function: ease-in;
    top: -20px;
}
@keyframes snowFall {
    0%   { transform: translateY(-10px) rotate(0deg);   opacity: 0; }
    10%  { opacity: 0.9; }
    100% { transform: translateY(640px) rotate(360deg); opacity: 0; }
}
.tropical-drop {
    width: 1.5px !important;
    background: linear-gradient(180deg, rgba(90,160,255,0) 0%, rgba(90,160,255,0.65) 60%, rgba(90,160,255,0) 100%);
    position: absolute; top: -30px;
    animation-name: fall; animation-timing-function: linear; animation-iteration-count: infinite;
}
.lush-particle {
    background: radial-gradient(circle, rgba(109,255,176,0.8), rgba(52,232,158,0));
    animation-name: sparkleFloat; animation-timing-function: ease-in-out;
}
@keyframes sparkleFloat {
    0%,100% { transform: translateY(0) scale(1);   opacity: 0.6; }
    40%     { transform: translateY(-20px) scale(1.6); opacity: 0; }
}

/* Hero text */
.hero-content { position: relative; z-index: 2; text-align: center; }
.hero-eyebrow {
    display: inline-block; padding: 6px 20px; border-radius: 999px;
    background: rgba(109,255,176,0.08); border: 1px solid var(--glass-border);
    color: var(--neon); font-size: 12.5px; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 24px;
}
.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 60px; font-weight: 900; line-height: 1.02; margin: 0 0 16px;
    background: linear-gradient(100deg, #ffffff 0%, #b3ffd8 40%, var(--neon) 70%, var(--emerald-soft) 100%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 30px rgba(109,255,176,0.18));
}
.hero-sub {
    font-size: 18px; font-weight: 600; color: var(--text-main);
    margin-bottom: 12px; letter-spacing: 0.01em;
}
.hero-desc {
    font-size: 15px; color: var(--text-dim);
    max-width: 580px; margin: 0 auto 30px; line-height: 1.65;
}
.hero-cta {
    display: inline-flex; gap: 14px; align-items: center;
    justify-content: center; flex-wrap: wrap;
}
.hero-pill {
    padding: 12px 28px; border-radius: 14px; font-weight: 700; font-size: 14px;
    background: linear-gradient(120deg, var(--emerald), var(--emerald-soft));
    color: #062712; box-shadow: 0 12px 32px -8px rgba(15,174,102,0.65);
    letter-spacing: 0.02em;
}
.hero-pill-outline {
    padding: 12px 28px; border-radius: 14px; font-weight: 600; font-size: 14px;
    border: 1px solid var(--glass-border); color: var(--text-main);
    background: rgba(255,255,255,0.025); letter-spacing: 0.01em;
}

/* ── SECTION HEADERS ────────────────────────────────────────────────────────── */
.sec-head { margin: 52px 0 24px; }
.sec-tag  {
    color: var(--neon); font-size: 12px; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
}
.sec-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 28px; font-weight: 800; margin: 7px 0 5px; color: var(--text-main);
}
.sec-desc { color: var(--text-dim); font-size: 14.5px; line-height: 1.55; }

/* ── KPI CARDS ──────────────────────────────────────────────────────────────── */
.kpi-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 28px 20px;
    text-align: center;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    transition: transform 0.32s ease, box-shadow 0.32s ease, border-color 0.32s ease, background 0.32s ease;
    box-shadow: var(--shadow-card);
}
.kpi-card:hover {
    transform: translateY(-10px) scale(1.01);
    border-color: var(--glass-border-h);
    background: var(--glass-hover);
    box-shadow: 0 28px 60px -20px rgba(15,174,102,0.50), var(--shadow-card);
}
.kpi-icon  { font-size: 28px; margin-bottom: 10px; }
.kpi-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 34px; font-weight: 900;
    background: linear-gradient(100deg, #ffffff, var(--neon));
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
/* Animated counter effect */
.kpi-value.counting { animation: countUp 1.2s ease forwards; }
@keyframes countUp {
    0%   { opacity: 0; transform: translateY(12px); }
    100% { opacity: 1; transform: translateY(0); }
}
.kpi-label {
    color: var(--text-dim); font-size: 12.5px; font-weight: 600;
    margin-top: 5px; letter-spacing: 0.03em;
}

/* ── INSIGHT PANEL ──────────────────────────────────────────────────────────── */
.insight-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 28px 24px;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    height: 100%;
    box-shadow: var(--shadow-card);
}
.insight-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 14px 0; border-bottom: 1px solid rgba(255,255,255,0.055);
}
.insight-row:last-child { border-bottom: none; }
.insight-label { color: var(--text-dim); font-size: 13px; font-weight: 600; }
.insight-value { color: var(--neon); font-size: 13.5px; font-weight: 700; }
.bar-track {
    width: 100%; height: 7px; border-radius: 6px;
    background: rgba(255,255,255,0.05); margin-top: 9px; overflow: hidden;
}
.bar-fill {
    height: 100%; border-radius: 6px;
    background: linear-gradient(90deg, var(--forest), var(--emerald), var(--neon));
    animation: barGrow 1.4s cubic-bezier(.4,0,.2,1) forwards;
    transform-origin: left;
}
@keyframes barGrow { 0% { width: 0 !important; } }

/* ── AI INSIGHT RECOMMENDATION CARDS ───────────────────────────────────────── */
.ai-insight-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-top: 20px;
}
.ai-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 18px;
    padding: 22px 20px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    transition: transform 0.28s ease, border-color 0.28s ease, box-shadow 0.28s ease;
    animation: fadeSlideUp 0.6s ease forwards;
    opacity: 0;
}
.ai-card:nth-child(1) { animation-delay: 0.05s; }
.ai-card:nth-child(2) { animation-delay: 0.15s; }
.ai-card:nth-child(3) { animation-delay: 0.25s; }
.ai-card:nth-child(4) { animation-delay: 0.35s; }
@keyframes fadeSlideUp {
    0%   { opacity: 0; transform: translateY(18px); }
    100% { opacity: 1; transform: translateY(0); }
}
.ai-card:hover {
    transform: translateY(-6px);
    border-color: var(--glass-border-h);
    box-shadow: 0 18px 44px -16px rgba(15,174,102,0.40);
}
.ai-card-icon  { font-size: 24px; margin-bottom: 10px; }
.ai-card-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 14px; font-weight: 700; color: var(--text-main); margin-bottom: 7px;
}
.ai-card-body  { color: var(--text-dim); font-size: 12.5px; line-height: 1.6; }

/* ── WEATHER THEME BANNER ───────────────────────────────────────────────────── */
.weather-banner {
    display: flex; align-items: center; gap: 14px;
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 18px 22px;
    margin-top: 18px;
    animation: fadeSlideUp 0.5s ease forwards;
    backdrop-filter: blur(12px);
}
.weather-banner-icon { font-size: 30px; flex-shrink: 0; }
.weather-banner-title { font-weight: 700; color: var(--text-main); font-size: 15px; }
.weather-banner-desc  { color: var(--text-dim); font-size: 12.5px; margin-top: 2px; }

/* ── PREDICTION RESULT CARD ─────────────────────────────────────────────────── */
.result-card {
    position: relative; border-radius: 24px; padding: 38px 30px;
    text-align: center;
    background: linear-gradient(160deg, rgba(15,174,102,0.16), rgba(5,13,9,0.88));
    border: 1px solid rgba(109,255,176,0.38);
    animation: fadeInScale 0.65s cubic-bezier(.4,0,.2,1) forwards, glowPulse 3s ease-in-out infinite;
    box-shadow: var(--shadow-card);
}
@keyframes fadeInScale {
    0%   { opacity: 0; transform: scale(0.88); }
    100% { opacity: 1; transform: scale(1);    }
}
@keyframes glowPulse {
    0%,100% { box-shadow: 0 0 28px 0  rgba(109,255,176,0.16), var(--shadow-card); }
    50%     { box-shadow: 0 0 58px 8px rgba(109,255,176,0.36), var(--shadow-card); }
}
.result-label     { color: var(--text-dim); font-size: 13px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; }
.result-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 54px; font-weight: 900; margin: 10px 0 4px;
    background: linear-gradient(100deg, #ffffff, var(--neon));
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    animation: numberReveal 1s cubic-bezier(.4,0,.2,1) forwards;
}
@keyframes numberReveal {
    0%   { opacity: 0; transform: scale(0.7); }
    60%  { transform: scale(1.04); }
    100% { opacity: 1; transform: scale(1); }
}
.result-unit       { color: var(--text-dim); font-size: 14px; margin-bottom: 4px; }
.result-confidence { color: var(--text-dim); font-size: 12.5px; margin-bottom: 18px; font-style: italic; }
.status-badge {
    display: inline-block; padding: 9px 24px; border-radius: 999px;
    font-weight: 700; font-size: 13px; letter-spacing: 0.04em;
}
.status-low       { background: rgba(255,107,107,0.12); color: #ff8a8a; border: 1px solid rgba(255,107,107,0.32); }
.status-moderate  { background: rgba(255,210,122,0.12); color: #ffd27a; border: 1px solid rgba(255,210,122,0.32); }
.status-high      { background: rgba(109,255,176,0.12); color: var(--neon); border: 1px solid rgba(109,255,176,0.38); }
.status-excellent { background: rgba(52,232,158,0.16);  color: #7dffc0; border: 1px solid rgba(52,232,158,0.48); }

/* ── WARNING BOX ────────────────────────────────────────────────────────────── */
.warn-box {
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    border-radius: 14px;
    padding: 14px 18px;
    margin-top: 14px;
    font-size: 13px;
    color: var(--warn-text);
    line-height: 1.7;
}

/* ── MODEL METRIC CARDS ─────────────────────────────────────────────────────── */
.metric-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 18px;
    padding: 26px 20px;
    text-align: center;
    backdrop-filter: blur(14px);
    transition: transform 0.30s ease, box-shadow 0.30s ease, border-color 0.30s ease;
}
.metric-card:hover {
    transform: translateY(-7px);
    border-color: var(--glass-border-h);
    box-shadow: 0 20px 48px -18px rgba(15,174,102,0.45);
}
.metric-title { color: var(--text-dim); font-size: 12.5px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 7px; }
.metric-num   { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 28px; font-weight: 800; color: var(--neon); }

/* ── HISTORY TABLE ──────────────────────────────────────────────────────────── */
.history-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.history-table th {
    background: rgba(109,255,176,0.06); color: var(--neon);
    font-weight: 700; padding: 11px 14px; text-align: left;
    font-size: 11.5px; letter-spacing: 0.08em; text-transform: uppercase;
    border-bottom: 1px solid var(--glass-border);
}
.history-table td {
    padding: 11px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.045);
    color: var(--text-main);
}
.history-table tr:last-child td { border-bottom: none; }
.history-table tr:hover td { background: rgba(109,255,176,0.035); }

/* ── ABOUT CARD ─────────────────────────────────────────────────────────────── */
.about-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 32px 28px;
    backdrop-filter: blur(14px);
    line-height: 1.78; color: var(--text-dim); font-size: 14.5px;
}
.about-card b { color: var(--text-main); }
.tag-chip {
    display: inline-block; padding: 5px 14px; margin: 5px 5px 0 0;
    border-radius: 999px;
    background: rgba(109,255,176,0.07); border: 1px solid var(--glass-border);
    color: var(--neon); font-size: 12px; font-weight: 600;
}

/* ── GLASS WRAPPER ──────────────────────────────────────────────────────────── */
.glass-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    padding: 26px 24px;
}

/* ── STREAMLIT FORM CONTROLS ────────────────────────────────────────────────── */
[data-testid="stForm"] {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 20px !important;
    padding: 28px 26px !important;
    backdrop-filter: blur(14px) !important;
}
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: var(--text-dim) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.035) !important;
    border-color: var(--glass-border) !important;
    border-radius: 12px !important;
}
.stNumberInput input {
    background: rgba(255,255,255,0.035) !important;
    border-color: var(--glass-border) !important;
    border-radius: 12px !important;
    color: var(--text-main) !important;
}
.stButton button, .stFormSubmitButton button {
    background: linear-gradient(120deg, var(--emerald), var(--emerald-soft)) !important;
    color: #062712 !important; font-weight: 700 !important; font-size: 14px !important;
    border: none !important; border-radius: 14px !important; padding: 12px 0 !important;
    box-shadow: 0 14px 34px -10px rgba(15,174,102,0.65) !important;
    transition: transform 0.24s ease, box-shadow 0.24s ease !important;
    letter-spacing: 0.02em !important;
}
.stButton button:hover, .stFormSubmitButton button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 20px 40px -10px rgba(15,174,102,0.85) !important;
}

/* ── LOADING ANIMATION ──────────────────────────────────────────────────────── */
.loading-wrap {
    display: flex; align-items: center; gap: 14px;
    padding: 20px 24px;
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    margin: 16px 0;
}
.loading-dots span {
    display: inline-block;
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--neon);
    animation: dotBounce 1.2s ease-in-out infinite;
    margin: 0 3px;
}
.loading-dots span:nth-child(2) { animation-delay: 0.15s; }
.loading-dots span:nth-child(3) { animation-delay: 0.30s; }
@keyframes dotBounce {
    0%,80%,100% { transform: scale(0.6); opacity: 0.4; }
    40%         { transform: scale(1.2); opacity: 1;   }
}

/* ── TABS ────────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--glass) !important;
    border-radius: 14px !important;
    border: 1px solid var(--glass-border) !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-dim) !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    font-size: 13.5px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(109,255,176,0.12) !important;
    color: var(--neon) !important;
}

/* ── FOOTER ─────────────────────────────────────────────────────────────────── */
.app-footer {
    text-align: center; color: var(--text-dim);
    font-size: 12px; padding: 36px 0 12px; opacity: 0.6;
    letter-spacing: 0.04em;
}

/* ── DIVIDER ─────────────────────────────────────────────────────────────────── */
.fancy-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--glass-border), transparent);
    margin: 48px 0;
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
        <div class="hero-sub">Next-Generation Crop Yield Forecasting</div>
        <div class="hero-desc">
            Harness machine learning trained on <strong>{TOTAL_RECORDS:,} real agricultural records</strong>
            across <strong>{TOTAL_COUNTRIES} countries</strong> — predicting crop yield from climate,
            soil, and agronomic inputs with <strong>{R2_SCORE}% accuracy</strong>.
        </div>
        <div class="hero-cta">
            <span class="hero-pill">🔮 Start Predicting</span>
            <span class="hero-pill-outline">📊 R² = {R2_SCORE}% Accuracy</span>
        </div>
    </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# Model loading warning
if not MODEL_LOADED:
    st.warning(
        "⚠️ Model artifacts not found. Place `crop_yield_model.pkl`, "
        "`area_encoder.pkl`, and `item_encoder.pkl` alongside this app. "
        f"(Details: {LOAD_ERROR})"
    )

# =============================================================================
# SECTION 2 — KPI DASHBOARD (animated counters via CSS)
# =============================================================================
st.markdown(
    """
    <div class="sec-head">
        <div class="sec-tag">Platform Overview</div>
        <div class="sec-title">Trained on Real Agricultural Data</div>
        <div class="sec-desc">Built from a decade of global climate and harvest records spanning 101 nations.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_cols = st.columns(4)
kpis = [
    ("📚", f"{TOTAL_RECORDS:,}", "Training Records"),
    ("🌍", f"{TOTAL_COUNTRIES}+", "Countries Covered"),
    ("🌱", f"{TOTAL_CROPS}", "Crop Types"),
    ("🎯", f"{R2_SCORE}%", "Model Accuracy (R²)"),
]
for col, (icon, value, label) in zip(kpi_cols, kpis):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value counting">{value}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# =============================================================================
# SECTION 3 — PREDICTION WORKSPACE
# =============================================================================
st.markdown(
    """
    <div class="sec-head" style="margin-top:0;">
        <div class="sec-tag">Prediction Workspace</div>
        <div class="sec-title">Forecast Your Crop Yield</div>
        <div class="sec-desc">Configure environmental parameters and run an AI-powered yield forecast in seconds.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.35, 1], gap="large")

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
                "🌧️ Avg Rainfall (mm/yr)", min_value=0.0, max_value=5000.0,
                value=1100.0, step=10.0,
            )
        with f4:
            pesticides = st.number_input(
                "🧪 Pesticides (tonnes)", min_value=0.0, max_value=200000.0,
                value=1500.0, step=10.0,
            )

        temperature = st.number_input(
            "🌡️ Avg Temperature (°C)", min_value=-10.0, max_value=50.0,
            value=25.0, step=0.1,
        )

        submitted = st.form_submit_button("🔮 Generate Yield Forecast", use_container_width=True)

    if submitted:
        if not MODEL_LOADED:
            st.error("Prediction unavailable — model artifacts are missing from this directory.")
        else:
             with st.spinner("Running AI prediction model..."):
                  try:
                         area_encoded = area_encoder.transform([country])[0]
                         item_encoded = item_encoder.transform([crop])[0]

                         features = pd.DataFrame({
                         "Area": [area_encoded],
                         "Item": [item_encoded],
                         "Year": [year],
                         "average_rain_fall_mm_per_year": [rainfall],
                         "pesticides_tonnes": [pesticides],
                         "avg_temp": [temperature]
                       })

                        prediction = float(model.predict(features)[0])

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.write(features)   # TEMPORARY DEBUG

prediction = float(model.predict(features)[0])
                    confidence    = get_confidence_range(model, features)
                    warnings_list = get_warnings(crop, rainfall, pesticides, temperature)
                    weather_theme = get_weather_theme(rainfall, temperature)

                    st.session_state["prediction"]    = prediction
                    st.session_state["confidence"]    = confidence
                    st.session_state["warnings"]      = warnings_list
                    st.session_state["pred_country"]  = country
                    st.session_state["pred_crop"]     = crop
                    st.session_state["weather_theme"] = weather_theme
                    st.session_state["pred_inputs"]   = {
                        "year": year, "rainfall": rainfall,
                        "pesticides": pesticides, "temperature": temperature,
                    }
                    st.session_state["show_result"] = True

                    # Append to history
                    status_label, _, _, _ = yield_status(prediction)
                    st.session_state["prediction_history"].append({
                        "Timestamp":       datetime.now().strftime("%H:%M:%S"),
                        "Country":         country,
                        "Crop":            crop,
                        "Year":            year,
                        "Rainfall (mm)":   rainfall,
                        "Temp (°C)":       temperature,
                        "Pesticides (t)":  pesticides,
                        "Yield (hg/ha)":   f"{prediction:,.0f}",
                        "Status":          status_label,
                    })
                except Exception as e:
                    st.session_state["show_result"] = False
                    st.error(f"Prediction failed: {e}")

with right_col:
    fi_top_label, fi_top_val = list(FEATURE_IMPORTANCE.items())[0]
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="sec-tag" style="margin-bottom:14px; display:block;">AI Model Insights</div>
            <div class="insight-row">
                <span class="insight-label">Top Predictor</span>
                <span class="insight-value">{fi_top_label}</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Algorithm</span>
                <span class="insight-value">Random Forest</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">R² Accuracy</span>
                <span class="insight-value">{R2_SCORE}%</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Confidence Mode</span>
                <span class="insight-value">Per-tree Std Dev</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Training Records</span>
                <span class="insight-value">{TOTAL_RECORDS:,}</span>
            </div>
            <div style="margin-top:20px;">
                <span class="insight-label">{fi_top_label} Influence</span>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{fi_top_val}%;"></div>
                </div>
            </div>
            <div style="margin-top:16px; color:var(--text-dim); font-size:12.5px; line-height:1.65;">
                Crop genetics dominate yield variance. Climate inputs — especially temperature
                and rainfall — act as secondary amplifiers. Smart warnings flag inputs that
                fall outside the model's training distribution.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# SECTION 4 — PREDICTION RESULT + GAUGE + WEATHER + AI INSIGHTS
# =============================================================================
if st.session_state.get("show_result"):
    pred_value    = st.session_state.get("prediction", 0.0)
    pred_country  = st.session_state.get("pred_country", "")
    pred_crop     = st.session_state.get("pred_crop", "")
    confidence    = st.session_state.get("confidence")
    warnings_list = st.session_state.get("warnings", [])
    pred_inputs   = st.session_state.get("pred_inputs", {})
    weather_theme = st.session_state.get("weather_theme", "temperate")

    status_label, status_class, status_color, gauge_pct = yield_status(pred_value)

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sec-head" style="margin-top:0;">
            <div class="sec-tag">AI Forecast Result</div>
            <div class="sec-title">Yield Prediction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Result card + gauge ───────────────────────────────────────────────────
    res_left, res_right = st.columns([1, 1], gap="large")

    with res_left:
        confidence_html = ""
        if confidence is not None:
            lo = max(0, pred_value - confidence)
            hi = pred_value + confidence
            confidence_html = f'<div class="result-confidence">95% range: {lo:,.0f} – {hi:,.0f} hg/ha  (±{confidence:,.0f})</div>'

        weather_particles_html = build_weather_particles(weather_theme, pred_inputs.get("rainfall", 0), pred_inputs.get("temperature", 25))
        st.markdown(
            f"""
            <div class="result-card">
                {weather_particles_html}
                <div style="position:relative;z-index:2;">
                    <div class="result-label">🌾 {pred_crop} · {pred_country}</div>
                    <div class="result-value">{pred_value:,.0f}</div>
                    <div class="result-unit">hectograms per hectare (hg/ha)</div>
                    {confidence_html}
                    <span class="status-badge {status_class}">{status_label}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Smart Warnings
        if warnings_list:
            warn_html = "<br>".join(warnings_list)
            st.markdown(f'<div class="warn-box">{warn_html}</div>', unsafe_allow_html=True)

    with res_right:
        # Gauge chart
        gauge_fig = make_gauge_chart(pred_value)
        st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

        # Weather theme banner
        weather_title, weather_desc = get_weather_description(weather_theme)
        st.markdown(
            f"""
            <div class="weather-banner">
                <div class="weather-banner-icon">{weather_title.split()[0]}</div>
                <div>
                    <div class="weather-banner-title">{" ".join(weather_title.split()[1:])}</div>
                    <div class="weather-banner-desc">{weather_desc}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── AI Recommendation Cards ───────────────────────────────────────────────
  
    # ==========================================================================
    # SECTION 4b — WHAT-IF SENSITIVITY ANALYSIS  ← UNCHANGED LOGIC
    # ==========================================================================
    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sec-head" style="margin-top:0;">
            <div class="sec-tag">Sensitivity Analysis</div>
            <div class="sec-title">What-If Explorer</div>
            <div class="sec-desc">See how predicted yield responds as each parameter shifts, holding all others constant.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if MODEL_LOADED:
        whatif_tabs = st.tabs(["🌡️ Temperature", "🌧️ Rainfall", "🧪 Pesticides", "📅 Year"])

        base_year        = pred_inputs.get("year", 2024)
        base_rainfall    = pred_inputs.get("rainfall", 1100.0)
        base_pesticides  = pred_inputs.get("pesticides", 1500.0)
        base_temp        = pred_inputs.get("temperature", 25.0)
        area_enc         = area_encoder.transform([pred_country])[0]
        item_enc         = item_encoder.transform([pred_crop])[0]

        def make_whatif_fig(x_vals, yields, x_label, current_x, current_y):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_vals, y=yields,
                mode="lines",
                line=dict(color="#0fae66", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(15,174,102,0.07)",
                name="Yield",
            ))
            fig.add_trace(go.Scatter(
                x=[current_x], y=[current_y],
                mode="markers",
                marker=dict(color="#6dffb0", size=13, symbol="circle",
                            line=dict(color="#ffffff", width=2.5)),
                name="Current Input",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#eaf6ee"),
                margin=dict(l=10, r=10, t=20, b=20),
                xaxis=dict(title=x_label, showgrid=True, gridcolor="rgba(255,255,255,0.055)"),
                yaxis=dict(title="Yield (hg/ha)", showgrid=True, gridcolor="rgba(255,255,255,0.055)"),
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            return fig

        with whatif_tabs[0]:
            temps   = np.linspace(max(-10, base_temp - 15), min(50, base_temp + 15), 60)
            yields_t = [float(model.predict([[base_year, base_rainfall, base_pesticides, t, area_enc, item_enc]])[0]) for t in temps]
            st.plotly_chart(make_whatif_fig(temps, yields_t, "Temperature (°C)", base_temp, pred_value), use_container_width=True)

        with whatif_tabs[1]:
            rains   = np.linspace(max(0, base_rainfall - 800), base_rainfall + 800, 60)
            yields_r = [float(model.predict([[base_year, r, base_pesticides, base_temp, area_enc, item_enc]])[0]) for r in rains]
            st.plotly_chart(make_whatif_fig(rains, yields_r, "Rainfall (mm/yr)", base_rainfall, pred_value), use_container_width=True)

        with whatif_tabs[2]:
            pests   = np.linspace(max(0, base_pesticides - 2000), base_pesticides + 2000, 60)
            yields_p = [float(model.predict([[base_year, base_rainfall, p, base_temp, area_enc, item_enc]])[0]) for p in pests]
            st.plotly_chart(make_whatif_fig(pests, yields_p, "Pesticides (tonnes)", base_pesticides, pred_value), use_container_width=True)

        with whatif_tabs[3]:
            years   = np.arange(1990, 2031, 1)
            yields_y = [float(model.predict([[int(y), base_rainfall, base_pesticides, base_temp, area_enc, item_enc]])[0]) for y in years]
            st.plotly_chart(make_whatif_fig(years, yields_y, "Year", base_year, pred_value), use_container_width=True)

# =============================================================================
# SECTION 5 — PREDICTION HISTORY
# =============================================================================
if st.session_state["prediction_history"]:
    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sec-head" style="margin-top:0;">
            <div class="sec-tag">Session Log</div>
            <div class="sec-title">Prediction History</div>
            <div class="sec-desc">All forecasts made this session — compare scenarios at a glance.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = st.session_state["prediction_history"]
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

    btn_left, btn_right = st.columns([1, 5])
    with btn_left:
        csv_data = export_history_csv(history)
        st.download_button(
            label="⬇️ Export CSV",
            data=csv_data,
            file_name=f"agripredict_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with btn_right:
        if st.button("🗑️ Clear History"):
            st.session_state["prediction_history"] = []
            st.rerun()

# =============================================================================
# SECTION 6 — FEATURE IMPORTANCE ANALYTICS
# =============================================================================
st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="sec-head" style="margin-top:0;">
        <div class="sec-tag">Model Explainability</div>
        <div class="sec-title">Feature Importance</div>
        <div class="sec-desc">Relative influence of each input variable on predicted crop yield.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

fi_sorted = dict(sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1]))
fig_fi = go.Figure(go.Bar(
    x=list(fi_sorted.values()),
    y=list(fi_sorted.keys()),
    orientation="h",
    marker=dict(
        color=list(fi_sorted.values()),
        colorscale=[[0, "#0c2016"], [0.45, "#0fae66"], [1, "#6dffb0"]],
        line=dict(color="rgba(109,255,176,0.30)", width=1),
    ),
    text=[f"{v}%" for v in fi_sorted.values()],
    textposition="outside",
    textfont=dict(color="#eaf6ee", size=13, family="Inter"),
))
fig_fi.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#eaf6ee"),
    margin=dict(l=10, r=40, t=16, b=16),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Importance (%)", range=[0, 72]),
    yaxis=dict(showgrid=False),
    height=360,
)
st.plotly_chart(fig_fi, use_container_width=True)

# =============================================================================
# SECTION 7 — MODEL PERFORMANCE
# =============================================================================
st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="sec-head" style="margin-top:0;">
        <div class="sec-tag">Validation Results</div>
        <div class="sec-title">Model Performance</div>
        <div class="sec-desc">Evaluated on a held-out test split of the full agricultural dataset.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3, gap="large")
for col, (title, num) in zip([m1, m2, m3], [
    ("R² Score", f"{R2_SCORE}%"),
    ("Mean Abs Error", f"{MAE:,.2f} hg/ha"),
    ("Root Mean Sq Error", f"{RMSE:,.2f} hg/ha"),
]):
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
st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="sec-head" style="margin-top:0;">
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
        AgriPredict AI is powered by a <b>Random Forest Regressor</b> — an ensemble of
        hundreds of decision trees that collectively model the complex, non-linear
        interplay between climate, agrochemical inputs, and crop productivity.
        <br><br>
        The model was trained on <b>{TOTAL_RECORDS:,} agricultural records</b> spanning
        <b>{TOTAL_COUNTRIES} countries</b> and <b>{TOTAL_CROPS} major crop types</b>,
        using annual rainfall, average temperature, pesticide volumes, country,
        crop variety, and year as predictive features. It achieves an
        <b>R² of {R2_SCORE}%</b> on a held-out test set.
        <br><br>
        <b>This version adds:</b> Plotly gauge charts · dynamic weather theme effects ·
        per-theme climate particles · AI recommendation cards · animated KPI counters ·
        glassmorphism design system · per-tree confidence intervals · what-if
        sensitivity explorer · session history with CSV export.
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
        🌾 AgriPredict AI &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Random Forest Regressor &nbsp;·&nbsp; Plotly Analytics &nbsp;·&nbsp; Glassmorphism UI
    </div>
    """,
    unsafe_allow_html=True,
)
