"""
app.py — RunnAing: Predicción de TRIMP y ACWR para corredores populares.
Modelo: models/best_model.pkl (XGBoost, R²=0.752, MAE=24.73 u.a.)
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import streamlit as st

from src.features import FEATURE_NAMES_GPS, assert_no_hr_leakage
from src.acwr import (
    compute_acwr_all_users,
    zone_distribution,
    ZONE_LOW, ZONE_HIGH_OPT, ZONE_HIGH_RISK,
)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RunnAing — Predictor TRIMP",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = Path("models")
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
FEEDBACK_PATH = Path("feedback_responses.csv")

ZONE_COLORS = {
    "insuficiente": "#3b82f6",
    "optima":       "#22c55e",
    "precaucion":   "#f59e0b",
    "riesgo":       "#ef4444",
    "sin_dato":     "#71717a",
}

ZONE_LABELS = {
    "insuficiente": "Insuficiente",
    "optima":       "Óptima",
    "precaucion":   "Precaución",
    "riesgo":       "Riesgo",
    "sin_dato":     "Sin datos",
}


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------
def apply_css():
    st.markdown("""
<style>
/* ─────────────────────────────────────────────
   Google Fonts
───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ─────────────────────────────────────────────
   Root variables
───────────────────────────────────────────── */
:root {
    --bg:          #f8f7ff;
    --surface:     #ffffff;
    --surface-2:   #f1f0fb;
    --border:      #e2e0f0;
    --border-2:    #c8c5e0;
    --accent:      #7c6ff7;
    --accent-hover:#5b52e0;
    --accent-muted:rgba(124,111,247,0.10);
    --text:        #1e1b3a;
    --text-2:      #4a4770;
    --text-3:      #8b88b0;
    --green:       #5ec486;
    --amber:       #f7c26b;
    --red:         #f4827a;
    --blue:        #7ab8f5;
    --radius:      12px;
    --radius-sm:   8px;
}

/* ─────────────────────────────────────────────
   Global resets
───────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: var(--bg);
    color: var(--text);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
}

[data-testid="stMain"] {
    background-color: var(--bg);
}

[data-testid="stHeader"] {
    background-color: var(--bg);
    border-bottom: 1px solid var(--border);
}

/* ─────────────────────────────────────────────
   Scrollbar
───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface-2); }
::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-3); }

/* ─────────────────────────────────────────────
   Sidebar
───────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-2);
    font-size: 0.85rem;
    line-height: 1.6;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text);
    font-weight: 600;
}

[data-testid="stSidebarContent"] {
    padding-top: 1.5rem;
}

/* ─────────────────────────────────────────────
   Typography
───────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', system-ui, sans-serif;
    color: var(--text);
    font-weight: 600;
    letter-spacing: -0.02em;
}

p, li {
    color: var(--text-2);
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ─────────────────────────────────────────────
   Tabs
───────────────────────────────────────────── */
[data-testid="stTabs"] {
    border-bottom: 1px solid var(--border);
}

[data-testid="stTabs"] [role="tablist"] {
    background: transparent;
    gap: 0;
    padding: 0;
}

[data-testid="stTabs"] [role="tab"] {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--text-3);
    font-size: 0.875rem;
    font-weight: 500;
    padding: 0.75rem 1.25rem;
    margin-bottom: -1px;
    border-radius: 0;
    transition: color 0.15s ease, border-color 0.15s ease;
    font-family: 'Inter', sans-serif;
}

[data-testid="stTabs"] [role="tab"]:hover {
    color: var(--text);
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--text);
    border-bottom-color: var(--accent);
    background: transparent;
}

/* ─────────────────────────────────────────────
   Buttons
───────────────────────────────────────────── */
[data-testid="stButton"] > button,
[data-testid="stFormSubmitButton"] > button {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 0.875rem;
    border-radius: var(--radius-sm);
    transition: background 0.15s ease, border-color 0.15s ease, opacity 0.15s ease;
    letter-spacing: -0.01em;
}

[data-testid="stButton"] > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primary"] {
    background-color: var(--accent);
    color: #ffffff;
    border: 1px solid var(--accent);
    padding: 0.5rem 1.25rem;
}

[data-testid="stButton"] > button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
    background-color: var(--accent-hover);
    border-color: var(--accent-hover);
}

[data-testid="stButton"] > button[kind="secondary"],
[data-testid="stFormSubmitButton"] > button[kind="secondary"] {
    background-color: transparent;
    color: var(--text-2);
    border: 1px solid var(--border-2);
    padding: 0.5rem 1.25rem;
}

[data-testid="stButton"] > button[kind="secondary"]:hover,
[data-testid="stFormSubmitButton"] > button[kind="secondary"]:hover {
    background-color: var(--surface-2);
    border-color: var(--text-3);
    color: var(--text);
}

/* ─────────────────────────────────────────────
   Number inputs
───────────────────────────────────────────── */
[data-testid="stNumberInput"] input {
    background-color: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: var(--radius-sm);
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    padding: 0.5rem 0.75rem;
    transition: border-color 0.15s ease;
}

[data-testid="stNumberInput"] input:focus {
    border-color: var(--accent);
    outline: none;
    box-shadow: 0 0 0 3px var(--accent-muted);
}

[data-testid="stNumberInput"] label {
    color: var(--text-2);
    font-size: 0.8125rem;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
}

/* ─────────────────────────────────────────────
   Selectbox
───────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background-color: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: var(--radius-sm);
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
}

[data-testid="stSelectbox"] > div > div:hover {
    border-color: var(--border-2);
}

[data-testid="stSelectbox"] label {
    color: var(--text-2);
    font-size: 0.8125rem;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
}

/* ─────────────────────────────────────────────
   Slider
───────────────────────────────────────────── */
[data-testid="stSlider"] label {
    color: var(--text-2);
    font-size: 0.8125rem;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    color: var(--text);
    background: var(--surface-2);
    border: 1px solid var(--border);
    font-size: 0.75rem;
    border-radius: 4px;
    padding: 2px 6px;
}

/* ─────────────────────────────────────────────
   Text area
───────────────────────────────────────────── */
[data-testid="stTextArea"] textarea {
    background-color: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: var(--radius-sm);
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    resize: vertical;
    transition: border-color 0.15s ease;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-muted);
}

[data-testid="stTextArea"] label {
    color: var(--text-2);
    font-size: 0.8125rem;
    font-weight: 500;
}

/* ─────────────────────────────────────────────
   File uploader
───────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background-color: var(--surface);
    border: 1px dashed var(--border-2);
    border-radius: var(--radius);
    padding: 1.5rem;
    text-align: center;
    transition: border-color 0.15s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent);
}

[data-testid="stFileUploader"] label {
    color: var(--text-2);
    font-size: 0.8125rem;
    font-weight: 500;
}

/* ─────────────────────────────────────────────
   Metric
───────────────────────────────────────────── */
[data-testid="metric-container"] {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
}

[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: var(--text-3);
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text);
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.03em;
}

/* ─────────────────────────────────────────────
   Alerts
───────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm);
    border-width: 1px;
    font-size: 0.875rem;
    font-family: 'Inter', sans-serif;
}

[data-testid="stAlert"][data-baseweb="notification"][aria-live="polite"] {
    background-color: rgba(34,197,94,0.08);
    border-color: rgba(34,197,94,0.25);
    color: #86efac;
}

/* ─────────────────────────────────────────────
   Info / warning / error boxes
───────────────────────────────────────────── */
.stAlert > div {
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
}

/* ─────────────────────────────────────────────
   Expander
───────────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
}

[data-testid="stExpander"] summary {
    color: var(--text-2);
    font-size: 0.875rem;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    padding: 0.75rem 1rem;
}

[data-testid="stExpander"] summary:hover {
    color: var(--text);
}

/* ─────────────────────────────────────────────
   Radio buttons
───────────────────────────────────────────── */
[data-testid="stRadio"] label {
    color: var(--text-2);
    font-size: 0.875rem;
    font-family: 'Inter', sans-serif;
}

[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-2);
    font-size: 0.8125rem;
}

/* ─────────────────────────────────────────────
   Caption / small text
───────────────────────────────────────────── */
[data-testid="stCaptionContainer"] {
    color: var(--text-3);
    font-size: 0.75rem;
    font-family: 'Inter', sans-serif;
}

/* ─────────────────────────────────────────────
   Download button
───────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background-color: transparent;
    color: var(--text-2);
    border: 1px solid var(--border-2);
    border-radius: var(--radius-sm);
    font-family: 'Inter', sans-serif;
    font-size: 0.8125rem;
    font-weight: 500;
    padding: 0.4rem 1rem;
    transition: all 0.15s ease;
}

[data-testid="stDownloadButton"] > button:hover {
    background-color: var(--surface-2);
    border-color: var(--text-3);
    color: var(--text);
}

/* ─────────────────────────────────────────────
   Form container
───────────────────────────────────────────── */
[data-testid="stForm"] {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
}

/* ─────────────────────────────────────────────
   Divider
───────────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* ─────────────────────────────────────────────
   Custom components
───────────────────────────────────────────── */

/* KPI card */
.kpi-card {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.kpi-card .kpi-label {
    font-size: 0.6875rem;
    font-weight: 600;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.kpi-card .kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.04em;
    line-height: 1.1;
}

.kpi-card .kpi-sublabel {
    font-size: 0.75rem;
    color: var(--text-3);
    margin-top: 2px;
}

/* TRIMP result card */
.trimp-result-card {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem;
    text-align: center;
}

.trimp-result-card .trimp-value {
    font-size: 4rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.06em;
    line-height: 1;
}

.trimp-result-card .trimp-unit {
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--text-3);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-top: 4px;
}

.trimp-result-card .trimp-interp {
    font-size: 0.9375rem;
    color: var(--text-2);
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
}

/* Section header */
.section-header {
    margin: 1.75rem 0 1rem 0;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
}

.section-header .sh-title {
    font-size: 0.9375rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.01em;
    margin: 0;
}

.section-header .sh-subtitle {
    font-size: 0.8125rem;
    color: var(--text-3);
    margin: 3px 0 0 0;
}

/* Zone pills */
.zone-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    line-height: 1.6;
}

.zone-pill-insuficiente {
    background-color: rgba(59,130,246,0.12);
    color: #93c5fd;
    border: 1px solid rgba(59,130,246,0.25);
}

.zone-pill-optima {
    background-color: rgba(34,197,94,0.12);
    color: #86efac;
    border: 1px solid rgba(34,197,94,0.25);
}

.zone-pill-precaucion {
    background-color: rgba(245,158,11,0.12);
    color: #fcd34d;
    border: 1px solid rgba(245,158,11,0.25);
}

.zone-pill-riesgo {
    background-color: rgba(239,68,68,0.12);
    color: #fca5a5;
    border: 1px solid rgba(239,68,68,0.25);
}

.zone-pill-sin_dato {
    background-color: rgba(113,113,122,0.12);
    color: #a1a1aa;
    border: 1px solid rgba(113,113,122,0.25);
}

/* Load bar */
.load-bar-wrap {
    margin: 1.25rem 0;
}

.load-bar-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-3);
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.load-bar-track {
    background: var(--surface-2);
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
    border: 1px solid var(--border);
}

.load-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.4s ease;
}

.load-bar-ticks {
    display: flex;
    justify-content: space-between;
    font-size: 0.6875rem;
    color: var(--text-3);
    margin-top: 4px;
}

/* Sidebar stat row */
.sidebar-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.8125rem;
}

.sidebar-stat .ss-label { color: var(--text-3); }
.sidebar-stat .ss-value { color: var(--text); font-weight: 600; font-variant-numeric: tabular-nums; }

/* App wordmark */
.app-wordmark {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.03em;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 0.25rem;
}

.app-wordmark .wm-dot {
    width: 8px;
    height: 8px;
    background: var(--accent);
    border-radius: 50%;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI helper components
# ---------------------------------------------------------------------------

def kpi_card(label: str, value: str, sublabel: str = "") -> None:
    sub_html = f'<div class="kpi-sublabel">{sublabel}</div>' if sublabel else ""
    st.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div>
    {sub_html}
</div>
""", unsafe_allow_html=True)


def zone_pill(zone: str) -> str:
    icons = {
        "insuficiente": "●",
        "optima":       "●",
        "precaucion":   "●",
        "riesgo":       "●",
        "sin_dato":     "●",
    }
    label = ZONE_LABELS.get(zone, zone.capitalize())
    icon  = icons.get(zone, "●")
    return (
        f'<span class="zone-pill zone-pill-{zone}">'
        f'<span>{icon}</span>{label}'
        f'</span>'
    )


def section_header(title: str, subtitle: str = "") -> None:
    sub_html = f'<p class="sh-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
<div class="section-header">
    <p class="sh-title">{title}</p>
    {sub_html}
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    if not BEST_MODEL_PATH.exists():
        return None
    with open(BEST_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def build_feature_row(duration_min, distance_km, speed_mean, speed_std,
                      pace_mean, elevation_gain, speed_max, altitude_mean) -> pd.DataFrame:
    grade_factor = (elevation_gain / distance_km) if distance_km > 0 else 0.0
    row = pd.DataFrame([{
        "duration_min":   duration_min,
        "distance_km":    distance_km,
        "speed_mean":     speed_mean,
        "speed_std":      speed_std,
        "pace_mean":      pace_mean,
        "elevation_gain": elevation_gain,
        "speed_max":      speed_max,
        "altitude_mean":  altitude_mean,
        "grade_factor":   grade_factor,
    }])
    row = row[FEATURE_NAMES_GPS]
    assert_no_hr_leakage(row)
    return row


def trimp_interpretation(trimp: float) -> str:
    if trimp < 40:
        return "Recuperación — sesión muy ligera"
    elif trimp < 80:
        return "Entrenamiento base — esfuerzo moderado"
    elif trimp < 140:
        return "Entrenamiento de calidad — sesión exigente"
    elif trimp < 200:
        return "Alta intensidad — sesión muy exigente"
    else:
        return "Carga extrema — competición o esfuerzo máximo"


def save_feedback(data: dict) -> None:
    file_exists = FEEDBACK_PATH.exists()
    with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


def plot_acwr(acwr_df: pd.DataFrame, user_id: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    fig.patch.set_facecolor("#f8f7ff")
    ax.set_facecolor("#f8f7ff")

    user_data = acwr_df[acwr_df["userId"] == user_id].copy()
    dates = pd.to_datetime(user_data["date"])
    acwr  = user_data["acwr"]

    ax.plot(dates, acwr, color="#6366f1", linewidth=2, zorder=3, solid_capstyle="round")
    ax.fill_between(dates, acwr, alpha=0.06, color="#6366f1")

    ax.axhspan(ZONE_LOW, ZONE_HIGH_OPT,      alpha=0.07, color="#22c55e", zorder=1)
    ax.axhspan(ZONE_HIGH_OPT, ZONE_HIGH_RISK, alpha=0.07, color="#f59e0b", zorder=1)
    ax.axhspan(ZONE_HIGH_RISK, ax.get_ylim()[1] if ax.get_ylim()[1] > ZONE_HIGH_RISK else ZONE_HIGH_RISK + 0.5,
               alpha=0.07, color="#ef4444", zorder=1)

    ax.axhline(ZONE_LOW,       color="#22c55e", linewidth=0.8, linestyle="--", alpha=0.45)
    ax.axhline(ZONE_HIGH_OPT,  color="#f59e0b", linewidth=0.8, linestyle="--", alpha=0.45)
    ax.axhline(ZONE_HIGH_RISK, color="#ef4444", linewidth=0.8, linestyle="--", alpha=0.45)

    ax.tick_params(colors="#4a4770", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#27272a")
    ax.set_xlabel("Fecha de sesión", color="#52525b", fontsize=8, labelpad=8)
    ax.set_ylabel("ACWR", color="#52525b", fontsize=8, labelpad=8)
    ax.set_title("Ratio Carga Aguda:Crónica (ACWR)", color="#4a4770", fontsize=9.5,
                 pad=12, fontweight="500", loc="left")
    ax.grid(alpha=0.08, color="#27272a", linewidth=0.8)
    fig.tight_layout(pad=1.5)
    return fig


# ---------------------------------------------------------------------------
# CSS injection (called here so it fires early)
# ---------------------------------------------------------------------------
apply_css()

# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------
model = load_model()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
<div class="app-wordmark">
    <span class="wm-dot"></span>RunnAing
</div>
<p style="color:#71717a;font-size:0.75rem;margin:0 0 1.25rem 0;">
    Predicción de Carga Interna
</p>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="sidebar-stat"><span class="ss-label">Modelo</span><span class="ss-value">XGBoost</span></div>
<div class="sidebar-stat"><span class="ss-label">R²</span><span class="ss-value">0.752</span></div>
<div class="sidebar-stat"><span class="ss-label">MAE</span><span class="ss-value">24.73 u.a.</span></div>
<div class="sidebar-stat"><span class="ss-label">Variables</span><span class="ss-value">Solo GPS</span></div>
<div class="sidebar-stat"><span class="ss-label">Dataset</span><span class="ss-value">FitRec / Endomondo</span></div>
<div class="sidebar-stat" style="border:none"><span class="ss-label">Versión</span><span class="ss-value">TFM 2026</span></div>
""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    st.caption("TRIMP: Banister (1991)  \nACWR: Gabbett (2016)  \nDataset: Ni et al. (2019)")

    if model is None:
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        st.error("Modelo no encontrado en `models/best_model.pkl`")

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
if model is None:
    st.error("Modelo no encontrado — verifica `models/best_model.pkl`.")
    st.stop()

# ---------------------------------------------------------------------------
# Tab navigation
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Predice tu sesión", "📈 Historial ACWR", "💬 Feedback"])


# ===========================================================================
# TAB 1 — Prediction form
# ===========================================================================
with tab1:
    section_header("Parámetros de la sesión", "Introduce los datos GPS de tu entrenamiento para estimar la carga interna (TRIMP)")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        duration_min = st.number_input(
            "⏱️ Duración (minutos)", min_value=5.0, max_value=600.0,
            value=45.0, step=1.0
        )
        elevation_gain = st.number_input(
            "⛰️ Desnivel acumulado (m)", min_value=0.0, max_value=5000.0,
            value=50.0, step=10.0
        )
        gender = st.selectbox("👤 Género", ["male", "female"])

    with col2:
        speed_mean = st.number_input(
            "💨 Velocidad media (km/h)", min_value=3.0, max_value=25.0,
            value=10.0, step=0.1
        )
        speed_std = st.number_input(
            "📉 Variabilidad velocidad (km/h)", min_value=0.0, max_value=10.0,
            value=1.5, step=0.1,
            help="~1 para ritmo constante, >2 para terreno variable o intervalos."
        )
        altitude_mean = st.number_input(
            "🗻 Altitud media (m s.n.m.)", min_value=0.0, max_value=5000.0,
            value=200.0, step=50.0
        )

    # Valores derivados — siempre consistentes entre sí
    distance_km = (duration_min / 60.0) * speed_mean      # distancia física real
    pace_mean   = (60.0 / speed_mean) if speed_mean > 0 else 6.0
    speed_max   = speed_mean * 1.35
    grade_fact  = (elevation_gain / distance_km) if distance_km > 0 else 0.0

    with st.expander("Valores calculados automáticamente"):
        dc1, dc2, dc3, dc4 = st.columns(4)
        with dc1:
            kpi_card("Distancia", f"{distance_km:.2f}", "km")
        with dc2:
            kpi_card("Ritmo medio", f"{pace_mean:.2f}", "min/km")
        with dc3:
            kpi_card("Vel. máx est.", f"{speed_max:.2f}", "km/h")
        with dc4:
            kpi_card("Factor pendiente", f"{grade_fact:.2f}", "m/km")

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Calcular TRIMP", type="primary", use_container_width=True)

    # --- Ejecutar predicción y guardar en session_state ---
    if predict_btn:
        X = build_feature_row(
            duration_min, distance_km, speed_mean, speed_std,
            pace_mean, elevation_gain, speed_max, altitude_mean
        )
        trimp_pred = float(model.predict(X)[0])
        trimp_pred = max(0.0, trimp_pred)

        if trimp_pred < 80:
            bar_color  = "#22c55e"
            zone_label = "Entrenamiento base"
        elif trimp_pred < 150:
            bar_color  = "#f59e0b"
            zone_label = "Entrenamiento de calidad"
        else:
            bar_color  = "#ef4444"
            zone_label = "Alta intensidad"

        st.session_state["last_trimp"]     = trimp_pred
        st.session_state["last_bar_color"] = bar_color
        st.session_state["last_zone"]      = zone_label
        st.session_state["last_session"]   = {
            "duration_min":   duration_min,
            "distance_km":    distance_km,
            "speed_mean":     speed_mean,
            "elevation_gain": elevation_gain,
            "gender":         gender,
            "trimp_pred":     trimp_pred,
        }

    # --- Renderizar resultados siempre que haya predicción guardada ---
    if "last_trimp" in st.session_state:
        trimp_pred = st.session_state["last_trimp"]
        bar_color  = st.session_state.get("last_bar_color", "#22c55e")
        zone_label = st.session_state.get("last_zone", "")
        sess       = st.session_state.get("last_session", {})
        pct        = min(trimp_pred / 500, 1.0) * 100

        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

        # TRIMP result card
        st.markdown(f"""
<div class="trimp-result-card">
    <div class="trimp-value">{trimp_pred:.1f}</div>
    <div class="trimp-unit">TRIMP &nbsp;·&nbsp; unidades arbitrarias</div>
    <div class="trimp-interp">
        {trimp_interpretation(trimp_pred)}
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

        # Load bar
        st.markdown(f"""
<div class="load-bar-wrap">
    <div class="load-bar-label">Escala de carga &nbsp;(0 – 500 u.a.)</div>
    <div class="load-bar-track">
        <div class="load-bar-fill" style="width:{pct:.1f}%; background:{bar_color};"></div>
    </div>
    <div class="load-bar-ticks">
        <span>0 · Recuperación</span>
        <span>80</span>
        <span>150</span>
        <span>500 · Máximo</span>
    </div>
</div>
""", unsafe_allow_html=True)

        # KPI summary row
        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi_card("TRIMP", f"{trimp_pred:.1f}", "u.a.")
        with k2:
            kpi_card("Duración", f"{sess.get('duration_min', duration_min):.0f}", "min")
        with k3:
            kpi_card("Distancia", f"{sess.get('distance_km', distance_km):.1f}", "km")
        with k4:
            kpi_card("Desnivel", f"{sess.get('elevation_gain', elevation_gain):.0f}", "m")

        if predict_btn:
            st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
            st.success("¡Predicción completada! Pasa a la pestaña **Feedback** para valorar el resultado.")


# ===========================================================================
# TAB 2 — ACWR history
# ===========================================================================
with tab2:
    section_header(
        "Ratio carga aguda : crónica (ACWR)",
        "Sube tu historial de sesiones para calcular la evolución del ACWR (mínimo 28 sesiones)"
    )

    uploaded = st.file_uploader(
        "CSV con columnas: date (YYYY-MM-DD) y trimp", type=["csv"]
    )

    if uploaded:
        df_hist = pd.read_csv(uploaded)
        df_hist["date"] = pd.to_datetime(df_hist["date"])
        if "userId" not in df_hist.columns:
            df_hist["userId"] = "athlete"
        if "trimp" not in df_hist.columns:
            st.error("El CSV debe contener una columna `trimp`.")
            st.stop()

        acwr_df  = compute_acwr_all_users(
            df_hist, date_col="date", trimp_col="trimp", user_col="userId"
        )
        user_ids = sorted(acwr_df["userId"].unique())

        sel_col, _ = st.columns([1, 2])
        with sel_col:
            selected = st.selectbox("Atleta", user_ids)

        user_acwr = acwr_df[acwr_df["userId"] == selected]
        last_row  = user_acwr.dropna(subset=["acwr"]).iloc[-1] if len(user_acwr) > 0 else None

        if last_row is not None:
            current_zone = last_row.get("zone", "sin_dato")

            st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                kpi_card("ACWR actual", f"{last_row['acwr']:.2f}", "media 7d / 28d")
            with m2:
                kpi_card("Sesiones", str(len(user_acwr)), "total cargadas")
            with m3:
                st.markdown(
                    f"<div class='kpi-card'>"
                    f"<div class='kpi-label'>Zona</div>"
                    f"<div style='margin-top:6px'>{zone_pill(current_zone)}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
        fig_acwr = plot_acwr(acwr_df, selected)
        st.pyplot(fig_acwr, use_container_width=True)
        plt.close(fig_acwr)

        section_header("Distribución histórica de zonas")
        dist = zone_distribution(acwr_df[acwr_df["userId"] == selected])
        dcols = st.columns(4)
        for col, zname in zip(dcols, ["insuficiente", "optima", "precaucion", "riesgo"]):
            with col:
                row = dist[dist["zone"] == zname]
                pct_val = row["pct"].values[0] if len(row) > 0 else 0.0
                kpi_card(ZONE_LABELS[zname], f"{pct_val:.0f}%", "de las sesiones")

    else:
        st.markdown("""
<div style="background:#111113;border:1px solid #27272a;border-radius:12px;padding:1.5rem;margin-top:0.5rem;">
    <p style="color:#a1a1aa;font-size:0.875rem;margin:0 0 0.75rem 0;font-weight:600;">Formato de CSV esperado</p>
    <pre style="background:#18181b;color:#a1a1aa;border:1px solid #27272a;
                border-radius:8px;padding:0.75rem 1rem;font-size:0.8125rem;
                overflow-x:auto;margin:0;">date,trimp
2024-01-01,85.3
2024-01-03,102.1
2024-01-06,74.8</pre>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# TAB 3 — Feedback
# ===========================================================================
with tab3:
    section_header(
        "Evaluación del modelo",
        "Ayúdanos a validar si predecir la carga interna sin pulsómetro es útil en la práctica"
    )

    trimp_mostrado = st.session_state.get("last_trimp", None)
    if trimp_mostrado:
        st.markdown(
            f"<div style='background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);"
            f"border-radius:8px;padding:0.75rem 1rem;font-size:0.875rem;color:#c7d2fe;"
            f"margin-bottom:1rem;'>"
            f"Último TRIMP predicho: <strong style='color:#e0e7ff'>{trimp_mostrado:.1f} u.a.</strong>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.warning("Calcula primero una sesión en la pestaña **Predice tu sesión** para dar feedback contextualizado.")

    with st.form("feedback_form", clear_on_submit=True):
        section_header("Sobre ti")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            es_corredor      = st.selectbox("¿Eres corredor/a?", ["Sí, amateur", "Sí, semipro / competitivo", "No practico running regularmente"])
            anos_experiencia = st.selectbox("Años corriendo", ["< 1 año", "1–3 años", "3–5 años", "> 5 años"])
        with c2:
            sesiones_semana = st.selectbox("Sesiones por semana", ["1–2", "3–4", "5 o más", "N/A"])
            usa_pulsometro  = st.selectbox("¿Usas pulsómetro?", ["Sí, siempre", "A veces", "No"])

        section_header("Sobre la predicción")

        valoracion = st.slider(
            "¿En qué medida te parece razonable la predicción de TRIMP?",
            min_value=1, max_value=5, value=3
        )
        st.caption("1 = Nada razonable   ·   3 = Neutral   ·   5 = Muy razonable")

        trimp_esperado = st.number_input(
            "¿Cuánto crees que fue tu TRIMP real? (0 = desconozco)",
            min_value=0.0, max_value=500.0, value=0.0, step=5.0
        )

        util_sin_pulso = st.radio(
            "¿Crees que predecir la carga interna sin pulsómetro es útil?",
            ["Sí, muy útil", "Útil en algunos casos", "Prefiero usar pulsómetro", "No estoy seguro/a"],
            horizontal=True,
        )

        comentario = st.text_area(
            "Comentarios adicionales (opcional)",
            placeholder="¿La predicción difiere de cómo te sentiste? ¿Algo que mejorar?"
        )

        submitted = st.form_submit_button("Enviar feedback", type="primary", use_container_width=True)

        if submitted:
            save_feedback({
                "timestamp":           datetime.now().isoformat(),
                "trimp_predicho":      round(trimp_mostrado, 2) if trimp_mostrado else "",
                "es_corredor":         es_corredor,
                "anos_experiencia":    anos_experiencia,
                "sesiones_semana":     sesiones_semana,
                "usa_pulsometro":      usa_pulsometro,
                "valoracion_1_5":      valoracion,
                "trimp_esperado":      trimp_esperado if trimp_esperado > 0 else "",
                "util_sin_pulsometro": util_sin_pulso,
                "comentario":          comentario,
            })
            st.success("Respuesta registrada — ¡gracias por tu feedback!")
            st.balloons()

    if FEEDBACK_PATH.exists():
        st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)
        with open(FEEDBACK_PATH, "rb") as f:
            st.download_button(
                "Descargar respuestas (CSV)",
                data=f,
                file_name="feedback_runnaing.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<hr style='margin-top:2.5rem;'>
<p style='text-align:center;color:#3f3f46;font-size:0.75rem;padding:0.5rem 0 1rem 0;'>
    RunnAing &nbsp;·&nbsp; TFM UNIR 2026 &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; scikit-learn &nbsp;·&nbsp; Streamlit
    &nbsp;·&nbsp; TRIMP: Banister (1991) &nbsp;·&nbsp; ACWR: Gabbett (2016) &nbsp;·&nbsp; FitRec: Ni et al. (2019)
</p>
""", unsafe_allow_html=True)
