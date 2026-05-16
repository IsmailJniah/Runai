"""
app.py — Dashboard Streamlit de RunnAing TFM.

Flujo:
  1. Carga CSV de sesiones (duración, distancia, velocidad, desnivel, ritmo, sexo).
  2. Predice TRIMP por sesión con el modelo ganador (best_model.pkl).
  3. Muestra ACWR diario con zonas de riesgo (Gabbett, 2016).
  4. Explica la predicción individual con SHAP (waterfall plot).

Arrancar:
    streamlit run app.py

El archivo best_model.pkl debe existir en models/ (generado por el notebook 05).
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Módulos del proyecto
from src.features import FEATURE_NAMES_GPS, assert_no_hr_leakage
from src.acwr import (
    filter_eligible_users,
    compute_acwr_all_users,
    zone_distribution,
    ZONE_LOW, ZONE_HIGH_OPT, ZONE_HIGH_RISK,
)

# SHAP opcional
try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MODELS_DIR = Path("models")
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

ZONE_COLORS = {
    "insuficiente": "#3498db",
    "optima": "#2ecc71",
    "precaucion": "#f39c12",
    "riesgo": "#e74c3c",
    "sin_dato": "#bdc3c7",
}

EXPECTED_COLS = {
    "duration_min": "Duración (min)",
    "distance_km": "Distancia (km)",
    "speed_mean": "Velocidad media (km/h)",
    "speed_std": "Variabilidad velocidad (km/h)",
    "pace_mean": "Ritmo medio (min/km)",
    "elevation_gain": "Desnivel acumulado (m)",
    "gender": "Sexo (male/female)",
    "date": "Fecha (YYYY-MM-DD)",
    "userId": "ID de usuario",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    if not BEST_MODEL_PATH.exists():
        return None
    with open(BEST_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye la matriz X a partir del CSV cargado por el usuario."""
    needed = ["duration_min", "distance_km", "speed_mean", "speed_std",
              "pace_mean", "elevation_gain"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error(f"Columnas faltantes en el CSV: {missing}")
        st.stop()

    # Derivar columnas opcionales si no están
    X = df[needed].copy()

    # speed_max y altitude_mean opcionales (relleno con medias si faltan)
    X["speed_max"] = df["speed_max"] if "speed_max" in df.columns else X["speed_mean"] * 1.3
    X["altitude_mean"] = df["altitude_mean"] if "altitude_mean" in df.columns else 400.0
    X["grade_factor"] = (X["elevation_gain"] / X["distance_km"].replace(0, np.nan)).fillna(0)

    X = X[FEATURE_NAMES_GPS]
    assert_no_hr_leakage(X)
    return X


def plot_acwr_timeline(acwr_user: pd.DataFrame, user_id: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    dates = pd.to_datetime(acwr_user["date"])
    acwr = acwr_user["acwr"]

    ax.plot(dates, acwr, color="#2c3e50", linewidth=1.4, zorder=3)
    ax.axhspan(ZONE_LOW, ZONE_HIGH_OPT, alpha=0.15, color=ZONE_COLORS["optima"], label="Zona óptima")
    ax.axhspan(ZONE_HIGH_OPT, ZONE_HIGH_RISK, alpha=0.15, color=ZONE_COLORS["precaucion"], label="Precaución")
    ax.axhline(ZONE_LOW, color=ZONE_COLORS["optima"], linewidth=0.8, linestyle="--")
    ax.axhline(ZONE_HIGH_OPT, color=ZONE_COLORS["precaucion"], linewidth=0.8, linestyle="--")
    ax.axhline(ZONE_HIGH_RISK, color=ZONE_COLORS["riesgo"], linewidth=0.8, linestyle="--")

    ax.set_title(f"ACWR diario — usuario {user_id}", fontsize=12)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("ACWR")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_shap_waterfall_single(model, X_row: pd.DataFrame) -> plt.Figure | None:
    if not _HAS_SHAP:
        return None
    underlying = model
    if hasattr(model, "named_steps"):
        underlying = model.named_steps[list(model.named_steps)[-1]]
    try:
        explainer = shap.TreeExplainer(underlying)
        sv = explainer(X_row)
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(sv[0], show=False)
        fig.tight_layout()
        return fig
    except Exception:
        return None


def zone_badge(zone: str) -> str:
    emojis = {
        "insuficiente": "🔵 Insuficiente",
        "optima": "🟢 Óptima",
        "precaucion": "🟡 Precaución",
        "riesgo": "🔴 Riesgo elevado",
        "sin_dato": "⚪ Sin dato",
    }
    return emojis.get(zone, zone)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RunnAing — Dashboard TRIMP & ACWR",
    page_icon="🏃",
    layout="wide",
)

st.title("🏃 RunnAing — Predicción de TRIMP y ACWR")
st.caption(
    "TFM: *Predicción de carga interna en corredores populares como proxy de fatiga acumulada* | "
    "Modelo: Banister (1991) | Dataset: FitRec Endomondo"
)

# ---- Sidebar ----
st.sidebar.header("Configuración")
st.sidebar.markdown(
    "**Columnas mínimas del CSV:**\n"
    + "\n".join(f"- `{k}`: {v}" for k, v in EXPECTED_COLS.items())
)

model = load_model()
if model is None:
    st.sidebar.warning(
        "⚠️ `models/best_model.pkl` no encontrado.\n"
        "Ejecuta el notebook 05 para generar el modelo."
    )

# ---- Carga de datos ----
st.header("1. Cargar sesiones")
uploaded = st.file_uploader(
    "Sube un CSV con las sesiones de carrera", type=["csv"],
    help="Una fila por sesión. Ver columnas requeridas en la barra lateral."
)

if uploaded is None:
    st.info("Sube un CSV para comenzar. Formato: una fila por sesión.")
    st.stop()

df = pd.read_csv(uploaded)
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

st.success(f"✅ {len(df):,} sesiones cargadas — {df['userId'].nunique() if 'userId' in df.columns else '?'} usuarios")
with st.expander("Vista previa del CSV"):
    st.dataframe(df.head(10), use_container_width=True)

# ---- Predicción TRIMP ----
st.header("2. TRIMP predicho por sesión")

if model is None:
    st.warning("Modelo no cargado. No es posible predecir TRIMP.")
    st.stop()

X = prepare_features(df)
df["trimp_pred"] = model.predict(X)

col1, col2, col3 = st.columns(3)
col1.metric("TRIMP medio", f"{df['trimp_pred'].mean():.1f} u.a.")
col2.metric("TRIMP mínimo", f"{df['trimp_pred'].min():.1f} u.a.")
col3.metric("TRIMP máximo", f"{df['trimp_pred'].max():.1f} u.a.")

fig_trimp, ax_trimp = plt.subplots(figsize=(9, 3))
ax_trimp.hist(df["trimp_pred"].dropna(), bins=40, color="#2980b9", edgecolor="white", alpha=0.85)
ax_trimp.set_xlabel("TRIMP predicho (u.a.)")
ax_trimp.set_ylabel("Sesiones")
ax_trimp.set_title("Distribución del TRIMP predicho")
ax_trimp.grid(alpha=0.3)
fig_trimp.tight_layout()
st.pyplot(fig_trimp)
plt.close(fig_trimp)

with st.expander("Tabla TRIMP por sesión"):
    cols_show = ["userId", "date", "duration_min", "distance_km", "speed_mean", "trimp_pred"]
    cols_show = [c for c in cols_show if c in df.columns]
    st.dataframe(df[cols_show].round(2), use_container_width=True)

# ---- ACWR ----
st.header("3. ACWR — Carga Aguda:Crónica (Gabbett, 2016)")

if "userId" not in df.columns or "date" not in df.columns:
    st.warning("El CSV no tiene columnas `userId` y `date`. No se puede calcular el ACWR.")
    st.stop()

df_trimp = df.rename(columns={"trimp_pred": "trimp"}) if "trimp_pred" in df.columns else df

# Cohorte elegible
df_cohort, span_stats = filter_eligible_users(df_trimp, date_col="date", user_col="userId")
n_total = span_stats["userId"].nunique()
n_eligible = df_cohort["userId"].nunique()

st.markdown(
    f"**Cohorte elegible** (span ≥ 28 días): **{n_eligible:,}** de {n_total:,} usuarios "
    f"({100*n_eligible/n_total:.1f}%)"
)

if n_eligible == 0:
    st.warning("Ningún usuario tiene historial ≥ 28 días. ACWR no calculable.")
    st.stop()

acwr_df = compute_acwr_all_users(df_cohort, date_col="date", trimp_col="trimp", user_col="userId")

# Distribución de zonas
dist = zone_distribution(acwr_df)
cols_zone = st.columns(4)
zone_names = ["insuficiente", "optima", "precaucion", "riesgo"]
for col, zname in zip(cols_zone, zone_names):
    row = dist[dist["zone"] == zname]
    pct = row["pct"].values[0] if len(row) > 0 else 0.0
    col.metric(zone_badge(zname), f"{pct:.1f}%")

# Selección de usuario para timeline
user_ids = sorted(acwr_df["userId"].unique())
selected_user = st.selectbox("Selecciona un usuario para ver su evolución ACWR", user_ids)

user_acwr = acwr_df[acwr_df["userId"] == selected_user].copy()
last_zone = user_acwr["zone"].dropna().iloc[-1] if len(user_acwr) > 0 else "sin_dato"
st.markdown(f"**Zona ACWR actual:** {zone_badge(last_zone)}")

fig_acwr = plot_acwr_timeline(user_acwr, selected_user)
st.pyplot(fig_acwr)
plt.close(fig_acwr)

# ---- SHAP individual ----
st.header("4. Explicación SHAP — sesión individual")

if not _HAS_SHAP:
    st.warning("SHAP no instalado. Ejecuta `pip install shap` para activar este panel.")
else:
    user_sessions = df[df["userId"] == selected_user] if "userId" in df.columns else df
    if len(user_sessions) == 0:
        st.info("No hay sesiones para el usuario seleccionado.")
    else:
        session_idx = st.slider(
            "Sesión a explicar (índice)",
            min_value=0,
            max_value=len(user_sessions) - 1,
            value=0,
        )
        row = user_sessions.iloc[[session_idx]]
        X_row = prepare_features(row)

        st.markdown(f"**TRIMP predicho para esta sesión:** `{df.loc[row.index[0], 'trimp_pred']:.2f}` u.a.")

        with st.spinner("Calculando SHAP..."):
            fig_wf = plot_shap_waterfall_single(model, X_row)

        if fig_wf is not None:
            st.pyplot(fig_wf)
            plt.close(fig_wf)
        else:
            st.warning("No se pudo generar el waterfall SHAP (modelo no compatible con TreeExplainer).")

        st.markdown(
            "**Nota:** El waterfall plot muestra la contribución de cada variable "
            "externa al TRIMP predicho. Variables como `duration_min`, `elevation_gain` "
            "y `speed_mean` suelen dominar (Banister, 1991; Gabbett, 2016)."
        )

# ---- Footer ----
st.markdown("---")
st.caption(
    "RunnAing TFM · Python 3.11 · scikit-learn · XGBoost · LightGBM · Optuna · SHAP · Streamlit"
)
