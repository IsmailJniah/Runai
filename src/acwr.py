"""
acwr.py — Cálculo de ACWR (Acute:Chronic Workload Ratio) según Gabbett (2016).

Definición:
  ACWR = carga_aguda_7d / carga_crónica_28d

  donde carga = suma de TRIMP en la ventana temporal correspondiente.

Zonas de riesgo (Gabbett, 2016):
  < 0.80   → carga insuficiente / descanso
  0.80–1.30 → zona óptima ("sweet spot")
  1.30–1.50 → zona de precaución
  > 1.50   → zona de alto riesgo

Cohorte elegible: usuarios con span temporal ≥ 28 días de datos.

Referencia:
  Gabbett, T. J. (2016). The training—injury prevention paradox: should
  athletes be training smarter and harder? British Journal of Sports
  Medicine, 50(5), 273–280.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ventanas temporales
ACUTE_DAYS = 7
CHRONIC_DAYS = 28
MIN_SPAN_DAYS = 28  # mínimo span para incluir usuario en la cohorte

# Límites de zonas Gabbett
ZONE_LOW = 0.80
ZONE_HIGH_OPT = 1.30
ZONE_HIGH_RISK = 1.50

ZONE_LABELS = {
    "insuficiente": f"ACWR < {ZONE_LOW}",
    "optima": f"{ZONE_LOW} ≤ ACWR ≤ {ZONE_HIGH_OPT}",
    "precaucion": f"{ZONE_HIGH_OPT} < ACWR ≤ {ZONE_HIGH_RISK}",
    "riesgo": f"ACWR > {ZONE_HIGH_RISK}",
}


def filter_eligible_users(
    df: pd.DataFrame,
    date_col: str = "date",
    user_col: str = "userId",
    min_span_days: int = MIN_SPAN_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filtra usuarios con span temporal ≥ min_span_days.

    Returns
    -------
    (df_eligible, cohorte_stats)
        df_eligible : sesiones de los usuarios aptos.
        cohorte_stats : DataFrame con span_days por usuario.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    spans = (
        df.groupby(user_col)[date_col]
        .agg(span_days=lambda x: (x.max() - x.min()).days)
        .reset_index()
    )
    spans["eligible"] = spans["span_days"] >= min_span_days

    eligible_users = spans.loc[spans["eligible"], user_col]
    df_eligible = df[df[user_col].isin(eligible_users)].copy()

    logger.info(
        "Usuarios totales: %d | Con span ≥ %d días: %d (%.1f%%)",
        spans.shape[0],
        min_span_days,
        eligible_users.shape[0],
        100 * eligible_users.shape[0] / spans.shape[0] if spans.shape[0] > 0 else 0,
    )
    return df_eligible, spans


def compute_acwr_per_user(
    df_user: pd.DataFrame,
    date_col: str = "date",
    trimp_col: str = "trimp",
    acute_days: int = ACUTE_DAYS,
    chronic_days: int = CHRONIC_DAYS,
) -> pd.DataFrame:
    """
    Calcula el ACWR diario para un único usuario.

    Parameters
    ----------
    df_user : sesiones de un único usuario, ordenadas por fecha.
    date_col : columna con la fecha de la sesión.
    trimp_col : columna con el TRIMP de Banister por sesión.

    Returns
    -------
    pd.DataFrame diario con columnas: date, trimp_acute, trimp_chronic, acwr, zone.
    """
    df_user = df_user.copy()
    df_user[date_col] = pd.to_datetime(df_user[date_col])
    df_user = df_user.sort_values(date_col)

    # Resamplear a suma diaria de TRIMP
    daily = (
        df_user.set_index(date_col)[trimp_col]
        .resample("D")
        .sum()
        .fillna(0)
    )

    # Rellenar el rango completo de fechas con 0 (días sin sesión)
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx, fill_value=0)

    # Ventanas deslizantes
    acute = daily.rolling(window=acute_days, min_periods=1).sum()
    chronic = daily.rolling(window=chronic_days, min_periods=1).sum()

    acwr = acute / chronic.replace(0, np.nan)

    result = pd.DataFrame({
        "date": daily.index,
        "trimp_acute": acute.values,
        "trimp_chronic": chronic.values,
        "acwr": acwr.values,
    })
    result["zone"] = result["acwr"].apply(_assign_zone)
    return result.reset_index(drop=True)


def compute_acwr_all_users(
    df: pd.DataFrame,
    date_col: str = "date",
    trimp_col: str = "trimp",
    user_col: str = "userId",
) -> pd.DataFrame:
    """
    Aplica compute_acwr_per_user a todos los usuarios del DataFrame.

    Returns
    -------
    pd.DataFrame con columnas: userId, date, trimp_acute, trimp_chronic, acwr, zone.
    """
    results = []
    for user_id, group in df.groupby(user_col):
        user_acwr = compute_acwr_per_user(group, date_col=date_col, trimp_col=trimp_col)
        user_acwr.insert(0, user_col, user_id)
        results.append(user_acwr)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def zone_distribution(acwr_df: pd.DataFrame, acwr_col: str = "acwr") -> pd.DataFrame:
    """
    Calcula la distribución de sesiones/días por zona Gabbett.

    Returns
    -------
    pd.DataFrame con columnas: zone, n, pct.
    """
    zones = acwr_df[acwr_col].dropna().apply(_assign_zone)
    counts = zones.value_counts().rename_axis("zone").reset_index(name="n")
    counts["pct"] = 100 * counts["n"] / counts["n"].sum()
    return counts


def _assign_zone(acwr: float) -> str:
    """Asigna zona Gabbett a un valor de ACWR."""
    if pd.isna(acwr):
        return "sin_dato"
    if acwr < ZONE_LOW:
        return "insuficiente"
    if acwr <= ZONE_HIGH_OPT:
        return "optima"
    if acwr <= ZONE_HIGH_RISK:
        return "precaucion"
    return "riesgo"
