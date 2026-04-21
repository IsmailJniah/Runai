"""
features.py — Construcción de features por sesión para RunnAing TFM.

Features GPS (9):
  1. duration_min       — duración de la sesión (minutos)
  2. distance_km        — distancia total recorrida (km)
  3. speed_mean         — velocidad media (km/h)
  4. speed_max          — velocidad máxima (km/h)
  5. speed_std          — desviación estándar de la velocidad instantánea
  6. pace_mean          — ritmo medio (min/km)  =  60 / speed_mean
  7. elevation_gain     — desnivel positivo acumulado (m)
  8. altitude_mean      — altitud media (m)
  9. grade_factor       — factor de pendiente  =  elevation_gain / distance_km

Features cardíacas (4):
  10. hr_mean           — FC media de la sesión (bpm)
  11. hr_max            — FC máxima de la sesión (bpm)
  12. hr_min            — FC mínima de la sesión (bpm)
  13. hrv_estimate      — HRV estimado: std(RR_intervals), donde RR = 60/FC_inst
                          (Shcherbina et al., 2017)

NOTA: hr_mean, hr_max, hr_min y hrv_estimate se calculan a partir de la FC
instantánea. El target TRIMP también usa FC, pero por rutas distintas
(FC media ponderada vs. distribución instantánea), por lo que la correlación
es moderada, no perfecta. Si se desea un modelo puramente externo (sin FC),
usar FEATURE_NAMES_GPS únicamente.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Contrato público: todos los nombres de columna de salida
FEATURE_NAMES_GPS: list[str] = [
    "duration_min",
    "distance_km",
    "speed_mean",
    "speed_max",
    "speed_std",
    "pace_mean",
    "elevation_gain",
    "altitude_mean",
    "grade_factor",
]

FEATURE_NAMES_HR: list[str] = [
    "hr_mean",
    "hr_max",
    "hr_min",
    "hrv_estimate",
]

FEATURE_NAMES: list[str] = FEATURE_NAMES_GPS


# ---------------------------------------------------------------------------
# Cálculo de features GPS
# ---------------------------------------------------------------------------

def compute_gps_features(
    timestamps: Sequence[float],
    latitudes: Sequence[float],
    longitudes: Sequence[float],
    altitudes: Sequence[float],
) -> dict:
    """
    Calcula las 9 features GPS a partir de las series temporales de una sesión.

    Parameters
    ----------
    timestamps : epoch-segundos (o milisegundos).
    latitudes, longitudes : grados decimales.
    altitudes : metros sobre el nivel del mar.

    Returns
    -------
    dict con claves == FEATURE_NAMES_GPS.
    """
    ts = np.asarray(timestamps, dtype=float)
    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    alt = np.asarray(altitudes, dtype=float)

    mask = ~(np.isnan(ts) | np.isnan(lat) | np.isnan(lon) | np.isnan(alt))
    ts, lat, lon, alt = ts[mask], lat[mask], lon[mask], alt[mask]

    n = len(ts)
    if n < 2:
        return {k: np.nan for k in FEATURE_NAMES_GPS}

    # Normalizar timestamps a segundos si vienen en milisegundos
    if ts[-1] - ts[0] > 1e8:
        ts = ts / 1000.0

    duration_s = ts[-1] - ts[0]
    duration_min = duration_s / 60.0

    seg_dist_km = _haversine_consecutive(lat, lon)
    distance_km = float(np.sum(seg_dist_km))

    dt_s = np.diff(ts)
    dt_s = np.where(dt_s <= 0, np.nan, dt_s)
    speed_kph = seg_dist_km / (dt_s / 3600.0)
    speed_kph = speed_kph[~np.isnan(speed_kph)]
    speed_kph = speed_kph[speed_kph < 60.0]  # >60 km/h no es running

    speed_mean = float(np.nanmean(speed_kph)) if len(speed_kph) > 0 else np.nan
    speed_max = float(np.nanmax(speed_kph)) if len(speed_kph) > 0 else np.nan
    speed_std = float(np.nanstd(speed_kph)) if len(speed_kph) > 1 else 0.0
    pace_mean = (60.0 / speed_mean) if (speed_mean and speed_mean > 0) else np.nan

    d_alt = np.diff(alt)
    elevation_gain = float(np.sum(d_alt[d_alt > 0]))
    altitude_mean = float(np.nanmean(alt))
    grade_factor = (elevation_gain / distance_km) if distance_km > 0 else 0.0

    return {
        "duration_min": duration_min,
        "distance_km": distance_km,
        "speed_mean": speed_mean,
        "speed_max": speed_max,
        "speed_std": speed_std,
        "pace_mean": pace_mean,
        "elevation_gain": elevation_gain,
        "altitude_mean": altitude_mean,
        "grade_factor": grade_factor,
    }


# ---------------------------------------------------------------------------
# Cálculo de features cardíacas
# ---------------------------------------------------------------------------

def compute_hr_features(heart_rate: Sequence[float]) -> dict:
    """
    Calcula hr_mean, hr_max, hr_min y hrv_estimate a partir de la serie
    instantánea de FC de una sesión.

    HRV estimado = std(RR_intervals), donde RR_interval = 60 / FC_inst (seg).
    Metodología: Shcherbina et al. (2017).

    Parameters
    ----------
    heart_rate : serie de FC instantánea (bpm) a lo largo de la sesión.

    Returns
    -------
    dict con claves == FEATURE_NAMES_HR.
    """
    hr = np.asarray(heart_rate, dtype=float)
    hr = hr[~np.isnan(hr)]
    hr = hr[hr > 0]  # evitar división por cero

    if len(hr) == 0:
        return {k: np.nan for k in FEATURE_NAMES_HR}

    hr_mean = float(np.mean(hr))
    hr_max = float(np.max(hr))
    hr_min = float(np.min(hr))

    # RR en segundos: RR = 60 / FC_inst
    rr_intervals = 60.0 / hr
    hrv_estimate = float(np.std(rr_intervals)) if len(rr_intervals) > 1 else 0.0

    return {
        "hr_mean": hr_mean,
        "hr_max": hr_max,
        "hr_min": hr_min,
        "hrv_estimate": hrv_estimate,
    }


# ---------------------------------------------------------------------------
# Función principal: calcula todas las features por sesión
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame, include_hr: bool = True) -> pd.DataFrame:
    """
    Aplica compute_gps_features (y opcionalmente compute_hr_features) a cada
    sesión y devuelve un DataFrame con FEATURE_NAMES como columnas.

    Espera columnas en df:
        timestamp, latitude, longitude, altitude (listas por timestep)
        heart_rate (lista por timestep, requerido si include_hr=True)

    Parameters
    ----------
    include_hr : bool
        Si True (default), incluye las 4 features cardíacas.
        Si False, devuelve solo las 9 features GPS (útil para ablación).

    Returns
    -------
    pd.DataFrame con columnas FEATURE_NAMES (o FEATURE_NAMES_GPS).
    """
    rows = []
    for _, row in df.iterrows():
        gps_feats = compute_gps_features(
            timestamps=row.get("timestamp", []),
            latitudes=row.get("latitude", []),
            longitudes=row.get("longitude", []),
            altitudes=row.get("altitude", []),
        )
        if include_hr:
            hr_feats = compute_hr_features(row.get("heart_rate", []))
            rows.append({**gps_feats, **hr_feats})
        else:
            rows.append(gps_feats)

    expected_cols = FEATURE_NAMES if include_hr else FEATURE_NAMES_GPS
    return pd.DataFrame(rows, index=df.index, columns=expected_cols)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _haversine_consecutive(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Distancias consecutivas en km usando la fórmula de Haversine."""
    R = 6371.0
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = np.diff(lat_r)
    dlon = np.diff(lon_r)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r[:-1]) * np.cos(lat_r[1:]) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


_HR_TERMS = ["hr", "heart_rate", "fc", "trimp", "hrv"]


def assert_no_hr_leakage(X: pd.DataFrame) -> None:
    leaking_cols = [
        col for col in X.columns
        if any(term in col.lower() for term in _HR_TERMS)
    ]
    assert not leaking_cols, (
        f"BUG CRÍTICO — columnas con HR/FC/target detectadas en X: {leaking_cols}"
    )


# Alias retrocompatible
compute_features_from_series = compute_gps_features
