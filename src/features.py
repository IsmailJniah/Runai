"""
features.py — Construcción de las 9 features externas (sin FC) para RunnAing.

Features:
  1. duration_min       — duración de la sesión (minutos)
  2. distance_km        — distancia total recorrida (km)
  3. speed_mean         — velocidad media (km/h)
  4. speed_std          — desviación estándar de la velocidad instantánea
  5. pace_mean          — ritmo medio (min/km)  =  60 / speed_mean
  6. elevation_gain     — desnivel positivo acumulado (m)
  7. elevation_loss     — desnivel negativo acumulado (m, valor positivo)
  8. altitude_mean      — altitud media (m)
  9. grade_factor       — factor de pendiente  =  elevation_gain / distance_km

NINGUNA de estas features contiene FC ni derivados de FC.
Al final del módulo hay un assert de guardia que falla si se detecta
cualquier nombre relacionado con la frecuencia cardíaca.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Nombres de columna de salida (contrato público)
FEATURE_NAMES: list[str] = [
    "duration_min",
    "speed_mean",
    "speed_std",
    "pace_mean",
    "distance_km",
    "elevation_gain",
    "elevation_loss",
    "altitude_mean",
    "grade_factor",
]

# Términos relacionados con FC que NO deben aparecer en X
_HR_KEYWORDS = frozenset({
    "heart_rate", "hr", "fc", "bpm", "heart", "pulse",
    "trimp",  # el target tampoco debe ir en X
})


# ---------------------------------------------------------------------------
# Cálculo de features desde series temporales GPS
# ---------------------------------------------------------------------------

def compute_features_from_series(
    timestamps: Sequence[float],
    latitudes: Sequence[float],
    longitudes: Sequence[float],
    altitudes: Sequence[float],
) -> dict:
    """
    Calcula las 9 features a partir de las series temporales GPS de una sesión.

    Parameters
    ----------
    timestamps : secuencia de epoch-segundos (o milisegundos).
    latitudes, longitudes : grados decimales.
    altitudes : metros sobre el nivel del mar.

    Returns
    -------
    dict con claves == FEATURE_NAMES.
    """
    ts = np.asarray(timestamps, dtype=float)
    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    alt = np.asarray(altitudes, dtype=float)

    # Eliminar NaN en cualquier canal
    mask = ~(np.isnan(ts) | np.isnan(lat) | np.isnan(lon) | np.isnan(alt))
    ts, lat, lon, alt = ts[mask], lat[mask], lon[mask], alt[mask]

    n = len(ts)
    if n < 2:
        return {k: np.nan for k in FEATURE_NAMES}

    # Normalizar timestamps a segundos si vienen en milisegundos
    if ts[-1] - ts[0] > 1e8:
        ts = ts / 1000.0

    # --- Duración ---
    duration_s = ts[-1] - ts[0]
    duration_min = duration_s / 60.0

    # --- Distancia por segmentos (Haversine) ---
    seg_dist_km = _haversine_consecutive(lat, lon)
    distance_km = float(np.sum(seg_dist_km))

    # --- Velocidad instantánea (km/h) ---
    dt_s = np.diff(ts)
    dt_s = np.where(dt_s <= 0, np.nan, dt_s)
    speed_kph = (seg_dist_km / (dt_s / 3600.0))
    speed_kph = speed_kph[~np.isnan(speed_kph)]
    speed_kph = speed_kph[speed_kph < 60.0]  # filtro de outliers (>60 km/h no es running)

    speed_mean = float(np.nanmean(speed_kph)) if len(speed_kph) > 0 else np.nan
    speed_std = float(np.nanstd(speed_kph)) if len(speed_kph) > 1 else 0.0
    pace_mean = (60.0 / speed_mean) if (speed_mean and speed_mean > 0) else np.nan

    # --- Desniveles ---
    d_alt = np.diff(alt)
    elevation_gain = float(np.sum(d_alt[d_alt > 0]))
    elevation_loss = float(np.abs(np.sum(d_alt[d_alt < 0])))
    altitude_mean = float(np.nanmean(alt))

    # --- Factor de pendiente ---
    grade_factor = (elevation_gain / distance_km) if distance_km > 0 else 0.0

    return {
        "duration_min": duration_min,
        "distance_km": distance_km,
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "pace_mean": pace_mean,
        "elevation_gain": elevation_gain,
        "elevation_loss": elevation_loss,
        "altitude_mean": altitude_mean,
        "grade_factor": grade_factor,
    }


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica compute_features_from_series a cada sesión del DataFrame FitRec
    y devuelve un DataFrame con FEATURE_NAMES como columnas.

    Espera columnas en df: timestamp, latitude, longitude, altitude
    (cada una como lista de valores por timestep).

    Returns
    -------
    pd.DataFrame con columnas FEATURE_NAMES, índice == df.index.
    """
    rows = []
    for _, row in df.iterrows():
        feats = compute_features_from_series(
            timestamps=row.get("timestamp", []),
            latitudes=row.get("latitude", []),
            longitudes=row.get("longitude", []),
            altitudes=row.get("altitude", []),
        )
        rows.append(feats)

    X = pd.DataFrame(rows, index=df.index)
    assert_no_hr_leakage(X)
    return X


# ---------------------------------------------------------------------------
# Guardia anti-leakage de FC
# ---------------------------------------------------------------------------

def assert_no_hr_leakage(X: pd.DataFrame) -> None:
    """
    Falla con AssertionError si cualquier columna de X tiene nombre relacionado
    con la frecuencia cardíaca. Llamar siempre antes de entrenar modelos.
    """
    leaking_cols = [
        col for col in X.columns
        if any(kw in col.lower() for kw in _HR_KEYWORDS)
    ]
    assert not leaking_cols, (
        f"BUG CRÍTICO — Las siguientes columnas contienen FC o derivados "
        f"y NO deben estar en X: {leaking_cols}"
    )


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _haversine_consecutive(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Distancias consecutivas en km usando la fórmula de Haversine.
    Devuelve array de longitud len(lat)-1.
    """
    R = 6371.0  # radio terrestre en km
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = np.diff(lat_r)
    dlon = np.diff(lon_r)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r[:-1]) * np.cos(lat_r[1:]) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
