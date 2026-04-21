"""
trimp.py — Cálculo del TRIMP de Banister (1991) diferenciado por sexo.

Fórmula incremental (preferida, más precisa):
    TRIMP = Σ_t ( Δt · HRR_t · e^(b · HRR_t) )

Fórmula de sesión agregada (legacy, cuando solo hay HR media):
    TRIMP = D · HRR_medio · e^(b · HRR_medio)

Donde:
    Δt          = fragmento de tiempo en minutos
    HRR_t       = (FC_inst_t - FC_reposo) / (FC_max - FC_reposo)  [fracción 0-1]
    D           = duración total de la sesión en minutos
    b           = 1.92 (hombres) | 1.67 (mujeres)   [Banister 1991]

Referencia:
    Banister, E. W. (1991). Modeling elite athletic performance. En H. J. Green,
    J. D. McDougal, & H. Wenger (Eds.), Physiological testing of the high-
    performance athlete (pp. 403–424). Human Kinetics.

Supuesto documentado:
    FitRec no proporciona FC_reposo ni FC_max por usuario.
    Se usan valores poblacionales: HR_rest=60 bpm, HR_max=185 bpm.
    (HR_max ≈ 220 − 35 años, valor conservador para adultos recreativos)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

B_MALE = 1.92
B_FEMALE = 1.67

DEFAULT_HR_REST = 60.0   # bpm
DEFAULT_HR_MAX = 185.0   # bpm


def banister_trimp_incremental(
    heart_rate: list | np.ndarray,
    timestamps: list | np.ndarray,
    gender: str,
    hr_rest: float = DEFAULT_HR_REST,
    hr_max: float = DEFAULT_HR_MAX,
) -> float:
    """
    Calcula el TRIMP de Banister usando la FC instantánea en cada timestep.

    Esta versión es más precisa que la de sesión agregada porque captura la
    distribución real de la intensidad a lo largo de la sesión.

    Parameters
    ----------
    heart_rate : array de FC instantánea (bpm) para cada timestep.
    timestamps : array de epoch-segundos (o milisegundos) correspondientes.
    gender : str  — 'male'/'M'/'h' → b=1.92; cualquier otro → b=1.67.
    hr_rest : float  — FC en reposo (bpm). Default=60.
    hr_max : float   — FC máxima (bpm). Default=185.

    Returns
    -------
    float — TRIMP de la sesión. NaN si datos insuficientes.
    """
    hr = np.asarray(heart_rate, dtype=float)
    ts = np.asarray(timestamps, dtype=float)

    # Alinear longitudes
    min_len = min(len(hr), len(ts))
    hr = hr[:min_len]
    ts = ts[:min_len]

    # Eliminar NaN
    valid = ~(np.isnan(hr) | np.isnan(ts))
    hr = hr[valid]
    ts = ts[valid]

    if len(hr) < 2:
        return np.nan

    # Normalizar timestamps a segundos
    if ts[-1] - ts[0] > 1e8:
        ts = ts / 1000.0

    # Intervalos de tiempo en minutos (entre samples consecutivos)
    dt_min = np.diff(ts) / 60.0
    dt_min = np.where(dt_min <= 0, np.nan, dt_min)

    # FC instantánea en el intervalo (valor al inicio del segmento)
    hr_seg = hr[:-1]

    # HRR = fracción de reserva cardíaca
    hrr = (hr_seg - hr_rest) / (hr_max - hr_rest)
    hrr = np.clip(hrr, 1e-6, 1.0)

    b = _select_b(gender)

    # TRIMP = Σ Δt · HRR · e^(b·HRR)
    contributions = dt_min * hrr * np.exp(b * hrr)
    valid_contribs = contributions[~np.isnan(contributions)]

    return float(np.sum(valid_contribs)) if len(valid_contribs) > 0 else np.nan


def banister_trimp(
    duration_min: float,
    hr_mean: float,
    gender: str,
    hr_rest: float = DEFAULT_HR_REST,
    hr_max: float = DEFAULT_HR_MAX,
) -> float:
    """
    Calcula el TRIMP de Banister a partir de la FC media (versión agregada).

    Usar cuando no se dispone de la serie temporal de FC.
    Para mayor precisión, usar banister_trimp_incremental().

    Parameters
    ----------
    duration_min : duración de la sesión en minutos.
    hr_mean : FC media durante la sesión (bpm).
    gender : 'male'/'M'/'h' → b=1.92; cualquier otro → b=1.67.
    hr_rest : FC en reposo (bpm). Default=60.
    hr_max : FC máxima (bpm). Default=185.

    Returns
    -------
    float — TRIMP. NaN si los datos son inválidos.
    """
    if not _valid_inputs(duration_min, hr_mean, hr_rest, hr_max):
        return np.nan

    b = _select_b(gender)
    hrr = (hr_mean - hr_rest) / (hr_max - hr_rest)
    hrr = np.clip(hrr, 1e-6, 1.0)

    return float(duration_min * hrr * 0.64 * np.exp(b * hrr))


def compute_trimp_column(
    df: pd.DataFrame,
    duration_col: str = "duration_min",
    hr_mean_col: str = "hr_mean",
    gender_col: str = "gender",
    hr_rest_col: str | None = None,
    hr_max_col: str | None = None,
) -> pd.Series:
    """
    Vectorizado: aplica banister_trimp a cada fila (versión sesión agregada).

    Para la versión incremental, usar compute_trimp_incremental_column().
    """
    hr_rest = df[hr_rest_col] if hr_rest_col and hr_rest_col in df.columns else DEFAULT_HR_REST
    hr_max = df[hr_max_col] if hr_max_col and hr_max_col in df.columns else DEFAULT_HR_MAX

    return pd.Series(
        [
            banister_trimp(
                duration_min=row[duration_col],
                hr_mean=row[hr_mean_col],
                gender=row[gender_col],
                hr_rest=hr_rest if np.isscalar(hr_rest) else hr_rest.iloc[i],
                hr_max=hr_max if np.isscalar(hr_max) else hr_max.iloc[i],
            )
            for i, (_, row) in enumerate(df.iterrows())
        ],
        index=df.index,
        name="trimp",
    )


def compute_trimp_incremental_column(
    df: pd.DataFrame,
    hr_col: str = "heart_rate",
    ts_col: str = "timestamp",
    gender_col: str = "gender",
    hr_rest_col: str | None = None,
    hr_max_col: str | None = None,
) -> pd.Series:
    """
    Vectorizado: aplica banister_trimp_incremental a cada fila.

    Parameters
    ----------
    df : DataFrame con columnas de listas (heart_rate, timestamp, gender).

    Returns
    -------
    pd.Series con TRIMP incremental por sesión.
    """
    hr_rest_default = DEFAULT_HR_REST
    hr_max_default = DEFAULT_HR_MAX

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        hr_rest = (
            row[hr_rest_col] if hr_rest_col and hr_rest_col in df.columns
            else hr_rest_default
        )
        hr_max = (
            row[hr_max_col] if hr_max_col and hr_max_col in df.columns
            else hr_max_default
        )
        results.append(
            banister_trimp_incremental(
                heart_rate=row.get(hr_col, []),
                timestamps=row.get(ts_col, []),
                gender=row.get(gender_col, ""),
                hr_rest=hr_rest,
                hr_max=hr_max,
            )
        )

    return pd.Series(results, index=df.index, name="trimp")


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _select_b(gender: str) -> float:
    if isinstance(gender, str) and gender.strip().lower() in ("male", "m", "h", "hombre"):
        return B_MALE
    return B_FEMALE


def _valid_inputs(duration_min, hr_mean, hr_rest, hr_max) -> bool:
    try:
        vals = [float(v) for v in (duration_min, hr_mean, hr_rest, hr_max)]
    except (TypeError, ValueError):
        return False
    d, hr, rest, mx = vals
    return d > 0 and hr > 0 and rest > 0 and mx > rest and hr >= rest
