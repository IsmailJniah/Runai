"""
trimp.py — Cálculo del TRIMP de Banister (1991) diferenciado por sexo.

Fórmula:
    TRIMP = D * ΔHR_ratio * 0.64 * exp(b * ΔHR_ratio)

Donde:
    D           = duración de la sesión en minutos
    ΔHR_ratio   = (HR_media - HR_reposo) / (HR_max - HR_reposo)
    b           = 1.92 (hombres) | 1.67 (mujeres)   [Banister 1991]

Referencia:
    Banister, E. W. (1991). Modeling elite athletic performance. En H. J. Green,
    J. D. McDougal, & H. Wenger (Eds.), Physiological testing of the high-
    performance athlete (pp. 403–424). Human Kinetics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Constantes de la fórmula de Banister
B_MALE = 1.92
B_FEMALE = 1.67
K = 0.64  # constante de ponderación

# Valores poblacionales de referencia cuando no hay datos de usuario
DEFAULT_HR_REST = 60.0   # bpm
DEFAULT_HR_MAX = 185.0   # bpm (≈ 220 − 35 años, valor conservador)


def banister_trimp(
    duration_min: float,
    hr_mean: float,
    gender: str,
    hr_rest: float = DEFAULT_HR_REST,
    hr_max: float = DEFAULT_HR_MAX,
) -> float:
    """
    Calcula el TRIMP de Banister para una única sesión.

    Parameters
    ----------
    duration_min : float
        Duración de la sesión en minutos.
    hr_mean : float
        Frecuencia cardíaca media durante la sesión (bpm).
    gender : str
        'male' / 'M' / 'h' → b=1.92; cualquier otro valor → b=1.67.
    hr_rest : float
        FC en reposo del sujeto (bpm). Default=60.
    hr_max : float
        FC máxima del sujeto (bpm). Default=185.

    Returns
    -------
    float
        Valor TRIMP. Devuelve NaN si los datos son inválidos.
    """
    if not _valid_inputs(duration_min, hr_mean, hr_rest, hr_max):
        return np.nan

    b = _select_b(gender)
    delta_hr_ratio = (hr_mean - hr_rest) / (hr_max - hr_rest)

    # ΔHR_ratio debe estar en (0, 1]; clamp por seguridad
    delta_hr_ratio = np.clip(delta_hr_ratio, 1e-6, 1.0)

    return duration_min * delta_hr_ratio * K * np.exp(b * delta_hr_ratio)


def compute_trimp_column(
    df: pd.DataFrame,
    duration_col: str = "duration_min",
    hr_mean_col: str = "hr_mean",
    gender_col: str = "gender",
    hr_rest_col: str | None = None,
    hr_max_col: str | None = None,
) -> pd.Series:
    """
    Vectorizado: aplica banister_trimp a cada fila de un DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con al menos las columnas indicadas.
    hr_rest_col / hr_max_col : str | None
        Si None, se usan los valores por defecto DEFAULT_HR_REST / DEFAULT_HR_MAX.

    Returns
    -------
    pd.Series
        TRIMP por sesión, índice alineado con df.
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


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _select_b(gender: str) -> float:
    """Devuelve el coeficiente b según sexo."""
    if isinstance(gender, str) and gender.strip().lower() in ("male", "m", "h", "hombre"):
        return B_MALE
    return B_FEMALE


def _valid_inputs(duration_min, hr_mean, hr_rest, hr_max) -> bool:
    """Comprueba que los inputs sean numéricos y fisiológicamente sensatos."""
    try:
        vals = [float(v) for v in (duration_min, hr_mean, hr_rest, hr_max)]
    except (TypeError, ValueError):
        return False
    d, hr, rest, mx = vals
    return (
        d > 0
        and hr > 0
        and rest > 0
        and mx > rest
        and hr >= rest  # FC media ≥ FC reposo
    )
