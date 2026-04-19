"""
models.py — Definición de los 5 modelos de regresión a comparar.

Modelos:
  1. LinearRegression     — baseline
  2. RandomForest
  3. GradientBoosting     (sklearn)
  4. XGBoost
  5. LightGBM

Todos los modelos con random_state=42 donde aplique.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

RANDOM_STATE = 42


def get_linear_regression(**kwargs) -> Pipeline:
    """Regresión Lineal con estandarización (baseline)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression(**kwargs)),
    ])


def get_random_forest(**kwargs) -> RandomForestRegressor:
    kwargs.setdefault("random_state", RANDOM_STATE)
    kwargs.setdefault("n_jobs", -1)
    return RandomForestRegressor(**kwargs)


def get_gradient_boosting(**kwargs) -> GradientBoostingRegressor:
    kwargs.setdefault("random_state", RANDOM_STATE)
    return GradientBoostingRegressor(**kwargs)


def get_xgboost(**kwargs):
    if not _HAS_XGB:
        raise ImportError("xgboost no está instalado. Ejecuta: pip install xgboost==2.0")
    kwargs.setdefault("random_state", RANDOM_STATE)
    kwargs.setdefault("n_jobs", -1)
    return xgb.XGBRegressor(**kwargs)


def get_lightgbm(**kwargs):
    if not _HAS_LGB:
        raise ImportError("lightgbm no está instalado. Ejecuta: pip install lightgbm==4.3")
    kwargs.setdefault("random_state", RANDOM_STATE)
    kwargs.setdefault("n_jobs", -1)
    kwargs.setdefault("verbose", -1)
    return lgb.LGBMRegressor(**kwargs)


# Registro centralizado de modelos disponibles
MODEL_REGISTRY: dict[str, callable] = {
    "LinearRegression": get_linear_regression,
    "RandomForest": get_random_forest,
    "GradientBoosting": get_gradient_boosting,
    "XGBoost": get_xgboost,
    "LightGBM": get_lightgbm,
}


def get_model(name: str, **hyperparams):
    """
    Instancia un modelo por nombre desde el registro.

    Parameters
    ----------
    name : str
        Clave en MODEL_REGISTRY.
    **hyperparams
        Parámetros del modelo.

    Returns
    -------
    Estimador scikit-learn compatible.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo '{name}' no reconocido. Disponibles: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**hyperparams)
