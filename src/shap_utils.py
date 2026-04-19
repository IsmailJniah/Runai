"""
shap_utils.py — Interpretabilidad con SHAP TreeExplainer.

Genera:
  - Summary plot beeswarm (Figura 4.1)
  - Bar plot global de importancia media |SHAP|
  - Waterfall plot de un caso concreto

Referencia:
  Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting
  model predictions. Advances in Neural Information Processing Systems, 30.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

logger = logging.getLogger(__name__)

FIGURES_DIR = Path("outputs/figures")
RANDOM_STATE = 42


def _require_shap():
    if not _HAS_SHAP:
        raise ImportError("shap no está instalado. Ejecuta: pip install shap==0.45")


def compute_shap_values(model, X: pd.DataFrame) -> "shap.Explanation":
    """
    Calcula valores SHAP con TreeExplainer.

    Parameters
    ----------
    model : árbol de decisión entrenado (XGBoost, LightGBM, RF, GB).
    X : pd.DataFrame con las features de evaluación.

    Returns
    -------
    shap.Explanation
    """
    _require_shap()

    # Intentar acceder al modelo subyacente si está dentro de un Pipeline
    underlying = model
    if hasattr(model, "named_steps"):
        step_names = list(model.named_steps.keys())
        underlying = model.named_steps[step_names[-1]]

    explainer = shap.TreeExplainer(underlying)
    shap_values = explainer(X)
    return shap_values


def plot_shap_beeswarm(
    shap_values,
    X: pd.DataFrame,
    max_display: int = 10,
    save_path: str | Path | None = None,
) -> Path:
    """
    Genera el summary plot beeswarm (Figura 4.1).

    Returns
    -------
    Path del archivo guardado.
    """
    _require_shap()

    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "figura_4_1_shap_beeswarm.png"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Figura beeswarm guardada en %s", save_path)
    return save_path


def plot_shap_bar(
    shap_values,
    max_display: int = 10,
    save_path: str | Path | None = None,
) -> Path:
    """Bar plot de importancia global media |SHAP|."""
    _require_shap()

    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "shap_bar_global.png"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Figura bar global guardada en %s", save_path)
    return save_path


def plot_shap_waterfall(
    shap_values,
    sample_idx: int = 0,
    save_path: str | Path | None = None,
) -> Path:
    """Waterfall plot para un caso concreto."""
    _require_shap()

    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / f"shap_waterfall_sample{sample_idx}.png"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    shap.plots.waterfall(shap_values[sample_idx], show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Figura waterfall guardada en %s", save_path)
    return save_path


def top_features_by_shap(shap_values, feature_names: list[str], n: int = 5) -> pd.DataFrame:
    """
    Devuelve las top-N features ordenadas por importancia SHAP global (mean |SHAP|).

    Returns
    -------
    pd.DataFrame con columnas: feature, mean_abs_shap, rank.
    """
    _require_shap()

    if hasattr(shap_values, "values"):
        vals = np.abs(shap_values.values)
    else:
        vals = np.abs(shap_values)

    mean_abs = vals.mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df.head(n)
