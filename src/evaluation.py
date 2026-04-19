"""
evaluation.py — Métricas de regresión y test estadístico Wilcoxon-Bonferroni.

Métricas por modelo:
  MAE, RMSE, R², MAPE, Correlación de Pearson

Test estadístico:
  Wilcoxon signed-rank test pareado (Demšar, 2006) con corrección de Bonferroni.

Referencia:
  Demšar, J. (2006). Statistical comparisons of classifiers over multiple
  data sets. Journal of Machine Learning Research, 7, 1–30.
"""

from __future__ import annotations

import itertools
import logging
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Métricas individuales
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula MAE, RMSE, R², MAPE y Pearson r para un fold."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    # MAPE: evitar división por cero
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]))) * 100

    pearson_r, _ = stats.pearsonr(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "Pearson_r": pearson_r,
    }


def aggregate_cv_metrics(fold_metrics: list[dict]) -> dict:
    """Agrega métricas de varios folds (media ± std)."""
    keys = fold_metrics[0].keys()
    result = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        result[f"{k}_mean"] = float(np.mean(vals))
        result[f"{k}_std"] = float(np.std(vals))
    return result


# ---------------------------------------------------------------------------
# Evaluación cross-validation de un modelo
# ---------------------------------------------------------------------------

def evaluate_model_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list,
) -> tuple[dict, list[np.ndarray]]:
    """
    Evalúa un modelo sobre los splits GroupKFold.

    Returns
    -------
    (aggregated_metrics, list_of_fold_errors)
        list_of_fold_errors: residuos absolutos por fold (para Wilcoxon).
    """
    fold_metrics = []
    fold_abs_errors = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test.values, y_pred)
        fold_metrics.append(metrics)
        fold_abs_errors.append(np.abs(y_test.values - y_pred))

        logger.debug("Fold %d — MAE=%.3f R²=%.3f", fold_idx, metrics["MAE"], metrics["R2"])

    return aggregate_cv_metrics(fold_metrics), fold_abs_errors


# ---------------------------------------------------------------------------
# Tests estadísticos
# ---------------------------------------------------------------------------

def wilcoxon_bonferroni(
    model_errors: dict[str, list[np.ndarray]],
    reference_model: str | None = None,
) -> pd.DataFrame:
    """
    Wilcoxon signed-rank test pareado con corrección de Bonferroni.

    Compara todos los pares de modelos (o cada modelo vs. el de referencia).

    Parameters
    ----------
    model_errors : dict
        {model_name: list_of_fold_abs_errors}
    reference_model : str | None
        Si se especifica, compara solo cada modelo vs. la referencia.
        Si None, compara todos los pares.

    Returns
    -------
    pd.DataFrame con columnas: model_A, model_B, statistic, p_raw, p_bonferroni, significant.
    """
    model_names = list(model_errors.keys())

    if reference_model:
        pairs = [(reference_model, m) for m in model_names if m != reference_model]
    else:
        pairs = list(itertools.combinations(model_names, 2))

    n_comparisons = len(pairs)
    rows = []

    for model_a, model_b in pairs:
        # Concatenar errores a nivel de sesión (todos los folds juntos)
        errors_a = np.concatenate(model_errors[model_a])
        errors_b = np.concatenate(model_errors[model_b])

        # Wilcoxon signed-rank
        stat, p_raw = stats.wilcoxon(errors_a, errors_b, alternative="two-sided")
        p_bonf = min(p_raw * n_comparisons, 1.0)

        rows.append({
            "model_A": model_a,
            "model_B": model_b,
            "statistic": stat,
            "p_raw": p_raw,
            "p_bonferroni": p_bonf,
            "significant_0.05": p_bonf < 0.05,
        })

    return pd.DataFrame(rows)


def build_results_table(
    agg_metrics: dict[str, dict],
    wilcoxon_df: pd.DataFrame,
    reference_model: str = "LinearRegression",
) -> pd.DataFrame:
    """
    Construye la Tabla 4.3: métricas + p-valor Wilcoxon vs. baseline.

    Parameters
    ----------
    agg_metrics : dict {model_name: aggregated_metrics_dict}
    wilcoxon_df : resultado de wilcoxon_bonferroni()
    reference_model : baseline para la columna p_vs_baseline.

    Returns
    -------
    pd.DataFrame listo para guardar como tabla_4_3_modelos.csv
    """
    rows = []
    for model_name, metrics in agg_metrics.items():
        row = {"model": model_name}
        row.update(metrics)

        # Buscar p-valor vs. baseline
        mask = (
            ((wilcoxon_df["model_A"] == reference_model) & (wilcoxon_df["model_B"] == model_name))
            | ((wilcoxon_df["model_A"] == model_name) & (wilcoxon_df["model_B"] == reference_model))
        )
        subset = wilcoxon_df[mask]
        if len(subset) > 0:
            row["p_bonferroni_vs_baseline"] = subset.iloc[0]["p_bonferroni"]
        else:
            row["p_bonferroni_vs_baseline"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).set_index("model")
