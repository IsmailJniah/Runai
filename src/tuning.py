"""
tuning.py — Optimización de hiperparámetros con Optuna (50 trials por modelo).

Usa GroupKFold interno para evitar data leakage durante la búsqueda.
Todas las búsquedas usan random_state=42 para reproducibilidad.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from src.models import get_model

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_TRIALS = 50
N_CV_INNER = 3  # folds internos durante la búsqueda (más rápido que 5)


def _objective_factory(model_name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Devuelve la función objetivo de Optuna para un modelo dado."""

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_name)
        model = get_model(model_name, **params)

        gkf = GroupKFold(n_splits=N_CV_INNER)
        maes = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            maes.append(mean_absolute_error(y[val_idx], preds))

        return float(np.mean(maes))

    return objective


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Espacio de búsqueda de hiperparámetros por modelo."""

    if model_name == "LinearRegression":
        # Sin hiperparámetros clave; devolvemos un dict vacío
        return {}

    if model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }

    if model_name == "GradientBoosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }

    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
        }

    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", -1, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
        }

    raise ValueError(f"Modelo '{model_name}' sin espacio de búsqueda definido.")


def tune_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_trials: int = N_TRIALS,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Optimiza los hiperparámetros de un modelo con Optuna.

    Parameters
    ----------
    model_name : str
        Nombre del modelo (clave en MODEL_REGISTRY).
    X, y, groups : DataFrames/Series alineados.
    n_trials : int
        Número de trials de Optuna.
    seed : int
        Semilla para reproducibilidad.

    Returns
    -------
    dict con claves 'best_params' y 'best_mae'.
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    X_np = X.values if isinstance(X, pd.DataFrame) else X
    y_np = y.values if isinstance(y, pd.Series) else y
    g_np = groups.values if isinstance(groups, pd.Series) else groups

    objective = _objective_factory(model_name, X_np, y_np, g_np)

    logger.info("Optuna: optimizando %s (%d trials)...", model_name, n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = {
        "best_params": study.best_params,
        "best_mae": study.best_value,
    }
    logger.info(
        "%s → mejor MAE=%.4f, params=%s",
        model_name,
        best["best_mae"],
        best["best_params"],
    )
    return best


def tune_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_names: list[str] | None = None,
    n_trials: int = N_TRIALS,
    seed: int = RANDOM_STATE,
) -> dict[str, dict]:
    """
    Ejecuta tune_model para todos los modelos del registro.

    Returns
    -------
    dict {model_name: {'best_params': ..., 'best_mae': ...}}
    """
    from src.models import MODEL_REGISTRY
    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    results = {}
    for name in model_names:
        results[name] = tune_model(name, X, y, groups, n_trials=n_trials, seed=seed)
    return results
