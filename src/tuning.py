"""
tuning.py — Optimización de hiperparámetros con Optuna (100 trials por modelo).

Usa la partición de validación (X_val) del split 70/15/15 para el objetivo de
búsqueda, en lugar de GroupKFold interno, evitando leakage y respetando la
estrategia de evaluación del TFM.

Persistencia:
  - Estudio Optuna: models/optuna_studies/<model_name>_study.pkl
  - Mejores hiperparámetros: models/optuna_studies/<model_name>_best_params.json
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error

from src.models import get_model

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_TRIALS = 100
STUDY_DIR = Path("models/optuna_studies")


def _objective_factory(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
):
    """Devuelve la función objetivo de Optuna: entrena en train, evalúa en val."""

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_name)
        model = get_model(model_name, **params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return float(mean_absolute_error(y_val, preds))

    return objective


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Espacio de búsqueda de hiperparámetros por modelo (especificación TFM)."""

    if model_name == "LinearRegression":
        return {}

    if model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }

    if model_name == "GradientBoosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }

    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    raise ValueError(f"Modelo '{model_name}' sin espacio de búsqueda definido.")


def tune_model(
    model_name: str,
    X_train: np.ndarray | "pd.DataFrame",
    y_train: np.ndarray | "pd.Series",
    X_val: np.ndarray | "pd.DataFrame",
    y_val: np.ndarray | "pd.Series",
    n_trials: int = N_TRIALS,
    seed: int = RANDOM_STATE,
    study_dir: Path | str | None = STUDY_DIR,
) -> dict[str, Any]:
    """
    Optimiza los hiperparámetros de un modelo con Optuna usando X_val como
    conjunto de evaluación (métrica: MAE, dirección: minimize).

    Parameters
    ----------
    model_name : clave en MODEL_REGISTRY.
    X_train, y_train : datos de entrenamiento (sin validación ni test).
    X_val, y_val : datos de validación (para el objetivo Optuna).
    n_trials : número de trials. Default=100.
    seed : semilla TPE. Default=42.
    study_dir : carpeta donde guardar estudio y JSON. None para no guardar.

    Returns
    -------
    dict con 'best_params', 'best_mae' y 'study'.
    """
    import numpy as np
    import pandas as pd

    def _to_numpy(arr):
        return arr.values if hasattr(arr, "values") else np.asarray(arr)

    X_tr = _to_numpy(X_train)
    y_tr = _to_numpy(y_train)
    X_v = _to_numpy(X_val)
    y_v = _to_numpy(y_val)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    objective = _objective_factory(model_name, X_tr, y_tr, X_v, y_v)

    logger.info("Optuna: optimizando %s (%d trials)...", model_name, n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    result = {
        "best_params": study.best_params,
        "best_mae": study.best_value,
        "study": study,
    }

    logger.info(
        "%s → mejor MAE=%.4f en validación | params=%s",
        model_name, result["best_mae"], result["best_params"],
    )

    # Persistencia
    if study_dir is not None:
        study_dir = Path(study_dir)
        study_dir.mkdir(parents=True, exist_ok=True)
        _save_study(study, model_name, result["best_params"], result["best_mae"], study_dir)

    return result


def tune_all_models(
    X_train,
    y_train,
    X_val,
    y_val,
    model_names: list[str] | None = None,
    n_trials: int = N_TRIALS,
    seed: int = RANDOM_STATE,
    study_dir: Path | str | None = STUDY_DIR,
) -> dict[str, dict]:
    """
    Ejecuta tune_model para cada modelo y devuelve un dict con los resultados.

    Returns
    -------
    dict {model_name: {'best_params': ..., 'best_mae': ..., 'study': ...}}
    """
    from src.models import MODEL_REGISTRY
    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    results = {}
    for name in model_names:
        results[name] = tune_model(
            name, X_train, y_train, X_val, y_val,
            n_trials=n_trials, seed=seed, study_dir=study_dir,
        )
    return results


# ---------------------------------------------------------------------------
# Helpers de persistencia
# ---------------------------------------------------------------------------

def _save_study(
    study: optuna.Study,
    model_name: str,
    best_params: dict,
    best_mae: float,
    study_dir: Path,
) -> None:
    """Guarda el study de Optuna en pickle y los mejores params en JSON."""
    study_path = study_dir / f"{model_name}_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    logger.info("Study guardado: %s", study_path)

    params_path = study_dir / f"{model_name}_best_params.json"
    payload = {"model": model_name, "best_mae": best_mae, "best_params": best_params}
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Mejores params guardados: %s", params_path)


def load_best_params(model_name: str, study_dir: Path | str = STUDY_DIR) -> dict:
    """Carga los mejores hiperparámetros guardados en JSON."""
    path = Path(study_dir) / f"{model_name}_best_params.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
