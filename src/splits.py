"""
splits.py — Estrategias de partición por user_id (sin data leakage entre usuarios).

Dos estrategias disponibles:
  1. group_train_val_test_split  — split 70/15/15 en un único holdout (recomendado
                                   para evaluación final con Optuna).
  2. get_group_kfold_splits      — GroupKFold k=5 (útil para CV rápida).

Referencia: Scikit-learn GroupKFold garantiza que ningún grupo (usuario)
aparece simultáneamente en train y test.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)

N_SPLITS = 5
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Split 70 / 15 / 15 por usuario (estrategia principal TFM)
# ---------------------------------------------------------------------------

def group_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series,
           pd.Series, pd.Series, pd.Series]:
    """
    Divide X, y en train/val/test asignando usuarios completos a cada partición.

    Garantiza que ningún usuario aparece en más de una partición.
    La fracción de test es implícitamente 1 - train_frac - val_frac.

    Parameters
    ----------
    X : pd.DataFrame — matriz de features.
    y : pd.Series    — vector objetivo (TRIMP).
    groups : pd.Series — user_id, mismo índice que X e y.
    train_frac : float — fracción de usuarios en train. Default=0.70.
    val_frac : float   — fracción de usuarios en validación. Default=0.15.
    seed : int         — semilla para reproducibilidad.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test
    """
    unique_users = np.array(sorted(groups.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_users)

    n = len(unique_users)
    n_train = int(np.round(n * train_frac))
    n_val = int(np.round(n * val_frac))
    # test_frac es el resto

    train_users = set(unique_users[:n_train])
    val_users = set(unique_users[n_train: n_train + n_val])
    test_users = set(unique_users[n_train + n_val:])

    assert len(train_users & val_users) == 0
    assert len(train_users & test_users) == 0
    assert len(val_users & test_users) == 0

    train_mask = groups.isin(train_users)
    val_mask = groups.isin(val_users)
    test_mask = groups.isin(test_users)

    logger.info(
        "Split 70/15/15 — train: %d usuarios (%d sesiones) | val: %d usuarios (%d sesiones)"
        " | test: %d usuarios (%d sesiones)",
        len(train_users), train_mask.sum(),
        len(val_users), val_mask.sum(),
        len(test_users), test_mask.sum(),
    )

    return (
        X[train_mask], X[val_mask], X[test_mask],
        y[train_mask], y[val_mask], y[test_mask],
        groups[train_mask], groups[val_mask], groups[test_mask],
    )


def verify_no_user_overlap(
    g_train: pd.Series,
    g_val: pd.Series,
    g_test: pd.Series,
) -> None:
    """
    Aserta que ningún usuario aparece en más de una partición.
    Lanza AssertionError si detecta data leakage.
    """
    train_u = set(g_train.unique())
    val_u = set(g_val.unique())
    test_u = set(g_test.unique())

    overlap_tv = train_u & val_u
    overlap_tt = train_u & test_u
    overlap_vt = val_u & test_u

    assert not overlap_tv, f"LEAKAGE — usuarios en train ∩ val: {overlap_tv}"
    assert not overlap_tt, f"LEAKAGE — usuarios en train ∩ test: {overlap_tt}"
    assert not overlap_vt, f"LEAKAGE — usuarios en val ∩ test: {overlap_vt}"

    logger.info("Verificación de leakage OK — 0 usuarios solapados entre particiones.")


# ---------------------------------------------------------------------------
# GroupKFold k=5 (evaluación por validación cruzada)
# ---------------------------------------------------------------------------

def get_group_kfold_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = N_SPLITS,
) -> list[Tuple[np.ndarray, np.ndarray]]:
    """
    Genera índices de partición GroupKFold garantizando sin leakage por usuario.

    Returns
    -------
    lista de (train_idx, test_idx) como arrays de enteros positionales.
    """
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups=groups))
    _validate_no_leakage(splits, groups)
    logger.info("GroupKFold: %d folds, %d usuarios únicos", n_splits, groups.nunique())
    return splits


def _validate_no_leakage(
    splits: list[Tuple[np.ndarray, np.ndarray]],
    groups: pd.Series,
) -> None:
    group_array = groups.values
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_groups = set(group_array[train_idx])
        test_groups = set(group_array[test_idx])
        overlap = train_groups & test_groups
        assert not overlap, (
            f"DATA LEAKAGE en fold {fold_idx}: usuarios {overlap} en train y test."
        )


def fold_stats(splits: list, groups: pd.Series) -> pd.DataFrame:
    """Devuelve un DataFrame con estadísticas de cada fold."""
    group_array = groups.values
    stats = []
    for i, (train_idx, test_idx) in enumerate(splits):
        stats.append({
            "fold": i,
            "train_sessions": len(train_idx),
            "test_sessions": len(test_idx),
            "train_users": len(set(group_array[train_idx])),
            "test_users": len(set(group_array[test_idx])),
        })
    return pd.DataFrame(stats)
