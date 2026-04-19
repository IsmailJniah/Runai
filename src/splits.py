"""
splits.py — Partición GroupKFold por user_id (sin data leakage entre usuarios).

Referencia: Scikit-learn GroupKFold garantiza que ningún grupo (usuario)
aparece simultáneamente en train y test.
"""

from __future__ import annotations

import logging
from typing import Generator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)

N_SPLITS = 5  # número de folds (configurable)


def get_group_kfold_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = N_SPLITS,
) -> list[Tuple[np.ndarray, np.ndarray]]:
    """
    Genera los índices de partición GroupKFold.

    Parameters
    ----------
    X : pd.DataFrame
        Matriz de features.
    y : pd.Series
        Vector objetivo (TRIMP).
    groups : pd.Series
        Identificador de grupo = user_id. Mismo índice que X e y.
    n_splits : int
        Número de folds. Default=5.

    Returns
    -------
    lista de (train_idx, test_idx) como arrays de enteros positionales.
    """
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups=groups))
    _validate_no_leakage(splits, groups)
    logger.info(
        "GroupKFold: %d folds, %d usuarios únicos",
        n_splits,
        groups.nunique(),
    )
    return splits


def _validate_no_leakage(
    splits: list[Tuple[np.ndarray, np.ndarray]],
    groups: pd.Series,
) -> None:
    """
    Aserta que ningún usuario aparece a la vez en train y test en ningún fold.
    Lanza AssertionError si detecta leakage.
    """
    group_array = groups.values
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_groups = set(group_array[train_idx])
        test_groups = set(group_array[test_idx])
        overlap = train_groups & test_groups
        assert not overlap, (
            f"DATA LEAKAGE en fold {fold_idx}: los usuarios {overlap} "
            f"aparecen en train Y en test."
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
