"""
test_groupkfold_no_leakage.py

ASSERT: ningún user_id está simultáneamente en train y test en ningún fold.

Verifica que:
1. get_group_kfold_splits produce el número correcto de folds.
2. Ningún usuario aparece a la vez en train y test.
3. _validate_no_leakage lanza AssertionError si se introduce leakage manual.
"""

import numpy as np
import pandas as pd
import pytest

from src.splits import get_group_kfold_splits, _validate_no_leakage, N_SPLITS


def _make_dataset(n_users: int = 20, sessions_per_user: int = 10):
    """Dataset sintético con n_users usuarios y sessions_per_user sesiones cada uno."""
    np.random.seed(42)
    user_ids = np.repeat(np.arange(n_users), sessions_per_user)
    X = pd.DataFrame({
        "duration_min": np.random.uniform(20, 90, len(user_ids)),
        "distance_km": np.random.uniform(3, 20, len(user_ids)),
    })
    y = pd.Series(np.random.uniform(30, 200, len(user_ids)), name="trimp")
    groups = pd.Series(user_ids, name="userId")
    return X, y, groups


class TestGroupKFoldSplits:
    def test_correct_number_of_folds(self):
        X, y, groups = _make_dataset()
        splits = get_group_kfold_splits(X, y, groups, n_splits=N_SPLITS)
        assert len(splits) == N_SPLITS

    def test_no_user_in_train_and_test(self):
        X, y, groups = _make_dataset(n_users=25)
        splits = get_group_kfold_splits(X, y, groups)
        group_array = groups.values
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train_users = set(group_array[train_idx])
            test_users = set(group_array[test_idx])
            overlap = train_users & test_users
            assert not overlap, (
                f"Fold {fold_idx}: usuarios {overlap} en train Y test → leakage detectado"
            )

    def test_all_data_covered(self):
        """Todos los índices deben aparecer en test exactamente una vez."""
        X, y, groups = _make_dataset()
        splits = get_group_kfold_splits(X, y, groups)
        all_test = np.concatenate([test_idx for _, test_idx in splits])
        assert len(set(all_test)) == len(X), "No todos los índices aparecen en test."

    def test_custom_n_splits(self):
        X, y, groups = _make_dataset(n_users=15)
        splits = get_group_kfold_splits(X, y, groups, n_splits=3)
        assert len(splits) == 3


class TestValidateNoLeakage:
    def test_valid_splits_pass(self):
        groups = pd.Series([0, 0, 1, 1, 2, 2])
        valid_splits = [(np.array([0, 1, 2, 3]), np.array([4, 5]))]
        _validate_no_leakage(valid_splits, groups)  # no debe lanzar

    def test_leaking_splits_raise(self):
        """Si un usuario está en train y test, debe lanzar AssertionError."""
        groups = pd.Series([0, 0, 1, 1, 2, 2])
        # Usuario 1 (índices 2,3) está en ambos conjuntos: leakage
        leaking_splits = [(np.array([0, 1, 2, 3]), np.array([2, 3, 4, 5]))]
        with pytest.raises(AssertionError, match="DATA LEAKAGE"):
            _validate_no_leakage(leaking_splits, groups)
