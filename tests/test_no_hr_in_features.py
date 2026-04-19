"""
test_no_hr_in_features.py

ASSERT: FC no aparece en la matriz X de features.

Verifica que:
1. FEATURE_NAMES no contiene términos relacionados con FC.
2. assert_no_hr_leakage lanza AssertionError cuando se pasa una columna con FC.
3. La función de construcción de features no incluye FC en su salida.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_NAMES,
    assert_no_hr_leakage,
    compute_features_from_series,
)

_HR_TERMS = {"heart_rate", "hr", "fc", "bpm", "heart", "pulse", "trimp"}


class TestFeatureNamesNoHR:
    def test_feature_names_no_hr_keywords(self):
        """Ninguna feature en FEATURE_NAMES debe contener términos de FC."""
        bad = [
            name for name in FEATURE_NAMES
            if any(term in name.lower() for term in _HR_TERMS)
        ]
        assert not bad, f"FC encontrada en FEATURE_NAMES: {bad}"

    def test_feature_names_count(self):
        """Debe haber exactamente 9 features externas."""
        assert len(FEATURE_NAMES) == 9, (
            f"Se esperaban 9 features, hay {len(FEATURE_NAMES)}: {FEATURE_NAMES}"
        )


class TestAssertNoHRLeakage:
    def test_clean_df_passes(self):
        """Un DataFrame sin columnas de FC no debe lanzar excepción."""
        X = pd.DataFrame({"duration_min": [30], "distance_km": [5.0]})
        assert_no_hr_leakage(X)  # no debe lanzar

    def test_heart_rate_col_raises(self):
        """Columna 'heart_rate' debe lanzar AssertionError."""
        X = pd.DataFrame({"duration_min": [30], "heart_rate": [150]})
        with pytest.raises(AssertionError, match="BUG CRÍTICO"):
            assert_no_hr_leakage(X)

    def test_hr_mean_col_raises(self):
        """Columna 'hr_mean' debe lanzar AssertionError."""
        X = pd.DataFrame({"duration_min": [30], "hr_mean": [150]})
        with pytest.raises(AssertionError, match="BUG CRÍTICO"):
            assert_no_hr_leakage(X)

    def test_fc_col_raises(self):
        """Columna 'fc_media' debe lanzar AssertionError."""
        X = pd.DataFrame({"duration_min": [30], "fc_media": [150]})
        with pytest.raises(AssertionError, match="BUG CRÍTICO"):
            assert_no_hr_leakage(X)

    def test_trimp_col_raises(self):
        """El target 'trimp' tampoco debe estar en X."""
        X = pd.DataFrame({"duration_min": [30], "trimp": [85.0]})
        with pytest.raises(AssertionError, match="BUG CRÍTICO"):
            assert_no_hr_leakage(X)


class TestComputeFeaturesNoHR:
    def _make_series(self, n: int = 10):
        """Crea series sintéticas de GPS."""
        t = np.linspace(0, 3600, n)  # 1 hora en segundos
        lat = np.linspace(40.0, 40.05, n)
        lon = np.linspace(-3.0, -2.95, n)
        alt = 600 + np.random.RandomState(42).randn(n) * 5
        return t, lat, lon, alt

    def test_output_keys_match_feature_names(self):
        t, lat, lon, alt = self._make_series()
        result = compute_features_from_series(t, lat, lon, alt)
        assert set(result.keys()) == set(FEATURE_NAMES), (
            f"Keys inesperadas: {set(result.keys()) - set(FEATURE_NAMES)}"
        )

    def test_no_hr_in_output(self):
        t, lat, lon, alt = self._make_series()
        result = compute_features_from_series(t, lat, lon, alt)
        bad = [k for k in result if any(term in k.lower() for term in _HR_TERMS)]
        assert not bad, f"FC encontrada en output de compute_features: {bad}"
