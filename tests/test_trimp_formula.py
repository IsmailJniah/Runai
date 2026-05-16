"""
test_trimp_formula.py

Casos conocidos de la fórmula de Banister (1991) para verificar
que la implementación es correcta.

TRIMP = D * ΔHR_ratio * 0.64 * exp(b * ΔHR_ratio)

  ΔHR_ratio = (HR_mean - HR_rest) / (HR_max - HR_rest)
  b = 1.92 (male), 1.67 (female)
"""

import math
import numpy as np
import pytest

from src.trimp import banister_trimp, banister_trimp_incremental, _select_b, B_MALE, B_FEMALE


def _expected_trimp(duration_min, hr_mean, gender, hr_rest=60, hr_max=185):
    """Cálculo de referencia para tests (manual)."""
    b = B_MALE if gender.lower() in ("male", "m", "h") else B_FEMALE
    ratio = (hr_mean - hr_rest) / (hr_max - hr_rest)
    ratio = max(1e-6, min(ratio, 1.0))
    return duration_min * ratio * 0.64 * math.exp(b * ratio)


class TestBansisterTrimp:
    def test_male_moderate_effort(self):
        """Hombre, 30 min, FC=140, reposo=60, max=185."""
        expected = _expected_trimp(30, 140, "male")
        result = banister_trimp(30, 140, "male")
        assert abs(result - expected) < 1e-6

    def test_female_moderate_effort(self):
        """Mujer, 45 min, FC=145, reposo=60, max=185."""
        expected = _expected_trimp(45, 145, "female")
        result = banister_trimp(45, 145, "female")
        assert abs(result - expected) < 1e-6

    def test_b_coefficient_male(self):
        assert _select_b("male") == B_MALE
        assert _select_b("M") == B_MALE
        assert _select_b("h") == B_MALE
        assert _select_b("hombre") == B_MALE

    def test_b_coefficient_female(self):
        assert _select_b("female") == B_FEMALE
        assert _select_b("F") == B_FEMALE
        assert _select_b("mujer") == B_FEMALE

    def test_higher_intensity_higher_trimp(self):
        """Mayor FC → mayor TRIMP (misma duración)."""
        low = banister_trimp(30, 120, "male")
        high = banister_trimp(30, 160, "male")
        assert high > low

    def test_longer_duration_higher_trimp(self):
        """Mayor duración → mayor TRIMP (misma FC)."""
        short = banister_trimp(20, 140, "male")
        long_ = banister_trimp(60, 140, "male")
        assert long_ > short

    def test_invalid_inputs_return_nan(self):
        """Inputs inválidos deben devolver NaN."""
        assert math.isnan(banister_trimp(0, 140, "male"))        # duración = 0
        assert math.isnan(banister_trimp(30, -10, "male"))       # FC negativa
        assert math.isnan(banister_trimp(30, 140, "male", hr_max=50))  # max < rest

    def test_trimp_positive(self):
        """El TRIMP debe ser siempre positivo para inputs válidos."""
        result = banister_trimp(30, 130, "female")
        assert result > 0

    def test_custom_hr_rest_max(self):
        """Verificar con FC reposo y máxima personalizadas."""
        expected = _expected_trimp(40, 150, "male", hr_rest=55, hr_max=190)
        result = banister_trimp(40, 150, "male", hr_rest=55, hr_max=190)
        assert abs(result - expected) < 1e-6

    def test_known_value_male(self):
        """
        Valor de referencia calculado manualmente:
        D=60, HR=155, rest=60, max=185, b=1.92
        ratio = (155-60)/(185-60) = 0.76
        TRIMP = 60 * 0.76 * 0.64 * exp(1.92*0.76) ≈ 123.33
        """
        ratio = (155 - 60) / (185 - 60)
        expected = 60 * ratio * 0.64 * math.exp(1.92 * ratio)
        result = banister_trimp(60, 155, "male")
        assert abs(result - expected) < 0.01


class TestBanisterTrimpIncremental:
    """
    Verifica que banister_trimp_incremental incluye el factor 0.64 de Banister
    y que su resultado converge al de banister_trimp para FC constante.
    """

    def _make_constant_hr_session(self, duration_s: float, hr_bpm: float, n_samples: int = 100):
        """Serie temporal con FC constante durante duration_s segundos."""
        timestamps = np.linspace(0, duration_s, n_samples)
        heart_rate = np.full(n_samples, hr_bpm)
        return heart_rate, timestamps

    def test_includes_064_factor(self):
        """TRIMP incremental con FC constante debe ≈ versión agregada (ambas con 0.64)."""
        duration_min = 30.0
        hr = 140.0
        hr_t, ts = self._make_constant_hr_session(duration_min * 60, hr, n_samples=300)
        incremental = banister_trimp_incremental(hr_t, ts, "male")
        aggregate = banister_trimp(duration_min, hr, "male")
        # Convergencia dentro de 2% (diferencia por discretización)
        assert abs(incremental - aggregate) / aggregate < 0.02, (
            f"Incremental={incremental:.4f} vs Aggregate={aggregate:.4f}: "
            "discrepancia > 2%, probablemente falta el factor 0.64"
        )

    def test_male_vs_female_coefficients(self):
        """Misma sesión: TRIMP hombre > mujer por b mayor (1.92 > 1.67)."""
        hr_t, ts = self._make_constant_hr_session(1800, 150, n_samples=200)
        male = banister_trimp_incremental(hr_t, ts, "male")
        female = banister_trimp_incremental(hr_t, ts, "female")
        assert male > female

    def test_higher_intensity_higher_trimp(self):
        """FC más alta → TRIMP más alto."""
        _, ts = self._make_constant_hr_session(1800, 120, n_samples=200)
        low_hr = np.full(len(ts), 120.0)
        high_hr = np.full(len(ts), 160.0)
        assert banister_trimp_incremental(high_hr, ts, "male") > banister_trimp_incremental(low_hr, ts, "male")

    def test_nan_on_insufficient_data(self):
        """Con menos de 2 muestras válidas devuelve NaN."""
        result = banister_trimp_incremental([140.0], [0.0], "male")
        assert math.isnan(result)

    def test_positive_result(self):
        """El resultado debe ser positivo para inputs válidos."""
        hr_t, ts = self._make_constant_hr_session(3600, 150, n_samples=100)
        assert banister_trimp_incremental(hr_t, ts, "female") > 0
