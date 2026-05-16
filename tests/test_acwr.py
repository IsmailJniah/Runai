"""
test_acwr.py

Verifica la cohorte apta y las zonas Gabbett (2016).
  - Usuarios con span < 28 días quedan excluidos.
  - Las zonas ACWR se asignan correctamente.
  - ACWR = media_TRIMP(7d) / media_TRIMP(28d) — converge a 1.0 en carga constante.
"""

import numpy as np
import pandas as pd
import pytest

from src.acwr import (
    ZONE_LOW, ZONE_HIGH_OPT, ZONE_HIGH_RISK,
    _assign_zone,
    compute_acwr_per_user,
    filter_eligible_users,
    zone_distribution,
)


def _make_sessions(user_id: str, n_days: int, trimp_val: float = 80.0) -> pd.DataFrame:
    """Crea sesiones diarias sintéticas para un único usuario."""
    dates = pd.date_range("2016-01-01", periods=n_days, freq="D")
    return pd.DataFrame({
        "userId": user_id,
        "date": dates,
        "trimp": trimp_val,
    })


class TestAssignZone:
    def test_below_low_is_insuficiente(self):
        assert _assign_zone(0.5) == "insuficiente"
        assert _assign_zone(0.79) == "insuficiente"

    def test_optimal_zone(self):
        assert _assign_zone(0.80) == "optima"
        assert _assign_zone(1.05) == "optima"
        assert _assign_zone(1.30) == "optima"

    def test_precaution_zone(self):
        assert _assign_zone(1.31) == "precaucion"
        assert _assign_zone(1.50) == "precaucion"

    def test_risk_zone(self):
        assert _assign_zone(1.51) == "riesgo"
        assert _assign_zone(2.50) == "riesgo"

    def test_nan_returns_sin_dato(self):
        assert _assign_zone(float("nan")) == "sin_dato"


class TestFilterEligibleUsers:
    def test_short_span_excluded(self):
        """Usuario con solo 10 días de datos queda excluido."""
        df = _make_sessions("user_short", n_days=10)
        df_elig, spans = filter_eligible_users(df, min_span_days=28)
        assert len(df_elig) == 0

    def test_long_span_included(self):
        """Usuario con 60 días de datos queda incluido."""
        df = _make_sessions("user_long", n_days=60)
        df_elig, spans = filter_eligible_users(df, min_span_days=28)
        assert len(df_elig) == 60

    def test_mixed_users(self):
        """Solo los usuarios con span ≥ 28d pasan el filtro."""
        df_short = _make_sessions("user_a", n_days=15)
        df_long = _make_sessions("user_b", n_days=45)
        df = pd.concat([df_short, df_long], ignore_index=True)
        df_elig, _ = filter_eligible_users(df, min_span_days=28)
        assert set(df_elig["userId"].unique()) == {"user_b"}


class TestComputeACWRPerUser:
    def test_acwr_constant_load_converges_to_one(self):
        """Con carga diaria constante, ACWR debe converger a ~1.0 tras 28 días.

        Implementación canónica Gabbett (2016): medias rodantes.
        media(7d) / media(28d) = TRIMP_cte / TRIMP_cte = 1.0 en estabilidad.
        """
        df_user = _make_sessions("u1", n_days=60, trimp_val=100.0)
        result = compute_acwr_per_user(df_user)
        # Tras 28 días las dos ventanas están completamente rellenas
        stable_acwr = result.loc[result.index >= 27, "acwr"].dropna()
        assert len(stable_acwr) > 0
        assert abs(stable_acwr.iloc[-1] - 1.0) < 0.05, (
            f"ACWR estabilizado={stable_acwr.iloc[-1]:.4f}, esperado ≈ 1.0. "
            "Verificar que la implementación usa medias (no sumas)."
        )

    def test_acwr_columns_present(self):
        df_user = _make_sessions("u2", n_days=35)
        result = compute_acwr_per_user(df_user)
        required = {"date", "trimp_acute_mean", "trimp_chronic_mean", "acwr", "zone"}
        assert required.issubset(set(result.columns))

    def test_no_session_days_have_valid_acute(self):
        """Días sin sesión no deben producir NaN en trimp_acute_mean."""
        df_user = _make_sessions("u3", n_days=40)
        df_user = df_user[df_user["date"].dt.day != 15]
        result = compute_acwr_per_user(df_user)
        assert result["trimp_acute_mean"].notna().all()

    def test_acwr_zone_optima_at_steady_state(self):
        """Con carga constante ≥ 28 días, la zona debe ser 'optima' (ACWR ≈ 1.0)."""
        df_user = _make_sessions("u4", n_days=60, trimp_val=80.0)
        result = compute_acwr_per_user(df_user)
        # Últimos días (ventanas completas)
        last_zones = result.tail(10)["zone"]
        assert (last_zones == "optima").all(), (
            f"Zona esperada 'optima', encontrada: {last_zones.unique()}"
        )


class TestZoneDistribution:
    def test_all_zones_present(self):
        acwr_df = pd.DataFrame({"acwr": [0.5, 0.9, 1.4, 1.8, float("nan")]})
        dist = zone_distribution(acwr_df)
        zones_found = set(dist["zone"])
        assert "insuficiente" in zones_found
        assert "optima" in zones_found
        assert "precaucion" in zones_found
        assert "riesgo" in zones_found

    def test_pct_sums_to_100(self):
        acwr_df = pd.DataFrame({"acwr": [0.5, 0.9, 1.1, 1.4, 1.8]})
        dist = zone_distribution(acwr_df)
        assert abs(dist["pct"].sum() - 100.0) < 0.01
