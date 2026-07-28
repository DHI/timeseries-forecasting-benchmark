"""Unit tests for the new metric functions (MAPE, sMAPE, NSE, KGE)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.bench.eval.metrics import kge, mape, nse, smape  # noqa: E402
from tsforecastbench.data.schema import DOMAIN_RESTRICTED_METRICS, Case  # noqa: E402


def test_mape_normal_case():
    y = np.array([1.0, 2.0, 4.0])
    y_hat = np.array([1.0, 3.0, 3.0])
    assert mape(y, y_hat) == pytest.approx((0.0 + 0.5 + 0.25) / 3)


def test_mape_nan_when_all_targets_zero():
    y = np.zeros(3)
    y_hat = np.array([1.0, 2.0, 3.0])
    assert np.isnan(mape(y, y_hat))


def test_smape_normal_case():
    y = np.array([1.0, 2.0, 4.0])
    y_hat = np.array([1.0, 3.0, 3.0])
    assert smape(y, y_hat) == pytest.approx((0.0 + 1 / 2.5 + 1 / 3.5) / 3)


def test_smape_nan_when_target_and_prediction_both_zero():
    y = np.zeros(3)
    y_hat = np.zeros(3)
    assert np.isnan(smape(y, y_hat))


def test_nse_perfect_forecast_is_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert nse(y, y.copy()) == pytest.approx(1.0)


def test_nse_nan_when_target_is_constant():
    y = np.full(5, 3.0)
    y_hat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isnan(nse(y, y_hat))


def test_nse_worse_than_mean_is_negative():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_hat = np.array([10.0, -10.0, 20.0, -20.0])
    assert nse(y, y_hat) < 0.0


def test_kge_perfect_forecast_is_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert kge(y, y.copy()) == pytest.approx(1.0)


def test_kge_nan_when_target_is_constant():
    y = np.full(5, 3.0)
    y_hat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isnan(kge(y, y_hat))


def test_kge_nan_when_target_mean_is_zero():
    y = np.array([-2.0, -1.0, 1.0, 2.0])
    y_hat = np.array([-2.0, -1.0, 1.0, 2.0])
    assert np.isnan(kge(y, y_hat))


def _minimal_case_kwargs(**overrides):
    kwargs = dict(
        id="test_case",
        domain="hydrology",
        target="y",
        freq="1D",
        seasonalities=[365],
        horizon_primary="7D",
        splits={"train_end": "2020-01-01", "val_end": "2020-06-01", "test_end": "2020-12-01"},
        source={"name": "test"},
    )
    kwargs.update(overrides)
    return kwargs


def test_domain_metrics_accepts_known_names():
    case = Case(**_minimal_case_kwargs(domain_metrics=["NSE", "KGE"]))
    assert case.domain_metrics == ["NSE", "KGE"]


def test_domain_metrics_rejects_unknown_name():
    with pytest.raises(ValueError, match="domain_metrics"):
        Case(**_minimal_case_kwargs(domain_metrics=["NOT_A_METRIC"]))


def test_domain_restricted_metrics_is_nse_and_kge():
    assert frozenset({"NSE", "KGE"}) == DOMAIN_RESTRICTED_METRICS
