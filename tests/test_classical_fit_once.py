"""Unit tests for the classical methods' fit-once (`is_global`) conversion.

Each converted forecaster (AutoARIMA, AutoETS, Theta, TBATS, MSTL-ETS, MSTL-ARIMA) must:
  1. Genuinely condition its forecast on the history `predict` is handed, not just on
     state captured at `fit` -- proven by forecasting from two different-length histories
     and checking the results differ.
  2. Never re-run its order/spec search or numerical optimizer inside `predict` -- proven
     by spying on the relevant library entry point and checking it is called exactly once
     across several `predict` calls, all triggered by the single `fit` call.
  3. Choose a spec that stays valid for the whole history `predict` will be handed, not
     just for the capped window the search ran on -- the fixture below plants a zero
     before that window, which is what `sewerlevel_bellinge` does in practice.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.bench.methods.classical import (  # noqa: E402
    AutoARIMAForecaster,
    AutoETSForecaster,
    MSTLAutoARIMAForecaster,
    MSTLAutoETSForecaster,
    TBATSForecaster,
    ThetaForecaster,
)
from tsforecastbench.data.schema import Case  # noqa: E402


def _case(seasonalities: list[int]) -> Case:
    return Case(
        id="dummy",
        domain="test",
        target="y",
        freq="1h",
        seasonalities=seasonalities,
        horizon_primary="6h",
        splits={"train_end": "2020-01-01", "val_end": "2021-01-01", "test_end": "2022-01-01"},
        source={"name": "test"},
    )


def _synthetic_series(n: int = 500, period: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 50 + 5 * np.sin(2 * np.pi * t / period) + np.cumsum(rng.normal(scale=0.2, size=n))
    ds = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame({"unique_id": "a", "ds": ds, "y": y})


def _series_with_zero_before_the_search_window(n: int = 500, window: int = 200) -> pd.DataFrame:
    """Strictly positive over the most recent `window` points, but dipping to zero before
    them -- so a spec search capped at `window` sees only positive data while `predict`,
    handed the whole record, sees the zero. `sewerlevel_bellinge` has exactly this shape:
    one 0.000 reading, ~40k points before the end of train+val.
    """
    df = _synthetic_series(n=n)
    df.loc[(n - window) // 2, "y"] = 0.0
    return df


def _assert_conditions_on_growing_history(method, case, train) -> None:
    assert method.is_global
    method.fit(case, train)
    short = method.predict(case, train.iloc[:400], 6)["y_hat"].to_numpy()
    long = method.predict(case, train.iloc[:450], 6)["y_hat"].to_numpy()
    assert np.isfinite(short).all() and np.isfinite(long).all()
    assert not np.allclose(short, long)


def _multi_origin_predict(method, case, train) -> None:
    for cutoff in (350, 400, 450, 480):
        method.predict(case, train.iloc[:cutoff], 6)


def test_auto_arima_conditions_on_growing_history():
    case = _case([24])
    train = _synthetic_series()
    _assert_conditions_on_growing_history(AutoARIMAForecaster(season_length=24), case, train)


def test_auto_arima_fits_order_search_and_optimizer_exactly_once():
    from statsforecast.models import AutoARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    case = _case([24])
    train = _synthetic_series()
    method = AutoARIMAForecaster(season_length=24)

    with (
        patch.object(AutoARIMA, "fit", autospec=True, wraps=AutoARIMA.fit) as order_search,
        patch.object(SARIMAX, "fit", autospec=True, wraps=SARIMAX.fit) as optimizer,
    ):
        method.fit(case, train)
        _multi_origin_predict(method, case, train)

    assert order_search.call_count == 1
    assert optimizer.call_count == 1


def test_auto_ets_conditions_on_growing_history():
    case = _case([24])
    train = _synthetic_series()
    _assert_conditions_on_growing_history(AutoETSForecaster(season_length=24), case, train)


def test_auto_ets_fits_spec_search_and_optimizer_exactly_once():
    from statsforecast.models import AutoETS
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    case = _case([24])
    train = _synthetic_series()
    method = AutoETSForecaster(season_length=24)

    with (
        patch.object(AutoETS, "fit", autospec=True, wraps=AutoETS.fit) as spec_search,
        patch.object(ETSModel, "fit", autospec=True, wraps=ETSModel.fit) as optimizer,
    ):
        method.fit(case, train)
        _multi_origin_predict(method, case, train)

    assert spec_search.call_count == 1
    assert optimizer.call_count == 1


def test_theta_conditions_on_growing_history():
    case = _case([24])
    train = _synthetic_series()
    _assert_conditions_on_growing_history(ThetaForecaster(season_length=24), case, train)


def test_theta_fits_exactly_once():
    from statsforecast.models import AutoTheta

    case = _case([24])
    train = _synthetic_series()
    method = ThetaForecaster(season_length=24)

    with patch.object(AutoTheta, "fit", autospec=True, wraps=AutoTheta.fit) as fit:
        method.fit(case, train)
        _multi_origin_predict(method, case, train)

    assert fit.call_count == 1


def test_tbats_conditions_on_growing_history():
    case = _case([24])
    train = _synthetic_series()
    _assert_conditions_on_growing_history(TBATSForecaster(season_length=24), case, train)


def test_tbats_never_reoptimizes_in_predict():
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    case = _case([24])
    train = _synthetic_series()
    method = TBATSForecaster(season_length=24)
    method.fit(case, train)

    with patch.object(
        UnobservedComponents, "fit", autospec=True, wraps=UnobservedComponents.fit
    ) as optimizer:
        _multi_origin_predict(method, case, train)

    assert optimizer.call_count == 0


def test_tbats_transform_stays_valid_for_history_before_the_search_window():
    case = _case([24])
    train = _series_with_zero_before_the_search_window()
    method = TBATSForecaster(season_length=24, max_train_steps=200)
    method.fit(case, train)

    assert method._pt.method == "yeo-johnson"
    y_hat = method.predict(case, train, 6)["y_hat"].to_numpy()
    assert np.isfinite(y_hat).all()


def test_tbats_keeps_box_cox_when_the_whole_record_is_positive():
    case = _case([24])
    method = TBATSForecaster(season_length=24, max_train_steps=200)
    method.fit(case, _synthetic_series())

    assert method._pt.method == "box-cox"


def test_auto_ets_spec_stays_valid_for_history_before_the_search_window():
    case = _case([24])
    train = _series_with_zero_before_the_search_window()
    method = AutoETSForecaster(season_length=24, max_train_steps=200)
    method.fit(case, train)

    assert "mul" not in (method._error, method._trend, method._seasonal)
    y_hat = method.predict(case, train, 6)["y_hat"].to_numpy()
    assert np.isfinite(y_hat).all()


def test_auto_ets_keeps_a_multiplicative_spec_when_the_whole_record_is_positive():
    case = _case([24])
    method = AutoETSForecaster(season_length=24, max_train_steps=200)
    method.fit(case, _synthetic_series())

    # The downgrade above is driven by the record's sign, not applied unconditionally: on
    # this series statsforecast picks a multiplicative error, and it must survive.
    assert method._error == "mul"


def test_mstl_ets_conditions_on_growing_history():
    case = _case([24])
    train = _synthetic_series()
    _assert_conditions_on_growing_history(MSTLAutoETSForecaster(), case, train)


def test_mstl_arima_conditions_on_growing_history():
    case = _case([24])
    train = _synthetic_series()
    _assert_conditions_on_growing_history(MSTLAutoARIMAForecaster(), case, train)


def test_mstl_ets_decomposes_and_fits_trend_exactly_once():
    from statsforecast.models import AutoETS
    from statsmodels.tsa.seasonal import MSTL

    case = _case([24])
    train = _synthetic_series()
    method = MSTLAutoETSForecaster()

    with (
        patch.object(MSTL, "fit", autospec=True, wraps=MSTL.fit) as decompose,
        patch.object(AutoETS, "fit", autospec=True, wraps=AutoETS.fit) as trend_spec_search,
    ):
        method.fit(case, train)
        _multi_origin_predict(method, case, train)

    assert decompose.call_count == 1
    assert trend_spec_search.call_count == 1


def test_mstl_arima_decomposes_and_fits_trend_exactly_once():
    from statsforecast.models import AutoARIMA
    from statsmodels.tsa.seasonal import MSTL

    case = _case([24])
    train = _synthetic_series()
    method = MSTLAutoARIMAForecaster()

    with (
        patch.object(MSTL, "fit", autospec=True, wraps=MSTL.fit) as decompose,
        patch.object(AutoARIMA, "fit", autospec=True, wraps=AutoARIMA.fit) as trend_order_search,
    ):
        method.fit(case, train)
        _multi_origin_predict(method, case, train)

    assert decompose.call_count == 1
    assert trend_order_search.call_count == 1


def test_mstl_is_still_honestly_inapplicable_on_declared_nonseasonal_case():
    case = _case([])
    train = _synthetic_series()
    try:
        MSTLAutoETSForecaster().fit(case, train)
        raise AssertionError("expected MSTL to raise on a non-seasonal case")
    except ValueError as exc:
        assert "non-seasonal" in str(exc)
