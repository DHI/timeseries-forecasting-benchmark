"""Unit tests for the classical methods' fit-once (`is_global`) behaviour.

The five global classical forecasters (AutoARIMA, AutoETS, Theta, MSTL-ETS/MSTL-ARIMA,
TBATS) roll forward via a fixed, already-estimated model. Each must:
  1. Genuinely condition its forecast on the history `predict` is handed, not just on
     state captured at `fit` -- proven by forecasting from two different-length histories
     and checking the results differ.
  2. Never re-run its order/spec/component search or numerical optimizer inside `predict`
     -- proven by spying on the statsforecast model's own `fit` (or, for TBATS, on
     `tbats_selection`, the function its component search actually runs through) and
     checking it is called exactly once across several `predict` calls.
  3. Reproduce statsforecast's own forecast for the same fitted model, so the wrapper is
     the library's estimator and not a reimplementation of it.
  4. Stay finite when `predict` is handed history the capped search window never saw --
     including a non-positive value, which `sewerlevel_bellinge` has in practice.

AutoARIMA/AutoETS/Theta/MSTL roll forward through statsforecast's own `forward(y, h)`.
TBATS has no such public method -- statsforecast couples its component search to every
`.fit()` on the class API -- so it instead calls `tbats_selection` directly at fit time and
replicates `forward` by hand at predict time with two more of statsforecast's own internals,
`calcTBATSFaster`/`tbats_forecast`. `test_tbats_matches_statsforecasts_own_forecast` is what
proves point 3 for TBATS specifically: that this hand-rolled roll-forward reproduces
`AutoTBATS`'s own output bit-for-bit rather than approximating it (see
`TBATSForecaster`'s docstring for why that distinction matters here).
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

# None of the classical methods trains iteratively, so none of them reads `fit`'s val
# argument. These tests hand over an empty holdout rather than carving one out of a
# synthetic series that exists only to exercise the roll-forward path.
NO_VAL = pd.DataFrame({"unique_id": [], "ds": pd.to_datetime([]), "y": []})


def _case(seasonalities: list[int]) -> Case:
    return Case(
        id="dummy",
        domain="test",
        target="y",
        freq="1h",
        seasonalities=seasonalities,
        dominant_seasonality=seasonalities[0] if seasonalities else 1,
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
    method.fit(case, train, NO_VAL)
    short = method.predict(case, train.iloc[:400], 6)["y_hat"].to_numpy()
    long = method.predict(case, train.iloc[:450], 6)["y_hat"].to_numpy()
    assert np.isfinite(short).all() and np.isfinite(long).all()
    assert not np.allclose(short, long)


def _multi_origin_predict(method, case, train) -> None:
    for cutoff in (350, 400, 450, 480):
        method.predict(case, train.iloc[:cutoff], 6)


def _assert_survives_history_the_search_window_never_saw(method, case) -> None:
    """`fit` on a strictly positive tail, `predict` over the whole record including a zero
    -- the shape of the real rolling loop, and the one that used to break the methods that
    chose a positive-only spec (Box-Cox, multiplicative ETS) from the capped window alone.
    """
    train = _series_with_zero_before_the_search_window()
    method.fit(case, train, NO_VAL)
    y_hat = method.predict(case, train, 6)["y_hat"].to_numpy()
    assert np.isfinite(y_hat).all()


def test_auto_arima_conditions_on_growing_history():
    case = _case([24])
    train = _synthetic_series()
    _assert_conditions_on_growing_history(AutoARIMAForecaster(season_length=24), case, train)


def test_auto_arima_fits_order_search_exactly_once():
    from statsforecast.models import AutoARIMA

    case = _case([24])
    train = _synthetic_series()
    method = AutoARIMAForecaster(season_length=24)

    with patch.object(AutoARIMA, "fit", autospec=True, wraps=AutoARIMA.fit) as order_search:
        method.fit(case, train, NO_VAL)
        _multi_origin_predict(method, case, train)

    assert order_search.call_count == 1


def test_auto_arima_matches_statsforecasts_own_forecast():
    """The wrapper must BE statsforecast's AutoARIMA, not an approximation of it. A private
    reimplementation once drifted from the library here, so pin it.
    """
    from statsforecast.models import AutoARIMA

    case = _case([24])
    train = _synthetic_series()
    method = AutoARIMAForecaster(season_length=24)
    method.fit(case, train, NO_VAL)

    y = train["y"].to_numpy(dtype=float)
    reference = AutoARIMA(season_length=24)
    reference.fit(y)

    ours = method.predict(case, train, 6)["y_hat"].to_numpy()
    theirs = np.asarray(reference.predict(6)["mean"], dtype=float)
    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-9)


def test_auto_arima_survives_history_the_search_window_never_saw():
    _assert_survives_history_the_search_window_never_saw(
        AutoARIMAForecaster(season_length=24, max_train_steps=200), _case([24])
    )


def test_auto_ets_conditions_on_growing_history():
    case = _case([24])
    train = _synthetic_series()
    _assert_conditions_on_growing_history(AutoETSForecaster(season_length=24), case, train)


def test_auto_ets_fits_spec_search_exactly_once():
    from statsforecast.models import AutoETS

    case = _case([24])
    train = _synthetic_series()
    method = AutoETSForecaster(season_length=24)

    with patch.object(AutoETS, "fit", autospec=True, wraps=AutoETS.fit) as spec_search:
        method.fit(case, train, NO_VAL)
        _multi_origin_predict(method, case, train)

    assert spec_search.call_count == 1


def test_auto_ets_matches_statsforecasts_own_forecast():
    from statsforecast.models import AutoETS

    case = _case([24])
    train = _synthetic_series()
    method = AutoETSForecaster(season_length=24)
    method.fit(case, train, NO_VAL)

    reference = AutoETS(season_length=24)
    reference.fit(train["y"].to_numpy(dtype=float))

    ours = method.predict(case, train, 6)["y_hat"].to_numpy()
    theirs = np.asarray(reference.predict(6)["mean"], dtype=float)
    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-9)


def test_auto_ets_survives_history_the_search_window_never_saw():
    _assert_survives_history_the_search_window_never_saw(
        AutoETSForecaster(season_length=24, max_train_steps=200), _case([24])
    )


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
        method.fit(case, train, NO_VAL)
        _multi_origin_predict(method, case, train)

    assert fit.call_count == 1


def test_theta_survives_history_the_search_window_never_saw():
    _assert_survives_history_the_search_window_never_saw(
        ThetaForecaster(season_length=24, max_train_steps=200), _case([24])
    )


def test_mstl_ets_conditions_on_growing_history():
    case = _case([24])
    train = _synthetic_series()
    _assert_conditions_on_growing_history(MSTLAutoETSForecaster(), case, train)


def test_mstl_arima_conditions_on_growing_history():
    case = _case([24])
    train = _synthetic_series()
    _assert_conditions_on_growing_history(MSTLAutoARIMAForecaster(), case, train)


def test_mstl_ets_decomposes_and_fits_trend_exactly_once():
    from statsforecast.models import MSTL, AutoETS

    case = _case([24])
    train = _synthetic_series()
    method = MSTLAutoETSForecaster()

    with (
        patch.object(MSTL, "fit", autospec=True, wraps=MSTL.fit) as decompose,
        patch.object(AutoETS, "fit", autospec=True, wraps=AutoETS.fit) as trend_spec_search,
    ):
        method.fit(case, train, NO_VAL)
        _multi_origin_predict(method, case, train)

    assert decompose.call_count == 1
    assert trend_spec_search.call_count == 1


def test_mstl_arima_decomposes_and_fits_trend_exactly_once():
    from statsforecast.models import MSTL, AutoARIMA

    case = _case([24])
    train = _synthetic_series()
    method = MSTLAutoARIMAForecaster()

    with (
        patch.object(MSTL, "fit", autospec=True, wraps=MSTL.fit) as decompose,
        patch.object(AutoARIMA, "fit", autospec=True, wraps=AutoARIMA.fit) as trend_order_search,
    ):
        method.fit(case, train, NO_VAL)
        _multi_origin_predict(method, case, train)

    assert decompose.call_count == 1
    assert trend_order_search.call_count == 1


def test_forward_reads_the_whole_history_not_the_capped_search_window():
    """`max_train_steps` bounds the order/spec *search*, not the roll-forward: predict
    conditions on everything it is handed. Some of these are slow over a long history
    (Theta's forward is superlinear), and that cost is reported rather than capped away --
    capping would substitute an arbitrary re-initialization for the state the fitted model
    actually implies. Proven by forwarding histories longer than the search window and
    checking the forecast still moves when data older than that window changes. Compared
    bitwise rather than approximately, because how *much* it moves varies enormously: an
    ARIMA/ETS filter carries an old level shift forward visibly, while MSTL's STL trend is
    local enough that the same shift 600 points back moves the forecast only in the 8th
    decimal. Either way a truncating predict would return identical numbers.
    """
    case = _case([24])
    train = _synthetic_series(n=900)
    # Identical over the last 600 points — so identical over any capped window — and shifted
    # over the first 300. A capped predict cannot tell these apart; an uncapped one can.
    altered = train.copy()
    altered.loc[:299, "y"] = altered.loc[:299, "y"] + 20.0

    for method in (
        AutoARIMAForecaster(season_length=24, max_train_steps=200),
        AutoETSForecaster(season_length=24, max_train_steps=200),
        ThetaForecaster(season_length=24, max_train_steps=200),
        MSTLAutoETSForecaster(),
        TBATSForecaster(season_length=24, max_train_steps=200),
    ):
        method.max_train_steps = 200
        method.fit(case, train, NO_VAL)
        as_is = method.predict(case, train, 6)["y_hat"].to_numpy()
        shifted = method.predict(case, altered, 6)["y_hat"].to_numpy()
        assert np.isfinite(as_is).all() and np.isfinite(shifted).all()
        assert not np.array_equal(as_is, shifted), (
            f"{method.name} ignored history older than its search window"
        )


def test_mstl_is_still_honestly_inapplicable_on_declared_nonseasonal_case():
    case = _case([])
    train = _synthetic_series()
    try:
        MSTLAutoETSForecaster().fit(case, train, NO_VAL)
        raise AssertionError("expected MSTL to raise on a non-seasonal case")
    except ValueError as exc:
        assert "non-seasonal" in str(exc)


def test_tbats_conditions_on_growing_history():
    case = _case([24])
    train = _synthetic_series()
    _assert_conditions_on_growing_history(TBATSForecaster(season_length=24), case, train)


def test_tbats_component_search_runs_exactly_once():
    """TBATS has no statsforecast *class* to patch `.fit()` on -- its component search runs
    through the module-level `tbats_selection` function instead, called once from `fit` and
    never again from `predict`.
    """
    import statsforecast.tbats as tbats_module

    case = _case([24])
    train = _synthetic_series()
    method = TBATSForecaster(season_length=24)

    with patch.object(
        tbats_module, "tbats_selection", autospec=True, wraps=tbats_module.tbats_selection
    ) as component_search:
        method.fit(case, train, NO_VAL)
        _multi_origin_predict(method, case, train)

    assert component_search.call_count == 1


def test_tbats_matches_statsforecasts_own_forecast():
    """The wrapper must reproduce `AutoTBATS`'s own fitted model, not a reimplementation of
    it -- the exact concern a fit-once TBATS was previously rejected over (see
    `TBATSForecaster`'s docstring). Forecasting over the same window the model was fit on
    must equal `AutoTBATS(...).fit(y).predict(h)` bit-for-bit, since both paths run the
    identical `tbats_selection` -> `calcTBATSFaster`/`tbats_forecast` internals.
    """
    from statsforecast.models import AutoTBATS

    case = _case([24])
    train = _synthetic_series()
    method = TBATSForecaster(season_length=24)
    method.fit(case, train, NO_VAL)

    y = train["y"].to_numpy(dtype=float)
    reference = AutoTBATS(season_length=24)
    reference.fit(y)

    ours = method.predict(case, train, 6)["y_hat"].to_numpy()
    theirs = np.asarray(reference.predict(6)["mean"], dtype=float)
    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-9)


def test_tbats_survives_history_the_search_window_never_saw():
    _assert_survives_history_the_search_window_never_saw(
        TBATSForecaster(season_length=24, max_train_steps=200), _case([24])
    )
