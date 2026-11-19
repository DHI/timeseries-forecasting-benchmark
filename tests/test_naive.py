"""The naive baselines read the history handed to `predict`, not the frame `fit` saw (gh #120)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsforecastbench.bench.methods.naive import (
    Climatology,
    MeanBaseline,
    Persistence,
    SeasonalNaive,
)
from tsforecastbench.data.registry import get_case


@pytest.fixture(scope="module")
def case():
    return get_case("discharge_flashy_catskills")


def _frame(case, values: np.ndarray) -> pd.DataFrame:
    ds = pd.date_range("2000-01-01", periods=len(values), freq=case.freq, tz="UTC")
    return pd.DataFrame({"unique_id": "s", "ds": ds, "y": values})


@pytest.mark.parametrize("method_cls", [MeanBaseline, Persistence, SeasonalNaive, Climatology])
def test_forecast_follows_the_history_not_the_fit_frame(case, method_cls):
    n = 3 * case.dominant_seasonality
    method = method_cls()
    method.fit(case, _frame(case, np.zeros(n)), _frame(case, np.zeros(0)))

    y_hat = method.predict(case, _frame(case, np.full(n, 5.0)), 7)["y_hat"].to_numpy()

    np.testing.assert_allclose(y_hat, 5.0)


@pytest.mark.parametrize("method_cls", [MeanBaseline, Persistence, SeasonalNaive, Climatology])
def test_naive_baselines_are_fit_once(method_cls):
    assert method_cls.is_global is True
