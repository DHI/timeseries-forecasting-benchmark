"""The MLP's direct multi-horizon models must see the calendar of the time they forecast.

mlforecast's direct strategy trains model h on the feature row at t against the target at
t + h. Its `date_features` are computed at t, so model h learned the calendar of the time
h steps before its target, while at predict time it is handed the target's own calendar.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from tsforecastbench.bench.methods import MLPForecaster
from tsforecastbench.data.registry import get_case

NO_VAL = pd.DataFrame({"unique_id": [], "ds": pd.to_datetime([]), "y": []})
CALENDAR = ["dayofyear", "dayofweek", "month", "sin365", "cos365"]


class _Recorder(BaseEstimator, RegressorMixin):
    """Keeps the feature table it was fit on and every table it was asked to predict."""

    predicted: ClassVar[list[pd.DataFrame]] = []

    def fit(self, X, y):
        self.X_ = X.reset_index(drop=True)
        self.y_ = np.asarray(y)
        return self

    def predict(self, X):
        _Recorder.predicted.append(X.reset_index(drop=True))
        return np.zeros(len(X))


class _RecordingMLP(MLPForecaster):
    def _make_model(self):
        return _Recorder()


def _calendar(case, ds: pd.DatetimeIndex) -> pd.DataFrame:
    """The calendar features of `ds`, computed the way the forecaster computes them."""
    from tsforecastbench.bench.methods.ml import _calendar_frame

    return _calendar_frame(case, ds)[CALENDAR].reset_index(drop=True)


@pytest.fixture(scope="module")
def fitted():
    """The MLP fit on three years of daily data whose target is its own row number, so a
    training target names the timestamp it belongs to."""
    case = get_case("discharge_flashy_catskills")
    ds = pd.date_range("2001-01-01", periods=3 * 365, freq="D")
    train = pd.DataFrame({"unique_id": case.id, "ds": ds, "y": np.arange(len(ds), dtype=float)})
    mlp = _RecordingMLP()
    mlp.fit(case, train, NO_VAL)
    return case, train, mlp


def test_each_horizon_trains_on_the_calendar_of_its_target(fitted):
    case, train, mlp = fitted
    models = mlp._fcst.models_[mlp._model_col]
    assert len(models) == 7
    for model in models.values():
        target_ds = pd.DatetimeIndex(train["ds"].to_numpy()[model.y_.astype(int)])
        pd.testing.assert_frame_equal(
            model.X_[CALENDAR], _calendar(case, target_ds), check_dtype=False
        )


def test_each_horizon_predicts_with_the_calendar_of_its_target(fitted):
    case, train, mlp = fitted
    _Recorder.predicted.clear()
    mlp.predict(case, train, horizon=7)
    future = pd.date_range(train["ds"].iloc[-1], periods=8, freq="D")[1:]
    seen = pd.concat([X[CALENDAR] for X in _Recorder.predicted], ignore_index=True)
    pd.testing.assert_frame_equal(seen, _calendar(case, future), check_dtype=False)
