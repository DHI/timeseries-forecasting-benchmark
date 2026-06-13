"""Windowed ML methods - Nixtla mlforecast with three sklearn-style estimators.

LinearLagsForecaster, MLPForecaster, LightGBMForecaster all share the same lag +
calendar feature pipeline; they differ only in the underlying regressor. Univariate
for the pilot (mlforecast wants all dynamic exogenous at predict time; the rolling
evaluator will supply them in Phase B).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.data.schema import Case


def _date_features_for(freq: str) -> list[str]:
    f = freq.lower()
    if f in {"1d", "d"}:
        return ["dayofyear", "dayofweek", "month"]
    if f in {"1h", "h"}:
        return ["hour", "dayofweek", "dayofyear"]
    if f in {"15min", "15t"}:
        return ["hour", "minute", "dayofweek", "dayofyear"]
    return ["dayofweek", "month"]


def _lags_for(freq: str) -> list[int]:
    f = freq.lower()
    if f in {"1d", "d"}:
        return [1, 2, 3, 7, 14, 30]
    if f in {"1h", "h"}:
        return [1, 2, 3, 6, 12, 24, 48, 168]
    if f in {"15min", "15t"}:
        return [1, 2, 4, 8, 16, 96]
    return [1, 2, 3]


class _WindowedMLBase(Forecaster):
    """Shared lag-feature pipeline; subclasses provide the sklearn regressor.

    `max_train_steps` caps the fit to the most recent N points. None = no cap. Useful
    for heavy methods (MLP) on long high-frequency series; truncation does not move the
    forecast origin so test alignment holds.
    """

    name = "WindowedML"
    max_train_steps: int | None = None

    def _make_model(self):
        raise NotImplementedError

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from mlforecast import MLForecast

        df = train[["unique_id", "ds", "y"]].copy()
        if self.max_train_steps and len(df) > self.max_train_steps:
            df = df.iloc[-self.max_train_steps :]
        ds = pd.to_datetime(df["ds"])
        if ds.dt.tz is not None:
            ds = ds.dt.tz_convert("UTC").dt.tz_localize(None)
        df["ds"] = ds

        self._fcst = MLForecast(
            models=[self._make_model()],
            freq=case.freq,
            lags=_lags_for(case.freq),
            date_features=_date_features_for(case.freq),
        )
        self._fcst.fit(df, static_features=[])
        self._model_col = type(next(iter(self._fcst.models.values()))).__name__

    def predict(self, case: Case, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        preds = self._fcst.predict(h=horizon)
        y_hat = preds[self._model_col].to_numpy().astype(np.float64)
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})


class LightGBMForecaster(_WindowedMLBase):
    name = "LightGBM"

    def _make_model(self):
        from lightgbm import LGBMRegressor

        return LGBMRegressor(n_estimators=200, learning_rate=0.05, verbosity=-1)


class LinearLagsForecaster(_WindowedMLBase):
    name = "LinearLags"

    def _make_model(self):
        from sklearn.linear_model import Ridge

        # Ridge over plain LinearRegression to be safe with collinear lag features
        return Ridge(alpha=1.0)


class MLPForecaster(_WindowedMLBase):
    name = "MLP"
    max_train_steps = 20000

    def _make_model(self):
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            random_state=0,
        )
