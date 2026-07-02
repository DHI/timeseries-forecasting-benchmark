"""Classical statistical methods - Nixtla statsforecast AutoARIMA.

Two tractability choices, both standard practice:

1. Non-seasonal by default. AutoARIMA with a seasonal period m fits SARIMA models whose
   cost explodes with m; seasonal ARIMA at sub-daily frequency (m=96 for 15-min, even
   m=24) is impractical. Seasonality is covered by SeasonalNaive and Climatology, so
   ARIMA here is the short-memory autoregressive baseline (m=1). Pass season_length
   explicitly to force a seasonal fit on a low-period case.

2. Capped training window. Stepwise-search cost grows with length, so we fit on the most
   recent max_train_steps observations. Truncating from the front does not move the
   forecast origin, so test alignment is preserved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.data.schema import Case


class AutoARIMAForecaster(Forecaster):
    name = "AutoARIMA"

    def __init__(self, season_length: int = 1, max_train_steps: int = 5000):
        self.season_length = season_length
        self.max_train_steps = max_train_steps

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA

        df = train[["unique_id", "ds", "y"]].copy()
        if len(df) > self.max_train_steps:
            df = df.iloc[-self.max_train_steps :]
        ds = pd.to_datetime(df["ds"])
        if ds.dt.tz is not None:
            ds = ds.dt.tz_convert("UTC").dt.tz_localize(None)
        df["ds"] = ds

        self._sf = StatsForecast(
            models=[AutoARIMA(season_length=self.season_length)],
            freq=case.freq,
            n_jobs=1,
        )
        self._sf.fit(df)

    def predict(self, case: Case, history: pd.DataFrame, horizon: int,
                future_exog: pd.DataFrame | None = None) -> pd.DataFrame:
        fcst = self._sf.predict(h=horizon)
        y_hat = fcst["AutoARIMA"].to_numpy().astype(np.float64)
        last_ds = history["ds"].max()
        future = self._future_index(case, last_ds, horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})


class AutoETSForecaster(Forecaster):
    """Exponential smoothing state space (auto-selected) via statsforecast.

    Like AutoARIMA, season_length defaults to 1 (non-seasonal); set explicitly for
    cases where a small seasonal period is appropriate. max_train_steps caps fit cost.
    """

    name = "AutoETS"

    def __init__(self, season_length: int = 1, max_train_steps: int = 5000):
        self.season_length = season_length
        self.max_train_steps = max_train_steps

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS

        df = train[["unique_id", "ds", "y"]].copy()
        if len(df) > self.max_train_steps:
            df = df.iloc[-self.max_train_steps :]
        ds = pd.to_datetime(df["ds"])
        if ds.dt.tz is not None:
            ds = ds.dt.tz_convert("UTC").dt.tz_localize(None)
        df["ds"] = ds

        self._sf = StatsForecast(
            models=[AutoETS(season_length=self.season_length)],
            freq=case.freq,
            n_jobs=1,
        )
        self._sf.fit(df)

    def predict(self, case: Case, history: pd.DataFrame, horizon: int,
                future_exog: pd.DataFrame | None = None) -> pd.DataFrame:
        fcst = self._sf.predict(h=horizon)
        y_hat = fcst["AutoETS"].to_numpy().astype(np.float64)
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})


class TBATSForecaster(Forecaster):
    """TBATS (Trigonometric, Box-Cox, ARMA, Trend, Seasonal) via statsforecast.

    Handles complex and multiple seasonal patterns via trigonometric terms.
    season_length accepts an int, a list of ints (e.g. [24, 168] for
    daily+weekly in hourly data), or None to auto-detect from case.freq.
    max_train_steps is set lower than other classicals because TBATS fit
    cost is substantially higher.

    When the training data contains non-positive values, a Yeo-Johnson
    power transform is applied before fitting and inverted on the forecasts,
    since TBATS's internal Box-Cox transform requires strictly positive data.
    """

    name = "TBATS"

    def __init__(self, season_length: int | list[int] | None = None, max_train_steps: int = 3000):
        self.season_length = season_length
        self.max_train_steps = max_train_steps

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from sklearn.preprocessing import PowerTransformer
        from statsforecast import StatsForecast
        from statsforecast.models import TBATS

        from tsforecastbench.bench.methods.naive import _default_season_length

        season_length = self.season_length if self.season_length is not None else _default_season_length(case.freq)

        df = train[["unique_id", "ds", "y"]].copy()
        if len(df) > self.max_train_steps:
            df = df.iloc[-self.max_train_steps :]
        ds = pd.to_datetime(df["ds"])
        if ds.dt.tz is not None:
            ds = ds.dt.tz_convert("UTC").dt.tz_localize(None)
        df["ds"] = ds

        # Apply Yeo-Johnson when Box-Cox would be disabled (non-positive data).
        y_vals = df["y"].to_numpy(dtype=float)
        if np.any(y_vals <= 0):
            self._pt = PowerTransformer(method="yeo-johnson", standardize=False)
            df["y"] = self._pt.fit_transform(y_vals.reshape(-1, 1)).ravel()
        else:
            self._pt = None

        self._sf = StatsForecast(
            models=[TBATS(season_length=season_length)],
            freq=case.freq,
            n_jobs=1,
        )
        self._sf.fit(df)

    def predict(self, case: Case, history: pd.DataFrame, horizon: int,
                future_exog: pd.DataFrame | None = None) -> pd.DataFrame:
        fcst = self._sf.predict(h=horizon)
        y_hat = fcst["TBATS"].to_numpy().astype(np.float64)
        if self._pt is not None:
            y_hat = self._pt.inverse_transform(y_hat.reshape(-1, 1)).ravel()
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})


class ThetaForecaster(Forecaster):
    """Theta method (auto-optimized) via statsforecast — a strong, cheap classical
    benchmark, often the one to beat among non-ML methods. Non-seasonal by default."""

    name = "Theta"

    def __init__(self, season_length: int = 1, max_train_steps: int = 8000):
        self.season_length = season_length
        self.max_train_steps = max_train_steps

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoTheta

        df = train[["unique_id", "ds", "y"]].copy()
        if len(df) > self.max_train_steps:
            df = df.iloc[-self.max_train_steps :]
        ds = pd.to_datetime(df["ds"])
        if ds.dt.tz is not None:
            ds = ds.dt.tz_convert("UTC").dt.tz_localize(None)
        df["ds"] = ds
        self._sf = StatsForecast(
            models=[AutoTheta(season_length=self.season_length)], freq=case.freq, n_jobs=1,
        )
        self._sf.fit(df)
        self._col = "AutoTheta"

    def predict(self, case: Case, history: pd.DataFrame, horizon: int,
                future_exog: pd.DataFrame | None = None) -> pd.DataFrame:
        fcst = self._sf.predict(h=horizon)
        y_hat = fcst[self._col].to_numpy().astype(np.float64)
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})
