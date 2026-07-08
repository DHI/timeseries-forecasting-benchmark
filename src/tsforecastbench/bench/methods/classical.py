"""Classical statistical methods - Nixtla statsforecast AutoARIMA.

Two tractability choices, both standard practice:

1. Non-seasonal by default. AutoARIMA with a seasonal period m fits SARIMA models whose
   cost explodes with m; seasonal ARIMA at sub-daily frequency (m=96 for 15-min, even
   m=24) is impractical. AutoETS is likewise impractical at large m (its seasonal state
   grows with m). So AutoARIMA/AutoETS here are the short-memory autoregressive baselines
   (m=1); the seasonal classical counterparts are the MSTL forecasters below, which strip
   seasonality with a cheap decomposition before fitting a trend model, and AutoTheta,
   which deseasonalizes cheaply. Pass season_length explicitly to force a seasonal
   AutoARIMA/AutoETS fit on a low-period case.

2. Capped training window. Stepwise-search cost grows with length, so we fit on the most
   recent max_train_steps observations. Truncating from the front does not move the
   forecast origin, so test alignment is preserved.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.data.schema import Case


class AutoARIMAForecaster(Forecaster):
    name = "AutoARIMA"
    family = "classical"

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

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
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
    family = "classical"

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

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        fcst = self._sf.predict(h=horizon)
        y_hat = fcst["AutoETS"].to_numpy().astype(np.float64)
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})


class TBATSForecaster(Forecaster):
    """TBATS (Trigonometric, Box-Cox, ARMA, Trend, Seasonal) via statsforecast.

    Handles complex and multiple seasonal patterns via trigonometric terms.
    season_length accepts an int, a list of ints (e.g. [24, 168] for
    daily+weekly in hourly data), or None to use the case's declared
    dominant period (1, non-seasonal, when seasonalities = []).
    max_train_steps is set lower than other classicals because TBATS fit
    cost is substantially higher.

    When the training data contains non-positive values, a Yeo-Johnson
    power transform is applied before fitting and inverted on the forecasts,
    since TBATS's internal Box-Cox transform requires strictly positive data.
    """

    name = "TBATS"
    family = "classical"

    def __init__(self, season_length: int | list[int] | None = None, max_train_steps: int = 3000):
        self.season_length = season_length
        self.max_train_steps = max_train_steps

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from sklearn.preprocessing import PowerTransformer
        from statsforecast import StatsForecast
        from statsforecast.models import TBATS

        from tsforecastbench.data.seasonality import dominant_period

        df = train[["unique_id", "ds", "y"]].copy()
        if len(df) > self.max_train_steps:
            df = df.iloc[-self.max_train_steps :]
        season_length = (
            self.season_length
            if self.season_length is not None
            else dominant_period(case.seasonalities, df["y"].to_numpy())
        )
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

        # Yeo-Johnson output isn't guaranteed strictly positive (unlike Box-Cox), so
        # TBATS's own positivity check can still trip and re-disable its Box-Cox with a
        # UserWarning even though we've already handled the transform. Pass
        # use_boxcox=False explicitly in that case to get the same outcome silently.
        use_boxcox = not (self._pt is not None)
        self._sf = StatsForecast(
            models=[TBATS(season_length=season_length, use_boxcox=use_boxcox)],
            freq=case.freq,
            n_jobs=1,
        )
        # TBATS's internal optimizer transiently overflows float64 while probing
        # inadmissible parameter candidates it goes on to reject; harmless noise.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="overflow encountered in square", category=RuntimeWarning
            )
            self._sf.fit(df)

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        fcst = self._sf.predict(h=horizon)
        y_hat = fcst["TBATS"].to_numpy().astype(np.float64)
        if self._pt is not None:
            y_hat = self._pt.inverse_transform(y_hat.reshape(-1, 1)).ravel()
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})


class ThetaForecaster(Forecaster):
    """Theta method (auto-optimized) via statsforecast — a strong, cheap classical
    benchmark, often the one to beat among non-ML methods.

    Seasonal by default: AutoTheta handles seasonality with a cheap classical
    decomposition (deseasonalize, fit, re-seasonalize) rather than a costly seasonal
    state, so — unlike AutoARIMA/AutoETS here — wiring the case's seasonal period in is
    tractable at any frequency and is standard practice (the M3/M4 seasonally-adjusted
    Theta). season_length=None resolves to the case's declared dominant period (1, i.e.
    non-seasonal, when the case declares seasonalities = []); pass an int to override.
    AutoTheta still runs its own seasonality test and only deseasonalizes when the series
    warrants it.
    """

    name = "Theta"
    family = "classical"

    def __init__(self, season_length: int | None = None, max_train_steps: int = 8000):
        self.season_length = season_length
        self.max_train_steps = max_train_steps

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoTheta

        from tsforecastbench.data.seasonality import dominant_period

        df = train[["unique_id", "ds", "y"]].copy()
        if len(df) > self.max_train_steps:
            df = df.iloc[-self.max_train_steps :]
        season_length = (
            self.season_length
            if self.season_length is not None
            else dominant_period(case.seasonalities, df["y"].to_numpy())
        )
        ds = pd.to_datetime(df["ds"])
        if ds.dt.tz is not None:
            ds = ds.dt.tz_convert("UTC").dt.tz_localize(None)
        df["ds"] = ds
        self._sf = StatsForecast(
            models=[AutoTheta(season_length=season_length)],
            freq=case.freq,
            n_jobs=1,
        )
        self._sf.fit(df)
        self._col = "AutoTheta"

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        fcst = self._sf.predict(h=horizon)
        y_hat = fcst[self._col].to_numpy().astype(np.float64)
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})


class _MSTLBase(Forecaster):
    """MSTL decomposition + a classical trend-cycle/residual forecaster (statsforecast).

    MSTL removes one or more seasonal components with a fast STL-based decomposition,
    fits the subclass's `trend_forecaster` on the seasonally-adjusted series, and
    re-applies the seasonal pattern to the forecast. This is the seasonal classical
    counterpart to the non-seasonal AutoARIMA/AutoETS baselines, and works uniformly
    across frequencies with large or multiple seasonal periods (168h weekly, 365d yearly)
    that native seasonal ARIMA/ETS cannot afford.

    Periods come from the case's declared `seasonalities`, filtered to those the training
    series is long enough to support (STL needs >=2 full cycles). If the case is declared
    non-seasonal or none survive, MSTL is genuinely inapplicable and fit raises rather than
    silently degenerating into the plain non-seasonal trend forecaster (which would just
    duplicate the AutoARIMA/AutoETS column).
    """

    name = "MSTL"
    family = "classical"
    max_train_steps = 8000

    def _make_trend_forecaster(self):
        raise NotImplementedError

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from statsforecast import StatsForecast
        from statsforecast.models import MSTL

        df = train[["unique_id", "ds", "y"]].copy()
        if len(df) > self.max_train_steps:
            df = df.iloc[-self.max_train_steps :]
        ds = pd.to_datetime(df["ds"])
        if ds.dt.tz is not None:
            ds = ds.dt.tz_convert("UTC").dt.tz_localize(None)
        df["ds"] = ds

        if not case.seasonalities:
            raise ValueError(f"MSTL inapplicable: case {case.id} is declared non-seasonal")
        periods = sorted(p for p in case.seasonalities if len(df) >= 2 * p)
        if not periods:
            raise ValueError(
                f"MSTL inapplicable: no declared period with >=2 cycles in {len(df)} train "
                f"points (declared {case.seasonalities})"
            )
        # MSTL takes an int for a single period or a list for multiple.
        season_length = periods[0] if len(periods) == 1 else periods

        self._sf = StatsForecast(
            models=[
                MSTL(season_length=season_length, trend_forecaster=self._make_trend_forecaster())
            ],
            freq=case.freq,
            n_jobs=1,
        )
        self._sf.fit(df)

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        fcst = self._sf.predict(h=horizon)
        y_hat = fcst["MSTL"].to_numpy().astype(np.float64)
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})


class MSTLAutoETSForecaster(_MSTLBase):
    """MSTL with an AutoETS trend-cycle/residual forecaster (statsforecast's default)."""

    name = "MSTL-ETS"

    def _make_trend_forecaster(self):
        from statsforecast.models import AutoETS

        # model="ZZN": auto error/trend, no seasonal component — MSTL already removed
        # seasonality, and it rejects a seasonally-adjusting trend forecaster.
        return AutoETS(model="ZZN")


class MSTLAutoARIMAForecaster(_MSTLBase):
    """MSTL with an AutoARIMA trend-cycle/residual forecaster. Capped tighter than
    MSTL-ETS because the ARIMA stepwise search is the more expensive fit."""

    name = "MSTL-ARIMA"
    max_train_steps = 5000

    def _make_trend_forecaster(self):
        from statsforecast.models import AutoARIMA

        return AutoARIMA()
