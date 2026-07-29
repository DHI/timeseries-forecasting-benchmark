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


def _prepare_train(train: pd.DataFrame, max_train_steps: int) -> pd.DataFrame:
    """Slice to the most recent max_train_steps rows and strip tz (statsforecast wants naive ds)."""
    df = train[["unique_id", "ds", "y"]].copy()
    if len(df) > max_train_steps:
        df = df.iloc[-max_train_steps:]
    ds = pd.to_datetime(df["ds"])
    if ds.dt.tz is not None:
        ds = ds.dt.tz_convert("UTC").dt.tz_localize(None)
    df["ds"] = ds
    return df


class AutoARIMAForecaster(Forecaster):
    """Order-search-once ARIMA: statsforecast picks the (S)ARIMA spec, statsmodels holds
    the fixed-parameter Kalman filter that conditions on each origin's growing history.

    statsforecast's own AutoARIMA is a from-scratch reimplementation, not a statsmodels
    wrapper, so its fitted coefficients don't translate into statsmodels' parameterization
    directly. Instead, `fit` uses statsforecast only for what it's good at — stepwise
    order search — then fits `statsmodels.tsa.statespace.sarimax.SARIMAX` once on the same
    window to get params in a form `.filter()` can hold fixed. `predict` rebuilds a SARIMAX
    of that same order on the (longer) history it's handed and calls `.filter(params)` — a
    Kalman recursion with no optimizer call — so the forecast genuinely reflects the new
    observations without ever re-estimating order or coefficients.
    """

    name = "AutoARIMA"
    family = "classical"
    is_global = True
    library = "statsforecast (Nixtla) for order search + statsmodels for fixed-param roll-forward"
    compute_profile = (
        "CPU only; a stepwise order search and one statsmodels fit at fit time (capped "
        "at the most recent 5,000 points by default); each origin after that is a single "
        "Kalman filter pass over the fixed order/parameters, no re-optimization."
    )
    limitations = (
        "Non-seasonal by default (season_length=1) — a large seasonal period would "
        "explode SARIMA's search cost, so seasonal structure isn't modeled unless you "
        "pass season_length explicitly. Univariate: ignores exogenous inputs. Order and "
        "coefficients are fixed at fit time; a structural change in the series after fit "
        "is tracked through the Kalman state, not through re-estimation."
    )
    code_snippet = (
        "from statsforecast import StatsForecast\n"
        "from statsforecast.models import AutoARIMA\n"
        "from statsmodels.tsa.statespace.sarimax import SARIMAX\n\n"
        "sf = StatsForecast(models=[AutoARIMA(season_length=1)], freq=case.freq)\n"
        "sf.fit(train_df)                     # train_df: unique_id, ds, y -- order search only\n"
        "order, seasonal_order, trend = ...   # read off sf.fitted_[0, 0].model_\n"
        "params = SARIMAX(train_y, order=order, seasonal_order=seasonal_order, trend=trend)\\\n"
        "    .fit().params                    # one statsmodels fit for fixed params\n\n"
        "# each origin: rebuild on the growing history, hold params fixed\n"
        "y_hat = SARIMAX(history_y, order=order, seasonal_order=seasonal_order, trend=trend)\\\n"
        "    .filter(params).forecast(horizon)"
    )

    def __init__(self, season_length: int = 1, max_train_steps: int = 5000):
        self.season_length = season_length
        self.max_train_steps = max_train_steps

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA

        df = _prepare_train(train, self.max_train_steps)

        sf = StatsForecast(
            models=[AutoARIMA(season_length=self.season_length)],
            freq=case.freq,
            n_jobs=1,
        )
        sf.fit(df)
        fitted = sf.fitted_[0, 0]
        p, q, ar_p, ar_q, m, d, seasonal_d = (int(v) for v in fitted.model_["arma"])
        self._order = (p, d, q)
        self._seasonal_order = (ar_p, seasonal_d, ar_q, m) if m > 1 else (0, 0, 0, 0)
        coef = fitted.model_["coef"]
        self._trend = "t" if "drift" in coef else ("c" if "intercept" in coef else "n")

        y = df["y"].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = self._build_model(y)
            self._params = model.fit(disp=False).params

    def _build_model(self, y: np.ndarray):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        return SARIMAX(y, order=self._order, seasonal_order=self._seasonal_order, trend=self._trend)

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        y = pd.to_numeric(history["y"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._build_model(y).filter(self._params)
        y_hat = np.asarray(res.forecast(horizon), dtype=np.float64)
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})


_ETS_LETTER = {"A": "add", "M": "mul", "N": None}


class AutoETSForecaster(Forecaster):
    """Order-search-once ETS: statsforecast picks the error/trend/season spec,
    statsmodels holds the fixed-parameter state that conditions on each origin's growing
    history.

    `fit` runs statsforecast's AutoETS once to choose the spec (error/trend/season type
    letters + damped flag), then fits `statsmodels.tsa.exponential_smoothing.ets.ETSModel`
    once on the same window for params in a form `.smooth()` can hold fixed. Unlike
    AutoARIMA/SARIMAX, ETSModel has no incremental `.append()`/`.extend()`, so `predict`
    rebuilds the model on the longer history and re-runs the full filter each origin
    (still no optimizer call) — cheap relative to re-optimizing, but a real O(n) cost per
    origin worth knowing about for very long histories.

    Like AutoARIMA, season_length defaults to 1 (non-seasonal); set explicitly for
    cases where a small seasonal period is appropriate. max_train_steps caps fit cost.
    """

    name = "AutoETS"
    family = "classical"
    is_global = True
    library = "statsforecast (Nixtla) for spec search + statsmodels for fixed-param roll-forward"
    compute_profile = (
        "CPU only; searches over error/trend/season state combinations plus one "
        "statsmodels fit at fit time (capped at the most recent 5,000 points by "
        "default); each origin after that re-filters the fixed spec/parameters over the "
        "growing history — no re-optimization, but a full O(n) refilter (ETSModel has no "
        "incremental update), unlike AutoARIMA's cheaper Kalman-filter roll-forward."
    )
    limitations = (
        "Non-seasonal by default (season_length=1), like AutoARIMA — its seasonal "
        "state grows with the period, so a large seasonal period is impractical unless "
        "you pass season_length explicitly. Univariate: ignores exogenous inputs. Spec "
        "and coefficients are fixed at fit time."
    )
    code_snippet = (
        "from statsforecast import StatsForecast\n"
        "from statsforecast.models import AutoETS\n"
        "from statsmodels.tsa.exponential_smoothing.ets import ETSModel\n\n"
        "sf = StatsForecast(models=[AutoETS(season_length=1)], freq=case.freq)\n"
        "sf.fit(train_df)                     # train_df: unique_id, ds, y -- spec search only\n"
        "spec = ...                           # read off sf.fitted_[0, 0].model_['components']\n"
        "params = ETSModel(train_y, **spec).fit().params   # one statsmodels fit for fixed params\n\n"
        "# each origin: rebuild on the growing history, hold params fixed\n"
        "y_hat = ETSModel(history_y, **spec).smooth(params).forecast(horizon)"
    )

    def __init__(self, season_length: int = 1, max_train_steps: int = 5000):
        self.season_length = season_length
        self.max_train_steps = max_train_steps

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS

        df = _prepare_train(train, self.max_train_steps)

        sf = StatsForecast(
            models=[AutoETS(season_length=self.season_length)],
            freq=case.freq,
            n_jobs=1,
        )
        sf.fit(df)
        fitted = sf.fitted_[0, 0]
        error, trend, season, damped_flag = fitted.model_["components"]
        self._error = _ETS_LETTER[error]
        self._trend = _ETS_LETTER[trend]
        self._seasonal = _ETS_LETTER[season]
        self._damped = damped_flag == "D"
        self._season_periods = self.season_length if self._seasonal is not None else None

        y = df["y"].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = self._build_model(y)
            self._params = model.fit(disp=False).params

    def _build_model(self, y: np.ndarray):
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        return ETSModel(
            y,
            error=self._error,
            trend=self._trend,
            damped_trend=self._damped,
            seasonal=self._seasonal,
            seasonal_periods=self._season_periods,
        )

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        y = pd.to_numeric(history["y"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._build_model(y).smooth(self._params)
        y_hat = np.asarray(res.forecast(horizon), dtype=np.float64)
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
    library = "statsforecast (Nixtla)"
    compute_profile = (
        "CPU only; the heaviest of the classical methods here, so it's fit on a "
        "smaller window (3,000 points) than its siblings — still typically seconds "
        "per case, but noticeably slower than AutoARIMA/AutoETS/Theta."
    )
    limitations = (
        "Univariate: ignores exogenous inputs. Its internal Box-Cox transform assumes "
        "strictly positive data; a Yeo-Johnson fallback (see below) handles series "
        "that dip to zero or negative."
    )
    code_snippet = (
        "from statsforecast import StatsForecast\n"
        "from statsforecast.models import TBATS\n\n"
        "sf = StatsForecast(models=[TBATS(season_length=season_length)], freq=case.freq)\n"
        "sf.fit(train_df)                     # train_df: unique_id, ds, y\n"
        'y_hat = sf.predict(h=horizon)["TBATS"]'
    )

    def __init__(self, season_length: int | list[int] | None = None, max_train_steps: int = 3000):
        self.season_length = season_length
        self.max_train_steps = max_train_steps

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from sklearn.preprocessing import PowerTransformer
        from statsforecast import StatsForecast
        from statsforecast.models import TBATS

        from tsforecastbench.data.seasonality import dominant_period

        df = _prepare_train(train, self.max_train_steps)
        season_length = (
            self.season_length
            if self.season_length is not None
            else dominant_period(case.seasonalities, df["y"].to_numpy())
        )

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
    library = "statsforecast (Nixtla)"
    compute_profile = (
        "CPU only; a cheap deseasonalize/fit/re-seasonalize decomposition, typically "
        "the fastest of the classical methods here (fit is capped at 8,000 points)."
    )
    limitations = (
        "Univariate: ignores exogenous inputs. Deseasonalization is a fixed classical "
        "decomposition rather than a learned model, so it can lag a genuinely "
        "shifting seasonal shape."
    )
    code_snippet = (
        "from statsforecast import StatsForecast\n"
        "from statsforecast.models import AutoTheta\n\n"
        "sf = StatsForecast(models=[AutoTheta(season_length=season_length)], freq=case.freq)\n"
        "sf.fit(train_df)                     # train_df: unique_id, ds, y\n"
        'y_hat = sf.predict(h=horizon)["AutoTheta"]'
    )

    def __init__(self, season_length: int | None = None, max_train_steps: int = 8000):
        self.season_length = season_length
        self.max_train_steps = max_train_steps

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoTheta

        from tsforecastbench.data.seasonality import dominant_period

        df = _prepare_train(train, self.max_train_steps)
        season_length = (
            self.season_length
            if self.season_length is not None
            else dominant_period(case.seasonalities, df["y"].to_numpy())
        )
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
    library = "statsforecast (Nixtla)"
    compute_profile = (
        "CPU only; a fast STL-based decomposition plus the subclass's trend forecaster "
        "(fit is capped at 8,000 points) — typically seconds per case."
    )
    limitations = (
        "Univariate: ignores exogenous inputs. Needs at least 2 full cycles of each "
        "declared seasonal period in the training window to fit at all; raises rather "
        "than degenerating into a duplicate non-seasonal forecast when a case has no "
        "seasonality it can use."
    )
    code_snippet = (
        "from statsforecast import StatsForecast\n"
        "from statsforecast.models import MSTL\n\n"
        "sf = StatsForecast(\n"
        "    models=[MSTL(season_length=periods, trend_forecaster=<trend_forecaster>)],\n"
        "    freq=case.freq,\n"
        ")\n"
        "sf.fit(train_df)                     # train_df: unique_id, ds, y\n"
        'y_hat = sf.predict(h=horizon)["MSTL"]'
    )

    def _make_trend_forecaster(self):
        raise NotImplementedError

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from statsforecast import StatsForecast
        from statsforecast.models import MSTL

        df = _prepare_train(train, self.max_train_steps)

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
    library = "statsforecast (Nixtla) — trend forecaster: AutoETS"

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
    library = "statsforecast (Nixtla) — trend forecaster: AutoARIMA"

    def _make_trend_forecaster(self):
        from statsforecast.models import AutoARIMA

        return AutoARIMA()
