"""Classical statistical methods - Nixtla statsforecast AutoARIMA.

Three tractability choices, all standard practice:

1. Fit once, roll forward. statsforecast's Auto* wrappers couple order/parameter search
   to every `.fit()` call, but ARIMA, ETS, Theta, and TBATS are all mathematically
   representable as fixed parameters plus a state that can be updated (filtered) through
   new observations without re-estimation. So every forecaster here is `is_global`: order
   or spec search runs once, in `fit`, and `predict` conditions on each rolling origin's
   growing `history` by holding those parameters fixed (a Kalman/state filter pass, not a
   re-fit) -- see each class's docstring for its specific fixed-parameter mechanism.

2. Non-seasonal by default. AutoARIMA with a seasonal period m fits SARIMA models whose
   cost explodes with m; seasonal ARIMA at sub-daily frequency (m=96 for 15-min, even
   m=24) is impractical. AutoETS is likewise impractical at large m (its seasonal state
   grows with m). So AutoARIMA/AutoETS here are the short-memory autoregressive baselines
   (m=1); the seasonal classical counterparts are the MSTL forecasters below, which strip
   seasonality with a cheap decomposition before fitting a trend model, and AutoTheta,
   which deseasonalizes cheaply. Pass season_length explicitly to force a seasonal
   AutoARIMA/AutoETS fit on a low-period case.

3. Capped training window. Order/spec-search cost grows with length, so we search on the
   most recent max_train_steps observations. Truncating from the front does not move the
   forecast origin, so test alignment is preserved; `predict`'s fixed-parameter roll-forward
   is not capped this way -- it conditions on the real, uncapped history it is handed.
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


def _select_tbats_spec(
    y: np.ndarray, periods: list[int]
) -> tuple[list[int], int, np.ndarray, float]:
    """Harmonics-per-period and AR-error order for `TBATSForecaster`, chosen once by AIC
    over a couple of candidates each — affordable now that this fit runs once per case
    rather than once per rolling origin, unlike TBATS's own automatic component search.

    Returns `(harmonics, ar_order, params, aic)` for the winning spec so `fit` doesn't
    need to refit it a second time.
    """
    import itertools

    from statsmodels.tsa.statespace.structural import UnobservedComponents

    harmonics_candidates = [sorted({1, min(3, p // 2)}) for p in periods]
    ar_candidates = (0, 1)

    best_aic = np.inf
    best: tuple[list[int], int, np.ndarray, float] | None = None
    for harmonics in itertools.product(*harmonics_candidates):
        freq_seasonal = [
            {"period": p, "harmonics": h} for p, h in zip(periods, harmonics, strict=True)
        ]
        for ar_order in ar_candidates:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = UnobservedComponents(
                        y,
                        level="local linear trend",
                        freq_seasonal=freq_seasonal or None,
                        autoregressive=ar_order or None,
                    ).fit(disp=False)
            except Exception:
                continue
            if res.aic < best_aic:
                best_aic = res.aic
                best = (list(harmonics), ar_order, res.params, res.aic)
    if best is None:
        raise ValueError("TBATS: no candidate spec converged")
    return best


class TBATSForecaster(Forecaster):
    """Custom Box-Cox/Yeo-Johnson + trigonometric-seasonal + AR-errors state-space model,
    inspired by TBATS but fit once rather than reimplementing statsforecast's per-origin
    TBATS (which couples its own automatic component search to every fit call).

    Built directly on `statsmodels.tsa.statespace.structural.UnobservedComponents`: a
    local linear trend, one harmonic (trigonometric) seasonal block per declared period,
    and AR errors in place of TBATS's full ARMA errors. `fit` picks the transform, the
    harmonics per period, and the AR order once (see `_select_tbats_spec`), then fits the
    model once for parameters `predict` holds fixed while rebuilding on each origin's
    growing history — the same fixed-parameter roll-forward as AutoARIMA/AutoETS above.
    This is a documented simplification relative to the published TBATS estimator (AR
    errors instead of full ARMA, no automatic component-inclusion search), not
    statsforecast's implementation; `name` stays "TBATS" for continuity of existing
    result columns.

    season_length accepts an int, a list of ints (e.g. [24, 168] for
    daily+weekly in hourly data), or None to use the case's declared
    dominant period (1, non-seasonal, when seasonalities = []). Periods without at least
    2 full cycles in the training window are dropped, same as MSTL below.
    max_train_steps is set lower than other classicals because TBATS fit
    cost is substantially higher.

    When the training data contains non-positive values, a Yeo-Johnson
    power transform is used in place of Box-Cox before fitting and inverted on the
    forecasts, since Box-Cox requires strictly positive data. The transform (and its
    fitted parameter) is chosen once from the fit-time window and reused unchanged at
    every origin.
    """

    name = "TBATS"
    family = "classical"
    is_global = True
    library = (
        "statsmodels (custom Box-Cox/Yeo-Johnson + trigonometric-seasonal + AR-errors "
        "state-space model, inspired by TBATS)"
    )
    compute_profile = (
        "CPU only; the heaviest of the classical methods here at fit time — a small "
        "one-time AIC comparison over harmonics/AR-order candidates, fit on a smaller "
        "window (3,000 points) than its siblings; each origin after that is a single "
        "Kalman filter pass over the fixed spec/parameters, no re-optimization."
    )
    limitations = (
        "Univariate: ignores exogenous inputs. A custom simplification of the published "
        "TBATS estimator, not statsforecast's implementation: AR errors instead of full "
        "ARMA, and harmonics/AR order chosen by a small one-time AIC comparison rather "
        "than TBATS's exhaustive automatic component search. Its Box-Cox transform "
        "assumes strictly positive data; a Yeo-Johnson fallback (see below) handles "
        "series that dip to zero or negative. Transform and spec/parameters are fixed "
        "at fit time."
    )
    code_snippet = (
        "from statsmodels.tsa.statespace.structural import UnobservedComponents\n\n"
        "spec = dict(\n"
        "    level='local linear trend',\n"
        "    freq_seasonal=[{'period': p, 'harmonics': k_p} for p in periods],\n"
        "    autoregressive=ar_order,          # k_p, ar_order chosen once by AIC\n"
        ")\n"
        "params = UnobservedComponents(train_y_transformed, **spec).fit().params\n\n"
        "# each origin: rebuild on the growing history, hold params fixed\n"
        "y_hat_t = UnobservedComponents(history_y_transformed, **spec)\\\n"
        "    .smooth(params).forecast(horizon)\n"
        "y_hat = power_transformer.inverse_transform(y_hat_t)"
    )

    def __init__(self, season_length: int | list[int] | None = None, max_train_steps: int = 3000):
        self.season_length = season_length
        self.max_train_steps = max_train_steps

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from sklearn.preprocessing import PowerTransformer

        from tsforecastbench.data.seasonality import dominant_period

        df = _prepare_train(train, self.max_train_steps)
        resolved = (
            self.season_length
            if self.season_length is not None
            else dominant_period(case.seasonalities, df["y"].to_numpy())
        )
        periods = resolved if isinstance(resolved, list) else ([resolved] if resolved > 1 else [])
        self._periods = sorted(p for p in periods if len(df) >= 2 * p)

        # Box-Cox needs strictly positive data; Yeo-Johnson handles the rest. Chosen once
        # here (statsforecast's TBATS used to redo this check on every per-origin refit).
        y_vals = df["y"].to_numpy(dtype=float)
        method = "box-cox" if np.all(y_vals > 0) else "yeo-johnson"
        self._pt = PowerTransformer(method=method, standardize=False)
        y_t = self._pt.fit_transform(y_vals.reshape(-1, 1)).ravel()

        self._harmonics, self._ar_order, self._params, _aic = _select_tbats_spec(y_t, self._periods)

    def _build_model(self, y: np.ndarray):
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        freq_seasonal = [
            {"period": p, "harmonics": h}
            for p, h in zip(self._periods, self._harmonics, strict=True)
        ]
        return UnobservedComponents(
            y,
            level="local linear trend",
            freq_seasonal=freq_seasonal or None,
            autoregressive=self._ar_order or None,
        )

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        y_vals = pd.to_numeric(history["y"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
        y_t = self._pt.transform(y_vals.reshape(-1, 1)).ravel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._build_model(y_t).smooth(self._params)
        y_hat_t = np.asarray(res.forecast(horizon), dtype=np.float64)
        y_hat = self._pt.inverse_transform(y_hat_t.reshape(-1, 1)).ravel()
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})


class ThetaForecaster(Forecaster):
    """Theta method (auto-optimized) via statsforecast — a strong, cheap classical
    benchmark, often the one to beat among non-ML methods.

    Fit-once: AutoTheta's own `forward(y, h)` re-runs its fixed-parameter recursive
    filter over a new series without touching the optimizer, so `fit` picks the
    deseasonalization/parameters once and `predict` just calls `forward` on each origin's
    growing history — no re-instantiation or re-fitting needed, unlike AutoARIMA/AutoETS
    which needed a second statsmodels fit to get a form that could hold parameters fixed.

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
    is_global = True
    library = "statsforecast (Nixtla)"
    compute_profile = (
        "CPU only; a cheap deseasonalize/fit/re-seasonalize decomposition at fit time "
        "(capped at 8,000 points); each origin after that is a cheap fixed-parameter "
        "`forward` pass, no re-fitting."
    )
    limitations = (
        "Univariate: ignores exogenous inputs. Deseasonalization is a fixed classical "
        "decomposition rather than a learned model, so it can lag a genuinely "
        "shifting seasonal shape. Parameters are fixed at fit time."
    )
    code_snippet = (
        "from statsforecast.models import AutoTheta\n\n"
        "model = AutoTheta(season_length=season_length)\n"
        "model.fit(train_y)                   # train_y: 1D array -- picks params once\n\n"
        "# each origin: fixed-parameter filter over the growing history, no re-fit\n"
        "y_hat = model.forward(history_y, horizon)['mean']"
    )

    def __init__(self, season_length: int | None = None, max_train_steps: int = 8000):
        self.season_length = season_length
        self.max_train_steps = max_train_steps

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from statsforecast.models import AutoTheta

        from tsforecastbench.data.seasonality import dominant_period

        df = _prepare_train(train, self.max_train_steps)
        season_length = (
            self.season_length
            if self.season_length is not None
            else dominant_period(case.seasonalities, df["y"].to_numpy())
        )
        self._model = AutoTheta(season_length=season_length)
        self._model.fit(df["y"].to_numpy(dtype=float))

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        y = pd.to_numeric(history["y"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
        y_hat = self._model.forward(y, horizon)["mean"].astype(np.float64)
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})


class _MSTLBase(Forecaster):
    """MSTL decomposition, frozen once, plus a fit-once classical trend-cycle/residual
    forecaster from this module.

    Decouples decomposition (frozen at fit time) from the trend forecaster (itself
    fit-once, see `AutoETSForecaster`/`AutoARIMAForecaster` above): `fit` runs
    `statsmodels.tsa.seasonal.MSTL` once on the training window, freezes each seasonal
    component's shape as a position-in-cycle lookup (the same convention the Climatology
    baseline and the sin/cos ML features use — see `data.seasonality.position_in_cycle`),
    and fits the subclass's trend forecaster once on the seasonally-adjusted series.
    `predict` looks up the frozen seasonal value at each timestamp of the growing
    `history` and the forecast horizon, subtracts it from `history` before handing that to
    the (already fit-once) trend forecaster, and adds it back to the trend forecaster's
    output — so the seasonal shape is genuinely fixed at fit time, while the trend
    forecaster still conditions on new observations exactly as it does standalone. This
    is the seasonal classical counterpart to the non-seasonal AutoARIMA/AutoETS
    baselines, and works uniformly across frequencies with large or multiple seasonal
    periods (168h weekly, 365d yearly) that native seasonal ARIMA/ETS cannot afford.

    Periods come from the case's declared `seasonalities`, filtered to those the training
    series is long enough to support (MSTL needs >=2 full cycles). If the case is declared
    non-seasonal or none survive, MSTL is genuinely inapplicable and fit raises rather than
    silently degenerating into the plain non-seasonal trend forecaster (which would just
    duplicate the AutoARIMA/AutoETS column).
    """

    name = "MSTL"
    family = "classical"
    is_global = True
    max_train_steps = 8000
    library = "statsmodels (MSTL decomposition, frozen once)"
    compute_profile = (
        "CPU only; a fast STL-based decomposition plus the subclass's trend forecaster's "
        "own fit cost, at fit time (capped at 8,000 points); each origin after that looks "
        "up the frozen seasonal shape and calls the (already fit-once) trend forecaster's "
        "predict — no decomposition or trend re-fitting."
    )
    limitations = (
        "Univariate: ignores exogenous inputs. Needs at least 2 full cycles of each "
        "declared seasonal period in the training window to fit at all; raises rather "
        "than degenerating into a duplicate non-seasonal forecast when a case has no "
        "seasonality it can use. The seasonal shape is a fixed lookup frozen at fit "
        "time, not re-estimated as history grows."
    )
    code_snippet = (
        "from statsmodels.tsa.seasonal import MSTL\n"
        "from tsforecastbench.data.seasonality import position_in_cycle\n\n"
        "decomposition = MSTL(train_y, periods=periods).fit()   # once, at fit time\n"
        "seasonal_lookup = {\n"
        "    p: pd.Series(decomposition.seasonal[..., i]).groupby(\n"
        "        position_in_cycle(case.freq, p, train_ds)).mean()\n"
        "    for i, p in enumerate(periods)\n"
        "}\n"
        "trend_forecaster.fit(case, train_deseasonalized)       # once, e.g. AutoETS/AutoARIMA\n\n"
        "# each origin: look up the frozen seasonal shape, add to the trend forecast\n"
        "seasonal = sum(seasonal_lookup[p][position_in_cycle(case.freq, p, ds)] for p in periods)\n"
        "y_hat = trend_forecaster.predict(case, history_deseasonalized, horizon)['y_hat'] + seasonal"
    )

    def _make_trend_forecaster(self) -> Forecaster:
        raise NotImplementedError

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from statsmodels.tsa.seasonal import MSTL

        from tsforecastbench.data.seasonality import position_in_cycle

        df = _prepare_train(train, self.max_train_steps)

        if not case.seasonalities:
            raise ValueError(f"MSTL inapplicable: case {case.id} is declared non-seasonal")
        periods = sorted(p for p in case.seasonalities if len(df) >= 2 * p)
        if not periods:
            raise ValueError(
                f"MSTL inapplicable: no declared period with >=2 cycles in {len(df)} train "
                f"points (declared {case.seasonalities})"
            )
        self._freq = case.freq
        self._periods = periods

        y = df["y"].to_numpy(dtype=float)
        decomposition = MSTL(y, periods=periods).fit()
        seasonal = np.asarray(decomposition.seasonal)
        if seasonal.ndim == 1:
            seasonal = seasonal[:, None]

        ds = pd.DatetimeIndex(df["ds"])
        self._seasonal_lookup = [
            pd.Series(seasonal[:, i]).groupby(position_in_cycle(case.freq, p, ds)).mean()
            for i, p in enumerate(periods)
        ]

        train_adj = df[["unique_id", "ds"]].copy()
        train_adj["y"] = y - seasonal.sum(axis=1)
        self._trend_forecaster = self._make_trend_forecaster()
        self._trend_forecaster.fit(case, train_adj)

    def _seasonal_component(self, ds: pd.DatetimeIndex) -> np.ndarray:
        from tsforecastbench.data.seasonality import position_in_cycle

        total = np.zeros(len(ds))
        for p, lookup in zip(self._periods, self._seasonal_lookup, strict=True):
            keys = position_in_cycle(self._freq, p, ds)
            total += np.array([lookup.get(k, 0.0) for k in keys], dtype=np.float64)
        return total

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        hist_ds = pd.DatetimeIndex(pd.to_datetime(history["ds"]))
        history_adj = history.copy()
        history_adj["y"] = pd.to_numeric(history["y"], errors="coerce").to_numpy(
            dtype=float
        ) - self._seasonal_component(hist_ds)
        trend_preds = self._trend_forecaster.predict(
            case, history_adj, horizon, future_exog=future_exog
        )
        future = pd.DatetimeIndex(trend_preds["ds"])
        y_hat = trend_preds["y_hat"].to_numpy(dtype=float) + self._seasonal_component(future)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})


class MSTLAutoETSForecaster(_MSTLBase):
    """MSTL with a fit-once AutoETS trend-cycle/residual forecaster."""

    name = "MSTL-ETS"
    library = "statsmodels (MSTL, frozen once) — trend forecaster: AutoETS"

    def _make_trend_forecaster(self) -> Forecaster:
        # Non-seasonal (season_length=1): MSTL already removed seasonality, and the
        # trend forecaster should not re-fit a seasonal component of its own.
        return AutoETSForecaster()


class MSTLAutoARIMAForecaster(_MSTLBase):
    """MSTL with a fit-once AutoARIMA trend-cycle/residual forecaster. Capped tighter
    than MSTL-ETS because the ARIMA stepwise search is the more expensive fit."""

    name = "MSTL-ARIMA"
    max_train_steps = 5000
    library = "statsmodels (MSTL, frozen once) — trend forecaster: AutoARIMA"

    def _make_trend_forecaster(self) -> Forecaster:
        return AutoARIMAForecaster()
