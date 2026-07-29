"""Classical statistical methods - Nixtla statsforecast AutoARIMA.

Three tractability choices, all standard practice:

1. Fit once, roll forward. statsforecast's `AutoARIMA`, `AutoETS`, `AutoTheta` and `MSTL`
   all expose `forward(y, h)`: apply the already-fitted model to a new series. It reads the
   fitted parameters straight back out (`ets_f(y, m, model=fitted)` takes alpha/beta/gamma/
   phi and the initial state from the fit and runs one filter pass; `Arima(x=y,
   model=fitted)` is the same apply-without-re-estimating path R's `forecast::Arima(x,
   model=fit)` uses) and never calls an optimizer. So those four are `is_global`: the
   order/spec search runs once in `fit`, and `predict` conditions on each rolling origin's
   growing `history` through `forward`. TBATS is the exception -- statsforecast couples its
   automatic component search to every `.fit()` and offers no `forward` -- so it stays
   local, refit per origin; see its docstring.

   Two things `forward` deliberately does *not* freeze, both fine and worth knowing:

   - Deterministic decompositions are re-derived per origin. AutoTheta re-runs its
     seasonality test and `seasonal_decompose` inside `forward`, and MSTL re-runs STL, both
     over the origin's own history. No parameter is re-estimated -- this is the direct
     analogue of the ML methods recomputing their sin/cos features per origin (see
     `methods.ml`). One consequence for Theta: its decompose decision, and its
     multiplicative-vs-additive choice, can differ between origins.
   - The initial state stays pinned to the fit window's start (ETS's `init_state`, Theta's
     `initial_smoothed`) while `predict` filters a history beginning earlier. Exponential
     smoothing forgets initial conditions geometrically, so this washes out; it is a
     caveat, not a defect.

2. Non-seasonal by default. AutoARIMA with a seasonal period m fits SARIMA models whose
   cost explodes with m; seasonal ARIMA at sub-daily frequency (m=96 for 15-min, even
   m=24) is impractical. AutoETS is likewise impractical at large m (its seasonal state
   grows with m). So AutoARIMA/AutoETS here are the short-memory autoregressive baselines
   (m=1); the seasonal classical counterparts are the MSTL forecasters below, which strip
   seasonality with a cheap decomposition before fitting a trend model, and AutoTheta,
   which deseasonalizes cheaply. Pass season_length explicitly to force a seasonal
   AutoARIMA/AutoETS fit on a low-period case.

3. Capped *search* window, uncapped roll-forward. Order/spec-search cost grows with length,
   so we search on the most recent max_train_steps observations. Truncating from the front
   does not move the forecast origin, so test alignment is preserved. `predict` is
   deliberately *not* capped: it forwards over the whole history it is handed, because that
   is what the fitted model's state at the origin actually implies, and truncating would
   substitute an arbitrary re-initialization for it.

   That costs real time, unevenly, and the benchmark reports it rather than engineering
   around it -- `predict_seconds` is one of the numbers on display. On a 200,000-point
   history: AutoARIMA ~0.03s and AutoETS ~0.01s per origin, but AutoTheta ~110s (superlinear
   -- 0.02s at 5,000, 2.5s at 50,000) and MSTL ~28s for periods [24, 168], since STL cost
   grows with n x period. A rolling run over a long record is therefore slow for Theta and
   MSTL. That is a true fact about those methods at that history length, and capping their
   context to hide it would also change their forecasts.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.bench.methods.scorecard import Capabilities, Level, Scorecard
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


class _ForwardForecaster(Forecaster):
    """Shared fit/predict for the classical methods that roll forward via statsforecast's
    `forward(y, h)` (see choice 1 in the module docstring).

    Subclasses supply `_make_model`, returning an unfitted statsforecast model configured
    for the case; everything else -- capping the search window, fitting on its values once,
    and forwarding each origin's full history through the fitted model -- is identical across
    them, so it lives here rather than being copied four times.
    """

    family = "classical"
    is_global = True

    def _make_model(self, case: Case, df: pd.DataFrame):
        raise NotImplementedError

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        df = _prepare_train(train, self.max_train_steps)
        self._model = self._make_model(case, df)
        self._model.fit(df["y"].to_numpy(dtype=float))

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        y = pd.to_numeric(history["y"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
        y_hat = np.asarray(self._model.forward(y, horizon)["mean"], dtype=np.float64)
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})


class AutoARIMAForecaster(_ForwardForecaster):
    """Stepwise-order-search ARIMA via statsforecast, fit once and rolled forward.

    `fit` runs the stepwise order search once; each origin then calls `forward`, which
    rebuilds the ARIMA recursion over that origin's history with the fitted order and
    coefficients held fixed. That is statsforecast's own supported path (`Arima(x=y,
    model=fitted)`), not a reimplementation, so the forecast is exactly what statsforecast
    would produce for this model on this history.
    """

    name = "AutoARIMA"
    library = "statsforecast (Nixtla)"
    compute_profile = (
        "CPU only; a stepwise order search at fit time, typically seconds per case (searched "
        "on the most recent 5,000 points by default); each origin after that is one "
        "fixed-parameter `forward` pass over that origin's whole history — ~0.03s even on a "
        "200,000-point one, the cheapest roll-forward of the classical methods here."
    )
    limitations = (
        "Non-seasonal by default (season_length=1) — a large seasonal period would "
        "explode SARIMA's search cost, so seasonal structure isn't modeled unless you "
        "pass season_length explicitly. Univariate: ignores exogenous inputs. Order and "
        "coefficients are fixed at fit time, so a structural change in the series after "
        "fit is tracked through the recursion's state, not through re-estimation."
    )
    code_snippet = (
        "from statsforecast.models import AutoARIMA\n\n"
        "model = AutoARIMA(season_length=1)\n"
        "model.fit(train_y)                   # train_y: 1D array -- order search once\n\n"
        "# each origin: fixed-parameter recursion over the growing history, no re-fit\n"
        "y_hat = model.forward(history_y, horizon)['mean']"
    )
    # Non-seasonal here by design (season_length=1, see choice 2 in the module docstring).
    capabilities = Capabilities(
        trend=True, seasonality=False, multiple_seasonality=False, exogenous=False
    )
    # Both costs Low on measured numbers: seconds for the one order search, ~0.03s per
    # origin even on a 200,000-point history — the cheapest roll-forward of the classicals.
    scorecard = Scorecard(
        history_requirements=Level.MEDIUM,
        interpretability=Level.MEDIUM,
        fit_cost=Level.LOW,
        inference_cost=Level.LOW,
        tuning_effort=Level.LOW,
        robustness_to_noise=Level.MEDIUM,
    )

    def __init__(self, season_length: int = 1, max_train_steps: int = 5000):
        self.season_length = season_length
        self.max_train_steps = max_train_steps

    def _make_model(self, case: Case, df: pd.DataFrame):
        from statsforecast.models import AutoARIMA

        return AutoARIMA(season_length=self.season_length)


class AutoETSForecaster(_ForwardForecaster):
    """Exponential smoothing state space (auto-selected) via statsforecast, fit once and
    rolled forward.

    `fit` searches the error/trend/season combinations once; each origin then calls
    `forward`, which re-runs the smoothing recursion over that origin's history with the
    selected spec and its smoothing parameters held fixed.

    Like AutoARIMA, season_length defaults to 1 (non-seasonal); set explicitly for
    cases where a small seasonal period is appropriate. max_train_steps caps fit cost.
    """

    name = "AutoETS"
    library = "statsforecast (Nixtla)"
    compute_profile = (
        "CPU only; searches over error/trend/season state combinations at fit time, "
        "typically seconds per case (searched on the most recent 5,000 points by default); "
        "each origin after that is one fixed-parameter `forward` pass over that origin's "
        "whole history — ~0.01s even on a 200,000-point one."
    )
    limitations = (
        "Non-seasonal by default (season_length=1), like AutoARIMA — its seasonal "
        "state grows with the period, so a large seasonal period is impractical unless "
        "you pass season_length explicitly. Univariate: ignores exogenous inputs. Spec "
        "and smoothing parameters are fixed at fit time."
    )
    code_snippet = (
        "from statsforecast.models import AutoETS\n\n"
        "model = AutoETS(season_length=1)\n"
        "model.fit(train_y)                   # train_y: 1D array -- spec search once\n\n"
        "# each origin: fixed-parameter smoothing over the growing history, no re-fit\n"
        "y_hat = model.forward(history_y, horizon)['mean']"
    )

    # Non-seasonal here by design (season_length=1), like AutoARIMA.
    capabilities = Capabilities(
        trend=True, seasonality=False, multiple_seasonality=False, exogenous=False
    )
    # Named level/trend components make it the more readable of the two, and ~0.01s per
    # origin makes it the cheapest to roll forward.
    scorecard = Scorecard(
        history_requirements=Level.MEDIUM,
        interpretability=Level.HIGH,
        fit_cost=Level.LOW,
        inference_cost=Level.LOW,
        tuning_effort=Level.LOW,
        robustness_to_noise=Level.MEDIUM,
    )

    def __init__(self, season_length: int = 1, max_train_steps: int = 5000):
        self.season_length = season_length
        self.max_train_steps = max_train_steps

    def _make_model(self, case: Case, df: pd.DataFrame):
        from statsforecast.models import AutoETS

        return AutoETS(season_length=self.season_length)


class TBATSForecaster(Forecaster):
    """TBATS (Trigonometric, Box-Cox, ARMA, Trend, Seasonal) via statsforecast.

    Handles complex and multiple seasonal patterns via trigonometric terms.
    season_length accepts an int, a list of ints (e.g. [24, 168] for
    daily+weekly in hourly data), or None to use the case's declared
    dominant period (1, non-seasonal, when seasonalities = []).
    max_train_steps is set lower than other classicals because TBATS fit
    cost is substantially higher.

    The one classical method here that is *not* `is_global`. statsforecast couples TBATS's
    automatic component search to every `.fit()` call and exposes no `forward`, so there is
    no way to hold its parameters fixed while conditioning on new history -- the only
    fit-once option would be to substitute a different, simplified model under the TBATS
    name, which would stop the column meaning what it says. So it is refit at each rolling
    origin instead, and its numbers carry the caveat in `limitations` below.

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
        "per case, but noticeably slower than AutoARIMA/AutoETS/Theta. Refit at every "
        "rolling origin, so `fit_seconds` is the cost of one such refit."
    )
    limitations = (
        "Univariate: ignores exogenous inputs. Its internal Box-Cox transform assumes "
        "strictly positive data; a Yeo-Johnson fallback (see below) handles series "
        "that dip to zero or negative. Unlike the other classical methods, this one is "
        "refit from scratch at every rolling origin — statsforecast ties its component "
        "search to each fit and offers no way to roll a fitted TBATS forward — so it "
        "re-estimates on more recent data than the fit-once methods do, and its accuracy "
        "is not measured under quite the same protocol as theirs."
    )
    code_snippet = (
        "from statsforecast import StatsForecast\n"
        "from statsforecast.models import TBATS\n\n"
        "sf = StatsForecast(models=[TBATS(season_length=season_length)], freq=case.freq)\n"
        "sf.fit(train_df)                     # train_df: unique_id, ds, y\n"
        'y_hat = sf.predict(h=horizon)["TBATS"]'
    )
    # Trigonometric seasonal terms, and season_length takes a list — the classical method
    # built for several periods at once.
    capabilities = Capabilities(
        trend=True, seasonality=True, multiple_seasonality=True, exogenous=False
    )
    # The inverse profile of Theta and MSTL below: predict is a plain lookup off an already
    # fitted model, but there is no `forward`, so the component search reruns at every
    # origin and the whole cost sits in fit_cost.
    scorecard = Scorecard(
        history_requirements=Level.MEDIUM,
        interpretability=Level.MEDIUM,
        fit_cost=Level.MEDIUM,
        inference_cost=Level.LOW,
        tuning_effort=Level.LOW,
        robustness_to_noise=Level.MEDIUM,
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


class ThetaForecaster(_ForwardForecaster):
    """Theta method (auto-optimized) via statsforecast — a strong, cheap classical
    benchmark, often the one to beat among non-ML methods.

    Fit once: `fit` optimizes the Theta parameters, and each origin calls `forward`, which
    re-runs the recursive filter over that origin's history without touching the optimizer.
    Note that `forward` also re-runs AutoTheta's seasonality test and seasonal
    decomposition on that history, so the deseasonalization -- unlike the parameters -- is
    re-derived per origin (see choice 1 in the module docstring).

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
    library = "statsforecast (Nixtla)"
    compute_profile = (
        "CPU only; a cheap deseasonalize/fit/re-seasonalize decomposition at fit time "
        "(searched on the most recent 8,000 points). Cheap to fit, but the most expensive "
        "roll-forward here: each origin re-derives the decomposition and re-runs the "
        "fixed-parameter filter over that origin's whole history, at a cost superlinear in "
        "its length — 0.02s at 5,000 points, ~2.5s at 50,000, ~110s at 200,000."
    )
    limitations = (
        "Univariate: ignores exogenous inputs. Parameters are fixed at fit time, but the "
        "deseasonalization is a fixed classical decomposition re-derived from each "
        "origin's history rather than a learned model, so it can lag a genuinely shifting "
        "seasonal shape and its multiplicative-vs-additive choice can differ between "
        "origins."
    )
    code_snippet = (
        "from statsforecast.models import AutoTheta\n\n"
        "model = AutoTheta(season_length=season_length)\n"
        "model.fit(train_y)                   # train_y: 1D array -- picks params once\n\n"
        "# each origin: fixed-parameter filter over the growing history, no re-fit\n"
        "y_hat = model.forward(history_y, horizon)['mean']"
    )

    # Deseasonalizes via a cheap classical decomposition on the case's dominant period —
    # one period, not several.
    capabilities = Capabilities(
        trend=True, seasonality=True, multiple_seasonality=False, exogenous=False
    )
    # Cheap to fit, but the most expensive roll-forward here: the per-origin cost is
    # superlinear in history length — 0.02s at 5,000 points, ~110s at 200,000.
    scorecard = Scorecard(
        history_requirements=Level.MEDIUM,
        interpretability=Level.HIGH,
        fit_cost=Level.LOW,
        inference_cost=Level.HIGH,
        tuning_effort=Level.LOW,
        robustness_to_noise=Level.MEDIUM,
    )

    def __init__(self, season_length: int | None = None, max_train_steps: int = 8000):
        self.season_length = season_length
        self.max_train_steps = max_train_steps

    def _make_model(self, case: Case, df: pd.DataFrame):
        from statsforecast.models import AutoTheta

        from tsforecastbench.data.seasonality import dominant_period

        season_length = (
            self.season_length
            if self.season_length is not None
            else dominant_period(case.seasonalities, df["y"].to_numpy())
        )
        return AutoTheta(season_length=season_length)


class _MSTLBase(_ForwardForecaster):
    """MSTL decomposition + a classical trend-cycle/residual forecaster (statsforecast),
    fit once and rolled forward.

    MSTL removes one or more seasonal components with a fast STL-based decomposition,
    fits the subclass's `trend_forecaster` on the seasonally-adjusted series, and
    re-applies the seasonal pattern to the forecast. This is the seasonal classical
    counterpart to the non-seasonal AutoARIMA/AutoETS baselines, and works uniformly
    across frequencies with large or multiple seasonal periods (168h weekly, 365d yearly)
    that native seasonal ARIMA/ETS cannot afford.

    `fit` runs the decomposition and the trend fit once. Each origin then calls MSTL's
    `forward`, which re-derives STL over that origin's history, forwards the
    seasonally-adjusted series through the trend forecaster's own fixed-parameter
    `forward`, and re-seasonalizes -- so the seasonal shape stays current while no
    parameter is re-estimated.

    Periods come from the case's declared `seasonalities`, filtered to those the training
    series is long enough to support (STL needs >=2 full cycles). If the case is declared
    non-seasonal or none survive, MSTL is genuinely inapplicable and fit raises rather than
    silently degenerating into the plain non-seasonal trend forecaster (which would just
    duplicate the AutoARIMA/AutoETS column).
    """

    name = "MSTL"
    max_train_steps = 8000
    library = "statsforecast (Nixtla)"
    compute_profile = (
        "CPU only; a fast STL-based decomposition plus the subclass's trend forecaster, on "
        "the most recent 8,000 points — typically seconds per case. Each origin after that "
        "re-derives STL over that origin's whole history and forwards the trend model with "
        "its parameters fixed; STL cost grows with history x period, so a long record is "
        "slow — ~0.7s at 5,000 points up to ~28s at 200,000 for periods [24, 168]."
    )
    limitations = (
        "Univariate: ignores exogenous inputs. Needs at least 2 full cycles of each "
        "declared seasonal period in the training window to fit at all; raises rather "
        "than degenerating into a duplicate non-seasonal forecast when a case has no "
        "seasonality it can use. The trend model's parameters are fixed at fit time; the "
        "STL decomposition around them is re-derived from each origin's full history, "
        "which keeps the seasonal shape current but makes predict the expensive half on a "
        "long record."
    )
    code_snippet = (
        "from statsforecast.models import MSTL\n\n"
        "model = MSTL(season_length=periods, trend_forecaster=<trend_forecaster>)\n"
        "model.fit(train_y)                   # train_y: 1D array -- decompose + fit once\n\n"
        "# each origin: re-derive STL, forward the fixed-parameter trend model, reseasonalize\n"
        "y_hat = model.forward(history_y, horizon)['mean']"
    )
    # Shared by both MSTL variants: the seasonal handling and the cost profile come from the
    # decomposition, not from which trend model sits inside it. Declared here rather than
    # twice below so the two can't drift apart.
    capabilities = Capabilities(
        trend=True, seasonality=True, multiple_seasonality=True, exogenous=False
    )
    # Needs >=2 full cycles of every declared period before it will fit at all, and STL cost
    # grows with history x period, so predict is the expensive half — ~0.7s per origin at
    # 5,000 points up to ~28s at 200,000 for periods [24, 168].
    scorecard = Scorecard(
        history_requirements=Level.MEDIUM,
        interpretability=Level.MEDIUM,
        fit_cost=Level.LOW,
        inference_cost=Level.HIGH,
        tuning_effort=Level.LOW,
        robustness_to_noise=Level.MEDIUM,
    )

    def _make_trend_forecaster(self):
        raise NotImplementedError

    def _make_model(self, case: Case, df: pd.DataFrame):
        from statsforecast.models import MSTL

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

        return MSTL(season_length=season_length, trend_forecaster=self._make_trend_forecaster())


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
