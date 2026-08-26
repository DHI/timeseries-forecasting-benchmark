"""Windowed ML methods - Nixtla mlforecast with three sklearn-style estimators.

LinearLags (Ridge), LightGBM and MLP share one lag + calendar feature pipeline; they
differ only in the underlying regressor. Calendar/seasonal features (date parts plus a
sin/cos encoding of the case's dominant period) are always included — deterministic and
free for any future timestamp, unlike exogenous covariates. Each comes in two flavours:

  * univariate (target lags + calendar features only) — the Tier-1 column;
  * exogenous-aware (``use_exogenous=True``) — additionally feeds the case's declared
    covariates as dynamic features. At predict time their future values come from
    `future_exog` (a perfect-forecast proxy supplied by the eval loop), which is the
    Tier-2 column. The univariate/exogenous pair exposes how much the covariates help.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from tsforecastbench.bench.eval.budget import MLP_MAX_EPOCHS, TRAIN_POINT_CEILING
from tsforecastbench.bench.methods.base import Forecaster, _lags_for
from tsforecastbench.bench.methods.scorecard import (
    Capabilities,
    ForecastStrategy,
    GrowthShape,
    Level,
    RunningCost,
    Scorecard,
)
from tsforecastbench.data.schema import Case
from tsforecastbench.data.seasonality import dominant_period, position_in_cycle


def _date_features_for(freq: str) -> list[str]:
    f = freq.lower()
    if f in {"1d", "d"}:
        return ["dayofyear", "dayofweek", "month"]
    if f in {"1h", "h"}:
        return ["hour", "dayofweek", "dayofyear"]
    if f in {"15min", "15t"}:
        return ["hour", "minute", "dayofweek", "dayofyear"]
    return ["dayofweek", "month"]


def _sincos_features_for(case: Case, y: np.ndarray) -> list:
    """Sin/cos encoding of the case's dominant period ([] -> no cyclic features). The
    dominant period is chosen from the declared seasonalities by variance explained on the
    fit-window target `y` (see `dominant_period`).

    MLForecast passes a DatetimeIndex to each callable (no .dt accessor) and names the
    feature column after the callable's __name__. Position within the cycle comes from
    the shared `position_in_cycle` helper, the same rule the Climatology baseline keys on.
    """
    p = dominant_period(case.seasonalities, y)
    if p == 1:
        return []
    freq = case.freq

    def sin_feat(ds):
        return np.sin(2 * np.pi * position_in_cycle(freq, p, ds) / p)

    def cos_feat(ds):
        return np.cos(2 * np.pi * position_in_cycle(freq, p, ds) / p)

    sin_feat.__name__ = f"sin{p}"
    cos_feat.__name__ = f"cos{p}"
    return [sin_feat, cos_feat]


def _exo_cols(case: Case) -> list[str]:
    """Declared covariates, de-duplicated (a column may be both historical and known-future)."""
    return list(dict.fromkeys([*case.exogenous, *case.exogenous_known_future]))


def _naive_ds(ds) -> pd.Series:
    ds = pd.to_datetime(pd.Series(ds) if not isinstance(ds, pd.Series) else ds, utc=True)
    return ds.dt.tz_convert("UTC").dt.tz_localize(None)


def _fit_len(vals: np.ndarray, n: int, fill: float) -> np.ndarray:
    if len(vals) >= n:
        return vals[:n]
    out = np.full(n, fill, dtype=float)
    out[: len(vals)] = vals
    return out


class _WindowedMLBase(Forecaster):
    """Shared lag-feature pipeline; subclasses provide the sklearn regressor.

    `max_train_steps` bounds the fit to the most recent N points. It defaults to
    `eval.budget`'s ceiling, which is a guard against a future oversized case rather than a
    training budget, and does not bind on any case here — see that module on why the old
    per-method caps were removed (issue #95). Truncation does not move the forecast origin,
    so test alignment holds wherever it does bind.
    """

    name = "WindowedML"
    family = "ml"
    is_global = True
    pools_segments = True
    max_train_steps: int = TRAIN_POINT_CEILING
    use_exogenous: bool = False
    direct_horizon: bool = False
    # BLAS threads allowed for the duration of the fit. `None` leaves the process default
    # untouched, which is what the tree and Ridge regressors want; a subclass sets a small
    # integer when its estimator's matmuls are too small to parallelise (see MLPForecaster).
    blas_threads: int | None = None
    compute_profile = (
        "CPU only; a lag+calendar feature table is built once, then the regressor "
        "fits on it directly — seconds for the linear/tree models here. The MLP costs "
        "far more: it holds out an early-stopping split, and its direct multi-horizon "
        "strategy trains one network per horizon step, so its fit scales with the "
        "horizon as well as with the history. Its fit is pinned to a single BLAS thread, "
        "whose matmuls are too small for multi-threading to do anything but add "
        "synchronisation overhead."
    )
    limitations = (
        "Needs enough history to fill its lag windows (shortest lag 1, longest tied "
        "to the case's declared seasonal periods). The univariate variant ignores "
        "exogenous inputs entirely; the +exo variant sees only the future covariate "
        "values supplied at predict time (a perfect-forecast proxy in this "
        "benchmark, not a real forecast of those covariates)."
    )
    # Shared by every regressor in this pipeline, since these follow from the feature table
    # rather than from the estimator. Lags cover all of the case's declared periods and the
    # date features span several cycles at once, so multiple seasonality is handled. The
    # pipeline can feed covariates for any of these regressors (that is all `use_exogenous`
    # switches), whether or not a +exo column happens to be registered for it — but subclasses
    # whose estimator cannot leave the training target's range override `trend` to False.
    capabilities = Capabilities(
        trend=True, seasonality=True, multiple_seasonality=True, exogenous=True
    )

    def __init__(self):
        self.uses_exogenous = self.use_exogenous
        self._exo: list[str] = []

    def _make_model(self):
        raise NotImplementedError

    def fit(self, case: Case, train: pd.DataFrame, val: pd.DataFrame) -> None:
        from mlforecast import MLForecast

        from tsforecastbench.bench.eval.loop import _horizon_to_steps

        self._exo = _exo_cols(case) if self.use_exogenous else []
        cols = ["unique_id", "ds", "y", *self._exo]
        df = train[[c for c in cols if c in train.columns]].copy()
        # The extended-history frame carries one unique_id per contiguous segment;
        # mlforecast treats them as separate series, so no lag feature ever spans a gap.
        # The tail segment (latest ds) is the one whose forecast we return at predict.
        self._target_uid = df["unique_id"].iloc[-1]
        df = df.groupby("unique_id", sort=False).tail(self.max_train_steps)
        df["ds"] = _naive_ds(df["ds"]).to_numpy()
        for c in self._exo:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Last known covariate value — the predict-time fallback when a future exogenous
        # value is missing from the perfect-forecast proxy (not training gap-fill: the
        # curated frame already delivers complete exogenous columns within the span).
        self._last_exo = {
            c: float(df[c].ffill().iloc[-1]) if c in df and df[c].notna().any() else 0.0
            for c in self._exo
        }

        date_feats = [
            *_date_features_for(case.freq),
            *_sincos_features_for(case, df["y"].to_numpy()),
        ]

        self._fcst = MLForecast(
            models=[self._make_model()],
            freq=case.freq,
            lags=_lags_for(case),
            date_features=date_feats,
        )
        # direct_horizon trains one model per horizon step from the origin's lag features
        # instead of one recursive model whose own predictions feed back in as later lag
        # inputs (see MLPForecaster's docstring for why it needs this).
        max_horizon = (
            _horizon_to_steps(case.horizon_primary, case.freq) if self.direct_horizon else None
        )
        # `limits=None` is a documented no-op, so the unrestricted regressors take the same
        # path. Only "blas" is filtered: LightGBM and CatBoost run their own OpenMP pools,
        # which are productive and must not be capped along with numpy's.
        with threadpool_limits(limits=self.blas_threads, user_api="blas"):
            self._fcst.fit(df, static_features=[], max_horizon=max_horizon)
        # mlforecast names prediction columns after its own alias for each model (the
        # underlying estimator's class name, unwrapped from any Pipeline) — read that
        # alias directly rather than re-deriving it from the fitted object's type, which
        # would report "Pipeline" for models wrapped in one.
        self._model_col = next(iter(self._fcst.models))

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        # Inference-only: condition the trained regressor on the tail segment's history as
        # handed in at this origin (mlforecast's `new_df`), never the frame captured at
        # fit — so one fit serves every rolling origin. Only the tail segment's series is
        # forecast; a multi-segment extended fit's other unique_ids were training-only.
        sub = history[history["unique_id"] == self._target_uid]
        new_df = sub[[c for c in ["unique_id", "ds", "y", *self._exo] if c in sub.columns]].copy()
        new_df["ds"] = _naive_ds(new_df["ds"]).to_numpy()
        for c in self._exo:
            new_df[c] = pd.to_numeric(new_df[c], errors="coerce")
        if self._exo:
            x_df = self._future_X(case, history, horizon, future_exog)
            preds = self._fcst.predict(h=horizon, new_df=new_df, X_df=x_df)
        else:
            preds = self._fcst.predict(h=horizon, new_df=new_df)
        y_hat = preds[self._model_col].to_numpy().astype(np.float64)
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})

    def _future_X(
        self, case: Case, history: pd.DataFrame, horizon: int, future_exog: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Build the mlforecast X_df of future covariate values for the horizon."""
        last = _naive_ds(history["ds"].max()).iloc[0]
        fidx = pd.date_range(start=last, periods=horizon + 1, freq=case.freq)[1:]
        data: dict = {"unique_id": self._target_uid, "ds": fidx}
        for c in self._exo:
            fill = self._last_exo.get(c, 0.0)
            if future_exog is not None and c in future_exog.columns:
                vals = _fit_len(
                    pd.to_numeric(future_exog[c], errors="coerce").to_numpy(), horizon, fill
                )
            else:
                vals = np.full(horizon, fill)
            data[c] = np.where(np.isfinite(vals), vals, fill)
        return pd.DataFrame(data)


class LightGBMForecaster(_WindowedMLBase):
    """Gradient-boosted decision trees (LightGBM) over lag and calendar features."""

    name = "LightGBM"
    library = "mlforecast + lightgbm"
    code_snippet = (
        "from lightgbm import LGBMRegressor\n"
        "from mlforecast import MLForecast\n\n"
        "fcst = MLForecast(\n"
        "    models=[LGBMRegressor(verbosity=-1)],          # LightGBM's own defaults\n"
        "    freq=case.freq,\n"
        "    lags=[1, 2, 4, 8, 16, 32, *case.seasonalities],\n"
        '    date_features=["dayofweek", "month", ...],\n'
        ")\n"
        "fcst.fit(train_df, static_features=[])\n"
        "# +exo: train_df also carries the covariate columns; mlforecast picks them up\n"
        "#       automatically as dynamic features since they're not in static_features\n\n"
        "y_hat = fcst.predict(h=horizon, new_df=new_df)   # new_df: this origin's history so far\n"
        "# +exo: fcst.predict(h=horizon, new_df=new_df, X_df=future_exog) -- X_df supplies\n"
        "#       each future timestamp's covariate values (a perfect-forecast proxy here,\n"
        "#       not a real forecast of those covariates)"
    )
    # A tree's prediction is an average of training targets, so it cannot output a value
    # outside the range it was fit on — no trend extrapolation, by construction.
    capabilities = _WindowedMLBase.capabilities.model_copy(update={"trend": False})
    # mlforecast's default lag-feedback strategy: each step's prediction is fed back in as
    # the next step's lag input. A tree's leaf-bound output keeps this well-behaved.
    forecast_strategy = ForecastStrategy.RECURSIVE
    # Fit is seconds; the per-origin recursive prediction is what adds up, at ~7 minutes for a
    # thousand forecasts on a 100,000-point record. Both terms track the record length.
    scorecard = Scorecard(
        history_requirements=Level.HIGH,
        interpretability=Level.MEDIUM,
        running_cost=RunningCost(level=Level.MEDIUM, growth=GrowthShape.LINEAR),
        tuning_effort=Level.MEDIUM,
    )

    def _make_model(self):
        from lightgbm import LGBMRegressor

        # LightGBM's own defaults (n_estimators=100, learning_rate=0.1). The 200 rounds at
        # 0.05 that stood here were picked rather than taken from anywhere, and applied
        # unchanged to a 1,500-point case and a 150,000-point one. This benchmark does not
        # tune, so each library's default is the honest configuration to compare — see
        # CatBoost, whose default round count is an order of magnitude higher.
        return LGBMRegressor(verbosity=-1)


class CatBoostForecaster(_WindowedMLBase):
    """Gradient-boosted decision trees (CatBoost) over lag and calendar features."""

    name = "CatBoost"
    # Unlike LightGBM, whose sampling defaults are off, CatBoost's default bootstrap (MVS,
    # subsample ~0.8 for RMSE on CPU) resamples rows at every iteration, so its forecast
    # moves with the seed. It only looked deterministic because the library's own default
    # seed is 0 — reproducible, but one draw of many (issue #106).
    is_stochastic = True
    library = "mlforecast + catboost"
    code_snippet = (
        "from catboost import CatBoostRegressor\n"
        "from mlforecast import MLForecast\n\n"
        "fcst = MLForecast(\n"
        "    models=[CatBoostRegressor(verbose=False, random_seed=seed)],  # else its defaults\n"
        "    freq=case.freq,\n"
        "    lags=[1, 2, 4, 8, 16, 32, *case.seasonalities],\n"
        '    date_features=["dayofweek", "month", ...],\n'
        ")\n"
        "fcst.fit(train_df, static_features=[])\n"
        "# +exo: train_df also carries the covariate columns; mlforecast picks them up\n"
        "#       automatically as dynamic features since they're not in static_features\n\n"
        "y_hat = fcst.predict(h=horizon, new_df=new_df)   # new_df: this origin's history so far\n"
        "# +exo: fcst.predict(h=horizon, new_df=new_df, X_df=future_exog) -- X_df supplies\n"
        "#       each future timestamp's covariate values (a perfect-forecast proxy here,\n"
        "#       not a real forecast of those covariates)"
    )
    # Bounded by the training target range for the same reason LightGBM is.
    capabilities = _WindowedMLBase.capabilities.model_copy(update={"trend": False})
    # Same recursive lag-feedback strategy as LightGBM, same leaf-bound stability.
    forecast_strategy = ForecastStrategy.RECURSIVE
    # Same shape as LightGBM and about a third cheaper to run -- a two-second fit against a
    # per-origin recursion that grows with the record and accounts for nearly all of it.
    scorecard = Scorecard(
        history_requirements=Level.HIGH,
        interpretability=Level.MEDIUM,
        running_cost=RunningCost(level=Level.MEDIUM, growth=GrowthShape.LINEAR),
        tuning_effort=Level.MEDIUM,
    )

    def _make_model(self):
        from catboost import CatBoostRegressor

        # CatBoost's own defaults (iterations=1000, learning_rate auto-selected from the
        # data), for the reason given on LightGBM's `_make_model`. `verbose`/
        # `allow_writing_files` are kept: they silence output and stop it writing a
        # catboost_info/ directory, neither of which affects the model.
        return CatBoostRegressor(verbose=False, allow_writing_files=False, random_seed=self.seed)


class LinearLagsForecaster(_WindowedMLBase):
    """Ridge regression over lag and calendar features."""

    name = "LinearLags"
    library = "mlforecast + scikit-learn (Ridge)"
    code_snippet = (
        "from mlforecast import MLForecast\n"
        "from sklearn.linear_model import Ridge\n\n"
        "fcst = MLForecast(\n"
        "    models=[Ridge(alpha=1.0)],\n"
        "    freq=case.freq,\n"
        "    lags=[1, 2, 4, 8, 16, 32, *case.seasonalities],\n"
        '    date_features=["dayofweek", "month", ...],\n'
        ")\n"
        "fcst.fit(train_df, static_features=[])\n"
        "# +exo: train_df also carries the covariate columns; mlforecast picks them up\n"
        "#       automatically as dynamic features since they're not in static_features\n\n"
        "y_hat = fcst.predict(h=horizon, new_df=new_df)   # new_df: this origin's history so far\n"
        "# +exo: fcst.predict(h=horizon, new_df=new_df, X_df=future_exog) -- X_df supplies\n"
        "#       each future timestamp's covariate values (a perfect-forecast proxy here,\n"
        "#       not a real forecast of those covariates)"
    )
    # Keeps the base's trend=True: linear in its lag features, so unlike the tree models it
    # can extrapolate past the range it was fit on.
    # Same recursive lag-feedback strategy as LightGBM/CatBoost; Ridge's linearity keeps it
    # well-behaved instead of a leaf bound.
    forecast_strategy = ForecastStrategy.RECURSIVE
    # A closed-form fit in under a second, so essentially all of its ~4 minutes for a thousand
    # forecasts is the recursive prediction walking the lag features forward.
    scorecard = Scorecard(
        history_requirements=Level.MEDIUM,
        interpretability=Level.MEDIUM,
        running_cost=RunningCost(level=Level.MEDIUM, growth=GrowthShape.LINEAR),
        tuning_effort=Level.LOW,
    )

    def _make_model(self):
        from sklearn.linear_model import Ridge

        # Ridge over plain LinearRegression to be safe with collinear lag features
        return Ridge(alpha=1.0)


class MLPForecaster(_WindowedMLBase):
    """Multi-layer perceptron (scikit-learn MLPRegressor) over lag and calendar features."""

    name = "MLP"
    # `random_state` below drives weight init, the early-stopping holdout draw and Adam's
    # minibatch shuffling, so a different seed is a different fit.
    is_stochastic = True
    # Trains one model per horizon step from the origin's lag features instead of one
    # recursive model whose own predictions feed back in as later lag inputs — an
    # MLPRegressor's unbounded linear output layer had no guarantee against drifting
    # outside the training distribution under the recursive strategy every other
    # windowed model here uses (mlforecast's default), the way a tree's leaf-bound
    # output or Ridge's linearity keeps those well-behaved under it.
    direct_horizon = True
    # direct_horizon above is exactly this: mlforecast's direct multi-horizon strategy.
    forecast_strategy = ForecastStrategy.DIRECT
    # One BLAS thread, because this estimator's matmuls are far too small to parallelise:
    # a minibatch is 200 x ~15 features into a 64-wide layer, ~0.4 MFLOP, so spreading it
    # over a dozen cores costs more in thread synchronisation than the arithmetic it saves.
    # Measured on a 12-core box, one fit of identical data and iteration count: 37.3s
    # unrestricted against 0.50s here. Threading only starts paying for this pipeline at
    # roughly batch >= 2,000 *and* a 1,024-wide layer *and* ~500 features, all at once —
    # two of which sklearn and the feature table fix well below that. It also makes the fit
    # a stable measurement: unrestricted, four repeats of one fit ranged 2.72s to 0.38s as
    # the thread pool warmed, while this holds steady, and `running_cost` is meant to rest
    # on measured `fit_seconds` (issue #97).
    blas_threads = 1
    library = "mlforecast + scikit-learn (MLPRegressor)"
    code_snippet = (
        "from mlforecast import MLForecast\n"
        "from sklearn.neural_network import MLPRegressor\n"
        "from sklearn.pipeline import make_pipeline\n"
        "from sklearn.preprocessing import StandardScaler\n\n"
        "model = make_pipeline(\n"
        "    StandardScaler(),\n"
        "    MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True,\n"
        "                 max_iter=MLP_MAX_EPOCHS),   # guard, not a budget\n"
        ")\n"
        "fcst = MLForecast(\n"
        "    models=[model],\n"
        "    freq=case.freq,\n"
        "    lags=[1, 2, 4, 8, 16, 32, *case.seasonalities],\n"
        '    date_features=["dayofweek", "month", ...],\n'
        ")\n"
        "# one model per horizon step instead of one recursive model (see limitations)\n"
        "fcst.fit(train_df, static_features=[], max_horizon=horizon)\n"
        "# +exo: train_df also carries the covariate columns; mlforecast picks them up\n"
        "#       automatically as dynamic features since they're not in static_features\n\n"
        "y_hat = fcst.predict(h=horizon, new_df=new_df)   # new_df: this origin's history so far\n"
        "# +exo: fcst.predict(h=horizon, new_df=new_df, X_df=future_exog) -- X_df supplies\n"
        "#       each future timestamp's covariate values (a perfect-forecast proxy here,\n"
        "#       not a real forecast of those covariates)"
    )
    # The heaviest fit of the windowed-ML family (~37s at 100,000 points) against a direct
    # prediction, which keeps a thousand forecasts to ~4 minutes.
    scorecard = Scorecard(
        history_requirements=Level.HIGH,
        interpretability=Level.LOW,
        running_cost=RunningCost(level=Level.MEDIUM, growth=GrowthShape.LINEAR),
        tuning_effort=Level.HIGH,
    )

    def _make_model(self):
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # MLPRegressor is scale-sensitive; lags and calendar features (e.g. dayofyear up
        # to 366) span very different ranges, so scale them first.
        return make_pipeline(
            StandardScaler(),
            # `max_iter` is a guard against a fit that never converges, not a budget: see
            # `eval.budget.MLP_MAX_EPOCHS` for where it comes from. `early_stopping` is what
            # actually ends the fit, and it does so on a shuffled 10% of the training rows
            # rather than on the case's declared val split — scikit-learn exposes only
            # `validation_fraction`, with no way to pass an explicit holdout, so unlike the
            # torch and neuralforecast models this one validates on rows interleaved in time
            # with the rows it trains on. Known and deliberate for now (gh #104).
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=MLP_MAX_EPOCHS,
                early_stopping=True,
                random_state=self.seed,
            ),
        )


class MLPExoForecaster(MLPForecaster):
    name = "MLP+exo"
    use_exogenous = True


class LightGBMExoForecaster(LightGBMForecaster):
    name = "LightGBM+exo"
    use_exogenous = True


class LinearLagsExoForecaster(LinearLagsForecaster):
    name = "LinearLags+exo"
    use_exogenous = True


class CatBoostExoForecaster(CatBoostForecaster):
    name = "CatBoost+exo"
    use_exogenous = True
