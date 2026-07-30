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

from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.bench.methods.scorecard import (
    Capabilities,
    ForecastStrategy,
    Level,
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


def _lags_for(case: Case) -> list[int]:
    """Uniform geometric memory ladder plus the case's declared seasonal periods.

    The ladder is recent autoregressive state, identical for every case and frequency —
    it makes no cycle claim (the regressor is free to ignore any lag). Seasonal structure
    enters only through the declared periods, keeping the case.toml the single source of
    seasonal configuration."""
    ladder = [1, 2, 4, 8, 16, 32]
    return sorted(set(ladder) | set(case.seasonalities))


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

    `max_train_steps` caps the fit to the most recent N points. None = no cap. Useful
    for heavy methods (MLP) on long high-frequency series; truncation does not move the
    forecast origin so test alignment holds.
    """

    name = "WindowedML"
    family = "ml"
    is_global = True
    pools_segments = True
    max_train_steps: int | None = None
    use_exogenous: bool = False
    direct_horizon: bool = False
    compute_profile = (
        "CPU only; a lag+calendar feature table is built once, then the regressor "
        "fits on it directly — fast for the linear/tree models here, slower for the "
        "MLP (which also cross-validates an early-stopping split)."
    )
    limitations = (
        "Needs enough history to fill its lag windows (shortest lag 1, longest tied "
        "to the case's declared seasonal periods). The univariate variant ignores "
        "exogenous inputs entirely; the +exo variant sees only the future covariate "
        "values supplied at predict time (a perfect-forecast proxy in this "
        "benchmark, not a real forecast of those covariates)."
    )
    code_snippet = (
        "from mlforecast import MLForecast\n\n"
        "fcst = MLForecast(\n"
        "    models=[<regressor>],           # e.g. LGBMRegressor, Ridge, MLPRegressor\n"
        "    freq=case.freq,\n"
        "    lags=[1, 2, 4, 8, 16, 32, *case.seasonalities],\n"
        '    date_features=["dayofweek", "month", ...],\n'
        ")\n"
        "fcst.fit(train_df, static_features=[])\n"
        "y_hat = fcst.predict(h=horizon, ids=[target_uid])"
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

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from mlforecast import MLForecast

        from tsforecastbench.bench.eval.loop import _horizon_to_steps

        self._exo = _exo_cols(case) if self.use_exogenous else []
        cols = ["unique_id", "ds", "y", *self._exo]
        df = train[[c for c in cols if c in train.columns]].copy()
        # The extended-history frame carries one unique_id per contiguous segment;
        # mlforecast treats them as separate series, so no lag feature ever spans a gap.
        # The tail segment (latest ds) is the one whose forecast we return at predict.
        self._target_uid = df["unique_id"].iloc[-1]
        if self.max_train_steps:
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
    # A tree's prediction is an average of training targets, so it cannot output a value
    # outside the range it was fit on — no trend extrapolation, by construction.
    capabilities = _WindowedMLBase.capabilities.model_copy(update={"trend": False})
    # mlforecast's default lag-feedback strategy: each step's prediction is fed back in as
    # the next step's lag input. A tree's leaf-bound output keeps this well-behaved.
    forecast_strategy = ForecastStrategy.RECURSIVE
    # Feature importances make it inspectable but not readable; needs a lot of rows before
    # the lag table is worth fitting, and n_estimators/learning_rate are real choices.
    scorecard = Scorecard(
        history_requirements=Level.HIGH,
        interpretability=Level.MEDIUM,
        fit_cost=Level.LOW,
        inference_cost=Level.LOW,
        tuning_effort=Level.MEDIUM,
        robustness_to_noise=Level.MEDIUM,
    )

    def _make_model(self):
        from lightgbm import LGBMRegressor

        return LGBMRegressor(n_estimators=200, learning_rate=0.05, verbosity=-1)


class CatBoostForecaster(_WindowedMLBase):
    """Gradient-boosted decision trees (CatBoost) over lag and calendar features."""

    name = "CatBoost"
    library = "mlforecast + catboost"
    # Bounded by the training target range for the same reason LightGBM is.
    capabilities = _WindowedMLBase.capabilities.model_copy(update={"trend": False})
    # Same recursive lag-feedback strategy as LightGBM, same leaf-bound stability.
    forecast_strategy = ForecastStrategy.RECURSIVE
    scorecard = Scorecard(
        history_requirements=Level.HIGH,
        interpretability=Level.MEDIUM,
        fit_cost=Level.LOW,
        inference_cost=Level.LOW,
        tuning_effort=Level.MEDIUM,
        robustness_to_noise=Level.MEDIUM,
    )

    def _make_model(self):
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=200, learning_rate=0.05, verbose=False, allow_writing_files=False
        )


class LinearLagsForecaster(_WindowedMLBase):
    """Ridge regression over lag and calendar features."""

    name = "LinearLags"
    library = "mlforecast + scikit-learn (Ridge)"
    # Keeps the base's trend=True: linear in its lag features, so unlike the tree models it
    # can extrapolate past the range it was fit on.
    # Same recursive lag-feedback strategy as LightGBM/CatBoost; Ridge's linearity keeps it
    # well-behaved instead of a leaf bound.
    forecast_strategy = ForecastStrategy.RECURSIVE
    # Cheapest of the windowed models to fit, with one meaningful knob (alpha), and readable
    # coefficients — though dozens of them, so inspectable rather than at-a-glance.
    scorecard = Scorecard(
        history_requirements=Level.MEDIUM,
        interpretability=Level.MEDIUM,
        fit_cost=Level.LOW,
        inference_cost=Level.LOW,
        tuning_effort=Level.LOW,
        robustness_to_noise=Level.MEDIUM,
    )

    def _make_model(self):
        from sklearn.linear_model import Ridge

        # Ridge over plain LinearRegression to be safe with collinear lag features
        return Ridge(alpha=1.0)


class MLPForecaster(_WindowedMLBase):
    """Multi-layer perceptron (scikit-learn MLPRegressor) over lag and calendar features."""

    name = "MLP"
    max_train_steps = 20000
    # Trains one model per horizon step from the origin's lag features instead of one
    # recursive model whose own predictions feed back in as later lag inputs — an
    # MLPRegressor's unbounded linear output layer had no guarantee against drifting
    # outside the training distribution under the recursive strategy every other
    # windowed model here uses (mlforecast's default), the way a tree's leaf-bound
    # output or Ridge's linearity keeps those well-behaved under it.
    direct_horizon = True
    # direct_horizon above is exactly this: mlforecast's direct multi-horizon strategy.
    forecast_strategy = ForecastStrategy.DIRECT
    library = "mlforecast + scikit-learn (MLPRegressor)"
    # No override needed: an MLP's linear output layer can extrapolate past the training
    # range like the base's trend=True already claims, now that nothing clips it back.
    # Several interacting knobs (layer sizes, iterations, early stopping, feature
    # scaling), and the slowest of the four to fit — direct_horizon means training H
    # independent models instead of one, and it's capped to the most recent 20,000
    # points for the same reason.
    scorecard = Scorecard(
        history_requirements=Level.HIGH,
        interpretability=Level.LOW,
        fit_cost=Level.HIGH,
        inference_cost=Level.LOW,
        tuning_effort=Level.HIGH,
        robustness_to_noise=Level.LOW,
    )

    def _make_model(self):
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # MLPRegressor is scale-sensitive; lags and calendar features (e.g. dayofyear up
        # to 366) span very different ranges, so scale them first.
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=300,
                early_stopping=True,
                random_state=0,
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
