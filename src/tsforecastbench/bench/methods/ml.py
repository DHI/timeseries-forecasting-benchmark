"""Windowed ML methods - Nixtla mlforecast with three sklearn-style estimators.

LinearLags (Ridge), LightGBM and MLP share one lag + calendar feature pipeline; they
differ only in the underlying regressor. Each comes in two flavours:

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
from tsforecastbench.data.schema import Case
from tsforecastbench.data.seasonality import dominant_period, normalize_freq


def _date_features_for(freq: str) -> list[str]:
    f = freq.lower()
    if f in {"1d", "d"}:
        return ["dayofyear", "dayofweek", "month"]
    if f in {"1h", "h"}:
        return ["hour", "dayofweek", "dayofyear"]
    if f in {"15min", "15t"}:
        return ["hour", "minute", "dayofweek", "dayofyear"]
    return ["dayofweek", "month"]


# Position-in-cycle from calendar fields for the canonical (freq, period) pairs: exact
# across leap years and DST-free naive timestamps, unlike an epoch-anchored step count.
_CANONICAL_POSITIONS = {
    ("1d", 7): lambda ds: ds.dayofweek,
    ("1d", 365): lambda ds: ds.dayofyear - 1,
    ("1h", 12): lambda ds: ds.hour % 12,
    ("1h", 24): lambda ds: ds.hour,
    ("1h", 168): lambda ds: ds.dayofweek * 24 + ds.hour,
    ("15min", 96): lambda ds: ds.hour * 4 + ds.minute // 15,
    ("15min", 672): lambda ds: ds.dayofweek * 96 + ds.hour * 4 + ds.minute // 15,
}


def _sincos_features_for(case: Case) -> list:
    """Sin/cos encoding of the case's declared dominant period ([] -> no cyclic features).

    MLForecast passes a DatetimeIndex to each callable (no .dt accessor) and names the
    feature column after the callable's __name__.
    """
    p = dominant_period(case.seasonalities)
    if p == 1:
        return []
    pos = _CANONICAL_POSITIONS.get((normalize_freq(case.freq), p))
    if pos is None:
        # Non-canonical period: position = epoch-anchored step count modulo p, identical
        # at fit and predict time because both use the same naive timestamps.
        step_ns = pd.Timedelta(pd.tseries.frequencies.to_offset(case.freq)).value

        def pos(ds, _step_ns=step_ns, _p=p):
            return (ds.asi8 // _step_ns) % _p

    def sin_feat(ds):
        return np.sin(2 * np.pi * pos(ds) / p)

    def cos_feat(ds):
        return np.cos(2 * np.pi * pos(ds) / p)

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
    max_train_steps: int | None = None
    use_exogenous: bool = False
    use_calendar: bool = False

    def __init__(self):
        self.uses_exogenous = self.use_exogenous
        self._exo: list[str] = []

    def _make_model(self):
        raise NotImplementedError

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from mlforecast import MLForecast

        self._exo = _exo_cols(case) if self.use_exogenous else []
        cols = ["unique_id", "ds", "y", *self._exo]
        df = train[[c for c in cols if c in train.columns]].copy()
        if self.max_train_steps and len(df) > self.max_train_steps:
            df = df.iloc[-self.max_train_steps :]
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

        date_feats = list(_date_features_for(case.freq))
        if self.use_calendar:
            date_feats += _sincos_features_for(case)

        self._fcst = MLForecast(
            models=[self._make_model()],
            freq=case.freq,
            lags=_lags_for(case),
            date_features=date_feats,
        )
        self._fcst.fit(df, static_features=[])
        self._model_col = type(next(iter(self._fcst.models.values()))).__name__

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self._exo:
            x_df = self._future_X(case, history, horizon, future_exog)
            preds = self._fcst.predict(h=horizon, X_df=x_df)
        else:
            preds = self._fcst.predict(h=horizon)
        y_hat = preds[self._model_col].to_numpy().astype(np.float64)
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})

    def _future_X(
        self, case: Case, history: pd.DataFrame, horizon: int, future_exog: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Build the mlforecast X_df of future covariate values for the horizon."""
        last = _naive_ds(history["ds"].max()).iloc[0]
        fidx = pd.date_range(start=last, periods=horizon + 1, freq=case.freq)[1:]
        data: dict = {"unique_id": case.id, "ds": fidx}
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
    name = "LightGBM"

    def _make_model(self):
        from lightgbm import LGBMRegressor

        return LGBMRegressor(n_estimators=200, learning_rate=0.05, verbosity=-1)


class CatBoostForecaster(_WindowedMLBase):
    name = "CatBoost"

    def _make_model(self):
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=200, learning_rate=0.05, verbose=False, allow_writing_files=False
        )


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


class LightGBMExoForecaster(LightGBMForecaster):
    name = "LightGBM+exo"
    use_exogenous = True


class LightGBMCalendarForecaster(LightGBMForecaster):
    name = "LightGBM+calendar"
    use_calendar = True


class LightGBMCalendarExoForecaster(LightGBMCalendarForecaster):
    name = "LightGBM+calendar+exo"
    use_exogenous = True


class LinearLagsExoForecaster(LinearLagsForecaster):
    name = "LinearLags+exo"
    use_exogenous = True


class CatBoostExoForecaster(CatBoostForecaster):
    name = "CatBoost+exo"
    use_exogenous = True
