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
        # last known covariate value, used to fill any future gaps at predict time
        self._last_exo = {
            c: float(df[c].ffill().iloc[-1]) if c in df and df[c].notna().any() else 0.0
            for c in self._exo
        }
        if self._exo:
            df[self._exo] = df[self._exo].ffill().fillna(0.0)

        self._fcst = MLForecast(
            models=[self._make_model()],
            freq=case.freq,
            lags=_lags_for(case.freq),
            date_features=_date_features_for(case.freq),
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


class LinearLagsExoForecaster(LinearLagsForecaster):
    name = "LinearLags+exo"
    use_exogenous = True
