"""Naive baselines: mean, persistence, seasonal naive."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.methods.base import Forecaster
from tsforecastbench.schema import Case


def _default_season_length(freq: str) -> int:
    f = freq.lower()
    if f in {"1d", "d"}:
        return 365
    if f in {"1h", "h"}:
        return 24
    if f in {"15min", "15t"}:
        return 96
    if f in {"1w", "w"}:
        return 52
    return 24


class MeanBaseline(Forecaster):
    name = "Mean"

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        self._mean = float(train["y"].mean())

    def predict(self, case: Case, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        last_ds = history["ds"].max()
        future = self._future_index(case, last_ds, horizon)
        return pd.DataFrame({"ds": future, "y_hat": np.full(horizon, self._mean)})


class Persistence(Forecaster):
    name = "Persistence"

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        pass

    def predict(self, case: Case, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        last_value = float(history["y"].iloc[-1])
        last_ds = history["ds"].max()
        future = self._future_index(case, last_ds, horizon)
        return pd.DataFrame({"ds": future, "y_hat": np.full(horizon, last_value)})


class SeasonalNaive(Forecaster):
    name = "SeasonalNaive"

    def __init__(self, season_length: int | None = None):
        self.season_length = season_length

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        self._s = self.season_length or _default_season_length(case.freq)
        self._tail = train["y"].tail(self._s).to_numpy()

    def predict(self, case: Case, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        s = self._s
        h_tail = history["y"].tail(s).to_numpy()
        if len(h_tail) < s:
            pad = self._tail[-(s - len(h_tail)) :]
            seasonal_window = np.concatenate([pad, h_tail])
        else:
            seasonal_window = h_tail
        n_rep = horizon // s + 1
        y_hat = np.tile(seasonal_window, n_rep)[:horizon]
        last_ds = history["ds"].max()
        future = self._future_index(case, last_ds, horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat.astype(np.float64)})


class Climatology(Forecaster):
    """Mean value by seasonal position (day-of-year for daily, hour for hourly,
    day-of-week x hour for sub-daily). A stronger 'no-skill' reference than Mean
    for series with a stable seasonal/diurnal cycle.
    """

    name = "Climatology"

    def _key(self, idx: pd.DatetimeIndex) -> np.ndarray:
        f = self._freq
        if f in {"1d", "d"}:
            return idx.dayofyear.to_numpy()
        if f in {"1h", "h"}:
            return idx.hour.to_numpy()
        # sub-daily: combine weekday and hour
        return (idx.dayofweek * 24 + idx.hour).to_numpy()

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        self._freq = case.freq.lower()
        idx = pd.DatetimeIndex(pd.to_datetime(train["ds"]))
        keys = self._key(idx)
        s = pd.Series(pd.to_numeric(train["y"], errors="coerce").to_numpy())
        self._clim = s.groupby(keys).mean()
        self._global_mean = float(s.mean())

    def predict(self, case: Case, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        last_ds = history["ds"].max()
        future = self._future_index(case, last_ds, horizon)
        keys = self._key(pd.DatetimeIndex(future))
        y_hat = np.array([self._clim.get(k, self._global_mean) for k in keys], dtype=np.float64)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})
