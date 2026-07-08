"""Naive baselines: mean, persistence, seasonal naive."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.data.schema import Case
from tsforecastbench.data.seasonality import dominant_period, position_in_cycle


class MeanBaseline(Forecaster):
    name = "Mean"
    family = "baseline"

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        self._mean = float(train["y"].mean())

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        last_ds = history["ds"].max()
        future = self._future_index(case, last_ds, horizon)
        return pd.DataFrame({"ds": future, "y_hat": np.full(horizon, self._mean)})


class Persistence(Forecaster):
    name = "Persistence"
    family = "baseline"

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        pass

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        last_value = float(history["y"].iloc[-1])
        last_ds = history["ds"].max()
        future = self._future_index(case, last_ds, horizon)
        return pd.DataFrame({"ds": future, "y_hat": np.full(horizon, last_value)})


class SeasonalNaive(Forecaster):
    name = "SeasonalNaive"
    family = "baseline"

    def __init__(self, season_length: int | None = None):
        self.season_length = season_length

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        self._s = self.season_length or dominant_period(case.seasonalities, train["y"].to_numpy())
        if self._s == 1:
            raise ValueError(
                f"Case {case.id} is declared non-seasonal (seasonalities = []) — "
                "SeasonalNaive would degenerate into a duplicate Persistence column"
            )
        self._tail = train["y"].tail(self._s).to_numpy()

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
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
    """Mean target value by position within the case's dominant declared seasonal cycle
    (`case.seasonalities`). Keying on the declared seasonality means the baseline has the
    same meaning at every frequency, rather than shifting between day-of-year, hour, and
    weekday x hour depending on the sampling rate. A non-seasonal case (`seasonalities =
    []`) has no cycle to key on, so every timestamp shares one bucket and the baseline is
    the global training mean. A stronger 'no-skill' reference than Mean for series with a
    stable seasonal/diurnal cycle.
    """

    name = "Climatology"
    family = "baseline"

    def _key(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return position_in_cycle(self._freq, self._period, idx)

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        self._freq = case.freq
        self._period = dominant_period(case.seasonalities, train["y"].to_numpy())
        idx = pd.DatetimeIndex(pd.to_datetime(train["ds"]))
        keys = self._key(idx)
        s = pd.Series(pd.to_numeric(train["y"], errors="coerce").to_numpy())
        self._clim = s.groupby(keys).mean()
        self._global_mean = float(s.mean())

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        last_ds = history["ds"].max()
        future = self._future_index(case, last_ds, horizon)
        keys = self._key(pd.DatetimeIndex(future))
        y_hat = np.array([self._clim.get(k, self._global_mean) for k in keys], dtype=np.float64)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})
