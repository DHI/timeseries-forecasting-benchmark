"""Naive baselines: mean, persistence, seasonal naive."""

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


class MeanBaseline(Forecaster):
    name = "Mean"
    family = "baseline"
    library = "pandas/numpy only"
    compute_profile = "CPU, negligible — one mean over the training split."
    limitations = (
        "Ignores trend, seasonality, and recency entirely; useful only as a floor "
        "other methods should comfortably beat."
    )
    code_snippet = "mean = train['y'].mean()\ny_hat = [mean] * horizon"
    capabilities = Capabilities(
        trend=False, seasonality=False, multiple_seasonality=False, exogenous=False
    )
    # Every future timestamp gets the same constant, looked up independently — there's no
    # step-by-step mechanism for "recursive vs direct" to describe.
    forecast_strategy = ForecastStrategy.NOT_APPLICABLE
    # Averaging the whole split makes one bad point negligible, hence robust; the flip side
    # is that it has no mechanism for anything at all.
    scorecard = Scorecard(
        history_requirements=Level.LOW,
        interpretability=Level.HIGH,
        fit_cost=Level.LOW,
        inference_cost=Level.LOW,
        tuning_effort=Level.LOW,
        robustness_to_noise=Level.HIGH,
    )

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
    library = "pandas/numpy only"
    compute_profile = "CPU, negligible — no fitting at all."
    limitations = (
        "Repeats the last observed value for the whole horizon; degrades fast on any "
        "series with a trend, cycle, or fast dynamics beyond one step."
    )
    code_snippet = "last_value = history['y'].iloc[-1]\ny_hat = [last_value] * horizon"
    capabilities = Capabilities(
        trend=False, seasonality=False, multiple_seasonality=False, exogenous=False
    )
    # Every future timestamp repeats the same last value, looked up independently — no
    # step-by-step mechanism this field could describe.
    forecast_strategy = ForecastStrategy.NOT_APPLICABLE
    # The whole forecast rests on one observation, so an outlier at the origin propagates
    # across the entire horizon — the least robust method here despite being the simplest.
    scorecard = Scorecard(
        history_requirements=Level.LOW,
        interpretability=Level.HIGH,
        fit_cost=Level.LOW,
        inference_cost=Level.LOW,
        tuning_effort=Level.LOW,
        robustness_to_noise=Level.LOW,
    )

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
    library = "pandas/numpy only"
    compute_profile = "CPU, negligible — tiles the last observed seasonal cycle."
    limitations = (
        "Repeats the most recent full cycle of the case's dominant declared seasonal "
        "period; refuses to run on a case declared non-seasonal, and can't adapt if "
        "the cycle's shape drifts over time."
    )
    code_snippet = (
        "seasonal_window = train['y'].tail(season_length)\ny_hat = tile(seasonal_window, horizon)"
    )
    capabilities = Capabilities(
        trend=False, seasonality=True, multiple_seasonality=False, exogenous=False
    )
    # Tiles the last observed cycle verbatim, position by position — no step-by-step
    # mechanism this field could describe.
    forecast_strategy = ForecastStrategy.NOT_APPLICABLE
    # Needs a full cycle before it can run at all, and tiles that one cycle verbatim, so a
    # noisy last period is reproduced rather than averaged away.
    scorecard = Scorecard(
        history_requirements=Level.MEDIUM,
        interpretability=Level.HIGH,
        fit_cost=Level.LOW,
        inference_cost=Level.LOW,
        tuning_effort=Level.LOW,
        robustness_to_noise=Level.LOW,
    )

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
    library = "pandas/numpy only"
    compute_profile = "CPU, negligible — one groupby-mean over the training split."
    limitations = (
        "Averages history by position in the dominant seasonal cycle, so it captures a "
        "stable average shape but not trend, one-off events, or year-to-year drift; "
        "falls back to the global mean on a case declared non-seasonal."
    )
    code_snippet = (
        "position = position_in_cycle(freq, period, train['ds'])\n"
        "climatology = train.groupby(position)['y'].mean()\n"
        "y_hat = [climatology[position_in_cycle(freq, period, ds)] for ds in future_ds]"
    )
    capabilities = Capabilities(
        trend=False, seasonality=True, multiple_seasonality=False, exogenous=False
    )
    # Each future timestamp is an independent lookup by position in the cycle — no
    # step-by-step mechanism this field could describe.
    forecast_strategy = ForecastStrategy.NOT_APPLICABLE
    # Wants several cycles so each within-cycle position has something to average, and that
    # averaging is what makes it the robust seasonal baseline where SeasonalNaive isn't.
    scorecard = Scorecard(
        history_requirements=Level.MEDIUM,
        interpretability=Level.HIGH,
        fit_cost=Level.LOW,
        inference_cost=Level.LOW,
        tuning_effort=Level.LOW,
        robustness_to_noise=Level.HIGH,
    )

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
