"""Abstract Forecaster interface — every method implements fit and predict."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from tsforecastbench.data.schema import Case


def lookback_for(freq: str, h: int) -> int:
    """Default lookback window length for a windowed DL/ML forecaster.

    Frequency-tuned base, floored at 2*h so the window always covers at least two
    horizons even at large h (freqs not listed default to 4*h).
    """
    f = freq.lower()
    base = {"1d": 64, "1h": 168, "15min": 192, "1w": 52}.get(f, 4 * h)
    return max(base, 2 * h)


class Forecaster(ABC):
    name: str = "Forecaster"
    requires_gpu: bool = False
    is_probabilistic: bool = False
    # Methods that consume covariates set this True; the eval loop then passes the
    # horizon's exogenous values (a perfect-forecast proxy) to predict via future_exog.
    uses_exogenous: bool = False
    # Coarse grouping (baseline/classical/ml/deep_learning/foundation_model/ensemble) used
    # by the results viewer to filter/group methods. "unclassified" flags a method that
    # forgot to set it, rather than silently disappearing from a family filter.
    family: str = "unclassified"

    @abstractmethod
    def fit(self, case: Case, train: pd.DataFrame) -> None:
        """Fit on the train split. May store any state on self."""

    @abstractmethod
    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Forecast `horizon` steps ahead. Returns columns: ds (UTC), y_hat (float64).

        `future_exog`, when given, holds the exogenous columns for the `horizon` future
        timestamps (ds + each declared covariate). Univariate methods ignore it;
        exogenous-aware methods use it as a perfect-forecast proxy for the covariates.
        """

    @staticmethod
    def _future_index(case: Case, last_ds, horizon: int) -> pd.DatetimeIndex:
        last = pd.Timestamp(last_ds)
        return pd.date_range(start=last, periods=horizon + 1, freq=case.freq, tz="UTC")[1:]
