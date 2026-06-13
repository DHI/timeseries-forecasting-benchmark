"""Abstract Forecaster interface — every method implements fit and predict."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from tsforecastbench.data.schema import Case


class Forecaster(ABC):
    name: str = "Forecaster"
    requires_gpu: bool = False
    is_probabilistic: bool = False

    @abstractmethod
    def fit(self, case: Case, train: pd.DataFrame) -> None:
        """Fit on the train split. May store any state on self."""

    @abstractmethod
    def predict(self, case: Case, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Forecast `horizon` steps ahead. Returns columns: ds (UTC), y_hat (float64)."""

    @staticmethod
    def _future_index(case: Case, last_ds, horizon: int) -> pd.DatetimeIndex:
        last = pd.Timestamp(last_ds)
        return pd.date_range(start=last, periods=horizon + 1, freq=case.freq, tz="UTC")[1:]
