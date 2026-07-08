"""Ensemble-of-baselines forecaster: combine cheap members into one column.

A simple combination of the naive baselines (optionally plus a fast classical method) is
cheap to compute and is often an embarrassingly strong reference point — the "hard to beat
cheaply" column that calibrates how much the more sophisticated methods actually buy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.bench.methods.classical import ThetaForecaster
from tsforecastbench.bench.methods.naive import (
    Climatology,
    MeanBaseline,
    Persistence,
    SeasonalNaive,
)
from tsforecastbench.data.schema import Case


class EnsembleForecaster(Forecaster):
    """Combine several member forecasters into one via a per-step median (or mean).

    Members share the fit/predict contract and produce the same future index, so the
    combination is a column-wise reduction over the stacked member forecasts. A member
    that fails at predict time is dropped for that origin; if none succeed the error is
    re-raised so the eval loop records it.
    """

    family = "ensemble"

    def __init__(
        self,
        members: list[Forecaster],
        name: str,
        combine: str = "median",
    ):
        if combine not in {"median", "mean"}:
            raise ValueError(f"combine must be 'median' or 'mean', got {combine!r}")
        self.members = members
        self.name = name
        self.combine = combine

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        for member in self.members:
            member.fit(case, train)

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        forecasts: list[np.ndarray] = []
        future: pd.Series | None = None
        last_error: Exception | None = None
        for member in self.members:
            try:
                preds = member.predict(case, history, horizon, future_exog=future_exog)
            except Exception as exc:  # drop this member for this origin
                last_error = exc
                continue
            if future is None:
                future = preds["ds"]
            forecasts.append(preds["y_hat"].to_numpy(dtype=np.float64))

        if not forecasts:
            raise RuntimeError("all ensemble members failed to predict") from last_error

        stacked = np.vstack(forecasts)
        reducer = np.nanmedian if self.combine == "median" else np.nanmean
        y_hat = reducer(stacked, axis=0).astype(np.float64)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})


def _naive_members() -> list[Forecaster]:
    return [MeanBaseline(), Persistence(), SeasonalNaive(), Climatology()]


def ensemble_baselines() -> EnsembleForecaster:
    """Median of the naive baselines (Mean, Persistence, SeasonalNaive, Climatology)."""
    return EnsembleForecaster(_naive_members(), name="EnsembleBaselines")


def ensemble_baselines_theta() -> EnsembleForecaster:
    """Median of the naive baselines plus the fast classical Theta method."""
    return EnsembleForecaster([*_naive_members(), ThetaForecaster()], name="EnsembleBaselinesTheta")
