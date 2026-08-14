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
from tsforecastbench.bench.methods.scorecard import (
    Capabilities,
    ForecastStrategy,
    Level,
    Scorecard,
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
    library = "no additional library — combines member forecasters' own outputs"
    compute_profile = (
        "Sum of its members' costs; cheap in practice since the members are the "
        "naive baselines (and optionally the fast classical Theta method)."
    )
    limitations = (
        "Only as good as its members — a median/mean of weak forecasts is still "
        "weak. A member that fails at one origin is silently dropped from that "
        "step's combination rather than failing the whole prediction."
    )
    code_snippet = (
        "member_forecasts = [m.predict(case, history, horizon) for m in members]\n"
        "stacked = np.vstack([f['y_hat'] for f in member_forecasts])\n"
        "y_hat = np.nanmedian(stacked, axis=0)   # or np.nanmean if combine='mean'"
    )

    # `capabilities` and `scorecard` are per-instance for the same reason `name` is: what an
    # ensemble can model and what it costs are properties of its member list, which the
    # factories below choose.
    def __init__(
        self,
        members: list[Forecaster],
        name: str,
        capabilities: Capabilities,
        scorecard: Scorecard,
        combine: str = "median",
    ):
        if combine not in {"median", "mean"}:
            raise ValueError(f"combine must be 'median' or 'mean', got {combine!r}")
        self.members = members
        self.name = name
        self.capabilities = capabilities
        self.scorecard = scorecard
        self.combine = combine
        # Always N/A, unlike capabilities/scorecard: a composite's own strategy is just
        # whatever mix of members it combines, and there is only ever one EnsembleForecaster
        # class, not several sibling subclasses that could disagree.
        self.forecast_strategy = ForecastStrategy.NOT_APPLICABLE

    def fit(self, case: Case, train: pd.DataFrame, val: pd.DataFrame) -> None:
        for member in self.members:
            member.fit(case, train, val)

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
    return EnsembleForecaster(
        _naive_members(),
        name="EnsembleBaselines",
        # No member extrapolates level drift, so neither does the median of them.
        capabilities=Capabilities(
            trend=False, seasonality=True, multiple_seasonality=False, exogenous=False
        ),
        # A per-step median across four members is what buys the robustness: an outlier has
        # to fool most of them to move the forecast.
        scorecard=Scorecard(
            history_requirements=Level.MEDIUM,
            interpretability=Level.MEDIUM,
            fit_cost=Level.LOW,
            inference_cost=Level.LOW,
            tuning_effort=Level.LOW,
            robustness_to_noise=Level.HIGH,
        ),
    )


def ensemble_baselines_theta() -> EnsembleForecaster:
    """Median of the naive baselines plus the fast classical Theta method."""
    return EnsembleForecaster(
        [*_naive_members(), ThetaForecaster()],
        name="EnsembleBaselinesTheta",
        # Theta brings the trend mechanism the naive members lack.
        capabilities=Capabilities(
            trend=True, seasonality=True, multiple_seasonality=False, exogenous=False
        ),
        # Theta is the only member with a real fit, and it's a cheap one; the combination
        # stays under a second either way.
        scorecard=Scorecard(
            history_requirements=Level.MEDIUM,
            interpretability=Level.MEDIUM,
            fit_cost=Level.LOW,
            inference_cost=Level.LOW,
            tuning_effort=Level.LOW,
            robustness_to_noise=Level.HIGH,
        ),
    )
