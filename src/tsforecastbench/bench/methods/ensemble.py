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
    GrowthShape,
    Level,
    RunningCost,
    Scorecard,
)
from tsforecastbench.data.schema import Case


class EnsembleForecaster(Forecaster):
    """Combine several member forecasters into one via a per-step median (or mean).

    Members share the fit/predict contract and produce the same future index, so the
    combination is a column-wise reduction over the stacked member forecasts. A member
    that fails at predict time is dropped for that origin; if none succeed the error is
    re-raised so the eval loop records it.

    Every member is fit once, like the composite, and rolled forward by its own `predict`:
    `ThetaForecaster` through its fitted model, the naive baselines by reading the history
    up to each origin (gh #120).
    """

    family = "ensemble"
    library = "no additional library — combines member forecasters' own outputs"
    # Explicit rather than inherited: a member's fit takes one contiguous 1-D series, and the
    # naive members' statistics assume a contiguous history, so there is no multi-segment fit
    # to offer and the eval loop produces no extended-history variant for this column.
    pools_segments = False
    compute_profile = (
        "Sum of its members' costs. The naive members have nothing to fit and read each "
        "origin's history inside predict, so a naive-only ensemble is milliseconds per "
        "origin with no meaningful fit; the Theta variant "
        "inherits Theta's cost at both ends — roughly 210s to fit and ~5s per origin on a "
        "49,000-point record, tens of minutes and ~110s per origin on a 214,000-point one."
    )
    limitations = (
        "Only as good as its members — a median/mean of weak forecasts is still weak. A "
        "member that fails at predict is dropped from that origin's combination rather "
        "than failing the whole prediction, but a member that cannot be conditioned on the "
        "case at all raises instead, since silently combining fewer members would "
        "misreport what the column is."
    )
    code_snippet = (
        "# fit: every member, once\n"
        "for m in members:\n"
        "    m.fit(case, train, val)\n\n"
        "# each origin: every member reads the history up to it\n"
        "stacked = np.vstack([m.predict(case, history, h)['y_hat'] for m in members])\n"
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
        # Derived rather than declared, so adding an exogenous-aware, stochastic or GPU
        # member later cannot leave a stale flag published on the method card. Every current
        # member is deterministic, univariate and CPU-only, so all three are False today.
        self.uses_exogenous = any(m.uses_exogenous for m in members)
        self.is_stochastic = any(m.is_stochastic for m in members)
        self.requires_gpu = any(m.requires_gpu for m in members)
        self._fitted = False

    def fit(self, case: Case, train: pd.DataFrame, val: pd.DataFrame) -> None:
        # `SeasonalNaive.fit` raises on a case declared non-seasonal, and that has to surface
        # as a failed cell rather than as a quietly smaller median under a card that still
        # claims seasonality.
        self._fitted = False
        for member in self.members:
            member.seed = self.seed
            member.fit(case, train, val)
        self._fitted = True

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(
                f"{self.name}.predict called before fit; a composite has no forecast until "
                "its members have one"
            )
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
        # Four naive members, none of which costs anything: a fit plus a thousand forecasts
        # runs in under seven seconds on a 100,000-point record.
        scorecard=Scorecard(
            history_requirements=Level.MEDIUM,
            interpretability=Level.MEDIUM,
            running_cost=RunningCost(level=Level.LOW, growth=GrowthShape.FLAT),
            tuning_effort=Level.LOW,
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
        # Adding Theta costs three orders of magnitude: the four naive members together run
        # in seconds, and Theta's quadratic roll-forward takes the combination to ~3 hours
        # for a thousand forecasts on a 100,000-point record. A composite cannot be cheaper
        # than its most expensive member, and it inherits that member's growth shape too.
        scorecard=Scorecard(
            history_requirements=Level.MEDIUM,
            interpretability=Level.MEDIUM,
            running_cost=RunningCost(level=Level.HIGH, growth=GrowthShape.SUPERLINEAR),
            tuning_effort=Level.LOW,
        ),
    )
