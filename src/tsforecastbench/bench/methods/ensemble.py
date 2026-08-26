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

    Two cadences, one column. A member that keeps a trained model apart from the series it
    scores (`is_global`, e.g. `ThetaForecaster`, whose `AutoTheta` parameter search is the
    expensive part) is fit once, on the declared train split, and rolled forward. A member
    whose parameters are a closed-form statistic of whatever frame it is shown (the naive
    baselines: a sample mean, within-cycle means, the last observed cycle) is re-derived
    inside `predict` from that origin's history, which keeps it tracking the origin exactly
    as a per-origin refit did.

    That split is why this class is `is_global = True` rather than local. Before, the whole
    composite was refit at every origin, so the Theta member re-ran its optimizer once per
    origin — 100 times per cell for a forecast change of at most a fraction of a percent,
    and half the total runtime of a 49,000-point case. Fitting it once instead is not just
    cheaper: a model reused across origins has to be fit on the declared train split alone,
    and the only frame the local path ever hands `fit` is the history up to an origin, which
    runs through val (gh #99).
    """

    family = "ensemble"
    library = "no additional library — combines member forecasters' own outputs"
    # Set unconditionally rather than derived from the member list, so one flag cannot mean
    # different things for two instances of the same class. An all-local ensemble
    # (`ensemble_baselines`) therefore reports a fit that reuses nothing, which is the honest
    # description of it: every member is re-derived per origin inside predict, so its cost
    # lands in `predict_seconds` and its `fit_seconds` is near zero because there genuinely is
    # no reusable model. Its forecasts are unchanged by that move.
    is_global = True
    # Explicit rather than inherited: a member's fit takes one contiguous 1-D series, and the
    # naive members' statistics assume a contiguous history, so there is no multi-segment fit
    # to offer and the eval loop produces no extended-history variant for this column.
    pools_segments = False
    compute_profile = (
        "Sum of its members' costs, split across the two cadences: members that keep a "
        "trained model are fit once and rolled forward, while the naive members are "
        "re-derived from each origin's history inside predict. A naive-only ensemble is "
        "therefore milliseconds per origin with no meaningful fit; the Theta variant "
        "inherits Theta's cost at both ends — roughly 210s to fit and ~5s per origin on a "
        "49,000-point record, tens of minutes and ~110s per origin on a 214,000-point one."
    )
    limitations = (
        "Only as good as its members — a median/mean of weak forecasts is still weak. A "
        "member that fails at predict is dropped from that origin's combination rather "
        "than failing the whole prediction, but a member that cannot be conditioned on the "
        "case at all raises instead, since silently combining fewer members would "
        "misreport what the column is. The column also mixes cadences: a trained member is "
        "fit once on the declared train split while the naive members are re-derived per "
        "origin, so `fit_seconds` describes only the former."
    )
    code_snippet = (
        "# fit: every member, once, on the declared train split\n"
        "for m in members:\n"
        "    m.fit(case, train, val)\n\n"
        "# each origin: re-derive the members whose state is a statistic of the history\n"
        "for m in local_members:\n"
        "    m.fit(case, history, history.iloc[:0])\n"
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
        # The members re-derived per origin in `predict`. Partitioned on each member's own
        # `is_global` rather than a hardcoded list, so the cadence decision is delegated to
        # the same flag the eval loop would have used on that member directly.
        self._local_members = [m for m in members if not m.is_global]
        self._fitted = False

    def fit(self, case: Case, train: pd.DataFrame, val: pd.DataFrame) -> None:
        # Every member, including the ones `predict` re-derives: it keeps `fit`'s meaning
        # uniform, leaves the object in a valid post-fit state for any caller, and preserves
        # the failure timing this column had before — `SeasonalNaive.fit` raises on a case
        # declared non-seasonal, and that has to surface as a failed cell rather than as a
        # quietly smaller median under a card that still claims seasonality.
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
        # Re-derive the members whose parameters are a closed-form statistic of the frame
        # they are shown, on the history this origin handed us — the same frame, and the same
        # empty holdout, that the eval loop's per-origin refit used to pass (see
        # `eval.rolling` on why val is included: the history is contiguous and cannot skip a
        # period). Deliberately outside the per-member drop handler below: a member that
        # cannot be conditioned on this case at all is a broken ensemble, not a member to
        # silently drop for one origin.
        for member in self._local_members:
            member.seed = self.seed
            member.fit(case, history, history.iloc[:0])

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
        # Theta is the only member with a real fit, and its cost dominates at both ends: the
        # combination is not "under a second either way". `inference_cost` follows the
        # rubric's measured-`predict_seconds` anchor — ~5s per origin on a 49,000-point
        # record and ~110s on a 214,000-point one, which is High, and a composite cannot be
        # cheaper than its most expensive member (Theta declares High too).
        scorecard=Scorecard(
            history_requirements=Level.MEDIUM,
            interpretability=Level.MEDIUM,
            fit_cost=Level.LOW,
            inference_cost=Level.HIGH,
            tuning_effort=Level.LOW,
            robustness_to_noise=Level.HIGH,
        ),
    )
