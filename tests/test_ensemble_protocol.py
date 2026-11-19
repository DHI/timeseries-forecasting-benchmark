"""How `EnsembleForecaster` fits its members and rolls them forward.

The composite and every member are fit once, before the test period. The naive members have
nothing to fit and read the history up to each origin in their own `predict` (gh #120).

The regression this file guards is a `ThetaForecaster` member whose `AutoTheta` optimizer
re-ran once per origin -- 100 times per cell -- because the composite around it declared
itself local.

Everything here uses stub members, so no optimizer runs and the whole file is well under a
second. The real-Theta counterparts live in `test_classical_fit_once.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.bench.eval.rolling import run_case_rolling  # noqa: E402
from tsforecastbench.bench.methods.base import Forecaster  # noqa: E402
from tsforecastbench.bench.methods.ensemble import (  # noqa: E402
    EnsembleForecaster,
    ensemble_baselines,
    ensemble_baselines_theta,
)
from tsforecastbench.bench.methods.scorecard import (  # noqa: E402
    Capabilities,
    ForecastStrategy,
    GrowthShape,
    Level,
    RunningCost,
    Scorecard,
)
from tsforecastbench.data.registry import get_case  # noqa: E402
from tsforecastbench.data.schema import Case  # noqa: E402

N_ORIGINS = 4


class _Recorder(Forecaster):
    """Forecasts a constant, recording every frame its `fit` was handed."""

    name = "Recorder"
    family = "baseline"

    def __init__(self):
        self.fit_calls = 0
        self.train_lengths: list[int] = []
        self.val_lengths: list[int] = []
        self.cases_seen: list[str] = []
        self.seeds_seen: list[int] = []
        self._value = 0.0

    def fit(self, case: Case, train: pd.DataFrame, val: pd.DataFrame) -> None:
        self.fit_calls += 1
        self.train_lengths.append(len(train))
        self.val_lengths.append(len(val))
        self.cases_seen.append(case.id)
        self.seeds_seen.append(self.seed)
        self._value = float(np.nanmean(train["y"].to_numpy(dtype=float)))

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": np.full(horizon, self._value)})


def _ensemble(members, name="StubEnsemble") -> EnsembleForecaster:
    return EnsembleForecaster(
        members,
        name=name,
        capabilities=Capabilities(
            trend=False, seasonality=False, multiple_seasonality=False, exogenous=False
        ),
        scorecard=Scorecard(
            history_requirements=Level.LOW,
            interpretability=Level.HIGH,
            running_cost=RunningCost(level=Level.LOW, growth=GrowthShape.FLAT),
            tuning_effort=Level.LOW,
        ),
    )


@pytest.fixture(scope="module")
def case():
    return get_case("discharge_flashy_catskills")


def _rolling(case, method):
    return run_case_rolling(case, [method], n_origins=N_ORIGINS, prefer_synth=True, n_seeds=1)


def test_neither_factory_pools_segments():
    """`is` comparison, because `test_method_scorecards` compares `gaps` by identity."""
    for factory in (ensemble_baselines, ensemble_baselines_theta):
        ens = factory()
        assert ens.pools_segments is False, ens.name


def test_members_are_fit_once_across_origins(case):
    """The direct regression test: one optimizer run per cell, not one per origin."""
    trained, naive = _Recorder(), _Recorder()

    res, _ = _rolling(case, _ensemble([trained, naive]))

    assert (res["status"] == "ok").all()
    assert trained.fit_calls == 1
    assert naive.fit_calls == 1


def test_member_is_fit_on_the_same_frame_as_its_standalone_column(case):
    """What makes the member numerically comparable to its standalone column: all data
    before the test period, with no holdout (gh #120)."""
    trained = _Recorder()

    res, _ = _rolling(case, _ensemble([trained]))

    assert (res["status"] == "ok").all()
    assert trained.fit_calls == 1
    assert trained.val_lengths == [0]
    assert set(res["val_points"]) == {0}
    assert set(res["train_points"]) == {trained.train_lengths[0]}


def test_cached_member_fit_is_not_reused_across_cases():
    """Instances are built once in `build_methods()` and reused for all 12 cases, so a
    member fit that outlived its cell would silently score case B on case A's model."""
    trained = _Recorder()
    ens = _ensemble([trained])

    for case_id in ("discharge_flashy_catskills", "discharge_lowland_denmark"):
        run_case_rolling(
            get_case(case_id), [ens], n_origins=N_ORIGINS, prefer_synth=True, n_seeds=1
        )

    assert trained.fit_calls == 2
    assert trained.cases_seen == [
        "discharge_flashy_catskills",
        "discharge_lowland_denmark",
    ]


def test_predict_before_fit_raises(case):
    """Without this a missing fit would turn EnsembleBaselinesTheta into a quietly
    different column rather than a failed cell."""
    ens = _ensemble([_Recorder()])
    frame = case.synth().head(50)

    with pytest.raises(RuntimeError, match="before fit"):
        ens.predict(case, frame, 3)


def test_member_predict_failure_is_still_dropped(case):
    """Unchanged behaviour: predict-time failure drops that member for that origin."""

    class _PredictFails(_Recorder):
        name = "PredictFails"

        def predict(self, case, history, horizon, future_exog=None):
            raise ValueError("no forecast today")

    good = _Recorder()

    res, _ = _rolling(case, _ensemble([good, _PredictFails()]))

    assert (res["status"] == "ok").all()


def test_flags_are_derived_from_members():
    """So an exogenous-aware or stochastic member added later cannot publish a stale flag."""

    class _Exo(_Recorder):
        name = "Exo"
        uses_exogenous = True

    class _Stochastic(_Recorder):
        name = "Stochastic"
        is_stochastic = True

    plain = _ensemble([_Recorder()])
    assert plain.uses_exogenous is False
    assert plain.is_stochastic is False
    assert _ensemble([_Recorder(), _Exo()]).uses_exogenous is True
    assert _ensemble([_Recorder(), _Stochastic()]).is_stochastic is True

    # Both real factories are deterministic and univariate today, which is what keeps the
    # recorded `uses_exogenous` column and the single-seed cell unchanged.
    for factory in (ensemble_baselines, ensemble_baselines_theta):
        assert factory().uses_exogenous is False
        assert factory().is_stochastic is False


def test_seed_is_forwarded_to_members(case):
    """Every stochastic method must route randomness through `self.seed` (base.py), so a
    composite has to pass its own down or a member would repeat one draw."""
    member = _Recorder()
    ens = _ensemble([member])
    ens.seed = 3

    ens.fit(case, case.synth().head(50), case.synth().head(0))

    assert member.seeds_seen == [3]


def test_forecast_strategy_stays_not_applicable():
    for factory in (ensemble_baselines, ensemble_baselines_theta):
        assert factory().forecast_strategy is ForecastStrategy.NOT_APPLICABLE
