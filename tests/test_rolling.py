"""Unit tests for rolling-origin selection and the per-cell seed sweep."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.bench.eval.rolling import (  # noqa: E402
    _select_rolling_origins,
    run_case_rolling,
)
from tsforecastbench.bench.methods.base import Forecaster  # noqa: E402
from tsforecastbench.data.registry import get_case  # noqa: E402
from tsforecastbench.data.schema import Case  # noqa: E402


def test_origins_span_full_range_when_horizon_is_small_relative_to_test():
    # Regression test: previously the oversample-then-truncate logic collapsed the
    # selection onto the front ~50% of the range whenever H << len(test).
    test_start, last_origin, H, n_origins = 0, 1000, 5, 8
    raw_y = np.ones(last_origin + H)

    origins = _select_rolling_origins(test_start, last_origin, raw_y, H, n_origins)

    assert max(origins) >= 0.9 * last_origin
    assert min(origins) <= 0.1 * last_origin


def test_origins_stay_spread_when_some_candidates_are_dropped_for_missing_data():
    test_start, last_origin, H, n_origins = 0, 1000, 5, 8
    raw_y = np.ones(last_origin + H)
    raw_y[600:700] = np.nan  # a chunk of candidates in this region get filtered out

    origins = _select_rolling_origins(test_start, last_origin, raw_y, H, n_origins)

    assert len(origins) <= n_origins
    assert max(origins) >= 0.9 * last_origin
    assert min(origins) <= 0.1 * last_origin


def test_all_valid_candidates_returned_when_fewer_than_n_origins():
    test_start, last_origin, H, n_origins = 0, 3, 1, 8
    raw_y = np.ones(last_origin + H)

    origins = _select_rolling_origins(test_start, last_origin, raw_y, H, n_origins)

    assert origins == sorted(set(origins))
    assert set(origins) <= {0, 1, 2, 3}


def test_falls_back_to_first_candidate_when_all_filtered_out():
    test_start, last_origin, H, n_origins = 0, 10, 5, 4
    raw_y = np.full(last_origin + H, np.nan)

    origins = _select_rolling_origins(test_start, last_origin, raw_y, H, n_origins)

    assert origins == [test_start]


# ---------------------------------------------------------------------------
# Per-cell seed sweep (issue #106): a stochastic method is run several times and
# the median-MSE run is kept whole.
# ---------------------------------------------------------------------------

# Exponent per seed. Each stub forecasts `scale * 10**exponent`, well above the data, so
# pooled MSE grows with the exponent and the ranking of the five runs is known in advance:
# ascending MSE is seeds 1, 3, 4, 0, 2, whose middle element is seed 4. Deliberately
# neither seed 0 (which a single-seed run would have reported) nor the arithmetically
# middle seed 2, so a loop that quietly kept the first or the last run fails this.
_EXPONENT_BY_SEED = {0: 3, 1: 0, 2: 4, 3: 1, 4: 2}
_MEDIAN_SEED = 4


class _SeedSensitive(Forecaster):
    """Forecasts a constant whose distance from the data is set by `self.seed`."""

    name = "SeedSensitive"
    family = "baseline"
    is_global = True
    is_stochastic = True

    def __init__(self):
        self.fit_calls = 0
        self.seeds_seen: list[int] = []

    def _level(self, history: pd.DataFrame) -> float:
        scale = max(1.0, float(np.nanmax(np.abs(history["y"].to_numpy(dtype=float)))))
        return scale * 10.0 ** _EXPONENT_BY_SEED[self.seed]

    def fit(self, case: Case, train: pd.DataFrame, val: pd.DataFrame) -> None:
        self.fit_calls += 1
        self.seeds_seen.append(self.seed)
        self._value = self._level(train)

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": np.full(horizon, self._value)})


class _Deterministic(_SeedSensitive):
    name = "Deterministic"
    is_stochastic = False


class _FailsOnSeeds(_SeedSensitive):
    """Raises for the seeds in `bad`, forecasts normally for the rest."""

    name = "FailsOnSeeds"
    bad: frozenset[int] = frozenset()

    def fit(self, case: Case, train: pd.DataFrame, val: pd.DataFrame) -> None:
        if self.seed in self.bad:
            raise RuntimeError(f"seed {self.seed} is unlucky")
        super().fit(case, train, val)


@pytest.fixture(scope="module")
def case():
    return get_case("discharge_flashy_catskills")


def _rolling(case, method, n_seeds):
    return run_case_rolling(case, [method], n_origins=4, prefer_synth=True, n_seeds=n_seeds)


def test_stochastic_cell_reports_the_median_mse_run(case):
    method = _SeedSensitive()

    res, snaps = _rolling(case, method, n_seeds=5)

    assert (res["status"] == "ok").all()
    assert sorted(method.seeds_seen) == [0, 1, 2, 3, 4], "every seed must actually be fit"
    assert set(res["seed"]) == {_MEDIAN_SEED}
    assert set(res["n_seeds"]) == {5}
    # The kept run's own snapshots, not seed 0's: the forecast line the viewer draws is
    # the run the cell's number describes.
    scale = snaps["y_hat"].max() / 10.0 ** _EXPONENT_BY_SEED[_MEDIAN_SEED]
    assert np.isclose(
        snaps["y_hat"].to_numpy(), scale * 10.0 ** _EXPONENT_BY_SEED[_MEDIAN_SEED]
    ).all()


def test_the_reported_numbers_describe_the_drawn_forecast(case):
    """The row must stay coherent: nine metrics off one run, never a per-metric median.

    This method forecasts one constant, two orders of magnitude above every actual, so
    both MAE and RMSE must come out within a percent of the very constant the snapshots
    draw. A row assembled from several runs would put a different constant behind each.
    """
    method = _SeedSensitive()

    res, snaps = _rolling(case, method, n_seeds=5)

    drawn = float(snaps["y_hat"].iloc[0])
    for metric in ("MAE", "RMSE"):
        value = float(res.loc[res["metric"] == metric, "value"].iloc[0])
        assert abs(value - drawn) / drawn < 0.01, f"{metric} does not match the drawn line"


def test_deterministic_method_is_fit_once_whatever_n_seeds_says(case):
    method = _Deterministic()

    res, _ = _rolling(case, method, n_seeds=5)

    assert method.fit_calls == 1, "repeating a deterministic method is pure cost"
    assert set(res["seed"]) == {0}
    assert set(res["n_seeds"]) == {1}


def test_median_is_taken_over_the_seeds_that_succeeded(case):
    method = _FailsOnSeeds()
    method.bad = frozenset({0, 1})

    res, snaps = _rolling(case, method, n_seeds=5)

    assert (res["status"] == "ok").all()
    # Survivors are seeds 2, 3, 4 with exponents 4, 1, 2; ascending MSE is 3, 4, 2, whose
    # middle element is seed 4.
    assert set(res["seed"]) == {4}
    assert set(res["n_seeds"]) == {5}, "n_seeds records how many were run, not how many ran ok"
    assert not snaps.empty


def test_cell_fails_when_every_seed_fails(case):
    method = _FailsOnSeeds()
    method.bad = frozenset(range(5))

    res, snaps = _rolling(case, method, n_seeds=5)

    assert res["status"].str.startswith("error: RuntimeError").all()
    assert res["value"].isna().all()
    assert snaps.empty, "a failed cell must not leave a forecast line behind"
    assert set(res["seed"]) == {0}, "the first run's failure, as a single-seed run would report"
