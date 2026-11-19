"""Unit tests for the block-bootstrap intervals in `tsforecastbench.bench.eval.intervals`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsforecastbench.bench.eval.intervals import (
    block_length,
    case_intervals,
    matrix_intervals,
    resample_indices,
)
from tsforecastbench.bench.eval.metrics import POOLED_METRICS
from tsforecastbench.bench.eval.rolling import run_case_rolling
from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.data.registry import get_case
from tsforecastbench.data.schema import Case

LOWER_IS_BETTER = {m: not m.startswith(("Skill", "NSE", "KGE")) for m in POOLED_METRICS}
H = 24


def _origins(n: int, every: str) -> pd.Series:
    return pd.Series(pd.date_range("2020-01-01", periods=n, freq=every))


def test_block_spans_one_horizon_of_overlapping_starts():
    assert block_length(_origins(100, "4h"), pd.Timedelta("24h")) == 6


def test_block_is_at_least_the_cube_root_of_the_start_count():
    assert block_length(_origins(1000, "7D"), pd.Timedelta("1D")) == 10


def test_resampled_blocks_are_runs_of_consecutive_starts_wrapping_at_the_end():
    idx = resample_indices(n=20, block=5, n_boot=50, rng=np.random.default_rng(0))

    assert idx.shape == (50, 20)
    assert (np.diff(idx.reshape(50, 4, 5), axis=2) % 20 == 1).all()


def _snapshots(errors: dict[str, np.ndarray], n_origins: int) -> pd.DataFrame:
    """One cell per method, `n_origins` daily starts of H hourly steps, truth at zero."""
    frames = []
    starts = pd.date_range("2020-01-01", periods=n_origins, freq="1D")
    for method, err in errors.items():
        for i, start in enumerate(starts):
            frames.append(
                pd.DataFrame(
                    {
                        "case_id": "c",
                        "method": method,
                        "used_extended_history": False,
                        "origin_id": i,
                        "origin_ds": start,
                        "ds": pd.date_range(start, periods=H, freq="1h"),
                        "y_hat": -err[i * H : (i + 1) * H],
                        "y": 0.0,
                        "y_persistence": 1.0,
                        "y_mean": 2.0,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1) with unit marginal variance."""
    shocks = rng.normal(scale=np.sqrt(1 - phi**2), size=n)
    out = np.empty(n)
    out[0] = rng.normal()
    for t in range(1, n):
        out[t] = phi * out[t - 1] + shocks[t]
    return out


def _intervals(snaps: pd.DataFrame, n_boot: int = 300) -> pd.DataFrame:
    return case_intervals(
        snaps,
        mase_scale=1.0,
        horizon=pd.Timedelta("24h"),
        lower_is_better=LOWER_IS_BETTER,
        reference_methods=set(snaps["method"]),
        n_boot=n_boot,
    )


@pytest.mark.parametrize("n_origins", [19, 40])
def test_interval_covers_the_true_mae_about_95_percent_of_the_time(n_origins):
    rng = np.random.default_rng(1)
    true_mae = np.sqrt(2 / np.pi)  # E|e| for a unit-variance normal
    reps = 150

    hits = 0
    for _ in range(reps):
        errors = {"m": _ar1(n_origins * H, phi=0.9, rng=rng)}
        res = _intervals(_snapshots(errors, n_origins))
        row = res[res["metric"] == "MAE"].iloc[0]
        hits += row["ci_low"] <= true_mae <= row["ci_high"]

    assert 0.88 <= hits / reps <= 0.99


def test_an_equally_good_method_cannot_be_told_apart_from_the_best():
    rng = np.random.default_rng(2)
    errors = {"a": _ar1(30 * H, phi=0.9, rng=rng), "b": _ar1(30 * H, phi=0.9, rng=rng)}

    res = _intervals(_snapshots(errors, 30))

    mae = res[res["metric"] == "MAE"]
    other = mae[mae["method"] != mae["best_method"]].iloc[0]
    assert other["diff_low"] <= 0 <= other["diff_high"]


def test_a_much_worse_method_is_told_apart_from_the_best():
    err = _ar1(30 * H, phi=0.9, rng=np.random.default_rng(3))
    res = _intervals(_snapshots({"best": err, "worse": err * 3}, 30))

    worse = res[(res["method"] == "worse") & (res["metric"] == "MAE")].iloc[0]
    assert worse["diff_low"] > 0


def test_best_is_the_highest_score_for_a_skill_metric():
    err = _ar1(30 * H, phi=0.9, rng=np.random.default_rng(4))
    res = _intervals(_snapshots({"good": err, "bad": err * 3}, 30), n_boot=50)

    skill = res[res["metric"] == "Skill_vs_Persistence"]
    assert set(skill["best_method"]) == {"good"}


def test_no_difference_is_given_against_a_best_forecast_from_other_starts():
    err = _ar1(30 * H, phi=0.9, rng=np.random.default_rng(5))
    snaps = _snapshots({"best": err, "other": err * 3}, 30)
    snaps = snaps[~((snaps["method"] == "other") & (snaps["origin_id"] == 0))]

    res = _intervals(snaps, n_boot=50)

    other = res[(res["method"] == "other") & (res["metric"] == "MAE")].iloc[0]
    assert np.isnan(other["diff_low"])
    assert np.isfinite(other["ci_low"])


@pytest.mark.parametrize("metric", POOLED_METRICS)
def test_point_score_sits_inside_its_interval(metric):
    err = _ar1(30 * H, phi=0.5, rng=np.random.default_rng(6))
    snaps = _snapshots({"m": err}, 30)
    snaps["y"] = 10.0 + np.sin(np.arange(len(snaps)) / 5.0)  # varies, and away from zero
    snaps["y_hat"] += snaps["y"]

    res = _intervals(snaps, n_boot=200)

    row = res[res["metric"] == metric].iloc[0]
    assert row["ci_low"] <= row["value"] <= row["ci_high"]


class _Offset(Forecaster):
    """Forecasts the last observed value plus a fixed offset."""

    family = "baseline"

    def __init__(self, name: str, offset: float):
        self.name = name
        self.offset = offset

    def fit(self, case: Case, train: pd.DataFrame, val: pd.DataFrame) -> None:
        pass

    def predict(self, case, history, horizon, future_exog=None) -> pd.DataFrame:
        future = self._future_index(case, history["ds"].max(), horizon)
        level = float(history["y"].iloc[-1]) + self.offset
        return pd.DataFrame({"ds": future, "y_hat": np.full(horizon, level)})


def test_matrix_rows_get_intervals_around_their_own_scores():
    case = get_case("discharge_flashy_catskills")
    methods = [_Offset("Close", 0.0), _Offset("Far", 50.0)]
    results, snaps = run_case_rolling(case, methods, n_origins=12, prefer_synth=True)

    out = matrix_intervals(results, snaps, LOWER_IS_BETTER, n_boot=100)

    assert len(out) == len(results)
    mae = out[out["metric"] == "MAE"].set_index("method")
    assert (mae["ci_low"] <= mae["value"]).all() and (mae["value"] <= mae["ci_high"]).all()
    assert set(mae["best_method"]) == {"Close"}
    assert mae.loc["Far", "diff_low"] > 0
