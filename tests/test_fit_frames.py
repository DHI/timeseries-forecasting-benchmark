"""What `fit` is handed before the test period (gh #120).

Parameters are estimated on all data before the test period. A method that stops on val
first gets the train and val splits apart, to find its stopping step, then refits on both.
These tests pin both paths on both eval loops, for the single-series local fit and for the
multi-segment extended-history one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.bench.eval import run_case, run_case_rolling  # noqa: E402
from tsforecastbench.bench.eval.loop import _naive_ds  # noqa: E402
from tsforecastbench.bench.methods.base import Forecaster  # noqa: E402
from tsforecastbench.bench.methods.torchdl import _build_val_windows  # noqa: E402
from tsforecastbench.data.registry import get_case  # noqa: E402
from tsforecastbench.data.schema import Case  # noqa: E402

# One case whose record is a single span and two whose extended history splits into several
# segments, so the pooled-panel fit frame is covered as well as the local one.
CASES = ["surge_cuxhaven", "discharge_lowland_denmark", "waves_oregon"]
RUNNERS = {
    "single_holdout": lambda case, m: run_case(case, [m], prefer_synth=True),
    "rolling": lambda case, m: run_case_rolling(case, [m], n_origins=3, prefer_synth=True),
}


class _FitFrameRecorder(Forecaster):
    """Keeps every frame `fit` and `refit` are handed, and forecasts zeros.

    `pools_segments` is on so the eval loop builds the extended-history fit too.
    """

    name = "Recorder"
    family = "test"
    pools_segments = True

    def __init__(self) -> None:
        self.fit_frames: list[pd.DataFrame] = []
        self.val_frames: list[pd.DataFrame] = []
        self.refit_frames: list[pd.DataFrame] = []

    def fit(self, case: Case, train: pd.DataFrame, val: pd.DataFrame) -> None:
        self.fit_frames.append(train)
        self.val_frames.append(val)

    def refit(self, case: Case, train_val: pd.DataFrame) -> None:
        self.refit_frames.append(train_val)

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        step = pd.Timedelta(pd.tseries.frequencies.to_offset(case.freq))
        ds = pd.date_range(_naive_ds(history).max() + step, periods=horizon, freq=case.freq)
        return pd.DataFrame({"ds": ds, "y_hat": np.zeros(horizon)})


class _StopsOnValRecorder(_FitFrameRecorder):
    stops_on_val = True


def _bounds(case: Case) -> tuple[pd.Timestamp, pd.Timestamp]:
    train_end, val_end, _ = case.split_bounds(case.load(prefer_synth=True, curated=True))
    return train_end, val_end


def _ends_at_val_end(frame: pd.DataFrame, case: Case, val_end: pd.Timestamp) -> bool:
    """`val_end` is a quantile of the timestamps, so it can fall between two of them."""
    step = pd.Timedelta(pd.tseries.frequencies.to_offset(case.freq))
    last = _naive_ds(frame).max()
    return val_end - step < last <= val_end


@pytest.mark.parametrize("runner", RUNNERS)
@pytest.mark.parametrize("cid", CASES)
def test_fit_gets_everything_before_the_test_period(cid, runner):
    case = get_case(cid)
    recorder = _FitFrameRecorder()
    RUNNERS[runner](case, recorder)

    _, val_end = _bounds(case)
    assert recorder.fit_frames, "fit was never called"
    for frame, val in zip(recorder.fit_frames, recorder.val_frames, strict=True):
        assert _ends_at_val_end(frame, case, val_end)
        assert len(val) == 0
    assert recorder.refit_frames == []


@pytest.mark.parametrize("runner", RUNNERS)
@pytest.mark.parametrize("cid", CASES)
def test_a_method_that_stops_on_val_finds_its_step_on_val_then_refits_on_both(cid, runner):
    case = get_case(cid)
    recorder = _StopsOnValRecorder()
    RUNNERS[runner](case, recorder)

    train_end, val_end = _bounds(case)
    assert len(recorder.refit_frames) == len(recorder.fit_frames) > 0
    for frame, val, refit in zip(
        recorder.fit_frames, recorder.val_frames, recorder.refit_frames, strict=True
    ):
        assert _naive_ds(frame).max() <= train_end
        assert _naive_ds(val).min() > train_end
        assert _ends_at_val_end(val, case, val_end)
        assert _ends_at_val_end(refit, case, val_end)
        assert len(refit) == len(frame) + len(val)


def test_the_local_val_frame_is_the_declared_split():
    case = get_case("surge_cuxhaven")
    recorder = _StopsOnValRecorder()
    run_case(case, [recorder], prefer_synth=True)

    declared = case.split_frame(case.load(prefer_synth=True, curated=True))["val"]
    assert len(recorder.val_frames[0]) == len(declared)


def test_val_windows_target_only_the_val_split():
    """Targets inside val, inputs allowed to reach back into train, one window per horizon."""
    # Values equal indices, so a target value below 1000 would mean a target step drawn
    # from train -- i.e. early stopping scoring itself on data the model trained on.
    train_tail = np.arange(1000, dtype=np.float32)
    val_y = np.arange(1000, 1240, dtype=np.float32)
    L, H = 168, 24

    X, _Xf, Y = _build_val_windows(train_tail, val_y, L, H)

    assert (Y >= 1000).all()
    assert len(X) == len(val_y) // H  # disjoint targets tiling the val split
    assert X.min() < 1000, "the first window's input should reach back into train"
    # Disjoint, in order, and tiling: each target starts where the previous one ended.
    np.testing.assert_array_equal(Y[:, 0], np.arange(1000, 1240, H, dtype=np.float32))


def test_dlinear_ignores_the_val_split():
    """DLinear has no training loop to stop, so the holdout must not touch its ridge fit."""
    from tsforecastbench.bench.methods.dl import DLinearForecaster

    case = get_case("waves_oregon")
    splits = case.split_frame(case.synth())
    train, val = splits["train"], splits["val"]

    with_val = DLinearForecaster()
    with_val.fit(case, train, val)
    without_val = DLinearForecaster()
    without_val.fit(case, train, val.iloc[:0])

    np.testing.assert_array_equal(with_val._model.coef_, without_val._model.coef_)


def test_val_windows_reject_a_val_split_too_short_for_one_window():
    with pytest.raises(ValueError, match="too few for one"):
        _build_val_windows(
            np.arange(1000, dtype=np.float32), np.arange(10, dtype=np.float32), 168, 24
        )
