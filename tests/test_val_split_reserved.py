"""The declared val split must be a real holdout: no method may estimate parameters on it.

Before gh #99 every method was fit on `train + val` -- `eval.loop` concatenated the two into
one frame and the val boundary survived only as the offset locating where test begins -- while
the app displayed and described a three-way split. These tests pin the boundary on both eval
paths, for the single-series local fit and for the multi-segment extended-history one.
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
from tsforecastbench.bench.eval.budget import nf_val_size  # noqa: E402
from tsforecastbench.bench.methods.base import Forecaster  # noqa: E402
from tsforecastbench.bench.methods.torchdl import _build_val_windows  # noqa: E402
from tsforecastbench.data.registry import get_case  # noqa: E402
from tsforecastbench.data.schema import Case  # noqa: E402

# One case whose record is a single span and two whose extended history splits into several
# segments, so the pooled-panel fit frame is covered as well as the local one.
CASES = ["surge_cuxhaven", "discharge_lowland_denmark", "waves_oregon"]


def _naive_max_ds(frame: pd.DataFrame) -> pd.Timestamp:
    """Latest timestamp in `frame`, tz-naive UTC to match `Case.split_bounds`."""
    ds = pd.to_datetime(frame["ds"])
    if ds.dt.tz is not None:
        ds = ds.dt.tz_convert("UTC").dt.tz_localize(None)
    return ds.max()


class _FitFrameRecorder(Forecaster):
    """Keeps every pair of frames `fit` is handed, and forecasts zeros.

    `pools_segments` is on so the eval loop builds the extended-history fit too.
    """

    name = "Recorder"
    family = "test"
    is_global = True
    pools_segments = True

    def __init__(self) -> None:
        self.fit_frames: list[pd.DataFrame] = []
        self.val_frames: list[pd.DataFrame] = []

    def fit(self, case: Case, train: pd.DataFrame, val: pd.DataFrame) -> None:
        self.fit_frames.append(train)
        self.val_frames.append(val)

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        step = pd.Timedelta(pd.tseries.frequencies.to_offset(case.freq))
        ds = pd.date_range(_naive_max_ds(history) + step, periods=horizon, freq=case.freq)
        return pd.DataFrame({"ds": ds, "y_hat": np.zeros(horizon)})


def _train_end(case: Case) -> pd.Timestamp:
    return case.split_bounds(case.load(prefer_synth=True, curated=True))[0]


@pytest.mark.parametrize("cid", CASES)
def test_single_holdout_fit_stops_at_the_train_boundary(cid):
    case = get_case(cid)
    recorder = _FitFrameRecorder()
    run_case(case, [recorder], prefer_synth=True)

    train_end = _train_end(case)
    assert recorder.fit_frames, "fit was never called"
    for frame in recorder.fit_frames:
        assert _naive_max_ds(frame) <= train_end


@pytest.mark.parametrize("cid", CASES)
def test_rolling_fit_stops_at_the_train_boundary(cid):
    case = get_case(cid)
    recorder = _FitFrameRecorder()
    run_case_rolling(case, [recorder], n_origins=3, prefer_synth=True)

    train_end = _train_end(case)
    assert recorder.fit_frames, "fit was never called"
    for frame in recorder.fit_frames:
        assert _naive_max_ds(frame) <= train_end


@pytest.mark.parametrize("cid", CASES)
def test_fit_receives_the_declared_val_split(cid):
    """The other half of the contract: the holdout handed over IS the declared split."""
    case = get_case(cid)
    recorder = _FitFrameRecorder()
    run_case(case, [recorder], prefer_synth=True)

    df = case.load(prefer_synth=True, curated=True)
    train_end, val_end, _ = case.split_bounds(df)
    declared = case.split_frame(df)["val"]

    assert recorder.val_frames, "fit was never called"
    # The local fit's val frame is the declared split exactly. The extended fit's is the
    # same period taken from the segmented panel, so it carries segment `unique_id`s and
    # can be shorter where a gap falls inside the val period -- hence bounds, not equality.
    assert len(recorder.val_frames[0]) == len(declared)
    for frame in recorder.val_frames:
        ds = pd.to_datetime(frame["ds"])
        if ds.dt.tz is not None:
            ds = ds.dt.tz_convert("UTC").dt.tz_localize(None)
        assert ds.min() > train_end
        assert ds.max() <= val_end


def test_nf_val_size_uses_the_whole_val_split_when_the_panel_affords_it():
    # discharge_lowland_denmark's extended panel: one other segment, long enough to give up
    # the full declared val and still train.
    assert nf_val_size([3651], target_train_len=1534, lookback=64, h=7, val_len=329) == 329


def test_nf_val_size_backs_off_to_keep_every_segment():
    # waves_oregon: cutting the full 3,672-point val split would drop the four segments
    # shorter than 3,864 points, so the cut shrinks to what the 462-point one can spare.
    others = [462, 2019, 2808, 2975, 4580, 9099, 13456, 18473]
    val_size = nf_val_size(others, target_train_len=17176, lookback=168, h=24, val_len=3672)

    assert val_size == 270
    assert all(n >= 168 + 24 + val_size for n in others), "a segment would still be dropped"


def test_nf_val_size_never_exceeds_the_declared_val_split():
    # A panel of long segments could spare far more than the declared split holds; the cut
    # must not reach past it, since everything past it is test data.
    assert nf_val_size([50_000], target_train_len=40_000, lookback=168, h=24, val_len=458) == 458


def test_nf_val_size_rejects_a_val_split_below_the_floor():
    with pytest.raises(ValueError, match="floor"):
        nf_val_size([5000], target_train_len=5000, lookback=168, h=24, val_len=90)


def test_val_windows_target_only_the_val_split():
    """Targets inside val, inputs allowed to reach back into train, one window per horizon."""
    # Values equal indices, so a target value below 1000 would mean a target step drawn
    # from train -- i.e. early stopping scoring itself on data the model trained on.
    train_tail = np.arange(1000, dtype=np.float32)
    val_y = np.arange(1000, 1240, dtype=np.float32)
    L, H = 168, 24

    X, Y = _build_val_windows(train_tail, val_y, L, H)

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
