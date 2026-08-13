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
from tsforecastbench.bench.methods.base import Forecaster  # noqa: E402
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
    """Keeps every frame `fit` is handed, and forecasts zeros.

    `pools_segments` is on so the eval loop builds the extended-history fit too. `val` is
    optional so that against the pre-#99 two-argument `fit` these tests fail on the frame
    boundary -- the actual defect -- rather than on a TypeError.
    """

    name = "Recorder"
    family = "test"
    is_global = True
    pools_segments = True

    def __init__(self) -> None:
        self.fit_frames: list[pd.DataFrame] = []

    def fit(self, case: Case, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        self.fit_frames.append(train)

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
