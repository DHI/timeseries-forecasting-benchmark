"""Unit tests for splitting the extended-history frame into gap-free segments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.bench.eval.loop import _segment_extended
from tsforecastbench.data.registry import get_case


def test_extended_history_splits_at_an_exogenous_gap():
    case = get_case("waves_oregon")
    n = 6000
    extended = pd.DataFrame(
        {
            "unique_id": case.id,
            "ds": pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC"),
            "y": np.ones(n),
            **{c: np.ones(n) for c in case.exogenous},
        }
    )
    extended.loc[2000:2999, "wspd"] = np.nan  # the buoy's anemometer was down

    segments = _segment_extended(extended, case, horizon=24)

    assert len(segments) == 2
    assert all(seg["wspd"].notna().all() for seg in segments)
