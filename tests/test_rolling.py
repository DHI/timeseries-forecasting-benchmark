"""Unit tests for rolling-origin selection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.bench.eval.rolling import _select_rolling_origins  # noqa: E402


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
