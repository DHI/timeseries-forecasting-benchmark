"""Unit tests for the gap policy in `tsforecastbench.data.curate`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsforecastbench.data.curate import (
    curate_frame,
    fraction_split_dates,
    gap_cap,
    longest_segment_bounds,
    split_fraction_for,
)

SPLIT = [0.7, 0.85]


def _frame(y: np.ndarray, **exog: np.ndarray) -> pd.DataFrame:
    ds = pd.date_range("2020-01-01", periods=len(y), freq="1h", tz="UTC")
    return pd.DataFrame({"unique_id": "u", "ds": ds, "y": y, **exog})


def _curate(df: pd.DataFrame, exog: list[str] | None = None):
    return curate_frame(df, "1h", seasonalities=[24], horizon_primary="24h", exog=exog)


def test_gap_cap_is_read_from_the_pre_test_record():
    rng = np.random.default_rng(0)
    smooth_history = np.cumsum(rng.normal(size=1700))  # autocorrelated: a long cap
    noisy_test = rng.normal(scale=200.0, size=300)  # white noise: a cap of 1
    df = _frame(np.concatenate([smooth_history, noisy_test]))

    curated, info = _curate(df)

    _, val_end = fraction_split_dates(curated["ds"], SPLIT)
    pre_test = df["y"][df["ds"].dt.tz_localize(None) <= val_end]
    assert info["cap"] == gap_cap(pre_test, "1h")
    assert info["cap"] != gap_cap(df["y"], "1h")


def _smooth(n: int) -> np.ndarray:
    return np.sin(np.arange(n) / 20.0)


def test_short_exogenous_gap_is_interpolated():
    wind = np.arange(2000, dtype=float)
    wind[500:502] = np.nan
    curated, info = _curate(_frame(_smooth(2000), wind=wind), exog=["wind"])

    assert info["cap"] >= 2
    assert curated["wind"].isna().sum() == 0
    assert curated["wind"].iloc[500] == 500.0


def test_exogenous_gap_longer_than_the_cap_fails():
    wind = np.ones(2000)
    wind[500:900] = np.nan  # straight-lining this would invent 400 h of wind

    with pytest.raises(ValueError, match="wind"):
        _curate(_frame(_smooth(2000), wind=wind), exog=["wind"])


def test_tail_length_is_checked_with_the_pre_test_cap():
    rng = np.random.default_rng(0)
    y = np.concatenate([_smooth(1700), rng.normal(scale=200.0, size=300)])
    y[1900:1902] = np.nan  # with the full record's cap of 1, this gap splits the record

    curated, info = _curate(_frame(y))

    assert info["cap"] >= 2
    assert len(curated) == 2000


def _daily(n: int) -> pd.Series:
    return pd.Series(pd.date_range("2000-01-01", periods=n, freq="1D"))


def test_split_is_15_percent_test_when_two_seasons_fit_inside_it():
    assert split_fraction_for(_daily(10_000), longest_period=7, freq="1D") == [0.7, 0.85]


def test_split_gives_two_seasons_of_test_when_15_percent_is_shorter():
    ds = _daily(2192)

    train_end, val_end = split_fraction_for(ds, longest_period=365, freq="1D")

    _, val_end_date = fraction_split_dates(ds, [train_end, val_end])
    assert (ds.max() - val_end_date).days >= 730
    assert val_end - train_end == pytest.approx(0.15, abs=1e-3)


def test_two_season_test_is_measured_in_time_when_observations_are_uneven():
    ds = _daily(2192)
    ds = ds[(ds < "2004-01-01") | (ds.dt.day % 2 == 0)]  # half the late days missing

    split = split_fraction_for(ds, longest_period=365, freq="1D")

    _, val_end_date = fraction_split_dates(ds, split)
    assert (ds.max() - val_end_date).days >= 730


def test_two_season_test_holds_when_a_gap_sits_at_the_cutoff():
    ds = _daily(2800)
    ds = ds.drop(range(2069, 2077)).reset_index(drop=True)  # 8 missing days at the cutoff

    split = split_fraction_for(ds, longest_period=365, freq="1D")

    _, val_end_date = fraction_split_dates(ds, split)
    assert ds.max() - val_end_date >= pd.Timedelta(days=730)


def test_longest_segment_is_kept_over_a_shorter_last_one():
    observed = np.ones(100, dtype=bool)
    observed[60:70] = False  # a 10-step gap splits the record into 60 and 30 steps

    assert longest_segment_bounds(observed, cap=5) == (0, 60)


def test_equal_segments_keep_the_latest():
    observed = np.ones(100, dtype=bool)
    observed[45:55] = False  # two 45-step segments

    assert longest_segment_bounds(observed, cap=5) == (55, 100)


def test_short_gaps_do_not_split_a_segment():
    observed = np.ones(100, dtype=bool)
    observed[60:63] = False
    observed[:2] = False  # leading and trailing unobserved rows are trimmed
    observed[-1] = False

    assert longest_segment_bounds(observed, cap=5) == (2, 99)


def test_last_segment_is_kept_when_long_enough():
    y = _smooth(2000)
    y[600:700] = np.nan  # leaves a 600 h first segment and a 1300 h last one

    curated, info = _curate(_frame(y))

    assert info["segment"] == "last"
    assert len(curated) == 1300


def test_longest_segment_is_kept_when_the_last_is_too_short():
    y = _smooth(2000)
    y[1900:1950] = np.nan  # the last 50 h cannot hold 4 daily cycles of train

    curated, info = _curate(_frame(y))

    assert info["segment"] == "longest"
    assert len(curated) == 1900


def test_curation_fails_when_no_segment_is_long_enough():
    y = _smooth(200)
    y[80:150] = np.nan  # segments of 80 h and 50 h

    with pytest.raises(ValueError, match="no segment is long enough"):
        _curate(_frame(y))


def test_split_is_reported_for_the_kept_segment():
    curated, info = _curate(_frame(_smooth(2000)))

    assert info["split_fraction"] == split_fraction_for(curated["ds"], 24, "1h")
