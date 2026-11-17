"""Unit tests for the gap policy in `tsforecastbench.data.curate`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsforecastbench.data.curate import curate_frame, fraction_split_dates, gap_cap

SPLIT = [0.7, 0.85]


def _frame(y: np.ndarray, **exog: np.ndarray) -> pd.DataFrame:
    ds = pd.date_range("2020-01-01", periods=len(y), freq="1h", tz="UTC")
    return pd.DataFrame({"unique_id": "u", "ds": ds, "y": y, **exog})


def _curate(df: pd.DataFrame, exog: list[str] | None = None):
    return curate_frame(
        df, "1h", seasonalities=[24], horizon_primary="24h", split_fraction=SPLIT, exog=exog
    )


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
