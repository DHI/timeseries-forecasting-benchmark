"""The surge target's tide model must be fitted on the pre-test record only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.data.cases.surge_cuxhaven.download import _tidal_residual

M2_HOURS = 12.4206012


def test_tide_is_fitted_only_before_fit_until():
    ds = pd.Series(pd.date_range("2020-01-01", periods=24 * 120, freq="1h", tz="UTC"))
    hours = np.arange(len(ds), dtype=float)
    amplitude = np.where(ds < pd.Timestamp("2020-04-01", tz="UTC"), 1.0, 3.0)
    df = pd.DataFrame({"ds": ds, "sea_level": amplitude * np.cos(2 * np.pi * hours / M2_HOURS)})

    out = _tidal_residual(df, lat=53.9, fit_until=pd.Timestamp("2020-04-01"))

    pre = out["ds"] < pd.Timestamp("2020-04-01", tz="UTC")
    assert out.loc[pre, "residual_m"].abs().max() < 0.05
    assert out.loc[~pre, "residual_m"].abs().max() > 1.5
