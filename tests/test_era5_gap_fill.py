"""Unit tests for filling buoy gaps from ERA5 in `tsforecastbench.data.cases._era5`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.data.cases._era5 import fill_gaps_from_era5


def _hourly(n: int) -> pd.Series:
    return pd.Series(pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC"))


def test_buoy_gap_is_filled_with_era5_mapped_onto_the_buoy_scale():
    n = 400
    era5_wind = 5.0 + 3.0 * np.sin(np.arange(n) / 10.0)
    buoy_wind = 0.8 * era5_wind + 1.0
    buoy_wind[100:300] = np.nan
    buoy = pd.DataFrame({"ds": _hourly(n), "wspd": buoy_wind})
    era5 = pd.DataFrame({"ds": _hourly(n), "wspd10": era5_wind})

    filled, _ = fill_gaps_from_era5(
        buoy, era5, {"wspd": "wspd10"}, fit_until=pd.Timestamp("2020-01-17")
    )

    np.testing.assert_allclose(filled["wspd"], 0.8 * era5_wind + 1.0)


def test_era5_mapping_is_fitted_only_before_fit_until():
    n = 400
    era5_wind = 5.0 + 3.0 * np.sin(np.arange(n) / 10.0)
    buoy_wind = 0.8 * era5_wind + 1.0
    buoy_wind[350:] = 100.0  # after fit_until: must not shift the mapping
    buoy_wind[100:120] = np.nan
    buoy = pd.DataFrame({"ds": _hourly(n), "wspd": buoy_wind})
    era5 = pd.DataFrame({"ds": _hourly(n), "wspd10": era5_wind})

    filled, info = fill_gaps_from_era5(
        buoy, era5, {"wspd": "wspd10"}, fit_until=pd.Timestamp("2020-01-13")
    )

    np.testing.assert_allclose(filled["wspd"][100:120], 0.8 * era5_wind[100:120] + 1.0)
    assert info["wspd"]["n_filled"] == 20
