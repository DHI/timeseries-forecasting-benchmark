"""Synthetic hourly significant wave height for a NDBC-style buoy.

Wind-sea + swell: Hs responds to wind speed (wind-sea) plus a slowly varying swell
background, with storm episodes. Exogenous wind/pressure/temperatures generated too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.data.schema import Case


def generate(case: Case, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = min(case.synth_steps * 24, 6 * 365 * 24)
    end = pd.Timestamp(case.splits.test_end, tz="UTC")
    idx = pd.date_range(end=end, periods=n, freq="1h")
    doy = idx.dayofyear.to_numpy()

    # Wind speed: seasonal (stormier winter) + synoptic wander + gusts
    base_wind = 6.0 + 3.0 * np.cos(2 * np.pi * (doy - 20) / 365)
    synoptic = np.abs(np.cumsum(rng.normal(0, 0.4, n)))
    synoptic = synoptic - np.linspace(synoptic[0], synoptic[-1], n)
    wspd = np.clip(base_wind + 0.5 * synoptic + rng.gamma(2.0, 1.0, n) - 2.0, 0.2, None)

    pres = 1015.0 + 5.0 * np.cos(2 * np.pi * (doy - 30) / 365) - 0.3 * (wspd - wspd.mean()) + rng.normal(0, 1.0, n)
    atmp = 11.0 + 7.0 * np.cos(2 * np.pi * (doy - 200) / 365) + rng.normal(0, 1.0, n)
    wtmp = 12.0 + 4.0 * np.cos(2 * np.pi * (doy - 220) / 365) + rng.normal(0, 0.5, n)

    # Wind-sea grows with wind^2 (smoothed), plus a swell background (slow), plus noise
    windsea = 0.02 * wspd**2
    swell = 1.0 + 0.6 * np.cos(2 * np.pi * (doy - 40) / 365)
    smoother = np.ones(8) / 8
    hs = np.convolve(windsea, smoother, mode="same") + swell
    # AR(1) memory + noise
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.8 * noise[t - 1] + rng.normal(0, 0.05)
    wvht = np.clip(hs + noise, 0.1, None)

    return pd.DataFrame(
        {
            "unique_id": [case.id] * n,
            "ds": idx,
            "y": wvht.astype(np.float64),
            "wvht": wvht.astype(np.float64),
            "wspd": wspd.astype(np.float64),
            "pres": pres.astype(np.float64),
            "atmp": atmp.astype(np.float64),
            "wtmp": wtmp.astype(np.float64),
        }
    )
