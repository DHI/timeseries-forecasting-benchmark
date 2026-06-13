"""Synthetic snowmelt-dominated daily discharge: a smooth temperature-driven melt hydrograph."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.data.schema import Case


def generate(case: Case, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = min(case.synth_steps, 6 * 365)
    end = pd.Timestamp(case.splits.test_end, tz="UTC")
    idx = pd.date_range(end=end, periods=n, freq="1D")
    doy = idx.dayofyear.to_numpy()

    temp = 8.0 + 12.0 * np.cos(2 * np.pi * (doy - 205) / 365) + rng.normal(0, 2.0, n)
    precip = np.where(rng.random(n) < 0.2, rng.gamma(1.3, 6.0, n), 0.0)

    # snowpack accumulates when cold, melts when warm; melt -> discharge with routing lag
    snow = 0.0
    melt = np.zeros(n)
    for t in range(n):
        if temp[t] < 0.5:
            snow += 0.6 * (precip[t] + 4.0)  # winter accumulation
        else:
            m = min(snow, 0.9 * max(temp[t], 0.0))
            snow -= m
            melt[t] = m
    route = np.array([0.3, 0.25, 0.2, 0.13, 0.07, 0.05])
    routed = np.convolve(melt, route, mode="full")[:n]
    baseflow = 1.5 + 0.02 * np.convolve(precip, np.ones(30) / 30, mode="same")
    discharge = np.clip(baseflow + 0.5 * routed + rng.normal(0, 0.3, n), 0.2, None)

    return pd.DataFrame(
        {
            "unique_id": [case.id] * n,
            "ds": idx,
            "y": discharge.astype(np.float64),
            "discharge": discharge.astype(np.float64),
            "temp_air": temp.astype(np.float64),
            "precip": precip.astype(np.float64),
        }
    )
