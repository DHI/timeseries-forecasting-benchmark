"""Synthetic flashy daily river discharge driven by synthetic precipitation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.data.schema import Case


def generate(case: Case, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = min(case.synth_steps, 6 * 365)
    end = pd.Timestamp(case.splits.test_end, tz="UTC")
    idx = pd.date_range(end=end, periods=n, freq="1D")
    doy = idx.dayofyear.to_numpy()

    # seasonal air temperature
    temp = 9.0 + 11.0 * np.cos(2 * np.pi * (doy - 205) / 365) + rng.normal(0, 2.0, n)
    # sparse, heavy-tailed precipitation (wetter in cool season)
    wet_bias = 0.25 + 0.12 * np.cos(2 * np.pi * (doy - 30) / 365)
    is_wet = rng.random(n) < wet_bias
    precip = np.where(is_wet, rng.gamma(1.4, 7.0, n), 0.0)

    # flashy runoff: fast unit hydrograph (short recession) + slow baseflow
    fast_kernel = np.array([0.45, 0.25, 0.14, 0.08, 0.05, 0.03])
    fast = np.convolve(precip, fast_kernel, mode="full")[:n]
    baseflow = np.zeros(n)
    b = 2.0
    for t in range(n):
        b = 0.985 * b + 0.02 * precip[t] + 0.01  # slow store, small recharge
        baseflow[t] = b
    discharge = baseflow + 0.6 * fast + rng.normal(0, 0.15, n).clip(-0.5, None)
    discharge = np.clip(discharge, 0.05, None)

    return pd.DataFrame(
        {
            "unique_id": [case.id] * n,
            "ds": idx,
            "y": discharge.astype(np.float64),
            "discharge": discharge.astype(np.float64),
            "precip": precip.astype(np.float64),
            "temp_air": temp.astype(np.float64),
        }
    )
