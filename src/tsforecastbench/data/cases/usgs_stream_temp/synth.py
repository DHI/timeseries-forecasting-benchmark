"""Synthetic daily river water temperature, air-temperature driven."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.data.schema import Case


def generate(case: Case, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = min(case.synth_steps, 6 * 365)
    end = pd.Timestamp(case.splits.test_end, tz="UTC")
    idx = pd.date_range(end=end, periods=n, freq="1D")
    doy = idx.dayofyear.to_numpy()

    temp_air = 14.0 + 13.0 * np.cos(2 * np.pi * (doy - 205) / 365) + rng.normal(0, 2.5, n)
    precip = np.where(rng.random(n) < 0.3, rng.gamma(1.4, 6.0, n), 0.0)

    # water temp lags and damps air temp (thermal inertia); storms cool slightly; floored at 0
    wt = np.zeros(n)
    w = 10.0
    for t in range(n):
        target = 0.85 * temp_air[t] + 2.0 - 0.05 * precip[t]
        w = w + 0.25 * (target - w)  # first-order thermal response
        wt[t] = w
    wt = np.clip(wt + rng.normal(0, 0.4, n), 0.0, None)

    return pd.DataFrame(
        {
            "unique_id": [case.id] * n,
            "ds": idx,
            "y": wt.astype(np.float64),
            "water_temp": wt.astype(np.float64),
            "temp_air": temp_air.astype(np.float64),
            "precip": precip.astype(np.float64),
        }
    )
