"""Synthetic daily groundwater depth-to-water driven by recharge from precipitation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.data.schema import Case


def generate(case: Case, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = min(case.synth_steps, 6 * 365)
    end = pd.Timestamp(case.splits.test_end, tz="UTC")
    idx = pd.date_range(end=end, periods=n, freq="1D")
    doy = idx.dayofyear.to_numpy()

    temp = 10.0 + 11.0 * np.cos(2 * np.pi * (doy - 205) / 365) + rng.normal(0, 2.0, n)
    wet_bias = 0.25 + 0.10 * np.cos(2 * np.pi * (doy - 30) / 365)
    precip = np.where(rng.random(n) < wet_bias, rng.gamma(1.4, 7.0, n), 0.0)

    # depth-to-water (m): rises (depth decreases) after recharge, slow decline (ET + drainage)
    # recharge is precip damped by warm-season ET
    et_factor = np.clip(1.0 - 0.03 * (temp - 10.0), 0.2, 1.0)
    recharge = 0.004 * precip * et_factor
    depth = np.zeros(n)
    d = 9.0
    for t in range(n):
        d = d + 0.004 - recharge[t]  # baseline decline minus recharge rise
        d = float(np.clip(d, 1.0, 18.0))
        depth[t] = d
    depth = depth + rng.normal(0, 0.02, n)

    return pd.DataFrame(
        {
            "unique_id": [case.id] * n,
            "ds": idx,
            "y": depth.astype(np.float64),
            "gwl_depth_m": depth.astype(np.float64),
            "precip": precip.astype(np.float64),
            "temp_air": temp.astype(np.float64),
        }
    )
