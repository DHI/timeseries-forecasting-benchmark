"""Synthetic daily discharge series for a groundwater-dominated DK catchment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.data.schema import Case


def generate(case: Case, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = case.synth_steps
    end = pd.Timestamp(case.splits.test_end, tz="UTC")
    idx = pd.date_range(end=end, periods=n, freq="D")
    doy = idx.dayofyear.to_numpy()

    seasonal = 1.2 + 0.6 * np.cos(2 * np.pi * (doy - 30) / 365)
    wet = rng.random(n) < 0.45
    precip = np.where(wet, rng.gamma(shape=1.2, scale=2.5, size=n), 0.0)
    pet = np.clip(
        1.5 + 1.5 * np.cos(2 * np.pi * (doy - 200) / 365) + rng.normal(0, 0.3, n), 0.0, None
    )
    temp = 8.5 + 8.0 * np.cos(2 * np.pi * (doy - 200) / 365) + rng.normal(0, 1.5, n)

    q = np.zeros(n)
    q[0] = seasonal[0]
    decay = 0.92
    for t in range(1, n):
        baseflow = decay * q[t - 1] + (1 - decay) * seasonal[t]
        lag_window = max(0, t - 5)
        recent_precip = precip[lag_window : t + 1].mean()
        q[t] = baseflow + 0.18 * recent_precip + rng.normal(0, 0.05)
    q = np.clip(q, 0.01, None)

    return pd.DataFrame(
        {
            "unique_id": [case.id] * n,
            "ds": idx,
            "y": q.astype(np.float64),
            "q_obs": q.astype(np.float64),
            "precip": precip.astype(np.float64),
            "temp": temp.astype(np.float64),
            "pet": pet.astype(np.float64),
        }
    )
