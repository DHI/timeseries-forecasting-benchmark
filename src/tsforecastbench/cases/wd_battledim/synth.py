"""Synthetic hourly L-Town inlet flow: strong diurnal + weekly + weak seasonal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.schema import Case


def generate(case: Case, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = min(case.synth_steps * 24, 2 * 365 * 24)
    end = pd.Timestamp(case.splits.test_end, tz="UTC")
    idx = pd.date_range(end=end, periods=n, freq="1h")
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    doy = idx.dayofyear.to_numpy()

    diurnal = (
        100.0
        + 35.0 * np.exp(-((hour - 8.5) ** 2) / 8.0)
        + 30.0 * np.exp(-((hour - 20.0) ** 2) / 10.0)
        - 30.0 * np.exp(-((hour - 4.0) ** 2) / 6.0)
    )
    weekly = 1.0 - 0.07 * (dow >= 5)  # lower weekends
    seasonal = 1.0 + 0.10 * np.cos(2 * np.pi * (doy - 210) / 365)  # summer high
    flow = diurnal * weekly * seasonal + rng.normal(0, 3.0, n)
    flow = np.clip(flow, 20.0, None)

    return pd.DataFrame(
        {
            "unique_id": [case.id] * n,
            "ds": idx,
            "y": flow.astype(np.float64),
            "inlet_flow": flow.astype(np.float64),
        }
    )
