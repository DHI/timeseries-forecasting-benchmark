"""Synthetic 15-min sewer water level for a Bellinge level meter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.data.schema import Case


def generate(case: Case, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = min(case.synth_steps * 96, 3 * 365 * 96)
    end = pd.Timestamp(case.splits.test_end, tz="UTC")
    idx = pd.date_range(end=end, periods=n, freq="15min")
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    doy = idx.dayofyear.to_numpy()

    diurnal = (
        0.25
        + 0.10 * np.exp(-((hour - 8.0) ** 2) / 4.0)
        + 0.12 * np.exp(-((hour - 19.0) ** 2) / 6.0)
    )
    seasonal = 0.04 * np.cos(2 * np.pi * (doy - 30) / 365)
    wet = rng.random(n) < 0.04
    rain = np.where(wet, rng.gamma(shape=0.8, scale=1.8, size=n), 0.0)
    kernel = np.array([0.0, 0.05, 0.15, 0.25, 0.22, 0.15, 0.10, 0.05, 0.02, 0.01])
    rain_response = np.convolve(rain, kernel, mode="full")[:n]
    temp = (
        8.5
        + 8.0 * np.cos(2 * np.pi * (doy - 200) / 365)
        + 3.0 * np.sin(2 * np.pi * hour / 24.0)
        + rng.normal(0, 0.5, n)
    )
    level = diurnal + seasonal + 0.12 * rain_response + rng.normal(0, 0.012, n)
    level = np.clip(level, 0.05, 1.5)
    nowcast = np.roll(rain, -1)
    nowcast[-1] = 0.0
    nowcast = np.clip(nowcast + rng.normal(0, 0.05, n), 0.0, None)

    return pd.DataFrame(
        {
            "unique_id": [case.id] * n,
            "ds": idx,
            "y": level.astype(np.float64),
            "level_m": level.astype(np.float64),
            "rain_gauge": rain.astype(np.float64),
            "rain_radar_nowcast": nowcast.astype(np.float64),
            "temp_air": temp.astype(np.float64),
        }
    )
