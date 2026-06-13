"""Synthetic hourly non-tidal residual for a NW European tide gauge."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.schema import Case


def generate(case: Case, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = min(case.synth_steps * 24, 4 * 365 * 24)
    end = pd.Timestamp(case.splits.test_end, tz="UTC")
    idx = pd.date_range(end=end, periods=n, freq="1h")
    doy = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy()

    base_mslp = 1013.0 + 4.0 * np.cos(2 * np.pi * (doy - 30) / 365)
    synoptic = np.cumsum(rng.normal(0, 0.6, n))
    synoptic = synoptic - np.linspace(synoptic[0], synoptic[-1], n)
    storms = np.zeros(n)
    n_storms = max(1, n // (24 * 30))
    storm_starts = rng.integers(0, max(1, n - 72), size=n_storms)
    for s in storm_starts:
        depth = rng.uniform(8, 25)
        duration = rng.integers(24, 72)
        kernel = -depth * np.exp(
            -((np.arange(duration) - duration / 2) ** 2) / (duration / 4) ** 2
        )
        endi = min(n, s + duration)
        storms[s:endi] += kernel[: endi - s]
    mslp = base_mslp + synoptic + storms

    dP = np.gradient(mslp)
    u10 = -2.0 * dP + 2.0 * np.sin(2 * np.pi * hour / 24.0) + rng.normal(0, 1.5, n)
    v10 = 1.2 * dP + rng.normal(0, 1.5, n)

    inv_baro = -0.01 * (mslp - 1013.0)
    wind_stress = 0.0008 * (u10**2 + v10**2) * np.sign(u10)
    smoother = np.ones(6) / 6
    residual = np.convolve(inv_baro + wind_stress, smoother, mode="same")

    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.7 * noise[t - 1] + rng.normal(0, 0.03)
    residual = residual + noise

    return pd.DataFrame(
        {
            "unique_id": [case.id] * n,
            "ds": idx,
            "y": residual.astype(np.float64),
            "residual_m": residual.astype(np.float64),
            "mslp": mslp.astype(np.float64),
            "u10": u10.astype(np.float64),
            "v10": v10.astype(np.float64),
        }
    )
