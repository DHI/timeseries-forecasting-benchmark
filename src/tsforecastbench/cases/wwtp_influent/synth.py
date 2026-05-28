"""Synthetic hourly WWTP influent flow: diurnal + weekly pattern + rain-driven peaks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.schema import Case


def generate(case: Case, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = min(case.synth_steps * 24, 2 * 365 * 24)
    end = pd.Timestamp(case.splits.test_end, tz="UTC")
    idx = pd.date_range(end=end, periods=n, freq="1h")
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()

    # Diurnal dry-weather flow: morning + evening peaks, lower at night
    diurnal = (
        1.0
        + 0.35 * np.exp(-((hour - 9.0) ** 2) / 6.0)
        + 0.30 * np.exp(-((hour - 20.0) ** 2) / 8.0)
        - 0.25 * np.exp(-((hour - 4.0) ** 2) / 6.0)
    )
    weekly = 1.0 - 0.08 * (dow >= 5)  # slightly lower on weekends
    base = 5000.0 * diurnal * weekly  # m3/h scale for ~350k PE

    # Rain-derived infiltration/inflow events
    wet = rng.random(n) < 0.03
    rain = np.where(wet, rng.gamma(1.0, 2.0, n), 0.0)
    kernel = np.array([0.0, 0.1, 0.25, 0.3, 0.2, 0.1, 0.05])
    rain_resp = np.convolve(rain, kernel, mode="same")
    flow = base * (1.0 + 0.15 * rain_resp) + rng.normal(0, 80.0, n)
    flow = np.clip(flow, 500.0, None)

    return pd.DataFrame(
        {
            "unique_id": [case.id] * n,
            "ds": idx,
            "y": flow.astype(np.float64),
            "influent_flow": flow.astype(np.float64),
        }
    )
