"""Synthetic daily river discharge for Saugeen River (Ontario, Canada).

Models a snowmelt-dominated regime: large spring freshet, summer low flows,
moderate autumn/winter baseflow with superimposed rainfall events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.data.schema import Case


def generate(case: Case, seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = min(case.synth_steps, 8 * 365)
    end = pd.Timestamp(case.splits.test_end, tz="UTC")
    idx = pd.date_range(end=end, periods=n, freq="1D")
    doy = idx.dayofyear.to_numpy()

    # spring snowmelt freshet (~day 110, late April)
    freshet = 40.0 * np.exp(-0.5 * ((doy - 110) / 15) ** 2)
    # secondary autumn pulse (~day 300, late October)
    autumn = 8.0 * np.exp(-0.5 * ((doy - 300) / 25) ** 2)
    seasonal = freshet + autumn + 5.0

    # stochastic rainfall events
    is_wet = rng.random(n) < 0.22
    rain_pulse = np.where(is_wet, rng.gamma(1.2, 6.0, n), 0.0)
    fast_kernel = np.array([0.35, 0.25, 0.17, 0.11, 0.07, 0.05])
    rain_contribution = np.convolve(rain_pulse, fast_kernel, mode="full")[:n]

    # slow baseflow storage
    baseflow = np.zeros(n)
    b = 10.0
    for t in range(n):
        b = 0.99 * b + 0.05 * rain_pulse[t] + 0.02 * seasonal[t]
        baseflow[t] = b

    discharge = baseflow + 0.4 * rain_contribution + rng.normal(0, 0.5, n).clip(-1.0, None)
    discharge = np.clip(discharge, 0.5, None)

    return pd.DataFrame({
        "unique_id": [case.id] * n,
        "ds": idx,
        "y": discharge.astype(np.float64),
        "discharge": discharge.astype(np.float64),
    })
