"""Lightweight deep-learning-style baselines that run on CPU without torch.

DLinear (Zeng et al., 2023) is the "embarrassingly simple" linear model that famously
matches or beats many transformers on long-horizon forecasting. It decomposes the input
window into a moving-average **trend** and a **remainder**, applies a separate linear map
from the lookback to the full horizon for each, and sums them. With both maps being linear,
the whole model is a single multi-output ridge regression on the concatenated decomposed
window — so we get a faithful DLinear with nothing heavier than scikit-learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.data.schema import Case


def _lookback_for(freq: str) -> int:
    f = freq.lower()
    if f in {"1d", "d"}:
        return 64
    if f in {"1h", "h"}:
        return 168
    if f in {"15min", "15t"}:
        return 192
    if f in {"1w", "w"}:
        return 52
    return 48


def _moving_avg(x: np.ndarray, k: int) -> np.ndarray:
    """Centered moving-average trend with edge padding (DLinear's decomposition kernel)."""
    if k <= 1:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    kernel = np.ones(k) / k
    return np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid")[: x.shape[1]], 1, xp)


class DLinearForecaster(Forecaster):
    name = "DLinear"
    max_samples: int = 8000

    def __init__(self, lookback: int | None = None, kernel: int = 25, alpha: float = 1.0):
        self.lookback = lookback
        self.kernel = kernel
        self.alpha = alpha

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from sklearn.linear_model import Ridge

        from tsforecastbench.bench.eval.loop import _horizon_to_steps

        self._H = _horizon_to_steps(case.horizon_primary, case.freq)
        self._L = self.lookback or _lookback_for(case.freq)
        y = pd.to_numeric(train["y"], errors="coerce").to_numpy(dtype=float)
        y = pd.Series(y).interpolate(limit_direction="both").bfill().ffill().to_numpy()
        self._last_window = y[-self._L :]

        L, H = self._L, self._H
        n = len(y) - L - H + 1
        if n <= 10:  # too short: fall back to persistence at predict time
            self._model = None
            return
        # stride so we keep at most max_samples training windows
        stride = max(1, n // self.max_samples)
        starts = np.arange(0, n, stride)
        X = np.stack([y[s : s + L] for s in starts])
        Y = np.stack([y[s + L : s + L + H] for s in starts])
        feats = self._decompose(X)
        self._model = Ridge(alpha=self.alpha).fit(feats, Y)

    def _decompose(self, X: np.ndarray) -> np.ndarray:
        trend = _moving_avg(X, self.kernel)
        remainder = X - trend
        return np.concatenate([trend, remainder], axis=1)

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        future = self._future_index(case, history["ds"].max(), horizon)
        if getattr(self, "_model", None) is None:
            last = float(pd.to_numeric(history["y"], errors="coerce").ffill().iloc[-1])
            return pd.DataFrame({"ds": future, "y_hat": np.full(horizon, last)})
        win = self._last_window.reshape(1, -1)
        y_hat = self._model.predict(self._decompose(win))[0]
        if len(y_hat) < horizon:  # horizon longer than trained H: pad with last value
            y_hat = np.concatenate([y_hat, np.full(horizon - len(y_hat), y_hat[-1])])
        return pd.DataFrame({"ds": future, "y_hat": y_hat[:horizon].astype(np.float64)})
