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

from tsforecastbench.bench.methods.base import Forecaster, lookback_for, segment_arrays
from tsforecastbench.data.schema import Case


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
    family = "deep_learning"
    is_global = True
    pools_segments = True
    max_samples: int = 8000
    library = "scikit-learn (Ridge) — no deep-learning framework"
    compute_profile = (
        "CPU only, fast; both linear maps collapse into a single closed-form ridge "
        "fit on the decomposed window, so there's no gradient-descent training loop."
    )
    limitations = (
        "Univariate — ignores exogenous inputs. The trend/remainder decomposition is "
        "a fixed moving average, not learned, so an unusually shaped trend can leak "
        "into the remainder term."
    )
    code_snippet = (
        "from sklearn.linear_model import Ridge\n\n"
        "trend = moving_average(window, kernel)\n"
        "remainder = window - trend\n"
        "model = Ridge(alpha=alpha).fit(concat([trend, remainder]), target_window)\n"
        "y_hat = model.predict(concat([trend, remainder]))"
    )

    def __init__(self, lookback: int | None = None, kernel: int = 25, alpha: float = 1.0):
        self.lookback = lookback
        self.kernel = kernel
        self.alpha = alpha

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from sklearn.linear_model import Ridge

        from tsforecastbench.bench.eval.loop import _horizon_to_steps

        self._H = _horizon_to_steps(case.horizon_primary, case.freq)
        self._L = self.lookback or lookback_for(case.freq, self._H)
        L, H = self._L, self._H

        # Windows are built per contiguous segment (unique_id) and never across a gap; a
        # plain local frame is a single segment, so this reduces to the original loop.
        segs, _tail = segment_arrays(train)
        counts = [max(0, len(y) - L - H + 1) for y in segs]
        total = sum(counts)
        if total <= 10:  # too little data: fall back to persistence at predict time
            self._model = None
            return
        # one shared stride across all segments so total windows stay <= max_samples
        stride = max(1, total // self.max_samples)
        Xs, Ys = [], []
        for y, n in zip(segs, counts, strict=True):
            if n <= 0:
                continue
            starts = np.arange(0, n, stride)
            Xs.append(np.stack([y[s : s + L] for s in starts]))
            Ys.append(np.stack([y[s + L : s + L + H] for s in starts]))
        X = np.concatenate(Xs)
        Y = np.concatenate(Ys)
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
        # Inference-only: the lookback window is the tail segment of the history handed in
        # at this origin, never state captured at fit — so one fit serves every origin.
        segs, tail = segment_arrays(history)
        win = segs[tail][-self._L :].reshape(1, -1)
        y_hat = self._model.predict(self._decompose(win))[0]
        if len(y_hat) < horizon:  # horizon longer than trained H: pad with last value
            y_hat = np.concatenate([y_hat, np.full(horizon - len(y_hat), y_hat[-1])])
        return pd.DataFrame({"ds": future, "y_hat": y_hat[:horizon].astype(np.float64)})
