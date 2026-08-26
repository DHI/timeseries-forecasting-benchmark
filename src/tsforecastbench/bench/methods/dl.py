"""Lightweight deep-learning-style baselines that run on CPU without torch.

DLinear (Zeng et al., 2023) is the "embarrassingly simple" linear model that famously
matches or beats many transformers on long-horizon forecasting. It decomposes the input
window into a moving-average **trend** and a **remainder**, applies a separate linear map
from the lookback to the full horizon for each, and sums them. With both maps being linear,
the whole model is a single multi-output ridge regression on the concatenated decomposed
window — so we get a faithful DLinear with nothing heavier than scikit-learn.

Comes in two flavours, like the windowed-ML methods in `ml.py`: univariate, and
exogenous-aware (``use_exogenous=True``) which appends the covariate values over the
forecast horizon to the decomposed window. The covariate contract is the ML one -- the value
at the step being predicted, supplied by the eval loop as a perfect-forecast proxy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.bench.methods.base import (
    Forecaster,
    _exo_cols,
    build_windows_fut,
    future_exo_block,
    last_observed_exo,
    lookback_for,
    segment_arrays,
    segment_exo_arrays,
)
from tsforecastbench.bench.methods.scorecard import (
    Capabilities,
    ForecastStrategy,
    GrowthShape,
    Level,
    RunningCost,
    Scorecard,
)
from tsforecastbench.data.schema import Case


def _moving_avg(x: np.ndarray, k: int) -> np.ndarray:
    """Centered moving-average trend with edge padding (DLinear's decomposition kernel)."""
    if k <= 1:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    # Kernel dtype follows the window matrix: a float64 kernel would silently upcast a
    # float32 design matrix here and undo what `build_windows` allocated (gh #101).
    kernel = np.ones(k, dtype=x.dtype) / k
    return np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid")[: x.shape[1]], 1, xp)


class DLinearForecaster(Forecaster):
    """Decomposes the lookback window into a moving-average trend and remainder, fits a
    separate linear map from each to the full horizon, and sums them."""

    name = "DLinear"
    family = "dl"
    is_global = True
    pools_segments = True
    library = "scikit-learn (Ridge) — no deep-learning framework"
    compute_profile = (
        "CPU only, fast; both linear maps collapse into a single closed-form ridge "
        "fit on the decomposed window, so there's no gradient-descent training loop."
    )
    limitations = (
        "The trend/remainder decomposition is a fixed moving average, not learned, so an "
        "unusually shaped trend can leak into the remainder term. Its kernel is a constant "
        "25 steps while the lookback now runs from 48 to 365 depending on the case, so the "
        "split between trend and remainder is sharper on a short window than a long one. "
        "The univariate variant ignores exogenous inputs; the +exo variant sees the "
        "covariate values over the forecast horizon and not their history."
    )
    code_snippet = (
        "from sklearn.linear_model import Ridge\n\n"
        "trend = moving_average(window, kernel)\n"
        "remainder = window - trend\n"
        "model = Ridge(alpha=alpha).fit(concat([trend, remainder]), target_window)\n"
        "y_hat = model.predict(concat([trend, remainder]))"
    )
    # An explicit trend branch, and a learned map from the whole lookback window to the whole
    # horizon, which can represent several interacting periods at once rather than one
    # declared period. `exogenous` is True for the estimator, not for this column: the ridge
    # takes covariate columns whenever `use_exogenous` is set, which is all the +exo variant
    # switches (same arrangement as `ml.py`'s `_WindowedMLBase`).
    capabilities = Capabilities(
        trend=True, seasonality=True, multiple_seasonality=True, exogenous=True
    )
    # Both linear maps predict the whole horizon from the lookback window in one closed-form
    # fit — no step's output feeds into computing another step.
    forecast_strategy = ForecastStrategy.DIRECT
    # The cheapest thing in the `dl` family by a wide margin: both linear maps collapse into
    # one closed-form ridge fit, so there is no training loop and nothing to tune, and a fit
    # plus a thousand forecasts comes in at ~26s. Needs many windows though -- under 10 it
    # gives up and falls back to persistence.
    scorecard = Scorecard(
        history_requirements=Level.HIGH,
        interpretability=Level.MEDIUM,
        running_cost=RunningCost(level=Level.LOW, growth=GrowthShape.LINEAR),
        tuning_effort=Level.LOW,
    )

    use_exogenous: bool = False

    def __init__(self, lookback: int | None = None, kernel: int = 25, alpha: float = 1.0):
        self.lookback = lookback
        self.kernel = kernel
        self.alpha = alpha
        self.uses_exogenous = self.use_exogenous
        self._exo: list[str] = []
        self._last_exo: dict[str, float] = {}

    def fit(self, case: Case, train: pd.DataFrame, val: pd.DataFrame) -> None:
        """Fit the ridge on every window in `train`.

        `val` goes unread: there is no training loop to stop, so a closed-form fit has
        nothing to decide with a holdout. It still never sees the val split -- `train` is
        the train split alone (gh #99) -- which is what keeps its training data the same as
        every other method's.
        """
        from sklearn.linear_model import Ridge

        from tsforecastbench.bench.eval.loop import _horizon_to_steps

        self._H = _horizon_to_steps(case.horizon_primary, case.freq)
        self._L = self.lookback or lookback_for(case, self._H)
        L, H = self._L, self._H

        # Windows are built per contiguous segment (unique_id) and never across a gap; a
        # plain local frame is a single segment, so this reduces to the original loop. Every
        # window is in the design matrix — the 15,000-window cap that used to stride it went
        # with gh #101.
        self._exo = _exo_cols(case) if self.use_exogenous else []
        self._last_exo = last_observed_exo(train, self._exo)
        segs, _tail = segment_arrays(train)
        exo_segs = segment_exo_arrays(train, self._exo) if self._exo else None
        X, Xf, Y = build_windows_fut(segs, exo_segs, L, H)
        if X is None:  # too little data: fall back to persistence at predict time
            self._model = None
            return
        self._model = Ridge(alpha=self.alpha).fit(self._features(X, Xf), Y)

    def _decompose(self, X: np.ndarray) -> np.ndarray:
        trend = _moving_avg(X, self.kernel)
        remainder = X - trend
        return np.concatenate([trend, remainder], axis=1)

    def _features(self, X: np.ndarray, Xf: np.ndarray | None) -> np.ndarray:
        """`[trend | remainder]`, plus the flattened horizon covariates for the +exo variant.

        The covariate block appends after the decomposed target rather than being folded into
        it: `_moving_avg` is a centred mean over adjacent time steps, and running it across a
        target-then-covariate boundary would average unrelated quantities.
        """
        feats = self._decompose(X)
        if not self._exo or Xf is None or Xf.shape[2] == 0:
            return feats
        # Raw, on the covariates' own scales, matching the raw target window beside them.
        # Standardising them first is the textbook move and it measured far worse: the ridge
        # penalty is absolute, so unit-variance columns need coefficients large enough that
        # `alpha=1` stops restraining the near-duplicate horizon columns a smooth covariate
        # produces. On `waves_oregon` that took MASE from 0.85 to ~3.7e6; on the other four
        # covariate cases tried it changed nothing to three decimals.
        flat = Xf.reshape(len(Xf), -1)
        flat = np.where(np.isfinite(flat), flat, 0.0)
        return np.concatenate([feats, flat.astype(feats.dtype, copy=False)], axis=1)

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
        xf = None
        if self._exo:
            xf = future_exo_block(self._exo, history, self._H, future_exog, self._last_exo).reshape(
                1, self._H, len(self._exo)
            )
        y_hat = self._model.predict(self._features(win, xf))[0]
        if len(y_hat) < horizon:  # horizon longer than trained H: pad with last value
            y_hat = np.concatenate([y_hat, np.full(horizon - len(y_hat), y_hat[-1])])
        return pd.DataFrame({"ds": future, "y_hat": y_hat[:horizon].astype(np.float64)})


class DLinearExoForecaster(DLinearForecaster):
    """DLinear with the case's declared covariates over the forecast horizon appended to the
    decomposed window."""

    name = "DLinear+exo"
    use_exogenous = True
