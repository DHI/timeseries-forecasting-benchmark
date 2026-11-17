"""Abstract Forecaster interface — every method implements fit and predict."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from tsforecastbench.bench.methods.scorecard import Capabilities, ForecastStrategy, Scorecard
from tsforecastbench.data.schema import Case


def segment_arrays(train: pd.DataFrame, value_col: str = "y") -> tuple[list[np.ndarray], int]:
    """Split a training frame into one array per contiguous segment (``unique_id``).

    The extended-history frame an `is_global` method fits on carries several segments,
    one per gap-bounded stretch of the record, each under its own ``unique_id`` (see
    `tsforecastbench.bench.eval.loop._segment_extended`). Building windows across a
    segment boundary would treat gap-separated rows as adjacent time steps, so windowed
    methods must build them per segment instead. A plain local frame is a single
    segment, so this returns one array and the loop below is unchanged.

    Returns the per-segment value arrays in first-appearance (time) order, and the index
    of the *tail* segment — the one containing the latest ``ds``, i.e. the segment the
    forecast origin sits at. The tail array is what predict conditions on.
    """
    segs = [
        pd.to_numeric(g[value_col], errors="coerce").to_numpy(dtype=float)
        for _, g in train.groupby("unique_id", sort=False)
    ]
    uids = list(dict.fromkeys(train["unique_id"]))
    # Frames arrive in time order with the tail segment (forecast origin) last.
    tail_uid = train["unique_id"].iloc[-1]
    return segs, uids.index(tail_uid)


def build_windows(
    segs: list[np.ndarray], L: int, H: int
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Stack (input, target) windows over every segment, never across a gap.

    Every window is built. There is no stride, no truncation and no cap on the total (gh
    #101, following #95 and #97). Consecutive windows overlap in all but one step, so
    thinning them out used to look free — but it made the size of the training pool a
    hand-picked constant rather than a property of the record, and the same constant applied
    to a 1,500-window case and a 280,000-window one.

    Both matrices are allocated float32 and filled in place. Stacking float64 slices and
    casting afterwards kept three copies of the same data live at once, which on the largest
    (frame, case) pair peaked at roughly five times the size of the matrix it returned.

    Returns `(None, None)` when there are too few windows to learn anything from; callers
    fall back to persistence.
    """
    counts = [max(0, len(y) - L - H + 1) for y in segs]
    total = sum(counts)
    if total <= 10:
        return None, None
    X = np.empty((total, L), dtype=np.float32)
    Y = np.empty((total, H), dtype=np.float32)
    at = 0
    for y, n in zip(segs, counts, strict=True):
        if n <= 0:
            continue
        # One strided view over the segment, copied out in a single vectorised assignment.
        # The per-window Python loop this replaces grew linearly in window count, which with
        # the cap gone is the whole record rather than a bounded sample of it.
        view = np.lib.stride_tricks.sliding_window_view(y, L + H)
        X[at : at + n] = view[:, :L]
        Y[at : at + n] = view[:, L:]
        at += n
    return X, Y


def _lags_for(case: Case) -> list[int]:
    """Uniform geometric memory ladder plus the case's declared seasonal periods.

    The ladder is recent autoregressive state, identical for every case and frequency —
    it makes no cycle claim (the regressor is free to ignore any lag). Seasonal structure
    enters only through the declared periods, keeping the case.toml the single source of
    seasonal configuration."""
    ladder = [1, 2, 4, 8, 16, 32]
    return sorted(set(ladder) | set(case.seasonalities))


def _exo_cols(case: Case) -> list[str]:
    """Declared covariates, de-duplicated (a column may be both historical and known-future)."""
    return list(dict.fromkeys([*case.exogenous, *case.exogenous_known_future]))


def segment_exo_arrays(train: pd.DataFrame, cols: list[str]) -> list[np.ndarray]:
    """Per-segment covariate arrays, aligned one-for-one with `segment_arrays`.

    Returns one ``(len(segment), len(cols))`` array per ``unique_id``, in the same
    first-appearance order `segment_arrays` uses, so window `i` of the target lines up with
    window `i` of the covariates. A missing column becomes all-NaN rather than shifting the
    remaining ones left, which would silently pair a window with the wrong covariate.
    """
    out = []
    for _, g in train.groupby("unique_id", sort=False):
        block = np.empty((len(g), len(cols)), dtype=float)
        for j, c in enumerate(cols):
            if c in g.columns:
                block[:, j] = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
            else:
                block[:, j] = np.nan
        out.append(block)
    return out


def build_windows_fut(
    segs: list[np.ndarray], exo_segs: list[np.ndarray] | None, L: int, H: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """`build_windows`, plus each window's horizon-aligned covariate block.

    Returns ``(X, Xf, Y)`` with ``X`` shaped ``(N, L)``, ``Y`` shaped ``(N, H)`` -- both
    identical to what `build_windows` returns for the same input -- and ``Xf`` shaped
    ``(N, H, n_exo)`` holding the covariate values at the *same* timestamps as ``Y``. That is
    the covariate contract the windowed-ML `+exo` methods use: the value at the step being
    predicted, not the covariate's history (see `methods/ml.py`).

    All three come out of one function so the window ordering cannot drift between the target
    and its covariates. An off-by-one in ``Xf`` would shift covariates against the target and
    still train to convergence, so there is nothing to catch it downstream.

    `exo_segs` is `None` (or a list of width-0 blocks) on a case with no covariates, and then
    ``Xf`` is ``(N, H, 0)`` and the caller's model reduces to univariate.
    """
    counts = [max(0, len(y) - L - H + 1) for y in segs]
    total = sum(counts)
    if total <= 10:
        return None, None, None
    n_exo = 0 if exo_segs is None else (exo_segs[0].shape[1] if exo_segs else 0)
    X = np.empty((total, L), dtype=np.float32)
    Y = np.empty((total, H), dtype=np.float32)
    Xf = np.empty((total, H, n_exo), dtype=np.float32)
    at = 0
    for k, (y, n) in enumerate(zip(segs, counts, strict=True)):
        if n <= 0:
            continue
        view = np.lib.stride_tricks.sliding_window_view(y, L + H)
        X[at : at + n] = view[:, :L]
        Y[at : at + n] = view[:, L:]
        if n_exo:
            # One strided view per covariate column over the same segment, sliced to the
            # horizon span [L, L+H) -- the timestamps Y covers.
            e = exo_segs[k]
            for j in range(n_exo):
                ev = np.lib.stride_tricks.sliding_window_view(e[:, j], L + H)
                Xf[at : at + n, :, j] = ev[:n, L:]
        at += n
    return X, Xf, Y


def future_exo_block(
    cols: list[str],
    history: pd.DataFrame,
    horizon: int,
    future_exog: pd.DataFrame | None,
    fallback: dict[str, float],
) -> np.ndarray:
    """Covariate values over the forecast horizon, as ``(horizon, len(cols))``.

    `future_exog` is the eval loop's perfect-forecast proxy -- the covariates' true future
    values, standing in for a forecast of them (see `Forecaster.predict`). Same convention as
    `methods/ml.py`'s `_future_X` and `methods/fm.py`'s `_future_covariate`: fall back to the
    last observed value per column when the proxy is absent, short, or non-finite, so a
    covariate gap never puts a NaN into the model's input.

    `fallback` is the per-column last-observed value captured at fit. A column missing from
    both the proxy and `fallback` becomes 0.0, which is the standardised mean once the caller
    scales the block.
    """
    out = np.empty((horizon, len(cols)), dtype=float)
    for j, c in enumerate(cols):
        fill = fallback.get(c, 0.0)
        if future_exog is not None and c in future_exog.columns:
            vals = pd.to_numeric(future_exog[c], errors="coerce").to_numpy(dtype=float)
            if len(vals) >= horizon:
                vals = vals[:horizon]
            else:
                padded = np.full(horizon, fill, dtype=float)
                padded[: len(vals)] = vals
                vals = padded
        else:
            vals = np.full(horizon, fill, dtype=float)
        out[:, j] = np.where(np.isfinite(vals), vals, fill)
    return out


def last_observed_exo(train: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """Per-column last observed covariate value, the predict-time fallback for `future_exo_block`.

    Not training gap-fill: the curated frame already delivers complete covariate columns
    within the span (see `data/curate.py`). This only covers a horizon the proxy cannot fill.
    """
    out: dict[str, float] = {}
    for c in cols:
        if c in train.columns:
            s = pd.to_numeric(train[c], errors="coerce")
            out[c] = float(s.ffill().iloc[-1]) if s.notna().any() else 0.0
        else:
            out[c] = 0.0
    return out


class ExoScaler:
    """Per-column covariate standardisation, fitted once on the training pool.

    For the torch models, whose RevIN rescales the *target* window by its own statistics. A
    covariate has no relation to that scale -- `waves_oregon` mixes air pressure near
    1,000 hPa with wind speed near 5 m/s against a target near 1 m -- so a small network fed
    raw values on those scales saturates before it learns anything.

    Not for a ridge: `methods/dl.py` feeds its covariates raw on purpose, because a ridge
    penalty is absolute and unit-variance columns need coefficients large enough that
    `alpha=1` stops restraining them (measured, see that module).

    A zero-variance column maps to zeros, which is the honest encoding of a covariate that
    carries no information.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, exo_segs: list[np.ndarray] | None, n_exo: int) -> ExoScaler:
        if not n_exo or not exo_segs:
            return cls(np.zeros(0), np.ones(0))
        stacked = np.concatenate(exo_segs, axis=0)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        # A constant column must divide by 1.0, so it standardises to 0 and drops out. The
        # test has to be *relative* to the column's magnitude: `waves_oregon`'s `atmp` is
        # 12.7 at every one of its 17,176 training points, and the std of that comes back as
        # 1.8e-15 from a contiguous 1-D slice but 3.4e-12 from the axis reduction over the
        # strided 2-D column -- the reduction's own rounding error. Any absolute epsilon
        # small enough to look safe sits inside that spread, and dividing a real covariate
        # deviation by 3.4e-12 put 1e12 into the network.
        floor = 1e-8 * np.maximum(np.abs(mean), 1.0)
        std = np.where(np.isfinite(std) & (std > floor), std, 1.0)
        return cls(mean, std)

    def transform(self, block: np.ndarray) -> np.ndarray:
        """Standardise a `(..., n_exo)` block, sending non-finite values to the column mean."""
        if block.shape[-1] == 0:
            return block
        out = (block - self.mean) / self.std
        return np.where(np.isfinite(out), out, 0.0)


def lookback_for(case: Case, h: int) -> int:
    """Lookback window length for a windowed DL forecaster: the ML lag reach, floored at 2*h.

    A DL method sees a contiguous window and the windowed-ML pipeline sees the sparse
    `_lags_for` ladder, so the two can never take the same input. What they can share is how
    far back the earliest input sits, and that is what this matches: `max(_lags_for(case))`.
    The pair used to disagree in both directions -- a frequency-keyed constant (64 daily, 168
    hourly, 192 15-min, 52 weekly, introduced with no rationale in 573ec28 and never
    revisited) left the DL methods unable to see the annual cycle the six daily cases declare,
    while reaching further than the ML ladder does on five others.

    The `2*h` floor stays, so the window always covers at least two horizons. It binds on the
    four hourly cases whose declared periods are short (largest lag 32), which is where the DL
    reach is still wider than the ML one -- exact on the other eight.

    `_lags_for` always contains the [1,2,4,8,16,32] ladder, so this never returns less than 32
    and needs no frequency fallback.
    """
    return max(max(_lags_for(case)), 2 * h)


class Forecaster(ABC):
    name: str = "Forecaster"
    requires_gpu: bool = False
    is_probabilistic: bool = False
    # Methods that consume covariates set this True; the eval loop then passes the
    # horizon's exogenous values (a perfect-forecast proxy) to predict via future_exog.
    uses_exogenous: bool = False
    # Coarse grouping (baseline/classical/ml/dl/fm/ensemble) used
    # by the results viewer to filter/group methods. "unclassified" flags a method that
    # forgot to set it, rather than silently disappearing from a family filter.
    family: str = "unclassified"
    # Capability flag: this method keeps a trained model apart from the series it scores,
    # the way a strictly local statistical method (whose parameters are a direct function
    # of the exact series passed to fit) does not. Two consequences follow:
    #   1. It can exploit more of a series' history than the shared curated (tail-segment)
    #      span. The results viewer scores `is_global` methods a second time on the
    #      extended-history fit (see `pools_segments`) in a separate matrix, alongside the
    #      shared local one every method is scored on; the gap is the gain from the history.
    #   2. The rolling-origin eval fits it once, on data strictly before the test period,
    #      then scores every origin by inference alone -- its predict conditions on the
    #      observed history up to that origin without re-estimating parameters. A non-global
    #      method is refit at each origin instead, because for it the fit IS the forecast
    #      and there is no reusable model to carry forward. Every `is_global` method must
    #      therefore make predict depend only on its trained model plus the `history` it is
    #      handed, never on per-origin state captured during fit.
    #      A composite (see `methods/ensemble.py`) satisfies that by fitting the members
    #      which keep a trained model once, and re-deriving the members whose state is a
    #      closed-form statistic of the history from the history predict was handed. That is
    #      conditioning, not estimation, in the same sense as the per-origin seasonal
    #      decomposition `methods/classical.py` re-runs inside `forward`.
    is_global: bool = False
    # Whether `fit` actually trains on the full multi-segment extended history (as
    # opposed to only conditioning on the tail segment). True for the methods that learn
    # parameters from training windows/rows (ml/dl/neuralforecast); False for
    # zero-shot foundation models, which condition on the recent context and cannot
    # bridge a gap, so extended history leaves their forecast unchanged. The eval loop
    # runs the second (extended) fit only for `is_global and pools_segments` methods.
    pools_segments: bool = False
    # Whether this method's forecast changes when `seed` changes -- weight init, minibatch
    # sampling, dropout, row subsampling. A cell for such a method is one draw from a
    # distribution, so the eval loop runs it several times and reports the median-MSE run
    # rather than whichever draw seed 0 happened to give (issue #106). False is the honest
    # default: most methods here estimate a direct function of their input and would return
    # identical numbers on every repeat, so repeating them is pure cost. Note that "seeded at
    # a fixed value" is not the same as deterministic -- CatBoost's default bootstrap
    # resamples rows and only looked deterministic because its library default seed is 0.
    is_stochastic: bool = False
    # The rest are documentation only, read by the results viewer's method-info
    # panel (tsforecastbench.webdata) — they have no effect on fit/predict. Kept as class attributes
    # (not a separate metadata file) so a method's implementation and its description of
    # itself can't drift apart.
    library: str = ""  # concrete package(s) doing the work, e.g. "statsforecast (Nixtla)"
    compute_profile: str = ""  # rough CPU/GPU + speed expectations, one sentence
    limitations: str = ""  # plain-language failure modes / assumptions
    code_snippet: str = ""  # short, truthful excerpt of the real fit/predict calls
    # The card's at-a-glance blocks (see bench/methods/scorecard.py for the rubric).
    # Annotated but deliberately unassigned: there is no sensible generic answer for any of
    # the three, so every concrete method states its own and forgetting one is an error, not
    # a default.
    capabilities: Capabilities
    scorecard: Scorecard
    forecast_strategy: ForecastStrategy
    # Gradient steps the last fit actually took, for the methods that train by gradient
    # descent and stop on convergence rather than at a fixed count. The eval loop reads it
    # into the results so a cell's training effort is visible in the data instead of only in
    # the code. Stays None for every method that does not train this way -- that is an honest
    # "not applicable", not a missing measurement.
    last_fit_steps: int | None = None
    # The step whose weights the last fit kept: its best validation check. Early stopping
    # ends a fit `DL_PATIENCE_CHECKS` checks after this, so `last_fit_steps` alone cannot
    # say how much training the scored model had. None wherever `last_fit_steps` is.
    last_best_step: int | None = None
    # The seed the next fit should use, set by the eval loop before each call to `fit` (the
    # same write-then-read arrangement as `last_fit_steps` above). Every `is_stochastic`
    # method must route all of its randomness through this -- a hard-coded literal would
    # silently give the same run back five times. Ignored by the rest.
    seed: int = 0

    @abstractmethod
    def fit(self, case: Case, train: pd.DataFrame, val: pd.DataFrame) -> None:
        """Estimate parameters on `train`. May store any state on self.

        `train` is the case's declared train split and nothing else. `val` is the declared
        validation split, for deciding when to stop training and for nothing else: a method
        that trains iteratively reads it to detect that its held-out loss has stopped
        improving, and every other method ignores it. **No method estimates parameters on
        `val`** — that is what makes the split the eval loop displays the split it runs
        (gh #99). `test` is scored and never reaches `fit` at all.

        Not seeing `val` at fit time is not the same as never reading it: an `is_global`
        method's `predict` conditions on the observed history running through val and into
        the test period up to each origin. Conditioning is not estimation.
        """

    @abstractmethod
    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Forecast `horizon` steps ahead. Returns columns: ds (UTC), y_hat (float64).

        Whether `history` is read depends on `is_global`. Global methods condition their
        forecast on it, so one fit serves every origin. Non-global methods read only its
        last timestamp (to place the forecast index) and forecast from state captured at
        `fit` — to move the origin, refit; passing fresh history to predict alone would
        silently forecast from the old origin.

        A global method may also re-derive deterministic state from `history` here —
        including by calling a local member's own `fit` on it, which is what a composite
        does for the members whose parameters are a closed-form statistic of what they are
        shown. What the `is_global` contract forbids is carrying origin information out of
        the method's *own* fit, not re-reading the history predict was handed.

        `future_exog`, when given, holds the exogenous columns for the `horizon` future
        timestamps (ds + each declared covariate). Univariate methods ignore it;
        exogenous-aware methods use it as a perfect-forecast proxy for the covariates.
        """

    @staticmethod
    def _future_index(case: Case, last_ds, horizon: int) -> pd.DatetimeIndex:
        last = pd.Timestamp(last_ds)
        return pd.date_range(start=last, periods=horizon + 1, freq=case.freq, tz="UTC")[1:]
