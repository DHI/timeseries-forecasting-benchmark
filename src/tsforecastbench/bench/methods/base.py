"""Abstract Forecaster interface — every method implements fit and predict."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

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


def lookback_for(freq: str, h: int) -> int:
    """Default lookback window length for a windowed DL/ML forecaster.

    Frequency-tuned base, floored at 2*h so the window always covers at least two
    horizons even at large h (freqs not listed default to 4*h).
    """
    f = freq.lower()
    base = {"1d": 64, "1h": 168, "15min": 192, "1w": 52}.get(f, 4 * h)
    return max(base, 2 * h)


class Forecaster(ABC):
    name: str = "Forecaster"
    requires_gpu: bool = False
    is_probabilistic: bool = False
    # Methods that consume covariates set this True; the eval loop then passes the
    # horizon's exogenous values (a perfect-forecast proxy) to predict via future_exog.
    uses_exogenous: bool = False
    # Coarse grouping (baseline/classical/ml/deep_learning/foundation_model/ensemble) used
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
    is_global: bool = False
    # Whether `fit` actually trains on the full multi-segment extended history (as
    # opposed to only conditioning on the tail segment). True for the methods that learn
    # parameters from training windows/rows (ml/deep_learning/neuralforecast); False for
    # zero-shot foundation models, which condition on the recent context and cannot
    # bridge a gap, so extended history leaves their forecast unchanged. The eval loop
    # runs the second (extended) fit only for `is_global and pools_segments` methods.
    pools_segments: bool = False
    # The next four are documentation only, read by the results viewer's method-info
    # panel (app/data.py) — they have no effect on fit/predict. Kept as class attributes
    # (not a separate metadata file) so a method's implementation and its description of
    # itself can't drift apart.
    library: str = ""  # concrete package(s) doing the work, e.g. "statsforecast (Nixtla)"
    compute_profile: str = ""  # rough CPU/GPU + speed expectations, one sentence
    limitations: str = ""  # plain-language failure modes / assumptions
    code_snippet: str = ""  # short, truthful excerpt of the real fit/predict calls

    @abstractmethod
    def fit(self, case: Case, train: pd.DataFrame) -> None:
        """Fit on the train split. May store any state on self."""

    @abstractmethod
    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Forecast `horizon` steps ahead. Returns columns: ds (UTC), y_hat (float64).

        `future_exog`, when given, holds the exogenous columns for the `horizon` future
        timestamps (ds + each declared covariate). Univariate methods ignore it;
        exogenous-aware methods use it as a perfect-forecast proxy for the covariates.
        """

    @staticmethod
    def _future_index(case: Case, last_ds, horizon: int) -> pd.DatetimeIndex:
        last = pd.Timestamp(last_ds)
        return pd.date_range(start=last, periods=horizon + 1, freq=case.freq, tz="UTC")[1:]
