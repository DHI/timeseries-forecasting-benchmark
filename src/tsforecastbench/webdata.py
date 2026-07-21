"""Data loading and curation behind the static JSON export for the React web app.

`scripts/export_web_data.py` imports these functions to produce the JSON `web/` reads —
`web/` has no backend. Reads `docs/results/scores.parquet` (the distilled metrics
committed to the repo) and, for per-origin snapshots, `results/pilot_snapshots.parquet`
(gitignored, produced by a rolling pilot run). Method family is not a column in the
scores file — it's read straight off the `Forecaster` classes in
`tsforecastbench.bench.methods`, the actual source of truth, and joined in memory at
load time.
"""

from __future__ import annotations

import inspect
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from tsforecastbench.bench import methods as _methods
from tsforecastbench.bench.eval.metrics import kge as _kge
from tsforecastbench.bench.eval.metrics import mae as _mae_score
from tsforecastbench.bench.eval.metrics import mape as _mape
from tsforecastbench.bench.eval.metrics import mase as _mase_score
from tsforecastbench.bench.eval.metrics import nse as _nse
from tsforecastbench.bench.eval.metrics import rmse as _rmse_score
from tsforecastbench.bench.eval.metrics import skill_score as _skill_score
from tsforecastbench.bench.eval.metrics import smape as _smape
from tsforecastbench.data.registry import get_case
from tsforecastbench.data.seasonality import fill_and_detrend, mase_season_for

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = REPO_ROOT / "docs" / "results" / "scores.parquet"

TRAINING_TAIL_N = 1000

# Seasonal-pattern plot tuning. PAD_FRAC: fraction of the cycle mirrored across each
# boundary before the LOESS fit, so the curve is continuous where the cycle wraps (e.g.
# Sunday night into Monday morning) instead of showing an edge artifact. LOESS_FRAC: the
# lowess bandwidth: adjust if rendered curves look under/over-smoothed once reviewed.
SEASONAL_PATTERN_PAD_FRAC = 0.1
SEASONAL_PATTERN_LOESS_FRAC = 0.15
SEASONAL_PATTERN_CURVE_POINTS = 200

# EnsembleForecaster's `name` is set per-instance by these factories, not as a class
# attribute, so it can't be read off the class the way every other method's can.
_ENSEMBLE_FACTORIES = ("ensemble_baselines", "ensemble_baselines_theta")

METRIC_LOWER_IS_BETTER = {
    "MAE": True,
    "RMSE": True,
    "MASE": True,
    "MAPE": True,
    "sMAPE": True,
    "Skill_vs_Persistence": False,
    "Skill_vs_Mean": False,
    "NSE": False,
    "KGE": False,
}


def _own_doc(cls: type) -> str | None:
    """First docstring found on `cls` or one of its Forecaster ancestors, stopping at
    `Forecaster` itself so a class with no docstring of its own doesn't fall through to
    `ABC`'s generic one (`inspect.getdoc` would otherwise walk that far)."""
    for klass in cls.__mro__:
        if klass is _methods.Forecaster:
            break
        doc = klass.__dict__.get("__doc__")
        if doc:
            return doc
    return None


@cache
def _method_registry() -> dict[str, dict]:
    """method name -> {family, description, uses_exogenous, library, compute_profile,
    limitations, code_snippet}, derived from the Forecaster classes. `description` is
    the class's own docstring (first paragraph only), via `_own_doc` so a subclass
    without one (e.g. LightGBM, which only adds a `_make_model`) inherits its base's;
    methods with no docstring anywhere in their Forecaster ancestry get a placeholder
    instead. The other four fields are ordinary class attributes (see `Forecaster` in
    bench/methods/base.py), so plain attribute lookup already inherits them through the
    MRO with no special-casing needed."""
    registry: dict[str, dict] = {}
    for attr_name in _methods.__all__:
        obj = getattr(_methods, attr_name)
        if attr_name in _ENSEMBLE_FACTORIES:
            instance = obj()
            cls = type(instance)
        elif (
            inspect.isclass(obj)
            and issubclass(obj, _methods.Forecaster)
            and obj not in (_methods.Forecaster, _methods.EnsembleForecaster)
        ):
            instance = obj()
            cls = obj
        else:
            continue
        doc = _own_doc(cls)
        registry[instance.name] = {
            "family": instance.family,
            "description": doc.split("\n\n")[0].strip() if doc else "No description yet.",
            "uses_exogenous": instance.uses_exogenous,
            "library": instance.library,
            "compute_profile": instance.compute_profile,
            "limitations": instance.limitations,
            "code_snippet": instance.code_snippet,
        }
    return registry


def _method_family_map() -> dict[str, str]:
    """method name -> family, derived from the Forecaster classes."""
    return {name: meta["family"] for name, meta in _method_registry().items()}


@cache
def load_metrics(version: int = 0) -> pd.DataFrame:
    """Load docs/results/scores.parquet and join each method's family.

    `version` has no effect on the result; bump it to land on a fresh cache key and
    re-read a pilot run that finished after the process started. Callers must treat the
    returned frame as read-only (it's a shared cached object).
    """
    del version
    df = pd.read_parquet(METRICS_PATH)
    family_map = _method_family_map()
    df["family"] = df["method"].map(family_map).fillna("unclassified")
    return df


@cache
def load_case_split(case_id: str, split: str, extended: bool = False) -> pd.DataFrame:
    """One split ("train"/"val"/"test") of a case's curated series.

    `extended=True` (meaningful only for `split="train"`) swaps in the case's full
    pre-curated history via `Case.load_extended`, cut at the curated train/val boundary
    so it never reaches into val or test; falls back to the plain curated train split
    if extended data can't be loaded for this case. Cached: like the case's curated data
    itself, this is static for the process lifetime.
    """
    case = get_case(case_id)
    curated = case.load(curated=True)
    splits = case.split_frame(curated)
    if not (extended and split == "train"):
        return splits[split]

    train_end, _, _ = case.split_bounds(curated)
    try:
        ext = case.load_extended()
    except Exception:
        return splits["train"]
    ds = pd.to_datetime(ext["ds"])
    if ds.dt.tz is not None:
        ds = ds.dt.tz_convert("UTC").dt.tz_localize(None)
    return ext.loc[ds <= train_end].reset_index(drop=True)


def _observed_mask(df: pd.DataFrame) -> pd.Series:
    return df["y_observed"] if "y_observed" in df.columns else df["y"].notna()


def load_case_split_values(case_id: str, split: str, extended: bool = False) -> pd.Series:
    """Observed y-values for one split."""
    frame = load_case_split(case_id, split, extended=extended)
    return frame.loc[_observed_mask(frame), "y"]


def load_case_split_counts(case_id: str, extended: bool = False) -> dict[str, int]:
    """Observed-point counts for train/val/test.

    `extended` only swaps in the extended-history train count; val/test always reflect
    the curated split, since extended history never reaches past the train/val boundary.
    """
    return {
        split: len(load_case_split_values(case_id, split, extended=extended and split == "train"))
        for split in ("train", "val", "test")
    }


@cache
def load_case_training_tail(case_id: str, n: int = 1000, extended: bool = False) -> pd.DataFrame:
    """Last `n` points of the case's training split."""
    train = load_case_split(case_id, "train", extended=extended)
    return train.tail(n)[["ds", "y"]].reset_index(drop=True)


@cache
def _detrended_observed_tail(
    case_id: str, extended: bool = False
) -> tuple[pd.DatetimeIndex, np.ndarray] | None:
    """Observed (ds, y) pairs from the training tail with the mean and linear trend
    removed -- shared across every declared period's seasonal-pattern subplot, since
    detrending doesn't depend on which period is being viewed. Two cycles with the same
    shape but offset by drift land on the same baseline this way. Reuses the same
    `fill_and_detrend` convention `dominant_period` uses to pick a case's dominant season.
    None if there isn't enough data to detrend (fewer than 3 finite training points).
    """
    tail = load_case_training_tail(case_id, TRAINING_TAIL_N, extended=extended)
    raw_y = tail["y"].to_numpy(dtype=float)
    observed = np.isfinite(raw_y)
    try:
        detrended = fill_and_detrend(raw_y)
    except ValueError:
        return None
    return pd.DatetimeIndex(tail.loc[observed, "ds"]), detrended[observed]


@cache
def _actual_series(case_id: str) -> pd.Series:
    """The case's real curated target, indexed by timestamp -- gap-filled, like the
    `y` column the eval pipeline fits models on and builds persistence/mean reference
    forecasts from (see `bench.eval.rolling.run_case_rolling`). Cached: curated data is
    static, and the same series is reused across every origin lookup and re-click."""
    case = get_case(case_id)
    df = case.load(curated=True)
    return pd.Series(df["y"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["ds"]))


@cache
def _observed_actual_series(case_id: str) -> pd.Series:
    """Like `_actual_series`, but with filled (non-observed) points masked back to NaN
    -- mirrors `bench.eval.loop._raw_y`, so only genuinely-observed points count towards
    a per-origin metric, matching how the matrix itself scores."""
    case = get_case(case_id)
    df = case.load(curated=True)
    mask = df["y_observed"] if "y_observed" in df.columns else df["y"].notna()
    y = np.where(mask.to_numpy(), df["y"].to_numpy(dtype=float), np.nan)
    return pd.Series(y, index=pd.DatetimeIndex(df["ds"]))


@cache
def _mase_context(case_id: str) -> tuple[np.ndarray, int]:
    """(train+val `y`, seasonal period) for this case's MASE denominator -- mirrors the
    once-per-case `naive_scale` in `bench.eval.rolling.run_case_rolling`, so a per-origin
    MASE uses the same fixed scale as the matrix's pooled MASE rather than a per-origin
    one."""
    case = get_case(case_id)
    df = case.load(curated=True)
    splits = case.split_frame(df)
    y_train = pd.concat([splits["train"], splits["val"]])["y"].to_numpy(dtype=float)
    return y_train, mase_season_for(case.seasonalities, len(y_train))


def _horizon_steps(case) -> int:
    return int(pd.Timedelta(case.horizon_primary) // pd.Timedelta(case.freq))


def _kde_curve(values: pd.Series, x_grid: np.ndarray) -> np.ndarray | None:
    """Gaussian-KDE density of `values` over `x_grid`, or None if there isn't enough
    variance to fit one (a near-empty split, or one with a single repeated value)."""
    from scipy.stats import gaussian_kde

    values = values.to_numpy(dtype=float)
    if len(values) < 2 or np.ptp(values) == 0:
        return None
    try:
        return gaussian_kde(values)(x_grid)
    except (np.linalg.LinAlgError, ValueError):
        return None


def _loess_curve(x: np.ndarray, y: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray] | None:
    """LOESS fit of `y` against its position-in-cycle `x` (`0..period-1`), evaluated on a
    fine grid spanning one full cycle -- for the case panel's seasonal-pattern plot.

    Before fitting, `x`/`y` are padded with points mirrored across each cycle boundary
    (`SEASONAL_PATTERN_PAD_FRAC` of the period from each end), so the curve is continuous
    where positions wrap around (e.g. `period - 1` back to `0`, Sunday night into Monday
    morning) instead of tapering at the edges; the padding never appears in the returned
    grid, which spans exactly `0..period-1`. None if there's nothing to fit.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    if len(x) < 2:
        return None
    pad = period * SEASONAL_PATTERN_PAD_FRAC
    wrap_from_end = (x >= period - pad) & (x < period)
    wrap_from_start = (x >= 0) & (x <= pad)
    x_padded = np.concatenate([x[wrap_from_end] - period, x, x[wrap_from_start] + period])
    y_padded = np.concatenate([y[wrap_from_end], y, y[wrap_from_start]])
    curve_x = np.linspace(0, period - 1, SEASONAL_PATTERN_CURVE_POINTS)
    try:
        curve_y = lowess(y_padded, x_padded, frac=SEASONAL_PATTERN_LOESS_FRAC, xvals=curve_x)
    except ValueError:
        return None
    return curve_x, curve_y


def _origin_metric(case_id: str, origin_preds: pd.DataFrame, metric: str) -> float:
    """One origin's score on `metric`, computed the same way the matrix's pooled score
    is (see `bench.eval.rolling.run_case_rolling`), just over a single origin's window
    instead of pooled across all of them. NaN if the window has no observed actual, or
    (MASE only) the case's train+val span can't support a seasonal-naive scale.

    MAE/RMSE/MASE compare against the observed-only actual (`_observed_actual_series`);
    MASE's denominator is the case-level `_mase_context` scale, not a per-origin one.
    The skill scores compare against a reference forecast built from the *filled*
    history strictly before this origin (`_actual_series`) -- persistence repeats its
    last value, mean repeats its average -- mirroring how the pipeline isolates the
    model's skill from a shifted baseline.
    """
    ds = pd.DatetimeIndex(origin_preds["ds"])
    aligned = _observed_actual_series(case_id).reindex(ds)
    mask = aligned.notna().to_numpy()
    if not mask.any():
        return float("nan")
    y = aligned.to_numpy()[mask]
    y_hat = origin_preds["y_hat"].to_numpy(dtype=float)[mask]

    if metric == "RMSE":
        return _rmse_score(y, y_hat)
    if metric == "MAPE":
        return _mape(y, y_hat)
    if metric == "sMAPE":
        return _smape(y, y_hat)
    if metric == "NSE":
        return _nse(y, y_hat)
    if metric == "KGE":
        return _kge(y, y_hat)
    if metric == "MASE":
        y_train, season = _mase_context(case_id)
        try:
            return _mase_score(y, y_hat, y_train, season=season)
        except ValueError:
            return float("nan")
    if metric in ("Skill_vs_Persistence", "Skill_vs_Mean"):
        origin_ds = pd.Timestamp(origin_preds["origin_ds"].iloc[0])
        history = _actual_series(case_id)
        local_hist = history[history.index < origin_ds].to_numpy()
        if len(local_hist) == 0:
            return float("nan")
        reference = local_hist[-1] if metric == "Skill_vs_Persistence" else np.nanmean(local_hist)
        return _skill_score(y, y_hat, np.full(mask.sum(), float(reference)))
    return _mae_score(y, y_hat)  # default / "MAE"


def curate_origins(
    case_preds: pd.DataFrame, anchor_method: str, case_id: str, metric: str = "MAE"
) -> pd.DataFrame:
    """Pick up to 3 representative origins for one case — the best, median, and worst
    performing by `metric` (whichever the matrix currently shows) for `anchor_method`
    (the first-selected/"anchor" method of the overlay) — so the snapshot grid shows a
    meaningful spread of outcomes rather than whatever origins happen to be first in
    storage order. All overlaid methods share this same set of origins so their
    forecasts remain directly comparable.

    Returns a frame with columns (origin_id, origin_ds, value, label), ordered
    Best -> Median -> Worst ("Best" always means most-favorable by `metric`, so lowest
    for an error metric but highest for a skill score -- see `METRIC_LOWER_IS_BETTER`).
    Origins with an undefined score are dropped before ranking; if fewer than 3 remain,
    only the labels that make sense are emitted (e.g. just Best/Worst for 2 origins).
    """
    anchor_preds = case_preds[case_preds["method"] == anchor_method]
    origins = anchor_preds[["origin_id", "origin_ds"]].drop_duplicates().reset_index(drop=True)
    origins["value"] = [
        _origin_metric(case_id, anchor_preds[anchor_preds["origin_id"] == oid], metric)
        for oid in origins["origin_id"]
    ]
    ascending = METRIC_LOWER_IS_BETTER.get(metric, True)
    origins = origins.dropna(subset=["value"]).sort_values("value", ascending=ascending)
    origins = origins.reset_index(drop=True)

    n = len(origins)
    if n == 0:
        return origins.assign(label=pd.Series(dtype=object))
    if n == 1:
        picks, labels = [0], ["Best"]
    elif n == 2:
        picks, labels = [0, 1], ["Best", "Worst"]
    else:
        picks = sorted({0, n // 2, n - 1})
        label_of = {0: "Best", n // 2: "Median", n - 1: "Worst"}
        labels = [label_of[i] for i in picks]

    curated = origins.loc[picks].copy()
    curated["label"] = labels
    return curated
