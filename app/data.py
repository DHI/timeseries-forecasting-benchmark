"""Data loading for the results-matrix app.

Reads `results/pilot_metrics.parquet` directly — no new artifacts, no pipeline changes.
Method family is not a column in that file (adding it would require rerunning the whole
pilot); instead it's read straight off the `Forecaster` classes in
`tsforecastbench.bench.methods`, which is the actual source of truth, and joined in
memory at load time.
"""

from __future__ import annotations

import inspect
from functools import cache
from pathlib import Path

import pandas as pd

from tsforecastbench.bench import methods as _methods
from tsforecastbench.data.registry import get_case

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
METRICS_PATH = RESULTS_DIR / "pilot_metrics.parquet"
SNAPSHOTS_PATH = RESULTS_DIR / "pilot_snapshots.parquet"

# Columns of the snapshot artifact written by the rolling eval (see
# tsforecastbench.bench.eval.rolling.SNAPSHOT_COLUMNS). Kept here so an empty frame has
# the right shape when the artifact is missing (e.g. an old results set, or a --fast run).
SNAPSHOT_COLUMNS = [
    "case_id",
    "method",
    "used_extended_history",
    "origin_id",
    "origin_ds",
    "ds",
    "y_hat",
]

# Display order for method families (left-to-right column grouping) and case domains
# (top-to-bottom row grouping). Anything not listed sorts after, alphabetically.
FAMILY_ORDER = ["baseline", "ensemble", "classical", "ml", "deep_learning", "foundation_model"]

# EnsembleForecaster's `name` is set per-instance by these factories, not as a class
# attribute, so it can't be read off the class the way every other method's can.
_ENSEMBLE_FACTORIES = ("ensemble_baselines", "ensemble_baselines_theta")


def _method_family_map() -> dict[str, str]:
    """method name -> family, derived from the Forecaster classes."""
    mapping: dict[str, str] = {}
    for attr_name in _methods.__all__:
        obj = getattr(_methods, attr_name)
        if attr_name in _ENSEMBLE_FACTORIES:
            instance = obj()
            mapping[instance.name] = instance.family
        elif (
            inspect.isclass(obj)
            and issubclass(obj, _methods.Forecaster)
            and obj not in (_methods.Forecaster, _methods.EnsembleForecaster)
        ):
            mapping[obj.name] = obj.family
    return mapping


def _rank(order: list[str]):
    return lambda value: order.index(value) if value in order else len(order)


@cache
def load_metrics(version: int = 0) -> pd.DataFrame:
    """Load pilot_metrics.parquet and join each method's family.

    `version` has no effect on the result; bump it (e.g. from a reload button) to land
    on a fresh cache key and re-read a pilot run that finished after the app started.
    Callers must treat the returned frame as read-only (it's a shared cached object).
    """
    del version
    df = pd.read_parquet(METRICS_PATH)
    family_map = _method_family_map()
    df["family"] = df["method"].map(family_map).fillna("unclassified")
    return df


@cache
def load_snapshots(version: int = 0) -> pd.DataFrame:
    """Load pilot_snapshots.parquet — the per-origin forecasts the rolling eval persists.

    Returns an empty (correctly-typed) frame when the artifact is absent, so the app still
    runs against a results set produced before snapshots existed (or a --fast run, which
    doesn't emit them). `version` works like `load_metrics`: bump it to land on a fresh
    cache key and re-read after a new run. Treat the result as read-only (shared cache).
    """
    del version
    if not SNAPSHOTS_PATH.exists():
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    return pd.read_parquet(SNAPSHOTS_PATH)


def ordered_cases(df: pd.DataFrame) -> list[str]:
    """Case ids sorted by domain (FAMILY_ORDER-style grouping isn't meaningful for
    domains, so alphabetical), then case id, for stable row grouping with separators."""
    pairs = df[["domain", "case_id"]].drop_duplicates()
    pairs = pairs.sort_values(["domain", "case_id"])
    return list(pairs["case_id"])


def ordered_methods(df: pd.DataFrame) -> list[str]:
    """Method names sorted by family (FAMILY_ORDER), then name, for stable column
    grouping with separators."""
    pairs = df[["family", "method"]].drop_duplicates()
    pairs["_family_rank"] = pairs["family"].map(_rank(FAMILY_ORDER))
    pairs = pairs.sort_values(["_family_rank", "method"])
    return list(pairs["method"])


@cache
def load_case_split(case_id: str, split: str, extended: bool = False) -> pd.DataFrame:
    """One split ("train"/"val"/"test") of a case's curated series, for the case-panel
    quick-look plots.

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
    """Observed y-values for one split, for the case-panel distribution histogram."""
    frame = load_case_split(case_id, split, extended=extended)
    return frame.loc[_observed_mask(frame), "y"]


def load_case_split_counts(case_id: str, extended: bool = False) -> dict[str, int]:
    """Observed-point counts for train/val/test, for the case-panel proportions bar.

    `extended` only swaps in the extended-history train count; val/test always reflect
    the curated split, since extended history never reaches past the train/val boundary.
    """
    return {
        split: len(load_case_split_values(case_id, split, extended=extended and split == "train"))
        for split in ("train", "val", "test")
    }


def case_has_extended_history(df: pd.DataFrame, case_id: str) -> bool:
    """Whether the pilot run recorded an extended-history fit for this case — i.e. some
    `is_global` method found real history beyond the curated span (see
    `bench.eval.rolling.run_case_rolling`)."""
    return bool(df.loc[df["case_id"] == case_id, "used_extended_history"].any())


@cache
def load_case_training_tail(case_id: str, n: int = 1000, extended: bool = False) -> pd.DataFrame:
    """Last `n` points of the case's training split, for a quick-look plot."""
    train = load_case_split(case_id, "train", extended=extended)
    return train.tail(n)[["ds", "y"]].reset_index(drop=True)


def case_summary(case_id: str) -> str:
    """Markdown summary of a case's static metadata: domain, target, horizon, exogenous
    inputs, and data source — the config in `case.toml`, not anything measured."""
    case = get_case(case_id)
    exo = f"{len(case.exogenous)} ({case.exo_tier})" if case.exogenous else "none"
    source = case.source.name
    if case.source.url:
        source = f"[{source}]({case.source.url})"
    lines = [
        f"**Domain**: {case.domain.replace('_', ' ')}",
        f"**Target**: `{case.target}`" + (f" ({case.target_unit})" if case.target_unit else ""),
        f"**Frequency**: `{case.freq}`  ·  **Horizon**: `{case.horizon_primary}`",
        f"**Exogenous inputs**: {exo}",
        f"**Source**: {source}",
    ]
    if case.notes:
        lines += ["", case.notes]
    return "\n\n".join(lines)
