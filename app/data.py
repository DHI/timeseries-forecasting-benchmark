"""Data loading for the results-matrix app.

Reads `docs/results/scores.parquet` — the distilled metrics committed to the repo, so a
fresh clone can open the app with real data before ever running the pilot. Per-origin
snapshots are the raw, regenerable artifact and stay gitignored under `results/`; a clone
that hasn't run the pilot just gets an empty snapshot frame (see `load_snapshots`).
Method family is not a column in the scores file (adding it would require rerunning the
whole pilot); instead it's read straight off the `Forecaster` classes in
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
from tsforecastbench.data.schema import DOMAIN_RESTRICTED_METRICS

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
METRICS_PATH = REPO_ROOT / "docs" / "results" / "scores.parquet"
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
FAMILY_ORDER = ["baseline", "ensemble", "classical", "ml", "dl", "fm"]

# EnsembleForecaster's `name` is set per-instance by these factories, not as a class
# attribute, so it can't be read off the class the way every other method's can.
_ENSEMBLE_FACTORIES = ("ensemble_baselines", "ensemble_baselines_theta")


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
    limitations, code_snippet}, derived from the Forecaster classes — the same
    introspection the old family-only map used, extended to cover what the method-info
    panel needs. `description` is the class's own docstring (first paragraph only), via
    `_own_doc` so a subclass without one (e.g. LightGBM, which only adds a
    `_make_model`) inherits its base's; methods with no docstring anywhere in their
    Forecaster ancestry get a placeholder instead. The other four fields are ordinary
    class attributes (see `Forecaster` in bench/methods/base.py), so plain attribute
    lookup already inherits them through the MRO with no special-casing needed."""
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


def _rank(order: list[str]):
    return lambda value: order.index(value) if value in order else len(order)


@cache
def load_metrics(version: int = 0) -> pd.DataFrame:
    """Load docs/results/scores.parquet and join each method's family.

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
    grouping with separators. Excludes "+exo" variants: they're folded into their base
    method's column in the matrix (see `render_matrix`'s exogenous-toggle swap), not shown
    as separate columns."""
    pairs = df[["family", "method"]].drop_duplicates()
    pairs = pairs[~pairs["method"].str.endswith("+exo")]
    pairs["_family_rank"] = pairs["family"].map(_rank(FAMILY_ORDER))
    pairs = pairs.sort_values(["_family_rank", "method"])
    return list(pairs["method"])


def cases_eligible_for_metric(metric: str, case_ids: list[str]) -> list[str]:
    """`case_ids` restricted to those declaring `metric` in their case.toml
    `domain_metrics` — a no-op for every metric outside `DOMAIN_RESTRICTED_METRICS`
    (MASE, MAPE, etc. apply to every case). Keeps domain-restricted metrics like NSE/KGE
    selectable in the same dropdown as everything else, while their matrix rows stay
    limited to cases that actually declare eligibility."""
    if metric not in DOMAIN_RESTRICTED_METRICS:
        return case_ids
    return [c for c in case_ids if metric in get_case(c).domain_metrics]


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


def method_training_seconds(
    df: pd.DataFrame, case_id: str, method: str, extended: bool = False
) -> float | None:
    """The persisted `seconds` timing for this case x method run — fit plus predict,
    pooled across every rolling origin (see `bench.eval.rolling.run_case_rolling`).
    Identical across every metric row for the same case/method/fit-variant, so any row
    will do. Falls back to the local (non-extended) fit's timing when `extended` is
    requested but this method never produced an extended-history variant (only
    global, segment-pooling methods do — see `case_has_extended_history`)."""
    mask = (df["case_id"] == case_id) & (df["method"] == method)
    rows = df[mask & (df["used_extended_history"] == extended)]
    if rows.empty and extended:
        rows = df[mask & ~df["used_extended_history"]]
    return float(rows["seconds"].iloc[0]) if not rows.empty else None


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


def case_training_stats(
    df: pd.DataFrame, case_id: str, methods: list[str], extended: bool = False
) -> str:
    """Markdown lines for the case-summary panel, below the split-proportions bar: the
    actual observed sample counts behind that bar's percentages (see
    `load_case_split_counts`), and the measured training time (fit + predict, pooled
    across rolling origins — see `method_training_seconds`) for each currently-selected
    method."""
    counts = load_case_split_counts(case_id, extended=extended)
    total = sum(counts.values())
    samples_line = (
        f"**Samples used**: train {counts['train']:,} · val {counts['val']:,} · "
        f"test {counts['test']:,} (total {total:,})"
    )
    if not methods:
        return samples_line

    times = []
    for method in methods:
        seconds = method_training_seconds(df, case_id, method, extended=extended)
        times.append(f"{method} {seconds:.2f}s" if seconds is not None else f"{method} —")
    time_line = "**Training time** (fit + predict, pooled across rolling origins): " + " · ".join(
        times
    )
    return f"{samples_line}\n\n{time_line}"


def method_info(case_id: str, method: str) -> str:
    """Markdown blurb for the bottom method-info panel: family, a brief description
    drawn from the Forecaster subclass's docstring, and which of the case's exogenous
    inputs (if any) this method actually consumes. Still a placeholder — the set of
    fields shown here is expected to grow."""
    case = get_case(case_id)
    meta = _method_registry().get(method, {})

    exo_line = "none"
    if meta.get("uses_exogenous"):
        exo_cols = list(dict.fromkeys([*case.exogenous, *case.exogenous_known_future]))
        if exo_cols:
            necessary = set(case.necessary_exogenous)
            exo_line = ", ".join(f"{c} (necessary)" if c in necessary else c for c in exo_cols)

    return "\n\n".join(
        [
            f"**Family**: {meta.get('family', 'unclassified').replace('_', ' ')}",
            meta.get("description", "No description yet."),
            f"**Exogenous inputs used**: {exo_line}",
        ]
    )


def method_info_detail(case_id: str, method: str) -> str:
    """Markdown for the method-info panel's collapsed "Implementation detail"
    section: which library actually does the work, a rough compute-cost profile,
    known limitations/assumptions, and a short excerpt of the real fit/predict
    calls. Blank fields (a method with nothing set yet) are omitted rather than
    shown as an empty heading."""
    del case_id
    meta = _method_registry().get(method, {})

    lines = []
    if meta.get("library"):
        lines.append(f"**Library**: {meta['library']}")
    if meta.get("compute_profile"):
        lines.append(f"**Compute profile**: {meta['compute_profile']}")
    if meta.get("limitations"):
        lines.append(f"**Limitations**: {meta['limitations']}")
    if meta.get("code_snippet"):
        lines.append(f"```python\n{meta['code_snippet']}\n```")
    return "\n\n".join(lines) if lines else "No implementation detail yet."
