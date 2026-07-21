"""Export the results-matrix data as static JSON for the React/GitHub Pages app.

`web/` has no backend — it reads plain JSON fetched from `web/public/data/`. This script
produces that JSON by importing `tsforecastbench.webdata` directly, so nothing here
recomputes statistics.

Usage (after a pilot run refreshes docs/results/scores.parquet):
    uv run python scripts/export_web_data.py --out web/public/data

Regenerate `scores.json` whenever the pilot re-runs. `methods.json` only changes when a
`Forecaster` subclass's attributes/docstring change (rarer). Review the JSON diff before
committing — unlike the source parquet, it's git-diffable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
# Allow running from a clone without installing.
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.bench.viz import METRIC_CONFIG  # noqa: E402
from tsforecastbench.data.registry import discover_cases, get_case  # noqa: E402
from tsforecastbench.webdata import (  # noqa: E402
    TRAINING_TAIL_N,
    _actual_series,
    _horizon_steps,
    _kde_curve,
    _method_registry,
    curate_origins,
    load_case_split,
    load_case_split_counts,
    load_case_split_values,
    load_case_training_tail,
    load_metrics,
)

SNAPSHOTS_PATH = REPO_ROOT / "results" / "pilot_snapshots.parquet"


def export_scores(out_dir: Path) -> None:
    """scores.json: docs/results/scores.parquet's rows with `family` already joined in
    (see `data.load_metrics`), so the browser never redoes that join."""
    df = load_metrics(0)
    out_dir.joinpath("scores.json").write_text(df.to_json(orient="records"), encoding="utf-8")


def export_methods(out_dir: Path) -> None:
    """methods.json: the method registry (family, description, uses_exogenous, library,
    compute_profile, limitations, code_snippet per method) — see `data._method_registry`."""
    registry = _method_registry()
    out_dir.joinpath("methods.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _finite_list(values) -> list[float | None]:
    """A numpy/pandas float array as a JSON-safe list — NaN/inf become None rather than
    the bare NaN/Infinity tokens Python's json module would otherwise emit (invalid JSON;
    browsers' JSON.parse rejects them)."""
    return [v if np.isfinite(v) else None for v in np.asarray(values, dtype=float).tolist()]


def _series_to_json(df: pd.DataFrame) -> dict:
    """A (ds, y) frame as parallel JSON-safe arrays, for the training-tail/test-split
    quick-look plots."""
    return {
        "ds": df["ds"].astype(str).tolist(),
        "y": _finite_list(df["y"]),
    }


def _distribution(case_id: str, extended: bool) -> dict:
    """(x_grid, train KDE, test KDE) for the train-vs-test distribution plot. x_grid is
    recomputed from whichever train variant (local/extended) is in play, so the
    extended-history distribution isn't just a re-slice of the local one when the case's
    value range actually differs."""
    train = load_case_split_values(case_id, "train", extended=extended)
    test = load_case_split_values(case_id, "test")
    combined = pd.concat([train, test])
    if combined.empty:
        return {"x_grid": [], "train_density": None, "test_density": None}
    x_grid = np.linspace(combined.min(), combined.max(), 200)
    train_density = _kde_curve(train, x_grid)
    test_density = _kde_curve(test, x_grid)
    return {
        "x_grid": _finite_list(x_grid),
        "train_density": _finite_list(train_density) if train_density is not None else None,
        "test_density": _finite_list(test_density) if test_density is not None else None,
    }


def export_cases(out_dir: Path) -> None:
    """cases.json: static case.toml metadata needed by the sidebar/case panels — domain,
    target, exogenous inputs, source, notes, which domain-restricted metrics
    (`cases_eligible_for_metric`) this case declares — plus bounded derived series for the
    case detail panel's plots (never the case's raw, possibly-huge history): the last
    `TRAINING_TAIL_N` training points, the full test split, train/val/test observed-point
    counts, and train-vs-test KDE distribution curves. Each of these has a plain and an
    `_extended` variant (extended-history train split); for cases with no real extended
    history, `data.load_case_split`'s own fallback makes the `_extended` variant identical
    to the plain one, so this doesn't special-case that here — same as the Dash app."""
    cases = {}
    for case_id, case in sorted(discover_cases().items()):
        test_split = load_case_split(case_id, "test")[["ds", "y"]]
        cases[case_id] = {
            "domain": case.domain,
            "target": case.target,
            "target_unit": case.target_unit,
            "freq": case.freq,
            "horizon_primary": case.horizon_primary,
            "exogenous": case.exogenous,
            "exogenous_known_future": case.exogenous_known_future,
            "necessary_exogenous": case.necessary_exogenous,
            "exo_tier": case.exo_tier,
            "domain_metrics": case.domain_metrics,
            "train_tail": _series_to_json(load_case_training_tail(case_id, TRAINING_TAIL_N)),
            "train_tail_extended": _series_to_json(
                load_case_training_tail(case_id, TRAINING_TAIL_N, extended=True)
            ),
            "test_split": _series_to_json(test_split),
            "split_counts": load_case_split_counts(case_id),
            "split_counts_extended": load_case_split_counts(case_id, extended=True),
            "distribution": _distribution(case_id, extended=False),
            "distribution_extended": _distribution(case_id, extended=True),
            "source": {"name": case.source.name, "url": case.source.url},
            "notes": case.notes,
        }
    out_dir.joinpath("cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")


def _round_sig(value: float, sig: int = 4) -> float:
    """Round to `sig` significant figures — unlike a fixed decimal-place round, this stays
    meaningful across cases whose values range from ~0.001 to ~100,000."""
    if value == 0 or not np.isfinite(value):
        return value
    return round(value, -int(np.floor(np.log10(abs(value)))) + (sig - 1))


def _round_and_finite_list(values, sig: int = 4) -> list[float | None]:
    return [
        None if not np.isfinite(v) else _round_sig(float(v), sig)
        for v in np.asarray(values, dtype=float)
    ]


def _predictions_by_origin(preds: pd.DataFrame) -> dict[str, list[float | None]]:
    """origin_id (as a string, since JSON object keys are always strings) -> that
    origin's y_hat values, ordered by ds — the horizon-length forecast the snapshot grid
    overlays on the actual-series window."""
    out = {}
    for origin_id, group in preds.groupby("origin_id"):
        out[str(int(origin_id))] = _round_and_finite_list(
            group.sort_values("ds")["y_hat"].to_numpy()
        )
    return out


def _curation_for(case_preds: pd.DataFrame, method: str, case_id: str, metric: str) -> list[dict]:
    curated = curate_origins(case_preds, method, case_id, metric)
    return [
        {"origin_id": int(o.origin_id), "value": _round_sig(float(o.value)), "label": o.label}
        for o in curated.itertuples(index=False)
    ]


def export_snapshots(out_dir: Path) -> None:
    """snapshots/<case_id>.json (one per case): bounded per-origin forecast data for the
    snapshot grid — every method's prediction at every one of the case's ~40 rolling
    origins is reachable under some anchor-method/metric combination, so this ships all of
    them rather than a curated subset, but eliminates the real waste in the raw parquet:
    the actual series is stored once per origin (not once per method-row), timestamps are
    dropped in favour of reconstructing them from `origin_ds` + `freq` client-side, and
    every value is rounded to 4 significant figures (matching the matrix's own display
    precision). `curation` is precomputed per (metric, method, local/extended variant) via
    `snapshot.curate_origins` so the client never re-derives MASE/skill-score, which needs
    a seasonal-naive train+val denominator.

    Requires `results/pilot_snapshots.parquet` (gitignored, produced by a rolling pilot
    run — see scripts/run_pilot.py); skipped with a warning if absent, same as the Dash
    app's own graceful fallback (`data.load_snapshots`) for a clone that hasn't run one."""
    if not SNAPSHOTS_PATH.exists():
        print(
            f"[export_web_data] {SNAPSHOTS_PATH} not found — skipping snapshot export "
            "(run a rolling pilot to produce it, e.g. `uv run python scripts/run_pilot.py`)"
        )
        return

    preds = pd.read_parquet(SNAPSHOTS_PATH)
    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    metrics = list(METRIC_CONFIG)

    for case_id in sorted(preds["case_id"].unique()):
        case = get_case(case_id)
        case_preds = preds[preds["case_id"] == case_id]
        horizon_steps = _horizon_steps(case)
        n_history = max(horizon_steps // 4, 1)
        step = pd.Timedelta(case.freq)
        actual_full = _actual_series(case_id)

        origins_df = (
            case_preds[["origin_id", "origin_ds"]].drop_duplicates().sort_values("origin_id")
        )
        origins = []
        actual = {}
        for row in origins_df.itertuples(index=False):
            origin_ds = pd.Timestamp(row.origin_ds)
            window_index = pd.date_range(
                origin_ds - n_history * step, periods=n_history + horizon_steps, freq=step
            )
            window = actual_full.reindex(window_index)
            origins.append({"origin_id": int(row.origin_id), "origin_ds": origin_ds.isoformat()})
            actual[str(int(row.origin_id))] = _round_and_finite_list(window.to_numpy())

        has_extended = case_preds.groupby("method")["used_extended_history"].any()
        predictions = {}
        variants_by_method: dict[str, list[tuple[str, bool]]] = {}
        for method in sorted(case_preds["method"].unique()):
            variants: list[tuple[str, bool]] = [("local", False)]
            if has_extended.get(method, False):
                variants.append(("extended", True))
            variants_by_method[method] = variants
            predictions[method] = {
                variant: _predictions_by_origin(
                    case_preds[
                        (case_preds["method"] == method)
                        & (case_preds["used_extended_history"] == is_ext)
                    ]
                )
                for variant, is_ext in variants
            }

        curation = {}
        for metric in metrics:
            curation[metric] = {}
            for method, variants in variants_by_method.items():
                curation[metric][method] = {
                    variant: _curation_for(
                        case_preds[
                            (case_preds["method"] == method)
                            & (case_preds["used_extended_history"] == is_ext)
                        ],
                        method,
                        case_id,
                        metric,
                    )
                    for variant, is_ext in variants
                }

        bundle = {
            "case_id": case_id,
            "n_history": n_history,
            "horizon_steps": horizon_steps,
            "origins": origins,
            "actual": actual,
            "predictions": predictions,
            "curation": curation,
        }
        snap_dir.joinpath(f"{case_id}.json").write_text(json.dumps(bundle), encoding="utf-8")
    print(f"[export_web_data] wrote {len(list(snap_dir.glob('*.json')))} snapshot bundle(s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="web/public/data", help="output directory (default: web/public/data)"
    )
    args = parser.parse_args(argv)

    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export_web_data] writing to {out_dir}")
    export_scores(out_dir)
    export_methods(out_dir)
    export_cases(out_dir)
    print("[export_web_data] wrote scores.json, methods.json, cases.json")
    export_snapshots(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
