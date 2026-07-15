"""Export the results-matrix app's data as static JSON for the React/GitHub Pages app.

`web/` has no backend — it reads plain JSON fetched from `web/public/data/`. This script
produces that JSON by importing `app/data.py` directly (the same module `app/app.py`
uses), so the two apps share one source of truth for what the numbers mean; nothing here
recomputes statistics.

Usage (after a pilot run refreshes docs/results/scores.parquet):
    uv sync --extra app
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
# app/data.py isn't a package — it's imported the same way app/app.py imports it, by
# putting app/ on sys.path (inserted ahead of the repo root so this never resolves to
# the unrelated top-level data/ directory instead).
sys.path.insert(0, str(REPO_ROOT / "app"))

from snapshot import TRAINING_TAIL_N, _kde_curve  # noqa: E402

from data import (  # noqa: E402
    _method_registry,
    load_case_split,
    load_case_split_counts,
    load_case_split_values,
    load_case_training_tail,
    load_metrics,
)
from tsforecastbench.data.registry import discover_cases  # noqa: E402


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
    """(x_grid, train KDE, test KDE) for the train-vs-test distribution plot — matches
    app/snapshot.py's `render_split_distribution`, which recomputes x_grid from whichever
    train variant (local/extended) is in play, so the extended-history distribution isn't
    just a re-slice of the local one when the case's value range actually differs."""
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
