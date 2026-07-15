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

REPO_ROOT = Path(__file__).resolve().parents[1]
# Allow running from a clone without installing.
sys.path.insert(0, str(REPO_ROOT / "src"))
# app/data.py isn't a package — it's imported the same way app/app.py imports it, by
# putting app/ on sys.path (inserted ahead of the repo root so this never resolves to
# the unrelated top-level data/ directory instead).
sys.path.insert(0, str(REPO_ROOT / "app"))

from data import _method_registry, load_metrics  # noqa: E402
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


def export_cases(out_dir: Path) -> None:
    """cases.json: static case.toml metadata needed by the sidebar/case panels — domain,
    target, exogenous inputs, source, notes, and which domain-restricted metrics
    (`cases_eligible_for_metric`) this case declares. Bounded derived series (training
    tail, test split, KDE curves) are added in a later export step, not here."""
    cases = {}
    for case_id, case in sorted(discover_cases().items()):
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
