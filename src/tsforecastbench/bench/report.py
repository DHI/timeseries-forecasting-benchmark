"""Shared "final write" logic for a pilot run: the merged metrics/snapshots parquet, the
committed docs/ mirror, and every matrix PNG.

Used by both `scripts/run_pilot.py` (a full run covering every method) and
`scripts/merge_pilot_results.py` (recombining tagged partial runs from separate machines).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tsforecastbench.bench.viz import render_matrix

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"
# Committed, distilled copies for onboarding: a new clone gets real scores and matrix
# figures without running the (multi-hour) pilot first. `results/` stays the gitignored
# working directory — raw snapshots and the same metrics, kept for local debugging.
DOCS_RESULTS_DIR = REPO_ROOT / "docs" / "results"
DOCS_FIGURES_DIR = REPO_ROOT / "docs" / "figures" / "benchmark"

# Two matrices per metric. The main one is every method on the shared curated span (the
# local fit only, `used_extended_history=False`), so all methods are compared on identical
# data. The `_global` one shows the same `is_global` methods scored on the full
# single-series history where the case has any beyond the curated span (the extended fit,
# `used_extended_history=True`), falling back to the local score for cases/methods with no
# extra history. Reading the two side by side shows the gain a global-capable method gets
# from the extra training data.
# `docs_stem` is the name these land under in docs/figures/benchmark/ (matches the names
# already referenced by docs/RESULTS.md).
MATRIX_SPECS = [
    ("MASE", "pilot_matrix_MASE", "MASE (lower is better)", "matrix_mase"),
    (
        "Skill_vs_Persistence",
        "pilot_matrix_skill",
        "skill vs persistence (higher is better)",
        "matrix_skill_vs_persistence",
    ),
    (
        "Skill_vs_Mean",
        "pilot_matrix_skill_vs_mean",
        "skill vs mean (higher is better)",
        "matrix_skill_vs_mean",
    ),
]


def write_pilot_artifacts(
    results: pd.DataFrame,
    snapshots: list[pd.DataFrame],
    method_order: list[str],
) -> None:
    """Write `results/pilot_metrics.parquet`, its `docs/results/scores.parquet` mirror,
    `results/pilot_snapshots.parquet` (if any snapshots), and every matrix PNG."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(RESULTS_DIR / "pilot_metrics.parquet", index=False)
    DOCS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(DOCS_RESULTS_DIR / "scores.parquet", index=False)
    if snapshots:
        pd.concat(snapshots, ignore_index=True).to_parquet(
            RESULTS_DIR / "pilot_snapshots.parquet", index=False
        )

    local_results = results[~results["used_extended_history"]]
    # For the global matrix, keep one row per (case, method, metric): the extended fit
    # when it ran, otherwise the local fit. Sorting False<True and keeping the last row
    # prefers the extended score wherever both exist.
    global_results = (
        results[results["is_global"]]
        .sort_values("used_extended_history")
        .drop_duplicates(subset=["case_id", "method", "metric"], keep="last")
    )
    global_method_order = [m for m in method_order if m in global_results["method"].unique()]

    DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for metric, stem, label, docs_stem in MATRIX_SPECS:
        render_matrix(
            local_results,
            metric=metric,
            out_path=RESULTS_DIR / f"{stem}.png",
            title=f"RK2026 pilot - {label}",
            method_order=method_order,
        )
        render_matrix(
            local_results,
            metric=metric,
            out_path=DOCS_FIGURES_DIR / f"{docs_stem}.png",
            title=f"RK2026 pilot - {label}",
            method_order=method_order,
        )
        written.append(f"{stem}.png")
        if global_method_order:
            render_matrix(
                global_results,
                metric=metric,
                out_path=RESULTS_DIR / f"{stem}_global.png",
                title=f"RK2026 pilot - {label} (is_global methods, extended-history fit)",
                method_order=global_method_order,
            )
            written.append(f"{stem}_global.png")

    print(f"\nWrote results to {RESULTS_DIR.relative_to(REPO_ROOT)}/")
    print(f"  pilot_metrics.parquet ({len(results)} rows)")
    if snapshots:
        n_snap = sum(len(s) for s in snapshots)
        print(f"  pilot_snapshots.parquet ({n_snap} rows)")
    if not global_method_order:
        print("  [note] no is_global methods in this run; skipped the _global matrices")
    print("  " + ", ".join(written))
    print(f"Wrote committed onboarding copies to {DOCS_RESULTS_DIR.relative_to(REPO_ROOT)}/ and")
    print(f"  {DOCS_FIGURES_DIR.relative_to(REPO_ROOT)}/ — commit these along with this run.")
