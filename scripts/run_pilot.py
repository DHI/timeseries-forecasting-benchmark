"""Run the Phase A pilot: 3 cases x 5 methods on (synthetic or real) data.

Usage:
    uv run python scripts/run_pilot.py              # real-or-synth fallback
    uv run python scripts/run_pilot.py --synth      # force synthetic generators
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running from a clone without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsforecastbench.bench.eval import DEFAULT_ORIGINS, run_case, run_case_rolling
from tsforecastbench.bench.methods import (
    AutoARIMAForecaster,
    AutoETSForecaster,
    CatBoostExoForecaster,
    CatBoostForecaster,
    Climatology,
    DLinearForecaster,
    LightGBMExoForecaster,
    LightGBMForecaster,
    LinearLagsExoForecaster,
    LinearLagsForecaster,
    MeanBaseline,
    MLPForecaster,
    MSTLAutoARIMAForecaster,
    MSTLAutoETSForecaster,
    Persistence,
    SeasonalNaive,
    TBATSForecaster,
    ThetaForecaster,
    ensemble_baselines,
    ensemble_baselines_theta,
)
from tsforecastbench.bench.viz import render_matrix
from tsforecastbench.data.registry import discover_cases

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
# Committed, distilled copies for onboarding: a new clone gets real scores and matrix
# figures without running the (multi-hour) pilot first. `results/` stays the gitignored
# working directory — raw snapshots and the same metrics, kept for local debugging.
DOCS_RESULTS_DIR = REPO_ROOT / "docs" / "results"
DOCS_FIGURES_DIR = REPO_ROOT / "docs" / "figures" / "benchmark"


def build_methods():
    # Univariate (Tier 1) baselines + windowed ML, then the exogenous-aware (Tier 2)
    # variants. On cases with no declared covariates the +exo methods reduce to univariate.
    return [
        MeanBaseline(),
        Persistence(),
        SeasonalNaive(),
        Climatology(),
        ensemble_baselines(),
        ensemble_baselines_theta(),
        AutoARIMAForecaster(),
        AutoETSForecaster(),
        MSTLAutoETSForecaster(),
        MSTLAutoARIMAForecaster(),
        TBATSForecaster(),
        ThetaForecaster(),
        LinearLagsForecaster(),
        LightGBMForecaster(),
        CatBoostForecaster(),
        MLPForecaster(),
        DLinearForecaster(),
        LinearLagsExoForecaster(),
        LightGBMExoForecaster(),
        CatBoostExoForecaster(),
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--synth", action="store_true", help="Force synthetic data generators")
    p.add_argument(
        "--cases", nargs="*", default=None, help="Case ids to run (default: all discovered cases)"
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Single hold-out instead of rolling-origin (faster, noisier).",
    )
    p.add_argument(
        "--origins",
        type=int,
        default=DEFAULT_ORIGINS,
        help=f"Rolling origins per cell (default {DEFAULT_ORIGINS}; "
        "override globally with TSFB_ORIGINS).",
    )
    p.add_argument(
        "--with-fm",
        action="store_true",
        help="Add the zero-shot foundation models (Chronos-Bolt, Chronos-2, TimesFM-2.5, "
        "Moirai-2.0; all installed via the 'fm' extra).",
    )
    p.add_argument(
        "--with-dl",
        action="store_true",
        help="Add the torch deep-learning models (N-HiTS, PatchTST; needs the 'dl' extra). "
        "Heavy on CPU — pair with a small --origins.",
    )
    p.add_argument(
        "--with-nf",
        action="store_true",
        help="Add the neuralforecast zoo (TFT, iTransformer, TimesNet; needs the 'nf' "
        "extra — Linux/WSL only). GPU-accelerated when available.",
    )
    args = p.parse_args()

    all_cases = discover_cases()
    ids = args.cases if args.cases else list(all_cases)
    cases = [all_cases[cid] for cid in ids if cid in all_cases]
    if not cases:
        print(f"No matching cases found. Available: {list(all_cases)}")
        return 1

    methods = build_methods()
    if args.with_dl:
        from tsforecastbench.bench.methods import NHITSForecaster, PatchTSTForecaster

        methods += [NHITSForecaster(), PatchTSTForecaster()]
    if args.with_nf:
        from tsforecastbench.bench.methods import (
            ITransformerForecaster,
            TFTForecaster,
            TimesNetForecaster,
        )

        methods += [TFTForecaster(), ITransformerForecaster(), TimesNetForecaster()]
    if args.with_fm:
        from tsforecastbench.bench.methods import (
            Chronos2ExoForecaster,
            Chronos2Forecaster,
            ChronosForecaster,
            Moirai2ExoForecaster,
            Moirai2Forecaster,
            TimesFMExoForecaster,
            TimesFMForecaster,
        )

        methods += [
            ChronosForecaster(),
            Chronos2Forecaster(),
            TimesFMForecaster(),
            Moirai2Forecaster(),
            Chronos2ExoForecaster(),
            TimesFMExoForecaster(),
            Moirai2ExoForecaster(),
        ]
    method_order = [m.name for m in methods]

    mode = "single hold-out" if args.fast else f"rolling-origin ({args.origins} origins)"
    print(f"Running pilot: {len(cases)} case(s) x {len(methods)} method(s) - {mode}")
    print(f"Data mode: {'synthetic' if args.synth else 'real-or-synth fallback'}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    all_snapshots = []
    for case in cases:
        print(
            f"=== {case.id} ({case.domain}, freq={case.freq}, horizon={case.horizon_primary}) ==="
        )
        try:
            if args.fast:
                # Single hold-out doesn't emit per-origin snapshots; the viewer degrades
                # to a "no forecast stored" note for these cases.
                res = run_case(case, methods, prefer_synth=args.synth)
                snaps = None
            else:
                res, snaps = run_case_rolling(
                    case, methods, n_origins=args.origins, prefer_synth=args.synth
                )
        except Exception as exc:  # keep a long unattended run going if one case fails
            print(f"  [case-error] {case.id}: {type(exc).__name__}: {exc}")
            continue
        all_results.append(res)
        if snaps is not None and not snaps.empty:
            all_snapshots.append(snaps)
        # incremental save so a multi-hour run is never all-or-nothing
        pd.concat(all_results, ignore_index=True).to_parquet(
            RESULTS_DIR / "pilot_metrics.parquet", index=False
        )
        if all_snapshots:
            pd.concat(all_snapshots, ignore_index=True).to_parquet(
                RESULTS_DIR / "pilot_snapshots.parquet", index=False
            )
        print()

    if not all_results:
        print("No case produced results.")
        return 1
    results = pd.concat(all_results, ignore_index=True)
    results.to_parquet(RESULTS_DIR / "pilot_metrics.parquet", index=False)
    DOCS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(DOCS_RESULTS_DIR / "scores.parquet", index=False)
    if all_snapshots:
        pd.concat(all_snapshots, ignore_index=True).to_parquet(
            RESULTS_DIR / "pilot_snapshots.parquet", index=False
        )

    # Two matrices per metric. The main one is every method on the shared curated span
    # (the local fit only, `used_extended_history=False`), so all methods are compared on
    # identical data. The `_global` one shows the same `is_global` methods scored on the
    # full single-series history where the case has any beyond the curated span (the
    # extended fit, `used_extended_history=True`), falling back to the local score for
    # cases/methods with no extra history. Reading the two side by side shows the gain a
    # global-capable method gets from the extra training data.
    # `docs_stem` is the name these land under in docs/figures/benchmark/ (matches the
    # names already referenced by docs/RESULTS.md).
    matrix_specs = [
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
    for metric, stem, label, docs_stem in matrix_specs:
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
    if all_snapshots:
        n_snap = sum(len(s) for s in all_snapshots)
        print(f"  pilot_snapshots.parquet ({n_snap} rows)")
    if not global_method_order:
        print("  [note] no is_global methods in this run; skipped the _global matrices")
    print("  " + ", ".join(written))
    print(f"Wrote committed onboarding copies to {DOCS_RESULTS_DIR.relative_to(REPO_ROOT)}/ and")
    print(f"  {DOCS_FIGURES_DIR.relative_to(REPO_ROOT)}/ — commit these along with this run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
