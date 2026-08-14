"""Run the Phase A pilot: 3 cases x 5 methods on (synthetic or real) data.

Usage:
    uv run python scripts/run_pilot.py              # real-or-synth fallback
    uv run python scripts/run_pilot.py --synth      # force synthetic generators

To split a run across machines (e.g. GPU methods on one, CPU methods on another) use
--tag to write a git-tracked partial result instead of the full pipeline, then combine
the tagged files later with scripts/merge_pilot_results.py:
    uv run python scripts/run_pilot.py --tag gpu_dl_nf --skip-baseline --with-dl --with-nf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running from a clone without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsforecastbench.bench.eval import (
    DEFAULT_ORIGINS,
    DEFAULT_SEEDS,
    run_case,
    run_case_rolling,
)
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
    MLPExoForecaster,
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
from tsforecastbench.bench.report import RESULTS_DIR, write_pilot_artifacts
from tsforecastbench.data.registry import discover_cases

REPO_ROOT = Path(__file__).resolve().parents[1]
# Tagged partial runs (--tag) land here instead of the gitignored results/ dir, so a
# `git add` + push actually carries this machine's output to whichever machine runs the
# merge (see scripts/merge_pilot_results.py).
DOCS_PARTIAL_DIR = REPO_ROOT / "docs" / "results" / "partial"


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
        MLPExoForecaster(),
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
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"Runs per stochastic method; the median-MSE run is kept whole "
        f"(default {DEFAULT_SEEDS}; override globally with TSFB_SEEDS). Deterministic "
        "methods run once whatever this is set to.",
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
    p.add_argument(
        "--tag",
        default=None,
        help="Write a tagged, git-tracked partial result to docs/results/partial/ instead "
        "of the full results/ pipeline, for a run that covers only some method families "
        "(e.g. a GPU-only machine running --with-dl --with-nf) and will be combined with "
        "other tagged partial runs later via scripts/merge_pilot_results.py. Skips the "
        "matrix PNGs and the docs/results/scores.parquet mirror, which only make sense "
        "for the final merged result.",
    )
    p.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the baseline/windowed-ML methods from build_methods() -- combine with "
        "--with-dl/--with-nf/--with-fm to run only those families, e.g. on a GPU-only "
        "machine where the baseline methods gain nothing from a GPU and are already "
        "covered by another machine's run.",
    )
    args = p.parse_args()

    all_cases = discover_cases()
    ids = args.cases if args.cases else list(all_cases)
    cases = [all_cases[cid] for cid in ids if cid in all_cases]
    if not cases:
        print(f"No matching cases found. Available: {list(all_cases)}")
        return 1

    methods = [] if args.skip_baseline else build_methods()
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
    if not methods:
        print("No methods selected -- --skip-baseline needs at least one --with-* flag too.")
        return 1
    method_order = [m.name for m in methods]

    mode = "single hold-out" if args.fast else f"rolling-origin ({args.origins} origins)"
    print(f"Running pilot: {len(cases)} case(s) x {len(methods)} method(s) - {mode}")
    print(f"Data mode: {'synthetic' if args.synth else 'real-or-synth fallback'}\n")

    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = DOCS_PARTIAL_DIR if args.tag else RESULTS_DIR
    metrics_path = out_dir / f"pilot_metrics{suffix}.parquet"
    snapshots_path = out_dir / f"pilot_snapshots{suffix}.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
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
                res = run_case(case, methods, prefer_synth=args.synth, n_seeds=args.seeds)
                snaps = None
            else:
                res, snaps = run_case_rolling(
                    case,
                    methods,
                    n_origins=args.origins,
                    prefer_synth=args.synth,
                    n_seeds=args.seeds,
                )
        except Exception as exc:  # keep a long unattended run going if one case fails
            print(f"  [case-error] {case.id}: {type(exc).__name__}: {exc}")
            continue
        all_results.append(res)
        if snaps is not None and not snaps.empty:
            all_snapshots.append(snaps)
        # incremental save so a multi-hour run is never all-or-nothing
        pd.concat(all_results, ignore_index=True).to_parquet(metrics_path, index=False)
        if all_snapshots:
            pd.concat(all_snapshots, ignore_index=True).to_parquet(snapshots_path, index=False)
        print()

    if not all_results:
        print("No case produced results.")
        return 1
    results = pd.concat(all_results, ignore_index=True)
    results.to_parquet(metrics_path, index=False)
    if all_snapshots:
        pd.concat(all_snapshots, ignore_index=True).to_parquet(snapshots_path, index=False)

    if args.tag:
        print(f"\nWrote tagged partial results to {metrics_path.relative_to(REPO_ROOT)}")
        if all_snapshots:
            print(f"  and {snapshots_path.relative_to(REPO_ROOT)}")
        print(
            "Commit these and combine with other tagged runs via "
            "scripts/merge_pilot_results.py once every machine's run has finished."
        )
        return 0

    write_pilot_artifacts(results, all_snapshots, method_order)
    return 0


if __name__ == "__main__":
    sys.exit(main())
