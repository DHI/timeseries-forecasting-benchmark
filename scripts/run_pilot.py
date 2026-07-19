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
    LightGBMCalendarExoForecaster,
    LightGBMCalendarForecaster,
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
)
from tsforecastbench.bench.viz import render_matrix
from tsforecastbench.data.registry import discover_cases

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def build_methods():
    # Univariate (Tier 1) baselines + windowed ML, then the exogenous-aware (Tier 2)
    # variants. On cases with no declared covariates the +exo methods reduce to univariate.
    return [
        MeanBaseline(),
        Persistence(),
        SeasonalNaive(),
        Climatology(),
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
        LightGBMCalendarForecaster(),
        LightGBMCalendarExoForecaster(),
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
        help="Add the zero-shot foundation model (Chronos-Bolt; needs the 'fm' extra).",
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
        from tsforecastbench.bench.methods import ChronosForecaster

        methods.append(ChronosForecaster())
    method_order = [m.name for m in methods]

    mode = "single hold-out" if args.fast else f"rolling-origin ({args.origins} origins)"
    print(f"Running pilot: {len(cases)} case(s) x {len(methods)} method(s) - {mode}")
    print(f"Data mode: {'synthetic' if args.synth else 'real-or-synth fallback'}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for case in cases:
        print(
            f"=== {case.id} ({case.domain}, freq={case.freq}, horizon={case.horizon_primary}) ==="
        )
        try:
            if args.fast:
                res = run_case(case, methods, prefer_synth=args.synth)
            else:
                res = run_case_rolling(
                    case, methods, n_origins=args.origins, prefer_synth=args.synth
                )
        except Exception as exc:  # keep a long unattended run going if one case fails
            print(f"  [case-error] {case.id}: {type(exc).__name__}: {exc}")
            continue
        all_results.append(res)
        # incremental save so a multi-hour run is never all-or-nothing
        pd.concat(all_results, ignore_index=True).to_parquet(
            RESULTS_DIR / "pilot_metrics.parquet", index=False
        )
        print()

    if not all_results:
        print("No case produced results.")
        return 1
    results = pd.concat(all_results, ignore_index=True)
    results.to_parquet(RESULTS_DIR / "pilot_metrics.parquet", index=False)

    render_matrix(
        results,
        metric="MASE",
        out_path=RESULTS_DIR / "pilot_matrix_MASE.png",
        title="RK2026 pilot - MASE (lower is better)",
        method_order=method_order,
    )
    render_matrix(
        results,
        metric="Skill_vs_Persistence",
        out_path=RESULTS_DIR / "pilot_matrix_skill.png",
        title="RK2026 pilot - skill vs persistence (higher is better)",
        method_order=method_order,
    )
    render_matrix(
        results,
        metric="Skill_vs_Mean",
        out_path=RESULTS_DIR / "pilot_matrix_skill_vs_mean.png",
        title="RK2026 pilot - skill vs mean (higher is better)",
        method_order=method_order,
    )

    print(f"\nWrote results to {RESULTS_DIR.relative_to(REPO_ROOT)}/")
    print(f"  pilot_metrics.parquet ({len(results)} rows)")
    print("  pilot_matrix_MASE.png, pilot_matrix_skill.png, pilot_matrix_skill_vs_mean.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
