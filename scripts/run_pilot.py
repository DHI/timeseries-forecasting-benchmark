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

from tsforecastbench.bench.eval import run_case, run_case_rolling
from tsforecastbench.bench.methods import (
    AutoARIMAForecaster,
    AutoETSForecaster,
    Climatology,
    DLinearForecaster,
    LightGBMExoForecaster,
    LightGBMForecaster,
    LinearLagsExoForecaster,
    LinearLagsForecaster,
    MeanBaseline,
    MLPForecaster,
    Persistence,
    SeasonalNaive,
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
        ThetaForecaster(),
        LinearLagsForecaster(),
        LightGBMForecaster(),
        MLPForecaster(),
        DLinearForecaster(),
        LinearLagsExoForecaster(),
        LightGBMExoForecaster(),
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--synth", action="store_true", help="Force synthetic data generators")
    p.add_argument("--cases", nargs="*", default=None,
                   help="Case ids to run (default: all discovered cases)")
    p.add_argument("--fast", action="store_true",
                   help="Single hold-out instead of rolling-origin (faster, noisier).")
    p.add_argument("--origins", type=int, default=16, help="Rolling origins per cell.")
    args = p.parse_args()

    all_cases = discover_cases()
    ids = args.cases if args.cases else list(all_cases)
    cases = [all_cases[cid] for cid in ids if cid in all_cases]
    if not cases:
        print(f"No matching cases found. Available: {list(all_cases)}")
        return 1

    methods = build_methods()
    method_order = [m.name for m in methods]

    mode = "single hold-out" if args.fast else f"rolling-origin ({args.origins} origins)"
    print(f"Running pilot: {len(cases)} case(s) x {len(methods)} method(s) - {mode}")
    print(f"Data mode: {'synthetic' if args.synth else 'real-or-synth fallback'}\n")

    all_results = []
    for case in cases:
        print(f"=== {case.id} ({case.domain}, freq={case.freq}, horizon={case.horizon_primary}) ===")
        if args.fast:
            res = run_case(case, methods, prefer_synth=args.synth)
        else:
            res = run_case_rolling(case, methods, n_origins=args.origins, prefer_synth=args.synth)
        all_results.append(res)
        print()

    results = pd.concat(all_results, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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
