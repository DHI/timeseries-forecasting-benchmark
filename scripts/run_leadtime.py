"""Lead-time skill curves: forecast error vs horizon, averaged over rolling origins.

Usage:
    uv run python scripts/run_leadtime.py --case discharge_lowland_denmark
    uv run python scripts/run_leadtime.py --case sewerlevel_bellinge --origins 30 --metric RMSE
    uv run python scripts/run_leadtime.py --case discharge_lowland_denmark --with-arima  # slow

By default runs the cheap references (Mean, Persistence, SeasonalNaive, Climatology)
plus LightGBM. ARIMA is opt-in because it refits at every origin.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsforecastbench.bench.eval.rolling import run_leadtime_curve
from tsforecastbench.bench.methods import (
    AutoARIMAForecaster,
    AutoETSForecaster,
    Climatology,
    LightGBMExoForecaster,
    LightGBMForecaster,
    LinearLagsForecaster,
    MeanBaseline,
    Persistence,
    SeasonalNaive,
)
from tsforecastbench.bench.viz.leadtime import plot_leadtime_curve
from tsforecastbench.data.registry import get_case

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

FREQ_LABEL = {"1D": "days", "1h": "hours", "15min": "15-min steps"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--origins", type=int, default=24)
    p.add_argument("--metric", default="MAE", choices=["MAE", "RMSE", "bias"])
    p.add_argument("--max-horizon", default=None, help="Override horizon, e.g. '14D', '48h'")
    p.add_argument("--with-arima", action="store_true")
    p.add_argument("--synth", action="store_true")
    args = p.parse_args()

    case = get_case(args.case)
    methods = [MeanBaseline(), Persistence(), SeasonalNaive(), Climatology(),
               LinearLagsForecaster(), LightGBMForecaster()]
    if case.exogenous or case.exogenous_known_future:
        methods.append(LightGBMExoForecaster())  # Tier-2 covariate-aware comparison
    if args.with_arima:
        methods.append(AutoARIMAForecaster())
        methods.append(AutoETSForecaster())

    print(f"Lead-time curve for {case.id} ({case.freq}, horizon {args.max_horizon or case.horizon_primary})")
    curve = run_leadtime_curve(
        case, methods, n_origins=args.origins, max_horizon=args.max_horizon, prefer_synth=args.synth
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    curve.to_parquet(RESULTS_DIR / f"leadtime_{case.id}.parquet", index=False)
    plot_leadtime_curve(
        curve,
        metric=args.metric,
        out_path=RESULTS_DIR / f"leadtime_{case.id}_{args.metric}.png",
        title=f"{case.id}: {args.metric} vs lead time (avg over {args.origins} origins)",
        freq_label=FREQ_LABEL.get(case.freq, "steps"),
    )
    print(f"\nWrote results/leadtime_{case.id}.parquet and _{args.metric}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
