"""Pretty-print the pilot and lead-time results for sharing or sanity-checking."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
R = REPO_ROOT / "results"


def main() -> int:
    m = R / "pilot_metrics.parquet"
    if m.exists():
        df = pd.read_parquet(m)
        print("\n=== Pilot matrix: MASE (lower is better; 1.0 = seasonal naive) ===")
        print(df[df["metric"] == "MASE"]
              .pivot(index="case_id", columns="method", values="value").round(2))
        print("\n=== Pilot matrix: Skill vs Persistence (higher is better; 0 = no better) ===")
        print(df[df["metric"] == "Skill_vs_Persistence"]
              .pivot(index="case_id", columns="method", values="value").round(2))
        print("\n=== Fraction of test window observed (NaN-masked) ===")
        print(df.groupby("case_id")["frac_finite_test"].first().round(3).to_string())
        print("\n=== Method runtimes (s) ===")
        print(df.groupby(["case_id", "method"])["seconds"].first().unstack().round(2))

    for cid in ("waves_oregon", "discharge_lowland_denmark", "sewerlevel_bellinge"):
        p = R / f"leadtime_{cid}.parquet"
        if p.exists():
            curve = pd.read_parquet(p)
            metric = "RMSE" if "RMSE" in curve.columns else "MAE"
            print(f"\n=== Lead-time curve for {cid} ({metric}) ===")
            print(curve.pivot(index="lead", columns="method", values=metric).round(3))

    return 0


if __name__ == "__main__":
    sys.exit(main())
