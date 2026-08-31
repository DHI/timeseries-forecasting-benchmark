"""Pretty-print the pilot and lead-time results for sharing or sanity-checking."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
R = REPO_ROOT / "results"
DOCS_R = REPO_ROOT / "docs" / "results"


def main() -> int:
    m = DOCS_R / "scores.parquet"
    if m.exists():
        df = pd.read_parquet(m)
        # is_global methods carry two rows per cell (local + extended fit); the matrices
        # below are the local fit, so every method is on the same curated span.
        local = df[~df["used_extended_history"]] if "used_extended_history" in df else df
        print("\n=== Pilot matrix: MASE (lower is better; 1.0 = seasonal naive in-sample) ===")
        print(
            local[local["metric"] == "MASE"]
            .pivot(index="case_id", columns="method", values="value")
            .round(2)
        )
        print("\n=== Pilot matrix: Skill vs Persistence (higher is better; 0 = no better) ===")
        print(
            local[local["metric"] == "Skill_vs_Persistence"]
            .pivot(index="case_id", columns="method", values="value")
            .round(2)
        )
        print("\n=== Fraction of test window observed (NaN-masked) ===")
        print(local.groupby("case_id")["frac_finite_test"].first().round(3).to_string())
        print("\n=== Method fit time, per origin (s) ===")
        print(local.groupby(["case_id", "method"])["fit_seconds"].first().unstack().round(2))
        print("\n=== Method predict time, per origin (s) ===")
        print(local.groupby(["case_id", "method"])["predict_seconds"].first().unstack().round(2))

        ext = df[df["used_extended_history"]] if "used_extended_history" in df else df.iloc[:0]
        if not ext.empty:
            key = ["case_id", "method"]
            loc_mase = local[local["metric"] == "MASE"].set_index(key)["value"]
            ext_mase = ext[ext["metric"] == "MASE"].set_index(key)["value"]
            cmp = pd.DataFrame({"local": loc_mase, "extended": ext_mase}).dropna()
            cmp["gain_%"] = (100 * (cmp["local"] - cmp["extended"]) / cmp["local"]).round(1)
            print("\n=== Extended-history MASE (is_global methods; full-series vs local fit) ===")
            print(cmp.round(3).to_string())

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
