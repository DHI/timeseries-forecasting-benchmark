"""Compute all 13 candidate morphological features for feature-selection analysis.

Outputs
-------
results/morphology_features_all.csv   case × 13 features (6 core + 7 extended)

Usage
-----
    uv run python scripts/compute_features_all.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.data.registry import discover_cases
from tsforecastbench.morphology.features import compute_features_all, period_for_freq

RESULTS_DIR = REPO_ROOT / "results"


def _extract_train_data(case) -> tuple[np.ndarray, np.ndarray | None]:
    df = case.load()
    train = case.split_frame(df)["train"]
    mask = train["y"].notna()
    y = train.loc[mask, "y"].to_numpy(dtype=float)
    if not case.exogenous:
        return y, None
    X = train.loc[mask, case.exogenous].to_numpy(dtype=float)
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        nan_idx = np.isnan(X[:, j])
        if nan_idx.any():
            X[nan_idx, j] = col_means[j]
    return y, X


def main() -> int:
    cases = discover_cases()
    records: dict[str, dict[str, float]] = {}
    domains: dict[str, str] = {}

    for case_id, case in cases.items():
        if not case.parquet_path.exists():
            print(f"[{case_id}] no parquet — skipping.")
            continue
        print(f"[{case_id}] computing ...", end="  ", flush=True)
        y, _ = _extract_train_data(case)
        period = period_for_freq(case.freq)
        feats = compute_features_all(y, period)
        records[case_id] = feats
        domains[case_id] = case.domain
        print("  ".join(f"{k}={v:+.3f}" for k, v in feats.items()))

    if not records:
        print("No cases with processed parquet found — nothing to do.")
        return 1

    features_df = pd.DataFrame.from_dict(records, orient="index")
    features_df.index.name = "case_id"
    features_df["domain"] = pd.Series(domains)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "morphology_features_all.csv"
    features_df.to_csv(csv_path)
    print(f"\nFeatures saved → {csv_path.relative_to(REPO_ROOT)}")
    print(features_df.drop(columns=["domain"]).to_string(float_format="{:+.3f}".format))
    return 0


if __name__ == "__main__":
    sys.exit(main())
