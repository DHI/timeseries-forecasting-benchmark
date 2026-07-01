"""Compute all 13 candidate morphological features for feature-selection analysis.

Gap-aware: short gaps (≤ period/12 steps) are filled by linear interpolation;
longer gaps become segment boundaries. Features are length-weighted averages
across segments — identical gap-handling to compute_morphology.py.

Outputs
-------
results/morphology_features_all.csv   case × 13 features + domain

Usage
-----
    uv run python scripts/compute_features_all.py
    uv run python scripts/compute_features_all.py --remaining-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.data.registry import discover_cases
from tsforecastbench.morphology import (
    GAP_INTERP_FRAC,
    GAP_INTERP_MIN,
    MIN_SEGMENT_ABS,
    MIN_SEGMENT_FRAC,
    aggregate_features,
    gap_aware_segments,
)
from tsforecastbench.morphology.features import (
    compute_features_all,
    period_for_freq,
)

RESULTS_DIR = REPO_ROOT / "results"

ALL_FEATURE_COLS = [
    "acf1", "trend_strength", "seasonal_strength", "spectral_entropy",
    "boxcox_lambda", "kurtosis",
    "hurst", "perm_entropy", "amplitude_dispersion", "phase_dispersion",
    "asymmetry", "l_skewness", "zero_proportion",
]

# The features already computed gap-aware by compute_morphology.py (must match
# its FEATURE_COLS).
SELECTED_COLS = ["seasonal_strength", "hurst", "amplitude_dispersion", "l_skewness"]
REMAINING_COLS = [c for c in ALL_FEATURE_COLS if c not in SELECTED_COLS]



def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--remaining-only", action="store_true",
        help=(
            "Load the gap-aware features from morphology_features.csv and "
            "compute only the remaining ones. Faster when the selected features "
            "are already up to date."
        ),
    )
    args = p.parse_args()

    cases = discover_cases()
    existing_df: pd.DataFrame | None = None

    if args.remaining_only:
        csv_path = RESULTS_DIR / "morphology_features.csv"
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found — run compute_morphology.py first.")
            return 1
        existing_df = pd.read_csv(csv_path, index_col="case_id")
        print(f"Loaded {len(existing_df)} cases from morphology_features.csv "
              f"({len(SELECTED_COLS)} gap-aware features).")
        print(f"Computing {len(REMAINING_COLS)} remaining features gap-aware: "
              f"{', '.join(REMAINING_COLS)}\n")
    else:
        print(f"Computing all {len(ALL_FEATURE_COLS)} features gap-aware.\n")

    available = [cid for cid, c in cases.items() if c.parquet_path.exists()]
    n_total = len(available)

    records: dict[str, dict[str, float]] = {}
    domains: dict[str, str] = {}

    for i, case_id in enumerate(available, 1):
        case = cases[case_id]
        period = period_for_freq(case.freq)
        gap_interp_max = max(GAP_INTERP_MIN, int(period * GAP_INTERP_FRAC))
        min_segment_len = max(MIN_SEGMENT_ABS, int(period * MIN_SEGMENT_FRAC))

        print(f"[{i}/{n_total}] {case_id}", end="  ", flush=True)
        y_segs, _ = gap_aware_segments(case, period, gap_interp_max, min_segment_len)

        if not y_segs:
            print("no valid segments — skipping.")
            continue

        n_segs = len(y_segs)
        lengths = [len(y) for y in y_segs]
        print(f"{n_segs} seg(s), gap_interp_max={gap_interp_max}", end="  ", flush=True)

        feat_dicts = [compute_features_all(y) for y in y_segs]

        feats = aggregate_features(feat_dicts, lengths)
        records[case_id] = feats
        domains[case_id] = case.domain
        print("  ".join(f"{k}={v:+.3f}" for k, v in feats.items()))

    if not records:
        print("No cases processed — nothing to write.")
        return 1

    features_df = pd.DataFrame.from_dict(records, orient="index")
    features_df.index.name = "case_id"

    if args.remaining_only and existing_df is not None:
        for col in SELECTED_COLS:
            if col in existing_df.columns:
                features_df[col] = existing_df[col]

    features_df["domain"] = pd.Series(domains)

    col_order = ALL_FEATURE_COLS + ["domain"]
    features_df = features_df[[c for c in col_order if c in features_df.columns]]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "morphology_features_all.csv"
    features_df.to_csv(out_path)
    print(f"\nFeatures saved → {out_path.relative_to(REPO_ROOT)}")
    print(features_df.drop(columns=["domain"]).to_string(float_format="{:+.3f}".format))
    return 0


if __name__ == "__main__":
    sys.exit(main())
