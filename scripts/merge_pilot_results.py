"""Combine tagged partial pilot runs (see `scripts/run_pilot.py --tag`) from separate
machines into the final `results/pilot_metrics.parquet`, `docs/results/scores.parquet`,
`results/pilot_snapshots.parquet`, and matrix PNGs.

Usage (run once every tagged file has been pulled onto one machine):
    uv run python scripts/merge_pilot_results.py \\
        docs/results/partial/pilot_metrics_gpu_dl_nf.parquet \\
        docs/results/partial/pilot_metrics_local.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
# Allow running from a clone without installing
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.bench.report import write_pilot_artifacts  # noqa: E402

METRICS_KEY = ["case_id", "method", "used_extended_history", "metric"]
SNAPSHOT_KEY = ["case_id", "method", "used_extended_history", "origin_id", "ds"]


def _snapshots_path(metrics_path: Path) -> Path:
    return metrics_path.with_name(metrics_path.name.replace("pilot_metrics", "pilot_snapshots"))


def _merge(frames: list[pd.DataFrame], key: list[str], label: str) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    dupes = combined.duplicated(subset=key, keep=False)
    if dupes.any():
        print(
            f"[merge_pilot_results] warning: {dupes.sum()} duplicate {label} row(s) on "
            f"{key} across inputs -- two machines likely ran the same method on the same "
            "case; keeping the last input file's version"
        )
    return combined.drop_duplicates(subset=key, keep="last")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics_paths",
        nargs="+",
        type=Path,
        help="Tagged pilot_metrics_*.parquet files, in precedence order (a later file "
        "wins on a duplicate case/method/metric key).",
    )
    args = parser.parse_args(argv)

    metrics_frames = []
    snapshot_frames = []
    method_order: list[str] = []
    for path in args.metrics_paths:
        df = pd.read_parquet(path)
        metrics_frames.append(df)
        for method in df["method"].unique():
            if method not in method_order:
                method_order.append(method)

        snap_path = _snapshots_path(path)
        if snap_path.exists():
            snapshot_frames.append(pd.read_parquet(snap_path))
        else:
            print(f"[merge_pilot_results] no snapshots at {snap_path}, skipping")

    results = _merge(metrics_frames, METRICS_KEY, "metrics")
    snapshots = [_merge(snapshot_frames, SNAPSHOT_KEY, "snapshot")] if snapshot_frames else []

    write_pilot_artifacts(results, snapshots, method_order)
    return 0


if __name__ == "__main__":
    sys.exit(main())
