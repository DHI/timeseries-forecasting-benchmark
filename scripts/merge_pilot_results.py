"""Combine tagged partial pilot runs (see `scripts/run_pilot.py --tag`) from separate
machines into the final `results/pilot_metrics.parquet`, `docs/results/scores.parquet`,
`results/pilot_snapshots.parquet`, and matrix PNGs.

Usage (run once every tagged file has been pulled onto one machine):
    uv run python scripts/merge_pilot_results.py \\
        docs/results/partial/pilot_metrics_gpu_dl_nf.parquet \\
        docs/results/partial/pilot_metrics_local.parquet

Every `MASE` row is recomputed from its `MAE` row and the case's current pinned
`dominant_seasonality`, so a partial written under a different pin (or before the field
existed) does not carry a stale denominator into the published matrix. No model is re-fitted:
MASE is MAE over one per-case constant. See `_recompute_mase`.

`--drop TAG:METHOD[,METHOD...]` (repeatable) leaves one method out of one input file, for a
run whose column is known bad — the file stays on disk with its rows intact, and an earlier
file's version of that method (if any) wins instead. It applies to the tagged file's metrics
and its snapshots alike, so a dropped method's forecast lines never outlive its scores.

The nine partials that make up the published matrix as of 2026-08-28, in precedence order
(research log entry 101 — `baselines_v5` and `gpu_fm_nf_dl_v5` are retired: they predate the
seed-median scoring of #107 and every cell they hold is superseded):

    uv run python scripts/merge_pilot_results.py \\
        docs/results/partial/pilot_metrics_first_six.parquet \\
        docs/results/partial/pilot_metrics_remaining_no_nf.parquet \\
        docs/results/partial/pilot_metrics_remaining_b.parquet \\
        docs/results/partial/pilot_metrics_surge_full.parquet \\
        docs/results/partial/pilot_metrics_nf_all.parquet \\
        docs/results/partial/pilot_metrics_ensembles_fitonce.parquet \\
        docs/results/partial/pilot_metrics_exo_new.parquet \\
        docs/results/partial/pilot_metrics_lookback_v2.parquet \\
        docs/results/partial/pilot_metrics_wm_full.parquet \\
        --drop exo_new:TFT+exo --drop lookback_v2:TFT --drop wm_full:TFT,TFT+exo
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
# A metric row's identity minus the metric itself -- pairs a MASE row with its MAE row.
MAE_KEY = ["case_id", "method", "used_extended_history"]
SNAPSHOT_KEY = ["case_id", "method", "used_extended_history", "origin_id", "ds"]


def _snapshots_path(metrics_path: Path) -> Path:
    return metrics_path.with_name(metrics_path.name.replace("pilot_metrics", "pilot_snapshots"))


def _tag(metrics_path: Path) -> str:
    """The `run_pilot.py --tag` name a partial was written under, from its filename."""
    return metrics_path.stem.replace("pilot_metrics_", "", 1)


def _parse_drops(specs: list[str], tags: list[str]) -> dict[str, set[str]]:
    """`["lookback_v2:TFT", "wm_full:TFT,TFT+exo"]` -> `{"lookback_v2": {"TFT"}, ...}`.

    An unknown tag is an error rather than a no-op: a mistyped `--drop` would otherwise
    silently republish the very column it was meant to hold back.
    """
    drops: dict[str, set[str]] = {}
    for spec in specs:
        tag, _, methods = spec.partition(":")
        if not tag or not methods:
            raise SystemExit(f"--drop expects TAG:METHOD[,METHOD...], got {spec!r}")
        if tag not in tags:
            raise SystemExit(
                f"--drop {spec!r}: no input file tagged {tag!r} (have: {', '.join(tags)})"
            )
        drops.setdefault(tag, set()).update(m.strip() for m in methods.split(","))
    return drops


def _apply_drops(df: pd.DataFrame, tag: str, methods: set[str], label: str) -> pd.DataFrame:
    """Remove `methods` from one input frame, reporting the row count per method."""
    kept = df[~df["method"].isin(methods)]
    for method in sorted(methods):
        n = int((df["method"] == method).sum())
        if n == 0:
            raise SystemExit(f"--drop {tag}:{method}: that file has no {method} {label} rows")
        print(f"[merge_pilot_results] dropped {n} {method} {label} row(s) from {tag}")
    return kept


def _recompute_mase(results: pd.DataFrame) -> pd.DataFrame:
    """Rescale every `MASE` row to the case's current pinned dominant period.

    MASE is that row's `MAE` over one per-case constant: the in-sample MAE of a
    seasonal-naive forecast at `case.dominant_seasonality`, over the pre-test history.
    Nothing about it depends on the model, so a partial written before the period was
    pinned (or under a different pin) carries a stale denominator that this corrects
    without re-fitting anything. Doing it here rather than in a one-off migration keeps
    every future merge self-healing, whatever checkout produced the inputs.

    The extended-history rows reuse the local scale, matching `eval.rolling`, which
    computes it once per case and shares it across both fits.
    """
    from tsforecastbench.data.registry import get_case

    mase = results["metric"] == "MASE"
    mae = results[results["metric"] == "MAE"].set_index(MAE_KEY)["value"]
    for case_id in sorted(results.loc[mase, "case_id"].unique()):
        case = get_case(case_id)
        splits = case.split_frame(case.load(curated=True))
        y = pd.concat([splits["train"], splits["val"]])["y"].to_numpy(dtype=float)
        s = case.dominant_seasonality
        if len(y) <= s:
            raise SystemExit(
                f"{case_id}: {len(y)} pre-test points cannot support a seasonal-naive "
                f"scale at dominant_seasonality {s}"
            )
        scale = float(pd.Series(abs(y[s:] - y[:-s])).mean())
        rows = mase & (results["case_id"] == case_id)
        keys = pd.MultiIndex.from_frame(results.loc[rows, MAE_KEY])
        missing = ~keys.isin(mae.index)
        if missing.any():
            raise SystemExit(f"{case_id}: {int(missing.sum())} MASE row(s) with no MAE row")
        before = results.loc[rows, "value"].to_numpy(dtype=float)
        results.loc[rows, "value"] = mae.reindex(keys).to_numpy(dtype=float) / scale
        after = results.loc[rows, "value"].to_numpy(dtype=float)
        factor = float((after / before)[0]) if len(before) else 1.0
        note = "unchanged" if abs(factor - 1.0) < 1e-9 else f"x{factor:.4g}"
        print(
            f"[merge_pilot_results] {case_id}: MASE scale {scale:.6g} "
            f"at period {s} ({len(before)} row(s), {note})"
        )
    return results


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
    parser.add_argument(
        "--drop",
        action="append",
        default=[],
        metavar="TAG:METHOD[,METHOD...]",
        help="Leave a method out of one input file (repeatable), for a run whose column is "
        "known bad -- an earlier file's version of that method wins instead. Applies to that "
        "file's snapshots too.",
    )
    args = parser.parse_args(argv)

    drops = _parse_drops(args.drop, [_tag(p) for p in args.metrics_paths])

    metrics_frames = []
    snapshot_frames = []
    method_order: list[str] = []
    for path in args.metrics_paths:
        tag = _tag(path)
        dropped = drops.get(tag, set())
        df = pd.read_parquet(path)
        if dropped:
            df = _apply_drops(df, tag, dropped, "metrics")
        metrics_frames.append(df)
        for method in df["method"].unique():
            if method not in method_order:
                method_order.append(method)

        snap_path = _snapshots_path(path)
        if snap_path.exists():
            snaps = pd.read_parquet(snap_path)
            if dropped:
                snaps = _apply_drops(snaps, tag, dropped, "snapshot")
            snapshot_frames.append(snaps)
        else:
            print(f"[merge_pilot_results] no snapshots at {snap_path}, skipping")

    results = _recompute_mase(_merge(metrics_frames, METRICS_KEY, "metrics"))
    snapshots = [_merge(snapshot_frames, SNAPSHOT_KEY, "snapshot")] if snapshot_frames else []

    write_pilot_artifacts(results, snapshots, method_order)
    return 0


if __name__ == "__main__":
    sys.exit(main())
