"""Materialize benchmark-ready series in data/curated/ (and data/curated_extended/)
from data/processed/.

Applies the gap policy in tsforecastbench.data.curate once per case: fill small gaps
(<= a per-case, autocorrelation-derived cap), keep the tail contiguous span (the one
ending at the record's most recent observation), drop the rest, and record which
points are genuine observations (`y_observed`). The eval loops read these; the raw
gappy Parquets in data/processed/ stay untouched for EDA and case review.

Also materializes the extended-history frame `Forecaster.is_global` methods use: the
full record up to the tail segment's `val_end`, gap-filled the same way but without
the segment trim (see `tsforecastbench.data.curate.extend_frame`).

Run after downloading/refreshing data, or after changing the policy:

    uv run python scripts/curate_data.py            # all cases
    uv run python scripts/curate_data.py waves_oregon surge_cuxhaven
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from tsforecastbench.data.curate import curate_frame, extend_frame
from tsforecastbench.data.registry import discover_cases
from tsforecastbench.data.schema import DATA_CURATED, DATA_CURATED_EXTENDED, REPO_ROOT

MANIFEST = DATA_CURATED / "manifest.json"
MANIFEST_EXTENDED = DATA_CURATED_EXTENDED / "manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cases = discover_cases()
    if argv:
        cases = {cid: cases[cid] for cid in argv if cid in cases}
        missing = [cid for cid in argv if cid not in discover_cases()]
        for cid in missing:
            print(f"[skip] unknown case {cid}")

    DATA_CURATED.mkdir(parents=True, exist_ok=True)
    DATA_CURATED_EXTENDED.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest_extended = (
        json.loads(MANIFEST_EXTENDED.read_text()) if MANIFEST_EXTENDED.exists() else {}
    )

    hdr = (
        f"{'case':30s} {'cap':>4s} {'orig':>8s} {'kept':>8s} {'kept%':>6s} {'filled':>7s} "
        f"{'bigg':>5s} {'ext_rows':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))
    rc = 0
    for cid, case in cases.items():
        if not case.parquet_path.exists():
            print(f"{cid:30s}  [skip] no processed parquet")
            continue
        try:
            raw = pd.read_parquet(case.parquet_path)
            exog = list(dict.fromkeys([*case.exogenous, *case.exogenous_known_future]))
            curated, info = curate_frame(
                raw,
                case.freq,
                seasonalities=case.seasonalities,
                horizon_primary=case.horizon_primary,
                split_fraction=case.split_fraction,
                exog=exog,
            )
            curated["ds"] = pd.to_datetime(curated["ds"], utc=True)
            out = DATA_CURATED / f"{cid}.parquet"
            curated.to_parquet(out, index=False)
            manifest[cid] = {
                "parquet": str(out.relative_to(REPO_ROOT)),
                "sha256": _sha256(out),
                **info,
            }

            _, val_end, _ = case.split_bounds(curated)
            extended, info_ext = extend_frame(raw, case.freq, cutoff=val_end, exog=exog)
            extended["ds"] = pd.to_datetime(extended["ds"], utc=True)
            out_ext = DATA_CURATED_EXTENDED / f"{cid}.parquet"
            extended.to_parquet(out_ext, index=False)
            manifest_extended[cid] = {
                "parquet": str(out_ext.relative_to(REPO_ROOT)),
                "sha256": _sha256(out_ext),
                **info_ext,
            }

            print(
                f"{cid:30s} {info['cap']:4d} {info['orig_rows']:8d} {info['rows']:8d} "
                f"{100 * info['kept_frac']:6.1f} {info['n_filled']:7d} {info['n_big_gaps']:5d} "
                f"{info_ext['rows']:8d}"
            )
        except Exception as exc:
            print(f"{cid:30s}  [error] {type(exc).__name__}: {exc}")
            rc = 1

    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    MANIFEST_EXTENDED.write_text(json.dumps(manifest_extended, indent=2, sort_keys=True))
    print(f"\nwrote {MANIFEST.relative_to(REPO_ROOT)}")
    print(f"wrote {MANIFEST_EXTENDED.relative_to(REPO_ROOT)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
