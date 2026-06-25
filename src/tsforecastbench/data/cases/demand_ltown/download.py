"""Download BattLeDIM L-Town SCADA from Zenodo and build the inlet-flow case.

Zenodo record 4017659 contains 2018 historical + 2019 evaluation SCADA CSVs
(pressures, flows, demands, levels), plus the EPANET model.

Target is the *total system demand* = sum of all PUMP_* inlet flow columns. L-Town
has multiple inlet pumps; any single one alternates with the others (operational
pump scheduling), so the single-pump signal is intermittent and not what you want
to forecast. The sum of pumps is the real demand signal at hourly resolution.

Run:
    uv run python -m tsforecastbench.data.cases.demand_ltown.download --list
    uv run python -m tsforecastbench.data.cases.demand_ltown.download --columns   # print flow columns
    uv run python -m tsforecastbench.data.cases.demand_ltown.download
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from tsforecastbench.data.cases import _io
from tsforecastbench.data.registry import get_case

ZENODO_RECORD = 4017659
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

CONFIG = {
    # Target = sum of per-node consumer demand from SCADA_Demands.csv. Each row in
    # that file is a 5-min snapshot of demand at every consumer node in the L-Town
    # network. The sum across nodes is the network's TOTAL CONSUMER DEMAND - a clean
    # diurnal/weekly signal that's the operationally meaningful forecasting target.
    #
    # We deliberately do NOT use SCADA_Flows.csv: it only has PUMP_1 (which cycles
    # ON/OFF on tank-setpoint, not demand) plus two internal pipe flowmeters that
    # measure sub-zone boundaries, not inlets.
    "demands_file_match": "Demands",
    # All columns other than the timestamp are per-node demands; sum them.
}


def _list_files() -> list[dict]:
    import requests

    r = requests.get(ZENODO_API, timeout=60)
    r.raise_for_status()
    files = r.json().get("files", [])
    return [
        {
            "name": f.get("key") or f.get("filename"),
            "url": (f.get("links") or {}).get("self") or (f.get("links") or {}).get("download"),
            "size": f.get("size"),
        }
        for f in files
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true", help="List Zenodo files")
    p.add_argument(
        "--columns", action="store_true", help="Print flow CSV column names after download"
    )
    args = p.parse_args(argv)

    case = get_case("demand_ltown")
    try:
        files = _list_files()
    except Exception as exc:
        print(f"[demand_ltown] Zenodo API error: {exc}")
        return 1

    if args.list:
        for f in files:
            print(f"  {f['name']}  ({f['size']} bytes)")
        return 0

    demand_files = [
        f
        for f in files
        if CONFIG["demands_file_match"].lower() in (f["name"] or "").lower()
        and (f["name"] or "").lower().endswith(".csv")
    ]
    if not demand_files:
        print(
            f"[demand_ltown] no demand CSVs matched {CONFIG['demands_file_match']!r}. Run --list."
        )
        return 1

    frames = []
    for f in demand_files:
        dest = _io.DATA_RAW / case.id / f["name"]
        if not dest.exists():
            print(f"[demand_ltown] downloading {f['name']} ({f['size'] / 1e6:.1f} MB) ...")
            _io.http_download(f["url"], dest)
        frames.append(_io.read_table(dest))
    raw = pd.concat(frames, ignore_index=True)

    if args.columns:
        for c in raw.columns[:30]:
            print(f"  {c}")
        if len(raw.columns) > 30:
            print(f"  ... and {len(raw.columns) - 30} more (per-node demand columns)")
        print(f"\n{len(raw.columns)} columns; {len(raw)} rows")
        return 0

    dcol = next(
        (c for c in raw.columns if c.lower() in {"timestamp", "datetime", "time", "date"}),
        raw.columns[0],
    )
    demand_cols = [c for c in raw.columns if c != dcol]
    print(f"[demand_ltown] summing demand across {len(demand_cols)} consumer nodes")

    total_demand = pd.DataFrame(
        {c: pd.to_numeric(raw[c], errors="coerce") for c in demand_cols}
    ).sum(axis=1)

    out = pd.DataFrame(
        {
            "unique_id": case.id,
            "ds": pd.to_datetime(raw[dcol], utc=True, errors="coerce"),
            "y": total_demand,
        }
    )
    out["inlet_flow"] = out["y"]
    out = out.dropna(subset=["ds"]).sort_values("ds")
    out = _io.resample_regular(out, case.freq)

    path = _io.write_canonical(case, out)
    print(
        f"[demand_ltown] wrote {path} ({len(out)} rows) - total consumer demand "
        f"summed across {len(demand_cols)} nodes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
