"""Download BattLeDIM L-Town SCADA from Zenodo and build the inlet-flow case.

Zenodo record 4017659 contains 2018 historical + 2019 evaluation SCADA CSVs
(pressures, flows, demands, levels), plus the EPANET model.

Target is the *total system demand* = sum of all PUMP_* inlet flow columns. L-Town
has multiple inlet pumps; any single one alternates with the others (operational
pump scheduling), so the single-pump signal is intermittent and not what you want
to forecast. The sum of pumps is the real demand signal at hourly resolution.

Run:
    uv run python -m tsforecastbench.cases.wd_battledim.download --list
    uv run python -m tsforecastbench.cases.wd_battledim.download --columns   # print flow columns
    uv run python -m tsforecastbench.cases.wd_battledim.download
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from tsforecastbench.cases import _io
from tsforecastbench.registry import get_case

ZENODO_RECORD = 4017659
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

CONFIG = {
    "flow_file_match": "Flows",
    # Sum all columns whose name matches one of these prefixes. PUMP_1 + PUMP_2 + ...
    # captures total network demand even when pumps alternate.
    "pump_col_prefixes": ["PUMP", "Pump"],
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
    p.add_argument("--columns", action="store_true",
                   help="Print flow CSV column names after download")
    args = p.parse_args(argv)

    case = get_case("wd_battledim")
    try:
        files = _list_files()
    except Exception as exc:  # noqa: BLE001
        print(f"[wd_battledim] Zenodo API error: {exc}")
        return 1

    if args.list:
        for f in files:
            print(f"  {f['name']}  ({f['size']} bytes)")
        return 0

    flow_files = [f for f in files if CONFIG["flow_file_match"].lower() in (f["name"] or "").lower()
                  and (f["name"] or "").lower().endswith(".csv")]
    if not flow_files:
        print(f"[wd_battledim] no flow CSVs matched {CONFIG['flow_file_match']!r}. Run --list.")
        return 1

    frames = []
    for f in flow_files:
        dest = _io.DATA_RAW / case.id / f["name"]
        if not dest.exists():
            print(f"[wd_battledim] downloading {f['name']} ({f['size']} bytes) ...")
            _io.http_download(f["url"], dest)
        frames.append(_io.read_table(dest))
    raw = pd.concat(frames, ignore_index=True)

    if args.columns:
        for c in raw.columns:
            print(f"  {c}")
        return 0

    dcol = next((c for c in raw.columns if c.lower() in {"timestamp", "datetime", "time", "date"}),
                raw.columns[0])
    pump_cols = [c for c in raw.columns
                 if any(c.upper().startswith(p.upper()) for p in CONFIG["pump_col_prefixes"])]
    if not pump_cols:
        print(f"[wd_battledim] no pump columns found (prefixes "
              f"{CONFIG['pump_col_prefixes']}). Columns: {list(raw.columns)}")
        return 1
    print(f"[wd_battledim] summing pumps: {pump_cols}")

    flow_total = sum(pd.to_numeric(raw[c], errors="coerce").fillna(0.0) for c in pump_cols)

    out = pd.DataFrame(
        {
            "unique_id": case.id,
            "ds": pd.to_datetime(raw[dcol], utc=True, errors="coerce"),
            "y": flow_total,
        }
    )
    out["inlet_flow"] = out["y"]
    out = out.dropna(subset=["ds"]).sort_values("ds")
    out = _io.resample_regular(out, case.freq)

    path = _io.write_canonical(case, out)
    print(f"[wd_battledim] wrote {path} ({len(out)} rows) - total inlet flow = "
          f"sum({', '.join(pump_cols)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
