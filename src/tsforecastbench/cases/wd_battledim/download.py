"""Download BattLeDIM L-Town SCADA from Zenodo and build the inlet-flow case.

Zenodo record 4017659 contains 2018 historical + 2019 evaluation SCADA CSVs
(pressures, flows, demands, levels), plus the EPANET model.

We pull the flows CSVs for both years, find an inlet column, concatenate,
resample 5-min -> hourly, and write canonical Parquet.

Run:
    uv run python -m tsforecastbench.cases.wd_battledim.download --list
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
    "flow_file_match": "Flows",       # picks 2018_SCADA_Flows.csv / 2019_SCADA_Flows.csv
    "inlet_col_candidates": ["n1", "n54", "n105", "inlet", "PUMP", "Pump"],
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
    p.add_argument("--list", action="store_true")
    args = p.parse_args(argv)

    case = get_case("wd_battledim")
    try:
        files = _list_files()
    except Exception as exc:
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

    dcol = next((c for c in raw.columns if c.lower() in {"timestamp", "datetime", "time", "date"}),
                raw.columns[0])
    inlet = next((c for c in raw.columns for cand in CONFIG["inlet_col_candidates"]
                  if cand.lower() in c.lower()), None)
    if inlet is None:
        print(f"[wd_battledim] no inlet column auto-detected. Columns: {list(raw.columns)}")
        return 1

    out = pd.DataFrame(
        {
            "unique_id": case.id,
            "ds": pd.to_datetime(raw[dcol], utc=True, errors="coerce"),
            "y": pd.to_numeric(raw[inlet], errors="coerce"),
        }
    )
    out["inlet_flow"] = out["y"]
    out = out.dropna(subset=["ds"]).sort_values("ds")
    out = _io.resample_regular(out, case.freq)

    path = _io.write_canonical(case, out)
    print(f"[wd_battledim] wrote {path} ({len(out)} rows) inlet={inlet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
