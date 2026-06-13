"""Download NDBC standard meteorological data for one buoy and build the waves case.

Historical yearly files: https://www.ndbc.noaa.gov/data/historical/stdmet/<id>h<year>.txt.gz
Whitespace-delimited; 2 header lines (#YY MM DD hh mm ... and a units line). Missing
values are sentinels (99, 999, 9999). We keep WVHT (target) + WSPD/PRES/ATMP/WTMP,
build a UTC hourly index, and write canonical Parquet.

Run:
    uv run python -m tsforecastbench.data.cases.ndbc_waves.download
    uv run python -m tsforecastbench.data.cases.ndbc_waves.download --station 46050 --years 2015-2023
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from tsforecastbench.data.cases import _io
from tsforecastbench.data.registry import get_case

BASE = "https://www.ndbc.noaa.gov/data/historical/stdmet"

CONFIG = {
    "station": "46050",        # Stonewall Bank, Oregon - long record, open-ocean waves
    "years": list(range(2015, 2024)),
    # NDBC standard columns -> our canonical names
    "col_map": {"WVHT": "wvht", "WSPD": "wspd", "PRES": "pres", "ATMP": "atmp", "WTMP": "wtmp"},
    # sentinel-by-column: values >= these are missing
    "missing_at": {"wvht": 90, "wspd": 90, "pres": 9000, "atmp": 90, "wtmp": 90},
}


def _read_year(path) -> pd.DataFrame:
    # row 0 = '#YY MM DD hh mm WDIR ...'; row 1 = units (also starts with '#')
    df = pd.read_csv(path, sep=r"\s+", header=0, skiprows=[1], compression="infer")
    df.columns = [c.lstrip("#") for c in df.columns]
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--station", default=CONFIG["station"])
    p.add_argument("--years", default=None, help="e.g. 2015-2023")
    args = p.parse_args(argv)

    case = get_case("ndbc_waves")
    station = args.station
    years = CONFIG["years"]
    if args.years:
        a, b = args.years.split("-")
        years = list(range(int(a), int(b) + 1))

    frames = []
    for yr in years:
        url = f"{BASE}/{station}h{yr}.txt.gz"
        dest = _io.DATA_RAW / case.id / f"{station}h{yr}.txt.gz"
        try:
            if not dest.exists():
                _io.http_download(url, dest)
            frames.append(_read_year(dest))
            print(f"[ndbc_waves] got {station} {yr}")
        except Exception as exc:
            print(f"[ndbc_waves] skip {yr}: {exc}")
    if not frames:
        print("[ndbc_waves] no years downloaded.")
        return 1

    raw = pd.concat(frames, ignore_index=True)
    # build datetime from YY MM DD hh mm
    ycol = "YY" if "YY" in raw.columns else "#YY"
    ds = pd.to_datetime(
        dict(
            year=raw[ycol].astype(int),
            month=raw["MM"].astype(int),
            day=raw["DD"].astype(int),
            hour=raw["hh"].astype(int),
            minute=raw.get("mm", 0),
        ),
        utc=True,
        errors="coerce",
    )
    out = pd.DataFrame({"unique_id": case.id, "ds": ds})
    for src, canon in CONFIG["col_map"].items():
        if src in raw.columns:
            v = pd.to_numeric(raw[src], errors="coerce")
            v = v.where(v < CONFIG["missing_at"].get(canon, 1e9), np.nan)
            out[canon] = v.to_numpy()
        else:
            print(f"[ndbc_waves] WARNING column {src} missing; got {list(raw.columns)}")
    out["y"] = out["wvht"]
    out = out.dropna(subset=["ds"]).sort_values("ds")
    out = _io.resample_regular(out, case.freq)

    path = _io.write_canonical(case, out)
    print(f"[ndbc_waves] wrote {path} ({len(out)} rows) station {station} {years[0]}-{years[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
