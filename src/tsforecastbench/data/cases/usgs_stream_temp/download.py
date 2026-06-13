"""Build the USGS stream-temperature case (James River, VA) + ERA5 forcing.

    uv run python -m tsforecastbench.data.cases.usgs_stream_temp.download
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from tsforecastbench.data.cases import _era5, _io, _usgs
from tsforecastbench.data.registry import get_case

SITE = "02035000"
START, END = "2011-01-01", "2020-12-31"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-era5", action="store_true", help="Skip ERA5 forcing (leave NaN).")
    args = p.parse_args(argv)

    case = get_case("usgs_stream_temp")
    info = _usgs.site_info(SITE)
    print(f"[usgs_stream_temp] {info['name']} ({info['lat']}, {info['lon']})")

    t = _usgs.fetch_nwis(SITE, "00010", START, END, service="dv", colname="water_temp")
    print(f"[usgs_stream_temp] {len(t)} daily rows, {t['water_temp'].notna().mean():.0%} finite")

    base = pd.DataFrame({"unique_id": case.id, "ds": t["ds"], "y": t["water_temp"],
                         "water_temp": t["water_temp"]})
    base = _io.resample_regular(base, case.freq)

    era5_vars = ["temp_air", "precip"]
    if args.no_era5 or info["lon"] is None:
        for c in era5_vars:
            base[c] = float("nan")
        print("[usgs_stream_temp] ERA5 skipped.")
    else:
        print(f"[usgs_stream_temp] fetching ERA5 {era5_vars} (daily) via Open-Meteo ...")
        hourly = _era5.fetch_era5_point(info["lat"], info["lon"], base["ds"].min(),
                                        base["ds"].max(), variables=era5_vars,
                                        cache_dir=_io.DATA_RAW / case.id / "era5_cache")
        daily = _era5.to_daily(hourly, era5_vars)
        base = _era5.merge_onto(base, daily, era5_vars, interpolate=False)
        cov = {c: f"{base[c].notna().mean():.0%}" for c in era5_vars}
        print(f"[usgs_stream_temp] ERA5 merged; coverage {cov}")

    path = _io.write_canonical(case, base)
    print(f"[usgs_stream_temp] wrote {path} ({len(base)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
