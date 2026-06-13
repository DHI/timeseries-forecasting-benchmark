"""Build the USGS groundwater-level case (well We-5020, NY) + ERA5 forcing.

Daily depth-to-water from USGS NWIS (parameter 72019, ft -> m) plus daily ERA5
precipitation + air temperature at the well point from Open-Meteo (credential-free).

    uv run python -m tsforecastbench.data.cases.usgs_groundwater.download
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from tsforecastbench.data.cases import _era5, _io, _usgs
from tsforecastbench.data.registry import get_case

SITE = "411421073481202"
FT_TO_M = 0.3048
START, END = "2008-01-01", "2020-12-31"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-era5", action="store_true", help="Skip ERA5 forcing (leave NaN).")
    args = p.parse_args(argv)

    case = get_case("usgs_groundwater")
    info = _usgs.site_info(SITE)
    print(f"[usgs_groundwater] {info['name']} ({info['lat']}, {info['lon']})")

    g = _usgs.fetch_nwis(SITE, "72019", START, END, service="dv", colname="gwl_depth_m")
    g["gwl_depth_m"] = g["gwl_depth_m"] * FT_TO_M
    print(f"[usgs_groundwater] {len(g)} daily rows, {g['gwl_depth_m'].notna().mean():.0%} finite")

    base = pd.DataFrame({"unique_id": case.id, "ds": g["ds"], "y": g["gwl_depth_m"],
                         "gwl_depth_m": g["gwl_depth_m"]})
    base = _io.resample_regular(base, case.freq)

    era5_vars = ["precip", "temp_air"]
    if args.no_era5 or info["lon"] is None:
        for c in era5_vars:
            base[c] = float("nan")
        print("[usgs_groundwater] ERA5 skipped.")
    else:
        print(f"[usgs_groundwater] fetching ERA5 {era5_vars} (daily) via Open-Meteo ...")
        hourly = _era5.fetch_era5_point(info["lat"], info["lon"], base["ds"].min(),
                                        base["ds"].max(), variables=era5_vars,
                                        cache_dir=_io.DATA_RAW / case.id / "era5_cache")
        daily = _era5.to_daily(hourly, era5_vars)
        base = _era5.merge_onto(base, daily, era5_vars, interpolate=False)
        cov = {c: f"{base[c].notna().mean():.0%}" for c in era5_vars}
        print(f"[usgs_groundwater] ERA5 merged; coverage {cov}")

    path = _io.write_canonical(case, base)
    print(f"[usgs_groundwater] wrote {path} ({len(base)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
