"""Build the USGS river-discharge case (Esopus Creek, NY) + ERA5 forcing.

Fetches daily discharge from USGS NWIS (cfs -> m3/s) and daily ERA5 precipitation + air
temperature at the gauge point from Open-Meteo (credential-free). Both observed; the
forcing is exposed as a perfect-forecast proxy.

    uv run python -m tsforecastbench.data.cases.usgs_river_discharge.download
    uv run python -m tsforecastbench.data.cases.usgs_river_discharge.download --no-era5
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from tsforecastbench.data.cases import _era5, _io, _usgs
from tsforecastbench.data.registry import get_case

SITE = "01362500"
CFS_TO_M3S = 0.0283168466
START, END = "2008-01-01", "2020-12-31"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-era5", action="store_true", help="Skip ERA5 forcing (leave NaN).")
    args = p.parse_args(argv)

    case = get_case("usgs_river_discharge")
    info = _usgs.site_info(SITE)
    print(f"[usgs_river_discharge] {info['name']} ({info['lat']}, {info['lon']})")

    q = _usgs.fetch_nwis(SITE, "00060", START, END, service="dv", colname="discharge")
    q["discharge"] = q["discharge"] * CFS_TO_M3S
    print(f"[usgs_river_discharge] {len(q)} daily discharge rows, "
          f"{q['discharge'].notna().mean():.0%} finite")

    base = pd.DataFrame({"unique_id": case.id, "ds": q["ds"], "y": q["discharge"],
                         "discharge": q["discharge"]})
    base = _io.resample_regular(base, case.freq)

    era5_vars = ["precip", "temp_air"]
    if args.no_era5 or info["lon"] is None:
        for c in era5_vars:
            base[c] = float("nan")
        print("[usgs_river_discharge] ERA5 skipped.")
    else:
        print(f"[usgs_river_discharge] fetching ERA5 {era5_vars} (daily) via Open-Meteo ...")
        hourly = _era5.fetch_era5_point(info["lat"], info["lon"], base["ds"].min(),
                                        base["ds"].max(), variables=era5_vars,
                                        cache_dir=_io.DATA_RAW / case.id / "era5_cache")
        daily = _era5.to_daily(hourly, era5_vars)
        base = _era5.merge_onto(base, daily, era5_vars, interpolate=False)
        cov = {c: f"{base[c].notna().mean():.0%}" for c in era5_vars}
        print(f"[usgs_river_discharge] ERA5 merged; coverage {cov}")

    path = _io.write_canonical(case, base)
    print(f"[usgs_river_discharge] wrote {path} ({len(base)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
