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

from tsforecastbench.data.cases import _io, _ndbc
from tsforecastbench.data.registry import get_case

CONFIG = {
    "station": "46050",        # Stonewall Bank, Oregon - long record, open-ocean swell
    "years": list(range(2015, 2024)),
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--station", default=CONFIG["station"])
    p.add_argument("--years", default=None, help="e.g. 2015-2023")
    args = p.parse_args(argv)

    case = get_case("ndbc_waves")
    years = CONFIG["years"]
    if args.years:
        a, b = args.years.split("-")
        years = list(range(int(a), int(b) + 1))

    out = _ndbc.build_ndbc(case, args.station, years)
    path = _io.write_canonical(case, out)
    print(f"[ndbc_waves] wrote {path} ({len(out)} rows) station {args.station} {years[0]}-{years[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
