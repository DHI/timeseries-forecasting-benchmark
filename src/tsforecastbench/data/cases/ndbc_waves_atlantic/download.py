"""Build the NW Atlantic wave case (NDBC buoy 44008, Nantucket SE).

    uv run python -m tsforecastbench.data.cases.ndbc_waves_atlantic.download
    uv run python -m tsforecastbench.data.cases.ndbc_waves_atlantic.download --station 44008 --years 2014-2023
"""

from __future__ import annotations

import argparse
import sys

from tsforecastbench.data.cases import _io, _ndbc
from tsforecastbench.data.registry import get_case

CONFIG = {"station": "44013", "years": list(range(2015, 2024))}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--station", default=CONFIG["station"])
    p.add_argument("--years", default=None, help="e.g. 2014-2023")
    args = p.parse_args(argv)

    case = get_case("ndbc_waves_atlantic")
    years = CONFIG["years"]
    if args.years:
        a, b = args.years.split("-")
        years = list(range(int(a), int(b) + 1))

    out = _ndbc.build_ndbc(case, args.station, years)
    path = _io.write_canonical(case, out)
    print(f"[ndbc_waves_atlantic] wrote {path} ({len(out)} rows) station {args.station} "
          f"{years[0]}-{years[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
