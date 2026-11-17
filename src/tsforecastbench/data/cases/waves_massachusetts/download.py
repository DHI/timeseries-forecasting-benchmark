"""Build the NW Atlantic wave case (NDBC buoy 44013, Boston / Massachusetts Bay).

The buoy's wind speed and pressure have gaps of up to 1818 h. They are filled from ERA5
10-m wind speed and mean sea-level pressure at the buoy, mapped onto the buoy's scale on
the pre-test record (gh #119).

uv run python -m tsforecastbench.data.cases.waves_massachusetts.download
uv run python -m tsforecastbench.data.cases.waves_massachusetts.download --station 44013 --years 2014-2023
"""

from __future__ import annotations

import argparse
import sys

from tsforecastbench.data.cases import _era5, _io, _ndbc
from tsforecastbench.data.curate import curate_frame
from tsforecastbench.data.registry import get_case

CONFIG = {
    "station": "44013",
    "years": list(range(2015, 2024)),
    "lat": 42.346,
    "lon": -70.651,
}
ERA5_FOR = {"wspd": "wspd10", "pres": "mslp"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--station", default=CONFIG["station"])
    p.add_argument("--years", default=None, help="e.g. 2014-2023")
    args = p.parse_args(argv)

    case = get_case("waves_massachusetts")
    years = CONFIG["years"]
    if args.years:
        a, b = args.years.split("-")
        years = list(range(int(a), int(b) + 1))

    out = _ndbc.build_ndbc(case, args.station, years)

    _, info = curate_frame(
        out,
        case.freq,
        seasonalities=case.seasonalities,
        horizon_primary=case.horizon_primary,
        split_fraction=case.split_fraction,
    )
    era5 = _era5.fetch_era5_point(
        CONFIG["lat"],
        CONFIG["lon"],
        out["ds"].min(),
        out["ds"].max(),
        variables=list(ERA5_FOR.values()),
        cache_dir=_io.DATA_RAW / case.id / "era5_cache",
    )
    out, fill = _era5.fill_gaps_from_era5(out, era5, ERA5_FOR, fit_until=info["val_end"])
    for col, f in fill.items():
        print(
            f"[waves_massachusetts] {col}: filled {f['n_filled']} h from ERA5 "
            f"({f['slope']:.3f} * era5 + {f['intercept']:.2f}, fitted on {f['n_fit']} h)"
        )

    path = _io.write_canonical(case, out)
    print(
        f"[waves_massachusetts] wrote {path} ({len(out)} rows) station {args.station} "
        f"{years[0]}-{years[-1]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
