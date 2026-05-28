"""Build the GESLA-3 storm-surge (non-tidal residual) case for one tide gauge.

GESLA-3 (Haigh et al. 2023) is distributed as a large bundle of per-station files
from https://gesla787883612.wordpress.com/downloads/ (the bulk archive is behind a
manual download link, so we don't automate the full fetch). This script expects the
single station file to already be present under data/raw/gesla_storm_surge/.

Pipeline:
  1. Read the GESLA-3 station file (header + whitespace-delimited records).
  2. Remove the tide with UTide harmonic analysis -> non-tidal residual (the target).
  3. (Optional) Merge ERA5 MSLP + 10 m winds if an ERA5 extract is present; otherwise
     write the residual alone and fill exogenous with NaN (add ERA5 in a later step).
  4. Write canonical Parquet.

ERA5 forcing needs a CDS account + the `cdsapi` package and a ~/.cdsapirc key; see
fetch_era5() below. Tide removal needs `utide` (pip install utide).

Run (after placing the station file in data/raw/gesla_storm_surge/):
    uv run python -m tsforecastbench.cases.gesla_storm_surge.download --station-file <name>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tsforecastbench.cases import _io
from tsforecastbench.registry import get_case

GESLA_DOWNLOADS = "https://gesla787883612.wordpress.com/downloads/"


def _read_gesla3(path: Path) -> pd.DataFrame:
    """Parse a GESLA-3 station file: ~41 header lines starting with '#', then records.

    Record columns: date time sea_level qc_flag use_flag
    """
    rows = []
    with path.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            ts = pd.to_datetime(parts[0] + " " + parts[1], utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            rows.append((ts, float(parts[2])))
    df = pd.DataFrame(rows, columns=["ds", "sea_level"])
    # GESLA uses -99.9999 as a null sentinel
    df.loc[df["sea_level"] < -99, "sea_level"] = np.nan
    return df


def _tidal_residual(df: pd.DataFrame, lat: float) -> pd.DataFrame:
    """Subtract the astronomical tide via UTide; return non-tidal residual."""
    import utide

    t = df["ds"].dt.tz_convert("UTC").dt.tz_localize(None)
    coef = utide.solve(t.to_numpy(), df["sea_level"].to_numpy(), lat=lat,
                       method="ols", conf_int="none", verbose=False)
    tide = utide.reconstruct(t.to_numpy(), coef, verbose=False)
    out = df.copy()
    out["residual_m"] = df["sea_level"].to_numpy() - tide.h
    return out


def fetch_era5(case, years: list[int], bbox: tuple[float, float, float, float]) -> Path | None:
    """Download ERA5 hourly MSLP + 10m winds for the station neighbourhood via cdsapi.

    Requires a CDS account and ~/.cdsapirc. Returns the NetCDF path, or None if cdsapi
    is unavailable. Wire the extracted point series into the canonical frame as
    mslp/u10/v10 (+ *_forecast for the known-future tier).
    """
    try:
        import cdsapi
    except ImportError:
        print("[gesla_storm_surge] cdsapi not installed; skipping ERA5 (exogenous=NaN).")
        return None
    dest = _io.DATA_RAW / case.id / "era5_mslp_wind.nc"
    dest.parent.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ["mean_sea_level_pressure", "10m_u_component_of_wind",
                         "10m_v_component_of_wind"],
            "year": [str(y) for y in years],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": list(bbox),  # N, W, S, E
            "format": "netcdf",
        },
        str(dest),
    )
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--station-file", help="GESLA-3 station filename under data/raw/<case>/")
    p.add_argument("--lat", type=float, default=55.46, help="Station latitude (for UTide)")
    args = p.parse_args(argv)

    case = get_case("gesla_storm_surge")
    raw_dir = _io.DATA_RAW / case.id

    if not args.station_file:
        print(f"[gesla_storm_surge] download the bulk archive from {GESLA_DOWNLOADS},")
        print(f"place a station file under {raw_dir}/ and re-run with --station-file <name>.")
        return 1

    station = raw_dir / args.station_file
    if not station.exists():
        print(f"[gesla_storm_surge] {station} not found.")
        return 1

    df = _read_gesla3(station)
    print(f"[gesla_storm_surge] read {len(df)} records; removing tide via UTide...")
    df = _tidal_residual(df, lat=args.lat)

    out = pd.DataFrame(
        {"unique_id": case.id, "ds": df["ds"], "y": df["residual_m"], "residual_m": df["residual_m"]}
    )
    out = _io.resample_regular(out, case.freq)

    # exogenous placeholders until ERA5 is merged (see fetch_era5)
    for col in ["mslp", "u10", "v10", "mslp_forecast", "u10_forecast", "v10_forecast"]:
        out[col] = float("nan")

    path = _io.write_canonical(case, out)
    print(f"[gesla_storm_surge] wrote {path} ({len(out)} rows). "
          "ERA5 exogenous still NaN — run fetch_era5() to fill.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
