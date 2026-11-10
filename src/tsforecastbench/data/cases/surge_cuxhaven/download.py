"""Build the GESLA storm-surge (non-tidal residual) case for one tide gauge.

The GESLA bulk archive (~6.8 GB zip, ~50 GB unzipped) holds one record file per station.
We use the published metadata CSV to pick a station, extract just that station's record
from the zip (no full unpack), remove the astronomical tide via UTide harmonic analysis,
and write canonical Parquet.

Setup
-----
Download from https://gesla787883612.wordpress.com/downloads/ :
  - GESLA*_ALL.zip   (the bulk record archive)
  - GESLA*_ALL.csv   (station metadata: name, code, lat, lon, span)
Save both under data/raw/surge_cuxhaven/.

Then:
    uv run python -m tsforecastbench.data.cases.surge_cuxhaven.download --scan
        # prints the metadata columns and ranks Danish + North Sea candidates
        # by record length
    uv run python -m tsforecastbench.data.cases.surge_cuxhaven.download \\
        --station-code <code>
        # extracts that station's record file from the zip, parses it,
        # subtracts the astronomical tide, writes data/processed/surge_cuxhaven.parquet
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from tsforecastbench.data.cases import _io
from tsforecastbench.data.registry import get_case

CONFIG = {
    # Cuxhaven_Steubenhoft (German Bight). Picked by --scan over the UK + Atlantic NW
    # Europe bounding box: 108 years (1917-2024), shallow North Sea shelf, surge-rich
    # operational climate (the 1962 Hamburg flood site). Alternative pin candidates:
    # 'NEW' (Newlyn, GBR, 110 yrs, Atlantic) or 'TRG' (Tregde, NOR, 99 yrs, Skagerrak).
    "station_code": "Cux",
    # UK + Ireland + Atlantic-facing NW Europe (incl. Danish North Sea coast through
    # Hirtshals). Deliberately stops at lon 11 E to exclude the Baltic / Kattegat,
    # whose long-period oscillations and freshwater forcing are different dynamics.
    "scan_bbox": {"lat": (49.0, 60.0), "lon": (-11.0, 11.0)},
    "scan_top_n": 25,
}


def _find_metadata_csv() -> Path | None:
    raw_dir = _io.DATA_RAW / "surge_cuxhaven"
    for p in raw_dir.glob("*.csv"):
        return p
    return None


def _find_bulk_zip() -> Path | None:
    raw_dir = _io.DATA_RAW / "surge_cuxhaven"
    for p in raw_dir.glob("*.zip"):
        return p
    return None


def _col(cols_lookup, *keys):
    for k in keys:
        if k in cols_lookup:
            return cols_lookup[k]
    return None


def _scan(meta_csv: Path) -> int:
    df = pd.read_csv(meta_csv)
    print(f"\n[scan] {meta_csv.name}: {len(df)} stations, {len(df.columns)} columns")
    print(f"[scan] columns: {list(df.columns)}\n")

    cols = {c.lower(): c for c in df.columns}
    c_lat = _col(cols, "latitude", "lat")
    c_lon = _col(cols, "longitude", "lon", "lng")
    c_name = _col(cols, "site name", "site_name", "station_name", "name")
    c_code = _col(
        cols, "site code", "site_code", "station_code", "code", "file name", "file_name", "filename"
    )
    c_country = _col(cols, "country")
    c_start = _col(cols, "start date/time", "start_date", "start date", "start", "earliest")
    c_end = _col(cols, "end date/time", "end_date", "end date", "end", "latest")
    c_years = _col(
        cols,
        "number of years",
        "number_of_years",
        "n_years",
        "years",
        "record_length",
        "length_years",
    )

    if c_lat is None or c_lon is None:
        print(
            "[scan] could not auto-detect lat/lon columns; printing the first 5 rows so "
            "you can edit the scan code to match the actual schema:"
        )
        print(df.head(5).to_string())
        return 0

    box = CONFIG["scan_bbox"]
    sel = df[df[c_lat].between(*box["lat"]) & df[c_lon].between(*box["lon"])].copy()
    print(f"[scan] {len(sel)} stations in box lat {box['lat']} lon {box['lon']}")

    rank_col = None
    if c_years and pd.api.types.is_numeric_dtype(sel[c_years]):
        sel = sel.sort_values(c_years, ascending=False)
        rank_col = c_years
    elif c_start and c_end:
        sel["__span_days"] = (
            pd.to_datetime(sel[c_end], errors="coerce")
            - pd.to_datetime(sel[c_start], errors="coerce")
        ).dt.days
        sel = sel.sort_values("__span_days", ascending=False)
        rank_col = "__span_days"

    # GESLA keeps both historic ("294") and real-time ("294A") series for the same
    # gauge - dedupe by site name, keeping the longest-record version.
    if c_name and rank_col:
        sel = sel.drop_duplicates(subset=c_name, keep="first")

    show = [
        c
        for c in (c_code, c_name, c_country, c_lat, c_lon, c_start, c_end, c_years, rank_col)
        if c is not None
    ]
    show = list(dict.fromkeys(show))
    n = min(CONFIG["scan_top_n"], len(sel))
    print(f"\n[scan] top {n} ranked by {rank_col!r}:")
    with pd.option_context("display.max_colwidth", 60, "display.width", 220):
        print(sel[show].head(n).to_string(index=False))
    if c_code:
        print(f"\n[scan] pick one and re-run with --station-code <{c_code}>")
    return 0


def _filename_from_metadata(meta_csv: Path, station_code: str) -> str | None:
    """Look up the FILE NAME for the given SITE CODE in the metadata CSV."""
    meta = pd.read_csv(meta_csv)
    cols = {c.lower(): c for c in meta.columns}
    c_code = _col(cols, "site code", "site_code", "station_code", "code")
    c_file = _col(cols, "file name", "file_name", "filename")
    if not c_code or not c_file:
        return None
    hit = meta[meta[c_code].astype(str).str.strip() == station_code.strip()]
    if hit.empty:
        # case-insensitive fall-back
        hit = meta[meta[c_code].astype(str).str.lower() == station_code.lower()]
    if hit.empty:
        return None
    return str(hit[c_file].iloc[0]).strip()


def _find_station_in_zip(
    bulk_zip: Path, station_code: str, meta_csv: Path | None = None
) -> str | None:
    """Find the member in the bulk zip for the given station code.

    Strategy:
      1. If we have metadata, look up the exact FILE NAME for this SITE CODE and
         match a member by that filename (most reliable).
      2. Otherwise fall back to exact basename match on the code.
      3. Finally a case-insensitive substring search (last resort; can be ambiguous
         when multiple GESLA series exist for the same physical gauge).
    """
    with zipfile.ZipFile(bulk_zip) as zf:
        members = zf.namelist()

    if meta_csv is not None:
        wanted = _filename_from_metadata(meta_csv, station_code)
        if wanted:
            wl = wanted.lower()
            for m in members:
                if Path(m).name.lower() == wl:
                    return m
            for m in members:
                if Path(m).name.lower().endswith(wl):
                    return m
            print(
                f"[surge_cuxhaven] metadata FILE NAME {wanted!r} not found in zip; "
                "falling back to code substring search."
            )

    code_l = station_code.lower()
    for m in members:
        if Path(m).name.lower() == code_l:
            return m
    for m in members:
        if code_l in Path(m).name.lower():
            return m
    return None


def _read_gesla_record(path: Path) -> pd.DataFrame:
    """Parse a GESLA-3/4 text record file.

    Format: ~40 header lines starting with '#', then whitespace-delimited records of
        date  time  sea_level  qc_flag  use_flag
    Sentinels (-99.9999) become NaN, and so does every value GESLA marks not for use
    (use_flag 0). GESLA derives that flag from the provider's QC flag: doubtful values,
    isolated spikes or wrong values, and missing values (QC 3, 4, 5) are not for use.
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
            try:
                sl = float(parts[2])
            except ValueError:
                continue
            use = int(parts[4]) if len(parts) >= 5 else 1
            rows.append((ts, sl, use))
    df = pd.DataFrame(rows, columns=["ds", "sea_level", "use_flag"])
    df.loc[df["sea_level"] < -99, "sea_level"] = np.nan
    n_rejected = int(((df["use_flag"] == 0) & df["sea_level"].notna()).sum())
    if n_rejected:
        print(f"[surge_cuxhaven] {n_rejected:,} values flagged not for use by GESLA -> NaN")
    df.loc[df["use_flag"] == 0, "sea_level"] = np.nan
    return df.drop(columns="use_flag")


def _tidal_residual(df: pd.DataFrame, lat: float) -> pd.DataFrame:
    try:
        import utide
    except ImportError as e:
        raise RuntimeError(
            "utide is required for GESLA tide removal. Install via: uv sync --extra data"
        ) from e

    t = df["ds"].dt.tz_convert("UTC").dt.tz_localize(None)
    coef = utide.solve(
        t.to_numpy(),
        df["sea_level"].to_numpy(),
        lat=lat,
        method="ols",
        conf_int="none",
        verbose=False,
    )
    tide = utide.reconstruct(t.to_numpy(), coef, verbose=False)
    out = df.copy()
    out["residual_m"] = df["sea_level"].to_numpy() - tide.h
    return out


def _latlon_from_metadata(meta_csv: Path, station_code: str) -> tuple[float | None, float | None]:
    meta = pd.read_csv(meta_csv)
    cols = {c.lower(): c for c in meta.columns}
    c_code = _col(cols, "site_code", "site code", "station_code", "code", "file_name", "filename")
    c_lat = _col(cols, "latitude", "lat")
    c_lon = _col(cols, "longitude", "lon", "lng")
    if not c_code or not c_lat:
        return None, None
    hit = meta[meta[c_code].astype(str).str.lower() == station_code.lower()]
    if hit.empty:
        return None, None
    lat = float(hit[c_lat].iloc[0])
    lon = float(hit[c_lon].iloc[0]) if c_lon else None
    return lat, lon


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scan",
        action="store_true",
        help="Read the metadata CSV, print Danish/North Sea candidates",
    )
    p.add_argument("--station-code", default=CONFIG["station_code"])
    p.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Station latitude for UTide (auto-detected from metadata if absent)",
    )
    p.add_argument(
        "--lon",
        type=float,
        default=None,
        help="Station longitude for ERA5 (auto-detected from metadata if absent)",
    )
    p.add_argument(
        "--era5-start",
        default="2000-01-01",
        help="Trim the residual record to this start so ERA5 forcing covers it "
        "(the full GESLA record is ~106 yr).",
    )
    p.add_argument(
        "--no-era5",
        action="store_true",
        help="Skip the ERA5 fetch (leave exogenous NaN). Offline fallback.",
    )
    args = p.parse_args(argv)

    case = get_case("surge_cuxhaven")
    raw_dir = _io.DATA_RAW / case.id
    raw_dir.mkdir(parents=True, exist_ok=True)

    meta_csv = _find_metadata_csv()
    if meta_csv is None:
        print(f"[surge_cuxhaven] no metadata CSV in {raw_dir}/. Place GESLA*_ALL.csv there.")
        return 1

    if args.scan:
        return _scan(meta_csv)

    if not args.station_code:
        print(
            "[surge_cuxhaven] --station-code <code> is required "
            "(run --scan first to see candidates)."
        )
        return 1

    bulk_zip = _find_bulk_zip()
    if bulk_zip is None:
        print(f"[surge_cuxhaven] no bulk zip in {raw_dir}/. Place GESLA*_ALL.zip there.")
        return 1

    member = _find_station_in_zip(bulk_zip, args.station_code, meta_csv)
    if member is None:
        print(f"[surge_cuxhaven] no member matching {args.station_code!r} in {bulk_zip.name}.")
        return 1

    extracted = _io.extract_member(bulk_zip, member, raw_dir / "extracted")
    print(f"[surge_cuxhaven] extracted {member}")

    lat = (
        args.lat if args.lat is not None else _latlon_from_metadata(meta_csv, args.station_code)[0]
    )
    lon = (
        args.lon if args.lon is not None else _latlon_from_metadata(meta_csv, args.station_code)[1]
    )
    if lat is None:
        print("[surge_cuxhaven] no latitude (pass --lat); UTide cannot run without it.")
        return 1
    print(f"[surge_cuxhaven] using station lat {lat:.3f}, lon {lon}")

    df = _read_gesla_record(extracted)
    # Trim to the ERA5-covered window before the (expensive) tide fit. The full record is
    # ~106 yr, and ERA5 forcing should span the whole series.
    start = pd.Timestamp(args.era5_start, tz="UTC")
    df = df[df["ds"] >= start].reset_index(drop=True)
    print(f"[surge_cuxhaven] {len(df)} records from {args.era5_start}; removing tide via UTide ...")
    df = _tidal_residual(df, lat=lat)

    out = pd.DataFrame(
        {
            "unique_id": case.id,
            "ds": df["ds"],
            "y": df["residual_m"],
            "residual_m": df["residual_m"],
        }
    )
    out = _io.resample_regular(out, case.freq)

    era5_vars = ["mslp", "u10", "v10"]
    if args.no_era5 or lon is None:
        for col in era5_vars:
            out[col] = float("nan")
        print("[surge_cuxhaven] ERA5 skipped; exogenous left NaN.")
    else:
        from tsforecastbench.data.cases import _era5

        print(
            f"[surge_cuxhaven] fetching ERA5 {era5_vars} at ({lat:.3f},{lon:.3f}) "
            f"{out['ds'].min().date()}..{out['ds'].max().date()} via Open-Meteo ..."
        )
        era5 = _era5.fetch_era5_point(
            lat,
            lon,
            out["ds"].min(),
            out["ds"].max(),
            variables=era5_vars,
            cache_dir=raw_dir / "era5_cache",
        )
        out = _era5.merge_onto(out, era5, era5_vars)
        cov = {c: f"{out[c].notna().mean():.0%}" for c in era5_vars}
        print(f"[surge_cuxhaven] ERA5 merged; coverage {cov}")

    path = _io.write_canonical(case, out)
    exo_state = "ERA5 exogenous NaN (--no-era5)" if args.no_era5 else "with ERA5 exogenous"
    print(f"[surge_cuxhaven] wrote {path} ({len(out)} rows), {exo_state}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
