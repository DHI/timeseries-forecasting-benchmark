"""Download Bellinge sensor + rain data from DTU Data (figshare) and build canonical.

Bellinge (Nedergaard Pedersen et al. 2021, ESSD), collection DOI 10.11583/DTU.c.5029124.
Relevant figshare articles (discovered via --list earlier):
  - 12509249  "2 - Sensor data"   -> 2_cleaned_data.zip (level/flow), 2_Coordinates_of_sensors.csv
  - 12509774  "3a - Rain gauge"    -> #3a_Raingauges.zip
  - 12513236  "3b - Radar data"    -> 3b_Meterologicalstation.zip (air temp), X-band radar

Pipeline:
  1. Download 2_cleaned_data.zip; extract the level-meter member for CONFIG['sensor_id'].
  2. Download #3a_Raingauges.zip; extract a rain-gauge member.
  3. Resample to 15-min, align, write canonical Parquet.

Run:
    uv run python -m tsforecastbench.data.cases.bellinge_sewer_level.download
    uv run python -m tsforecastbench.data.cases.bellinge_sewer_level.download --list-zip
    uv run python -m tsforecastbench.data.cases.bellinge_sewer_level.download --scan
        --scan reads every CSV member, reports time span and observed fraction per
        sensor instance, so we can pick a continuously-active sensor on evidence
        instead of by guessing the filename.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from tsforecastbench.data.cases import _io
from tsforecastbench.data.registry import get_case

FIGSHARE_API = "https://api.figshare.com/v2"
SENSOR_ARTICLE = 12509249
RAIN_ARTICLE = 12509774

CONFIG = {
    # G71F05R LevelInlet System2000: top of the --scan ranking with 100% active over
    # 10 years (Aug 2010 - Oct 2020). The original choice G71F04R_Level1_System2000p1
    # was only 20% active over the same span (sensor dormant most of the time under
    # the System2000 SCADA system).
    "sensor_id": "G71F05R",
    "sensor_zip_match": "cleaned_data",
    "rain_zip_match": "Raingauges",
    "level_member_match": "G71F05R_LevelInlet_System2000p1",
    # SVK 5425 (DMI standard gauge near Bellinge). Aabakken (the local Bellinge
    # waterworks gauge) only covers 2017-2018; 5425/5427 typically cover the full
    # Bellinge level-series period 2010-2020. If 5425 has gaps, try "5427"; if both
    # are short, concatenating multiple gauges is the next step.
    "rain_member_match": "5425",
    "level_col_candidates": ["level", "water_level", "value", "depth"],
    "rain_col_candidates": ["rain", "intensity", "precip", "value"],
    # Bellinge suburb, Odense, DK — for fetching ERA5 2-m air temperature at the point.
    # The dataset ships a met-station archive too, but ERA5 (credential-free via
    # Open-Meteo) gives a complete, gap-free temperature series over the whole record.
    "lat": 55.32,
    "lon": 10.30,
}


def _file_url(article_id: int, name_match: str):
    import requests

    r = requests.get(f"{FIGSHARE_API}/articles/{article_id}/files", timeout=60)
    r.raise_for_status()
    files = r.json()
    f = next((f for f in files if name_match.lower() in f["name"].lower()), None)
    return f


def _get_zip(case, article_id: int, name_match: str):
    f = _file_url(article_id, name_match)
    if f is None:
        print(f"[bellinge_sewer_level] no file matching {name_match!r} in article {article_id}")
        return None
    dest = _io.DATA_RAW / case.id / f["name"]
    if dest.exists():
        print(f"[bellinge_sewer_level] using cached {f['name']}")
        return dest
    print(f"[bellinge_sewer_level] downloading {f['name']} ({f['size']} bytes) ...")
    return _io.http_download(f["download_url"], dest)


def _pick_col(df, candidates):
    for c in df.columns:
        for cand in candidates:
            if cand.lower() in c.lower():
                return c
    return None


def _parse_svk_rain(path) -> pd.DataFrame:
    """Parse a Bellinge SVK-style rain text file.

    These files have an unstructured header (station info, dashes), then data lines of
    ``date time value`` or ``datetime value``. Pandas read_csv can't handle the header,
    so we walk the file ourselves: skip any line that doesn't start with a digit, then
    parse the remainder as datetime + value with flexible separators.

    Returns columns: ``ds`` (UTC, tz-aware), ``rain_gauge``.
    """
    rows = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            # Normalise separators: semicolon, comma, tab, multiple spaces -> single space
            parts = line.replace(";", " ").replace("\t", " ").split()
            if len(parts) < 2:
                continue
            # Try "date time value" (n=2 date tokens) then "datetime value" (n=1).
            # dayfirst=True because SVK / DMI Danish files use dd-mm-yyyy.
            for n in (2, 1):
                if len(parts) < n + 1:
                    continue
                try:
                    ts = pd.to_datetime(
                        " ".join(parts[:n]), utc=True, dayfirst=True, errors="raise"
                    )
                    val = float(parts[n].replace(",", "."))
                    rows.append((ts, val))
                    break
                except (ValueError, TypeError):
                    continue
    df = pd.DataFrame(rows, columns=["ds", "rain_gauge"])
    # Filter sentinel missing values (typical: -9999 / 999)
    df.loc[df["rain_gauge"] < 0, "rain_gauge"] = float("nan")
    df.loc[df["rain_gauge"] > 500, "rain_gauge"] = float("nan")
    return df


def _scan_coverage(sensor_zip, members):
    """Read every CSV member and report sensor-instance coverage stats."""
    rows = []
    for m in members:
        if not m.lower().endswith(".csv"):
            continue
        try:
            lp = _io.extract_member(sensor_zip, m, _io.DATA_RAW / "_scan_tmp")
            df = _io.read_table(lp)
        except Exception as exc:
            rows.append((m, f"read error: {exc}", 0, 0, 0.0))
            continue
        dcol = next((c for c in df.columns if c.lower() in
                     {"time", "datetime", "date", "ds", "timestamp"}), df.columns[0])
        lcol = _pick_col(df, ["level", "value", "depth"])
        if lcol is None:
            rows.append((m, "no level col", 0, 0, 0.0))
            continue
        ds = pd.to_datetime(df[dcol], errors="coerce")
        v = pd.to_numeric(df[lcol], errors="coerce")
        valid = ds.notna() & v.notna()
        if valid.sum() == 0:
            rows.append((m, "no valid rows", 0, 0, 0.0))
            continue
        ds_v = ds[valid]
        # "active" = non-zero value (sensor returning a real reading, not a sentinel zero)
        active = (v[valid].abs() > 1e-6).sum()
        rows.append((m, f"{ds_v.min().date()}..{ds_v.max().date()}",
                     int(valid.sum()), int(active), float(active / max(1, valid.sum()))))
    rows.sort(key=lambda r: r[3], reverse=True)  # by active rows desc
    print(f"\n{'member':<60s} {'span':<25s} {'rows':>10s} {'active':>10s}  active_frac")
    for m, span, total, active, frac in rows:
        print(f"  {m[-58:]:<58s} {span:<25s} {total:>10d} {active:>10d}  {frac:>6.0%}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list-zip", action="store_true")
    p.add_argument("--scan", action="store_true",
                   help="Scan every CSV member: time span + active (non-zero) row count")
    p.add_argument("--no-era5", action="store_true",
                   help="Skip the ERA5 temp_air fetch (leave it NaN). Offline fallback.")
    args = p.parse_args(argv)

    case = get_case("bellinge_sewer_level")
    try:
        sensor_zip = _get_zip(case, SENSOR_ARTICLE, CONFIG["sensor_zip_match"])
    except Exception as exc:
        print(f"[bellinge_sewer_level] figshare error: {exc}")
        return 1
    if sensor_zip is None:
        return 1

    members = [m for m in _io.list_zip(sensor_zip) if not m.endswith("/")]
    if args.list_zip:
        for m in members[:80]:
            print(f"  {m}")
        print(f"\n{len(members)} members in {sensor_zip.name}")
        return 0

    if args.scan:
        _scan_coverage(sensor_zip, members)
        return 0

    level_member = next((m for m in members if CONFIG["level_member_match"].lower() in m.lower()), None)
    if level_member is None:
        print(f"[bellinge_sewer_level] no member matched {CONFIG['level_member_match']!r}. "
              "Run --list-zip and edit CONFIG.")
        return 1
    lp = _io.extract_member(sensor_zip, level_member, _io.DATA_RAW / case.id / "extracted")
    level_raw = _io.read_table(lp)

    dcol = next((c for c in level_raw.columns if c.lower() in
                 {"time", "datetime", "date", "ds", "timestamp"}), level_raw.columns[0])
    lcol = _pick_col(level_raw, CONFIG["level_col_candidates"])
    if lcol is None:
        print(f"[bellinge_sewer_level] no level column in {list(level_raw.columns)}")
        return 1

    base = pd.DataFrame(
        {
            "unique_id": case.id,
            "ds": pd.to_datetime(level_raw[dcol], utc=True),
            "y": pd.to_numeric(level_raw[lcol], errors="coerce"),
        }
    )
    base["level_m"] = base["y"]
    base = _io.resample_regular(base, case.freq)

    # rain gauge
    try:
        rain_zip = _get_zip(case, RAIN_ARTICLE, CONFIG["rain_zip_match"])
    except Exception:
        rain_zip = None
    if rain_zip is not None:
        all_members = _io.list_zip(rain_zip)
        # Data-looking files only; exclude QC / metadata / event files
        skip_substrings = ("readme", "event", "summary", "missing", "closing",
                           "accepted", "coordinate")
        rmembers = [m for m in all_members
                    if m.lower().endswith((".csv", ".txt", ".dat"))
                    and not any(s in m.lower() for s in skip_substrings)]
        print(f"[bellinge_sewer_level] rain zip: {len(all_members)} members total, "
              f"{len(rmembers)} data-series candidates")
        print("[bellinge_sewer_level] candidates:")
        for m in rmembers[:15]:
            print(f"    {m}")
        # Prefer time-series files (`_ts.`) > minute > hour > other
        priority = ["_ts.", "minute", "1m", "min", "hour", "1h"]
        def _rank(name):
            low = name.lower()
            for i, key in enumerate(priority):
                if key in low:
                    return i
            return len(priority)
        rmembers_sorted = sorted(rmembers, key=_rank)
        rm = next(
            (m for m in rmembers_sorted if CONFIG["rain_member_match"].lower() in m.lower()),
            rmembers_sorted[0] if rmembers_sorted else None,
        )
        if rm:
            print(f"[bellinge_sewer_level] picked rain member: {rm}")
            rp = _io.extract_member(rain_zip, rm, _io.DATA_RAW / case.id / "extracted")
            rain = _parse_svk_rain(rp)
            print(f"[bellinge_sewer_level] rain parsed: {len(rain)} rows, "
                  f"{int(rain['rain_gauge'].notna().sum())} non-NaN, "
                  f"{int((rain['rain_gauge'] > 0).sum())} non-zero, "
                  f"span {rain['ds'].min()} .. {rain['ds'].max()}")
            rain.insert(0, "unique_id", case.id)
            rain = _io.resample_regular(rain, case.freq, agg={"rain_gauge": "sum"})
            print(f"[bellinge_sewer_level] rain resampled: {len(rain)} rows, "
                  f"{int(rain['rain_gauge'].notna().sum())} non-NaN, "
                  f"{int((rain['rain_gauge'] > 0).sum())} non-zero")
            base = base.merge(rain, on=["unique_id", "ds"], how="left")
            print(f"[bellinge_sewer_level] after merge: rain_gauge "
                  f"{int(base['rain_gauge'].notna().sum())} non-NaN of {len(base)}, "
                  f"{int((base['rain_gauge'] > 0).sum())} non-zero")
            base["rain_radar_nowcast"] = base["rain_gauge"]  # placeholder until X-band added

    # air temperature from ERA5 at the Bellinge point (credential-free via Open-Meteo)
    if not args.no_era5 and len(base):
        from tsforecastbench.data.cases import _era5

        print(f"[bellinge_sewer_level] fetching ERA5 temp_air at "
              f"({CONFIG['lat']},{CONFIG['lon']}) "
              f"{base['ds'].min().date()}..{base['ds'].max().date()} via Open-Meteo ...")
        era5 = _era5.fetch_era5_point(
            CONFIG["lat"], CONFIG["lon"], base["ds"].min(), base["ds"].max(),
            variables=["temp_air"], cache_dir=_io.DATA_RAW / case.id / "era5_cache",
        )
        base = _era5.merge_onto(base, era5, ["temp_air"])
        print(f"[bellinge_sewer_level] temp_air coverage {base['temp_air'].notna().mean():.0%}")

    for col in ["rain_gauge", "rain_radar_nowcast", "temp_air"]:
        if col not in base.columns:
            base[col] = float("nan")

    path = _io.write_canonical(case, base)
    print(f"[bellinge_sewer_level] wrote {path} ({len(base)} rows) from {level_member}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
