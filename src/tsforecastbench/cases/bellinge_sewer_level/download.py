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
    uv run python -m tsforecastbench.cases.bellinge_sewer_level.download
    uv run python -m tsforecastbench.cases.bellinge_sewer_level.download --list-zip
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from tsforecastbench.cases import _io
from tsforecastbench.registry import get_case

FIGSHARE_API = "https://api.figshare.com/v2"
SENSOR_ARTICLE = 12509249
RAIN_ARTICLE = 12509774

CONFIG = {
    "sensor_id": "G71F04R",  # paper F04 location, combined sewer
    "sensor_zip_match": "cleaned_data",
    "rain_zip_match": "Raingauges",
    "level_member_match": "G71F04R_Level1_System2000p1",  # continuous in-sewer level
    "rain_member_match": "",     # take first member if empty
    "level_col_candidates": ["F4", "level", "water_level", "value", "depth"],
    "rain_col_candidates": ["rain", "intensity", "precip", "value"],
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list-zip", action="store_true")
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
        rmembers = [m for m in _io.list_zip(rain_zip)
                    if m.lower().endswith(".csv") and "readme" not in m.lower()]
        rm = next((m for m in rmembers if CONFIG["rain_member_match"].lower() in m.lower()),
                  rmembers[0] if rmembers else None)
        if rm:
            rp = _io.extract_member(rain_zip, rm, _io.DATA_RAW / case.id / "extracted")
            rain_raw = _io.read_table(rp)
            rdcol = next((c for c in rain_raw.columns if c.lower() in
                          {"time", "datetime", "date", "ds", "timestamp"}), rain_raw.columns[0])
            rcol = _pick_col(rain_raw, CONFIG["rain_col_candidates"])
            if rcol:
                rain = pd.DataFrame(
                    {
                        "unique_id": case.id,
                        "ds": pd.to_datetime(rain_raw[rdcol], utc=True),
                        "rain_gauge": pd.to_numeric(rain_raw[rcol], errors="coerce"),
                    }
                )
                rain = _io.resample_regular(rain, case.freq, agg={"rain_gauge": "sum"})
                base = base.merge(rain, on=["unique_id", "ds"], how="left")
                base["rain_radar_nowcast"] = base["rain_gauge"]  # placeholder until X-band added

    for col in ["rain_gauge", "rain_radar_nowcast", "temp_air"]:
        if col not in base.columns:
            base[col] = float("nan")

    path = _io.write_canonical(case, base)
    print(f"[bellinge_sewer_level] wrote {path} ({len(base)} rows) from {level_member}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
