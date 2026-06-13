"""Shared NDBC standard-meteorological buoy fetch + parse, used by the wave cases.

Historical yearly files: https://www.ndbc.noaa.gov/data/historical/stdmet/<id>h<year>.txt.gz
Whitespace-delimited; 2 header lines (columns, then units). Missing values are sentinels
(99, 999, 9999). We keep WVHT (target) + WSPD/PRES/ATMP/WTMP as canonical columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.data.cases import _io
from tsforecastbench.data.schema import Case

BASE = "https://www.ndbc.noaa.gov/data/historical/stdmet"
COL_MAP = {"WVHT": "wvht", "WSPD": "wspd", "PRES": "pres", "ATMP": "atmp", "WTMP": "wtmp"}
MISSING_AT = {"wvht": 90, "wspd": 90, "pres": 9000, "atmp": 90, "wtmp": 90}


def _read_year(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=0, skiprows=[1], compression="infer")
    df.columns = [c.lstrip("#") for c in df.columns]
    return df


def build_ndbc(case: Case, station: str, years: list[int]) -> pd.DataFrame:
    """Download + parse the given buoy/years into a canonical frame on the case freq."""
    frames = []
    for yr in years:
        url = f"{BASE}/{station}h{yr}.txt.gz"
        dest = _io.DATA_RAW / case.id / f"{station}h{yr}.txt.gz"
        try:
            if not dest.exists():
                _io.http_download(url, dest)
            frames.append(_read_year(dest))
            print(f"[{case.id}] got {station} {yr}")
        except Exception as exc:  # missing year, network
            print(f"[{case.id}] skip {yr}: {exc}")
    if not frames:
        raise RuntimeError(f"[{case.id}] no NDBC years downloaded for {station}")

    raw = pd.concat(frames, ignore_index=True)
    ycol = "YY" if "YY" in raw.columns else "#YY"
    ds = pd.to_datetime(
        dict(
            year=raw[ycol].astype(int),
            month=raw["MM"].astype(int),
            day=raw["DD"].astype(int),
            hour=raw["hh"].astype(int),
            minute=raw.get("mm", 0),
        ),
        utc=True,
        errors="coerce",
    )
    out = pd.DataFrame({"unique_id": case.id, "ds": ds})
    for src, canon in COL_MAP.items():
        if src in raw.columns:
            v = pd.to_numeric(raw[src], errors="coerce")
            out[canon] = v.where(v < MISSING_AT.get(canon, 1e9), np.nan).to_numpy()
        else:
            print(f"[{case.id}] WARNING column {src} missing; got {list(raw.columns)}")
    out["y"] = out["wvht"]
    out = out.dropna(subset=["ds"]).sort_values("ds")
    return _io.resample_regular(out, case.freq)
