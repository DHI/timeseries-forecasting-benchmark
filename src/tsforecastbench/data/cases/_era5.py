"""Fetch ERA5 reanalysis as a point time series — credential-free via Open-Meteo.

Several cases need ERA5 atmospheric forcing (10-m wind, mean sea-level pressure, 2-m
air temperature) at a single lon/lat: storm surge (wind + pressure drive the residual),
urban drainage (air temperature), waves, etc. The canonical route is the Copernicus CDS
API (`cdsapi`), which needs a free account and an API key in `~/.cdsapirc`.

To keep the pipeline runnable with no credentials, we instead pull the same ERA5 hourly
fields from the **Open-Meteo Historical Weather API** (https://open-meteo.com/), whose
`archive` endpoint serves ERA5 (1940-present) for a point with no key. Open-Meteo is
CC-BY; ERA5 is the underlying reanalysis (Copernicus/ECMWF). For an operational forecast
these covariates would come from an NWP *forecast*; here the reanalysis is used as a
**perfect-forecast (oracle) proxy**, which the case docs state explicitly.

Canonical variables this module exposes (Open-Meteo source field in parentheses):

    mslp      mean sea-level pressure, hPa     (pressure_msl)
    temp_air  2-m air temperature, deg C       (temperature_2m)
    u10, v10  10-m wind components, m/s         (wind_speed_10m + wind_direction_10m)
    wspd10    10-m wind speed, m/s              (wind_speed_10m)

Wind is returned by Open-Meteo as speed + meteorological direction (the direction the
wind blows *from*); we convert to eastward/northward components:
    u = -speed * sin(dir);  v = -speed * cos(dir)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# canonical name -> the Open-Meteo hourly fields it needs
_CANON_TO_OM = {
    "mslp": ["pressure_msl"],
    "temp_air": ["temperature_2m"],
    "u10": ["wind_speed_10m", "wind_direction_10m"],
    "v10": ["wind_speed_10m", "wind_direction_10m"],
    "wspd10": ["wind_speed_10m"],
    "precip": ["precipitation"],
}

# how each canonical var aggregates when resampling hourly -> coarser (precip accumulates,
# everything else is an instantaneous mean)
_AGG = {"precip": "sum"}


def _om_fields(variables: list[str]) -> list[str]:
    fields: list[str] = []
    for v in variables:
        if v not in _CANON_TO_OM:
            raise ValueError(f"Unknown ERA5 variable {v!r}; known: {sorted(_CANON_TO_OM)}")
        for f in _CANON_TO_OM[v]:
            if f not in fields:
                fields.append(f)
    return fields


def _fetch_chunk(
    lat: float,
    lon: float,
    start: str,
    end: str,
    fields: list[str],
    timeout: int = 120,
    retries: int = 4,
) -> pd.DataFrame:
    import requests

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(fields),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
            r.raise_for_status()
            h = r.json()["hourly"]
            df = pd.DataFrame(h)
            df["ds"] = pd.to_datetime(df["time"], utc=True)
            return df.drop(columns=["time"])
        except Exception as exc:  # network / rate-limit / transient
            last_exc = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Open-Meteo ERA5 fetch failed for {start}..{end}: {last_exc!r}")


def fetch_era5_point(
    lat: float,
    lon: float,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    variables: list[str] | None = None,
    chunk_years: int = 5,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Return an hourly ERA5 point series on a UTC grid for [start, end].

    Columns: ``ds`` (UTC, tz-aware) plus the requested canonical ``variables``
    (default ``["mslp", "u10", "v10", "temp_air"]``). Long spans are fetched in
    ``chunk_years``-year blocks to keep each request bounded; if ``cache_dir`` is given,
    each chunk's raw response is cached there as Parquet so re-runs are offline.
    """
    variables = variables or ["mslp", "u10", "v10", "temp_air"]
    fields = _om_fields(variables)
    start = (
        pd.Timestamp(start).tz_localize(None) if pd.Timestamp(start).tzinfo else pd.Timestamp(start)
    )
    end = pd.Timestamp(end).tz_localize(None) if pd.Timestamp(end).tzinfo else pd.Timestamp(end)

    parts: list[pd.DataFrame] = []
    cur = pd.Timestamp(start)
    while cur <= end:
        chunk_end = min(
            pd.Timestamp(end), cur + pd.DateOffset(years=chunk_years) - pd.Timedelta(days=1)
        )
        s, e = cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cache = (cache_dir / f"era5_{lat:.3f}_{lon:.3f}_{s}_{e}.parquet") if cache_dir else None
        if cache and cache.exists():
            parts.append(pd.read_parquet(cache))
        else:
            raw = _fetch_chunk(lat, lon, s, e, fields)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                raw.to_parquet(cache, index=False)
            parts.append(raw)
        cur = chunk_end + pd.Timedelta(days=1)

    raw = pd.concat(parts, ignore_index=True).drop_duplicates("ds").sort_values("ds")
    return _to_canonical(raw, variables)


def _to_canonical(raw: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    out = pd.DataFrame({"ds": pd.to_datetime(raw["ds"], utc=True)})
    if "wind_speed_10m" in raw.columns:
        spd = pd.to_numeric(raw["wind_speed_10m"], errors="coerce").to_numpy()
        if "wspd10" in variables:
            out["wspd10"] = spd
        if "wind_direction_10m" in raw.columns:
            drad = np.deg2rad(pd.to_numeric(raw["wind_direction_10m"], errors="coerce").to_numpy())
            if "u10" in variables:
                out["u10"] = -spd * np.sin(drad)
            if "v10" in variables:
                out["v10"] = -spd * np.cos(drad)
    if "mslp" in variables:
        out["mslp"] = pd.to_numeric(raw.get("pressure_msl"), errors="coerce").to_numpy()
    if "temp_air" in variables:
        out["temp_air"] = pd.to_numeric(raw.get("temperature_2m"), errors="coerce").to_numpy()
    if "precip" in variables:
        out["precip"] = pd.to_numeric(raw.get("precipitation"), errors="coerce").to_numpy()
    return out


def to_daily(era5: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Aggregate an hourly canonical ERA5 frame to daily (UTC, anchored at 00:00).

    Precip is summed over the day; instantaneous fields (temp, pressure, wind) are averaged.
    """
    e = era5.copy()
    e["ds"] = pd.to_datetime(e["ds"], utc=True)
    agg = {v: _AGG.get(v, "mean") for v in variables if v in e.columns}
    out = e.set_index("ds").resample("1D").agg(agg).reset_index()
    return out


def merge_onto(
    base: pd.DataFrame,
    era5: pd.DataFrame,
    variables: list[str],
    interpolate: bool = True,
    interp_limit_hours: int = 6,
) -> pd.DataFrame:
    """Left-merge ERA5 canonical columns onto a base frame keyed on UTC ``ds``.

    The base frame keeps its own row grid (the case's freq); ERA5 hourly values are aligned
    by timestamp. When the base grid is finer than hourly (e.g. 15-min), an exact-timestamp
    merge would only fill the on-the-hour rows, so by default we **time-interpolate** the
    ERA5 columns onto the finer grid — valid because mslp / wind / temperature are
    continuous instantaneous fields. ``interp_limit_hours`` bounds how far a value is
    carried across a genuine ERA5 gap. Columns absent in ``era5`` are created as NaN so the
    canonical schema stays stable.
    """
    out = base.copy()
    out["ds"] = pd.to_datetime(out["ds"], utc=True)
    e = era5.copy()
    e["ds"] = pd.to_datetime(e["ds"], utc=True)
    have = [v for v in variables if v in e.columns]
    out = out.merge(e[["ds", *have]], on="ds", how="left")

    if interpolate and have and len(out) > 1:
        step = (out["ds"].iloc[1] - out["ds"].iloc[0]) or pd.Timedelta("1h")
        limit = max(1, int(pd.Timedelta(hours=interp_limit_hours) / step))
        s = out.set_index("ds")
        s[have] = s[have].interpolate(method="time", limit=limit, limit_direction="both")
        out = s.reset_index()

    for v in variables:
        if v not in out.columns:
            out[v] = float("nan")
    return out


def fill_gaps_from_era5(
    base: pd.DataFrame,
    era5: pd.DataFrame,
    pairs: dict[str, str],
    fit_until: pd.Timestamp,
) -> tuple[pd.DataFrame, dict]:
    """Fill NaN in each `base` column from the ERA5 column `pairs` maps it to.

    ERA5 values are mapped onto the measured column's scale by a least-squares line
    fitted where both are present at or before `fit_until`, so the test period does not
    shape the fill (gh #119). Returns the filled frame and, per column, the line and the
    number of values filled.
    """
    out = base.copy()
    ds = pd.to_datetime(out["ds"], utc=True)
    e = era5.assign(ds=pd.to_datetime(era5["ds"], utc=True)).set_index("ds")
    cutoff = pd.Timestamp(fit_until)
    cutoff = cutoff.tz_convert("UTC") if cutoff.tzinfo else cutoff.tz_localize("UTC")
    info: dict[str, dict] = {}
    for col, era_col in pairs.items():
        x = e[era_col].reindex(ds).to_numpy(dtype=float)
        y = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        fit = np.isfinite(x) & np.isfinite(y) & (ds <= cutoff).to_numpy()
        slope, intercept = np.polyfit(x[fit], y[fit], 1)
        gap = ~np.isfinite(y) & np.isfinite(x)
        y[gap] = slope * x[gap] + intercept
        out[col] = y
        info[col] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "n_fit": int(fit.sum()),
            "n_filled": int(gap.sum()),
        }
    return out, info
