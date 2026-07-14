"""Fetch USGS NWIS water data as a point time series — credential-free, public domain.

The USGS National Water Information System serves observed US water data over a public
REST API with no key. We use it for observed *targets* (river discharge, gage height,
water temperature, groundwater level) and pair them with ERA5 atmospheric forcing
(`_era5.py`) for the exogenous covariates.

Common parameter codes:
    00060  discharge, cubic feet per second
    00065  gage height, feet
    00010  water temperature, deg C
    72019  depth to water level, feet below land surface (groundwater)

Services: ``dv`` daily values, ``iv`` instantaneous (typically 15-min).
"""

from __future__ import annotations

import pandas as pd

NWIS = "https://waterservices.usgs.gov/nwis"
_MISSING = {"", "-999999", "-999999.0"}


def _get_json(url: str, timeout: int = 90, retries: int = 4) -> dict:
    import time

    import requests

    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"USGS NWIS request failed: {url}\n{last!r}")


def fetch_nwis(
    site: str,
    param: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    service: str = "dv",
    colname: str = "value",
) -> pd.DataFrame:
    """Return a tidy frame [ds (UTC), <colname>] for one USGS site + parameter.

    ``service='dv'`` gives daily values; ``'iv'`` gives instantaneous (~15-min). Missing
    sentinels become NaN. Daily timestamps are anchored at 00:00 UTC.
    """
    start = pd.Timestamp(start).strftime("%Y-%m-%d")
    end = pd.Timestamp(end).strftime("%Y-%m-%d")
    url = (
        f"{NWIS}/{service}/?format=json&sites={site}&parameterCd={param}"
        f"&startDT={start}&endDT={end}"
    )
    data = _get_json(url)
    ts = data.get("value", {}).get("timeSeries", [])
    if not ts:
        return pd.DataFrame({"ds": pd.to_datetime([], utc=True), colname: []})
    recs = ts[0]["values"][0]["value"]
    ds = pd.to_datetime([r["dateTime"] for r in recs], utc=True)
    vals = pd.to_numeric(
        pd.Series([r["value"] if r["value"] not in _MISSING else None for r in recs]),
        errors="coerce",
    )
    return pd.DataFrame({"ds": ds, colname: vals.to_numpy()})


def site_info(site: str) -> dict:
    """Return {name, lat, lon} for a USGS site from the site service."""
    url = f"{NWIS}/site/?format=rdb&sites={site}&siteOutput=expanded"
    import requests

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    rows = [ln for ln in r.text.splitlines() if ln and not ln.startswith("#")]
    if len(rows) < 3:
        return {"name": site, "lat": None, "lon": None}
    header = rows[0].split("\t")
    vals = rows[2].split("\t")  # row 1 is the format spec line
    rec = dict(zip(header, vals, strict=False))

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    return {
        "name": rec.get("station_nm", site),
        "lat": _f(rec.get("dec_lat_va")),
        "lon": _f(rec.get("dec_long_va")),
    }
