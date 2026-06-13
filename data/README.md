# data/

```
data/
  raw/<case_id>/        # source downloads (zips, CSVs, NetCDF, ERA5 cache, ...) - GITIGNORED
  processed/<case_id>.parquet   # canonical long-format series the benchmark reads - COMMITTED (~15 MB total)
  manifest.json         # SHA-256 + row count per processed case (written by downloads) - COMMITTED
```

## Why processed is committed

The canonical Parquets together are a few MB each (~15 MB total), small enough to ship in
git. Committing them lets a fresh clone run the full pilot and CI **without any downloads**:

```bash
git clone ...
uv sync
uv run python scripts/run_pilot.py   # works immediately, real data
```

The downloads then become an optional "rebuild from source" step — needed when you add
a new case, refresh a series, or want the raw observations behind a number.

## How to rebuild from source

Per case:

```bash
uv sync --extra data
uv run python -m tsforecastbench.data.cases.<case_id>.download
```

Or everything at once:

```bash
uv run python scripts/download_all.py
```

**Most cases fetch automatically with no credentials**: USGS NWIS (the `usgs_*` cases),
NOAA NDBC buoys (`ndbc_*`), and ERA5 atmospheric forcing for any case — the latter via the
**Open-Meteo** archive API, so no Copernicus/CDS account is needed (`_era5.py`).

A few need a manual one-time archive download:

- **GESLA-3 storm surge** — the bulk archive is behind a manual link
  (https://gesla787883612.wordpress.com/downloads/); save `GESLA*_ALL.zip` + `GESLA*_ALL.csv`
  under `data/raw/gesla_storm_surge/`, then run the download (defaults to `--station-code Cux`;
  ERA5 wind/pressure are fetched automatically).
- **Mendeley WWTP** — download the dataset zip from
  https://data.mendeley.com/datasets/xmbxhscgpr/1 (Download All) into
  `data/raw/wwtp_influent/`, then run the case download.
