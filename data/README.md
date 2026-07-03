# data/

```
data/
  raw/<case_id>/        # source downloads (zips, CSVs, NetCDF, ERA5 cache, ...) - GITIGNORED
  processed/<case_id>.parquet   # canonical long-format series, gaps and all - COMMITTED (~15 MB total)
  curated/<case_id>.parquet     # benchmark-ready series the eval loops read - COMMITTED
  manifest.json         # SHA-256 + row count per processed case (written by downloads) - COMMITTED
  curated/manifest.json # gap policy outcome per case (cap, gaps, span) - COMMITTED
```

## processed vs curated

`processed/` is the canonical record straight from the sources: a regular time grid with
real gaps left as NaN. It is what EDA, case review, and reports read, so the true gap
structure stays visible.

`curated/` is what the **benchmark eval loops** read (`Case.load(curated=True)`). It applies
one uniform gap policy — see `src/tsforecastbench/data/curate.py` and the RESEARCH_LOG:

- **Small gaps** (≤ a per-case cap, the lag where the target's autocorrelation drops below
  0.8, bounded by one seasonal cycle) are linearly interpolated.
- **Large gaps** are hard boundaries — never bridged. The record is split into contiguous
  segments and only the **longest** is kept.
- A **`y_observed`** boolean column marks genuine observations, so the target is complete
  enough for any method to fit while scoring still counts only measured points.

Rebuild it after downloading or refreshing data, or after changing the policy:

```bash
uv run python scripts/curate_data.py            # all cases
uv run python scripts/curate_data.py waves_oregon
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
uv run python scripts/curate_data.py   # refresh data/curated/ from the new processed/
```

**Most cases fetch automatically with no credentials**: USGS NWIS (the `usgs_*` cases),
NOAA NDBC buoys (`ndbc_*`), and ERA5 atmospheric forcing for any case — the latter via the
**Open-Meteo** archive API, so no Copernicus/CDS account is needed (`_era5.py`).

A few need a manual one-time archive download:

- **GESLA-3 storm surge** — the bulk archive is behind a manual link
  (https://gesla787883612.wordpress.com/downloads/); save `GESLA*_ALL.zip` + `GESLA*_ALL.csv`
  under `data/raw/surge_cuxhaven/`, then run the download (defaults to `--station-code Cux`;
  ERA5 wind/pressure are fetched automatically).
- **Mendeley WWTP** — download the dataset zip from
  https://data.mendeley.com/datasets/xmbxhscgpr/1 (Download All) into
  `data/raw/wwtp_influent_n2o_350k/`, then run the case download.
