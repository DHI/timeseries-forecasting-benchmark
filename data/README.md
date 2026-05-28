# data/

```
data/
  raw/<case_id>/        # source downloads (zips, CSVs, NetCDF, ...) - GITIGNORED, ~1 GB+ total
  processed/<case_id>.parquet   # canonical long-format series the benchmark reads - COMMITTED (~5 MB total)
  manifest.json         # SHA-256 + row count per processed case (written by downloads) - COMMITTED
```

## Why processed is committed

The canonical Parquets together are ~5 MB, small enough to ship in git. Committing them
lets a fresh clone run the full pilot and CI **without any downloads**:

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
uv run python -m tsforecastbench.cases.<case_id>.download
```

Or everything at once:

```bash
uv run python scripts/download_all.py
```

Some cases need manual one-time setup:

- **GESLA-3 storm surge** — bulk archive is behind a manual download link
  (https://gesla787883612.wordpress.com/downloads/); save a station file under
  `data/raw/gesla_storm_surge/` then run with `--station-file <name> --lat <lat>`.
- **Mendeley WWTP** — download the dataset zip from
  https://data.mendeley.com/datasets/xmbxhscgpr/1 (Download All button) and save under
  `data/raw/wwtp_influent/`, then run the case download.
- **Copernicus / ERA5** — needs `cdsapi` and `copernicusmarine` credentials.
