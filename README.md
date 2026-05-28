# tsforecastbench

**ML time series forecasting benchmark for water and environment.**

Comparing forecasting methods — from naive baselines through classical and windowed-ML
models to (soon) time series foundation models — across open datasets that span DHI's
water domains: hydrology, urban drainage, water distribution, wastewater treatment,
storm surge, spectral waves, and coastal ocean processes.

Developed under research outcome **RK2026 / KER 5** ("next-generation modelling").
See [`research_plan.md`](research_plan.md) for the full design and roadmap.

## Quick start

```bash
git clone ...
uv sync
uv run pytest                                # tests on synthetic + committed processed data
uv run python scripts/run_pilot.py           # real data is committed - works immediately
```

Processed Parquets (~5 MB total) are committed; raw downloads (~1 GB) are not. A fresh
clone runs the full pilot with no further downloads. Every case also ships a synthetic
generator so CI and laptop demos work without any data at all.

## Rebuild data from source

```bash
uv sync --extra data                         # xarray, utide, cdsapi, openpyxl, ...
uv run python scripts/download_all.py        # all cases
# or one at a time:
uv run python -m tsforecastbench.cases.camels_dk_discharge.download
```

GESLA-3 + Mendeley WWTP need a manual one-time archive; Copernicus/ERA5 need credentials.
See [`data/README.md`](data/README.md) and each case's `download.py` docstring.

## What's here

**Cases** (auto-discovered from `cases/*/case.toml`):

| Case | Domain | Target | Freq | Horizon | Source |
|---|---|---|---|---|---|
| `camels_dk_discharge` | hydrology | river discharge | daily | 7 d | CAMELS-DK |
| `bellinge_sewer_level` | urban drainage | sewer level | 15 min | 6 h | Bellinge (DTU) |
| `wd_battledim` | water distribution | inlet flow | hourly | 24 h | BattLeDIM L-Town |
| `wwtp_influent` | wastewater | influent flow | hourly | 24 h | Mendeley N₂O |
| `gesla_storm_surge` | storm surge | non-tidal residual | hourly | 48 h | GESLA-3 |
| `ndbc_waves` | spectral waves | significant wave height | hourly | 24 h | NOAA NDBC |

**Methods** (columns of the matrix):

- Naive: `Mean`, `Persistence`, `SeasonalNaive`, `Climatology`
- Classical: `AutoARIMA`, `AutoETS`
- Windowed ML: `LinearLags` (Ridge), `LightGBM`, `MLP`

Recurrent nets (LSTM/GRU), DLinear/N-HiTS/iTransformer, hybrid process+ML residual
correction, and time series foundation models are on the roadmap (see `research_plan.md`).

**Outputs**:

- `scripts/run_pilot.py` → colour-coded **cases × methods** matrix (MASE + skill vs persistence).
- `scripts/run_leadtime.py` → **error vs lead time** curves, averaged over rolling origins.
- `scripts/print_results.py` → plain-text dump of both.

```bash
uv run python scripts/run_pilot.py --cases camels_dk_discharge ndbc_waves
uv run python scripts/run_leadtime.py --case bellinge_sewer_level --origins 30 --metric RMSE
uv run python scripts/print_results.py
```

## Design notes

- **Storage**: canonical long-format Parquet (`unique_id`, `ds` UTC, `y`, exogenous);
  pandas in memory. SHA-256 manifest in `data/manifest.json`.
- **Splits**: calendar or observed-span fractions (`split_mode="fraction"`), robust to
  series whose record doesn't cover a fixed window.
- **Gaps**: explicit NaN rows; training is imputed, scoring is NaN-masked so only real
  observations count (each run reports the observed fraction of the test window).
- **Tractability**: AutoARIMA/AutoETS are non-seasonal by default and capped at 5000
  training points; MLP is capped at 20000. These keep classical/heavy methods feasible
  on high-frequency series; explicit overrides per method on a per-case basis are easy.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) — adding a case or a method is a local change
against a common interface.

## License

MIT — see [`LICENSE`](LICENSE). Built at DHI; public-funded and intended for open release.
