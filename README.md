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
uv run python -m tsforecastbench.data.cases.camels_dk_discharge.download
```

GESLA-3 + Mendeley WWTP need a manual one-time archive. ERA5 atmospheric forcing (GESLA
wind/pressure, Bellinge air temperature) is fetched **credential-free** at the station
point via the Open-Meteo archive API — no CDS account needed. See
[`data/README.md`](data/README.md) and each case's `download.py` docstring.

## What's here

**Cases** (auto-discovered from `data/cases/*/case.toml`):

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
- Windowed ML (Tier 1, univariate): `LinearLags` (Ridge), `LightGBM`, `MLP`
- Windowed ML (Tier 2, **exogenous-aware**): `LinearLags+exo`, `LightGBM+exo` — feed the
  case's declared covariates (rain, wind, pressure, ...). At prediction time the horizon's
  covariate values are supplied as a **perfect-forecast proxy** (observed reanalysis stands
  in for an NWP forecast). The Tier-1/Tier-2 pair exposes how much the covariates help.

Recurrent nets (LSTM/GRU), DLinear/N-HiTS/iTransformer, hybrid process+ML residual
correction, and time series foundation models are on the roadmap (see `research_plan.md`).

**Outputs**:

- `scripts/run_pilot.py` → colour-coded **cases × methods** matrices (MASE, skill vs
  persistence, **skill vs mean**) across all cases.
- `scripts/run_leadtime.py` → **error vs lead time** curves vs the simple benchmarks
  (mean, persistence), averaged over rolling origins; adds `LightGBM+exo` on cases with
  covariates.
- `scripts/print_results.py` → plain-text dump of both.

```bash
uv run python scripts/run_pilot.py                       # all cases x all methods
uv run python scripts/run_leadtime.py --case gesla_storm_surge --origins 20 --metric RMSE
uv run python scripts/print_results.py
```

Example result: ERA5 wind/pressure lift GESLA storm-surge skill-vs-persistence from
**0.30 → 0.66**, and precipitation flips CAMELS-DK `LinearLags` from **−3.7 → +0.83** —
the covariate value the benchmark is built to quantify.

## Design notes

- **Two concerns, two namespaces**: `tsforecastbench.data` (case-building — schema,
  registry, downloads; the eventual open-dataset half) and `tsforecastbench.bench` (model
  comparison — methods, eval, viz). `bench` depends on `data` only through the public
  `Case` interface, so the two are separable.
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
