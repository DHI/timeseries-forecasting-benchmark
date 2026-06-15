# RK2026 — ML Time Series Forecasting for Water & Environment

**Owner:** Jesper Mariegaard (DHI)
**Drafted:** 2026-05-27
**Status:** Plan for review — datasets curated, methods drafted, design sketched

---

## 1. Goal

Benchmark a wide range of time series forecasting methods — from trivial baselines to time series foundation models (TSFMs) — on 10 open datasets that span DHI's water and environment domains. The headline deliverable is a colour-coded matrix (cases × methods) of forecast skill, plus a written analysis of where TSFMs add value relative to classical and ML baselines (KER 5, "next generation modelling").

The user-facing decisions taken in scoping:

- **Datasets first.** Curate 10 candidate cases for approval before writing pipeline code.
- **Domain-appropriate horizons.** Each case forecasts at a horizon natural to its domain.
- **Zero-shot first, fine-tune the top performers.** Pragmatic on compute; matches the "applicability of emerging FMs" framing.
- **Internal report + notebooks → Streamlit app → external open release.** Public funding implies eventual external sharing; start small.

---

## 2. Candidate cases (10) — for approval

Selected to cover the water cycle from atmosphere to ocean, mix observed and modelled data, and provide meaningful exogenous covariates. Each case lists target, native frequency, proposed forecast horizon, exogenous inputs, and source.

| # | Domain | Target (what to forecast) | Freq | Horizon | Exogenous inputs | Dataset | Type |
|---|---|---|---|---|---|---|---|
| 1 | Hydrology — lowland | River discharge | Daily | 7 d | Precipitation, T, PET, soil moisture state, DK-model sim | **CAMELS-DK** (one groundwater-dominated DK catchment) | Obs + model |
| 2 | Hydrology — flashy | River discharge | Hourly | 24 h | Precipitation, T, ERA5-Land forcing | **Caravan / LamaH-CE** (one alpine basin) | Obs + reanalysis |
| 3 | Water resources | Groundwater head | Weekly | 12 w | Precipitation, PET, season | **GEMS-GER** (one well from 3000 in Germany) | Observed |
| 4 | Urban drainage | Sewer water level at pipe sensor | 1 min → resampled to 5–15 min | 1–6 h | Rain gauge + X-band radar nowcast, T | **Bellinge** (Odense, DK) | Observed |
| 5 | Combined sewer overflow | Overflow chamber level / spill volume | Event-based, 15 min | 1–6 h ahead, event hit rate | Same as #4 + antecedent dry-weather period | **Bellinge** (CSO chamber subset) | Observed |
| 6 | Water distribution | District demand / inlet flow | 5 min → hourly | 24 h | Day-of-week, hour-of-day, holiday flag, T | **BattLeDIM L-Town** (Cyprus, synthetic SCADA, 2 yrs) | Simulated |
| 7 | Wastewater treatment | Influent flow rate (and effluent NH₄ as secondary target) | 2 min → 15 min | 6–24 h | Rain at plant, T, dry-weather pattern, control signals | **Mendeley N₂O WWTP** (350 000 PE, 2 yrs) | Observed |
| 8 | Storm surge / sea level | Non-tidal residual at coastal gauge | Hourly | 48 h | ERA5 MSLP, 10 m wind, T, tide harmonic | **GESLA-3** (one NW European gauge) | Obs + reanalysis |
| 9 | Spectral waves | Significant wave height H_s (and peak period as secondary) | Hourly | 24–72 h | ERA5 10 m wind, MSLP, sea ice cover | **NDBC buoy** or **Copernicus NW Shelf wave hindcast** point | Obs or model |
| 10 | Coastal ocean | Chlorophyll-a at coastal pixel | Daily | 7–14 d | SST, PAR/light, river discharge proxy, season | **Copernicus Marine** (e.g. NW Shelf or Baltic) | Reanalysis |

**Optional 11th case** if you want sediments covered explicitly: suspended particulate matter at a Copernicus Marine coastal pixel, hourly–daily, exogenous = wind, waves, discharge.

**Notes on the choices.**

The two hydrology cases (1 & 2) are deliberately contrasting — one groundwater-dominated lowland, one flashy alpine — to expose whether the same forecasting method ranks consistently across hydrological regimes. Cases 4 & 5 share the Bellinge data but ask different questions (continuous pipe level vs. discrete overflow events), which is realistic for the urban water sector. Case 7 lets us optionally split into two targets (flow + NH₄) without spinning up a new pipeline.

Some cases are pure observations, some are model output, and several mix the two (CAMELS-DK includes DK-model simulations; ERA5 forcing is reanalysis). This is intentional — DHI's real use cases combine both.

---

## 3. Methods axis

The columns of the matrix. Roughly 16 methods, grouped:

**Naive baselines (sanity floor)**

- Mean of training period
- Persistence (last value)
- Seasonal naive (last value at same hour-of-day / day-of-week / day-of-year)
- Climatology (mean by season / hour-of-day)

**Classical statistical**

- ARIMA / auto-ARIMA
- ETS (exponential smoothing state space)
- (Prophet — optional, often weak on sub-daily but useful as a known reference)

**Windowed ML (lag-feature regression)**

- Linear regression with lagged target + exogenous
- Gradient boosting (LightGBM)
- MLP

**Recurrent neural networks**

- LSTM (sequence-to-one and sequence-to-sequence). Canonical baseline in operational hydrology — Kratzert et al. showed an LSTM trained on many basins beats locally-calibrated conceptual models. Should be a strong reference for cases 1, 2, 5, 7.
- GRU (lighter alternative to LSTM)
- DeepAR (probabilistic RNN with autoregressive sampling)

**Modern deep learning (point + probabilistic)**

- DLinear — embarrassingly simple linear decomposition that famously beats many transformers; cheap CPU baseline that often surprises.
- N-HiTS
- PatchTST
- iTransformer (or TSMixer as a simpler alternative)
- TimeMixer (more recent multiscale mixer; optional)
- TFT (heavy, but the most natural fit for multi-horizon + heterogeneous exogenous)

**Time series foundation models (zero-shot)**

- Chronos-2 (Amazon, currently #1 on GIFT-Eval among pretrained models)
- TimesFM 2.5 (Google)
- Moirai 2.0 (Salesforce, native multivariate / covariates)
- TimeMoE (open MoE FM)
- TimeGPT (Nixtla, API-only)
- Lag-Llama (open, smaller)

**Hybrid & operational reference (what clients actually receive today)**

These are the real benchmarks any DHI client-facing forecast has to beat. Easy to forget in a pure-ML benchmark, but their absence would make the matrix look unmoored from practice.

- **Process model output as a reference column.** For each case where an operational/reanalysis forecast already exists, pull it in unmodified as a "free" column.
  - Hydrology (1, 2) → DK-model simulation (CAMELS-DK already ships this) and ERA5-Land simulated runoff.
  - Storm surge (8) → Copernicus Marine global / regional surge forecast.
  - Waves (9) → Copernicus NW Shelf wave hindcast / ECMWF wave forecast.
  - Chlorophyll (10) → Copernicus biogeochemistry hindcast.
  - This is the column ML methods have to beat to be commercially interesting.
- **Process-model residual correction (hybrid ML).** Train ML on `obs − process_model`. Standard practice for operational storm-surge and discharge correction (a.k.a. model output statistics, MOS). Will run as a wrapper around LightGBM, MLP, and LSTM so we can compare hybrid vs. pure-ML for each case where a process baseline is available.
- **Conceptual hydrological model.** For cases 1 and 2, add a lumped conceptual model (GR4J or HBV via `airGR` / `hydromt`) as a domain-traditional baseline.

**Fine-tuning round.** After the zero-shot pass, pick the 1–2 most promising FMs and fine-tune on the train split of each case (or a pooled multi-case set). This gives a second "FM-FT" set of columns in the matrix.

**Why this matters.** A storm-surge forecast delivered to a port operator today is typically (a) a hydrodynamic model forced by NWP wind/pressure, often (b) bias-corrected against the recent few days of tide-gauge obs, sometimes (c) post-processed by ensemble MOS. The matrix needs to show whether and where TSFMs displace any of these steps — not just whether they beat ARIMA.

**Library choices.** Nixtla's `statsforecast` / `mlforecast` / `neuralforecast` give a unified interface across baselines, windowed ML, and most modern DL — same call signature, same exogenous handling — which keeps the benchmarking code clean. Foundation models come from their respective repos (`chronos-forecasting`, `timesfm`, `uni2ts`). `darts` is a fallback for anything that doesn't fit the Nixtla pattern. Evaluation goes through **ModelSkill** so we get the standard DHI plots and aggregation for free.

---

## 4. Benchmark design

**Per-case configuration object** — frequency, target column, exogenous columns, forecast horizon(s), input window, train/val/test cut dates, metric set, retraining cadence. All cases share the same interface so adding a method or a case is local work.

**Splits.** Rolling-origin evaluation with a fixed test window per case, then rolled forward N times. Reports the median and IQR of the metric per (case, method) cell so the matrix can show both central tendency and stability. A simple time-ordered hold-out is the fallback if rolling proves too expensive for the heavier DL models.

**Metrics.**

- MAE and RMSE — scale-dependent, reported per case.
- MASE relative to seasonal naive — scale-free, the primary cell value in the matrix.
- Skill score vs. persistence — interpretable for water professionals ("X% better than doing nothing").
- CRPS for the probabilistic models (most modern DL and all FMs).
- Domain-specific KPIs as a sidebar where they matter — e.g. event hit/miss/false-alarm for CSO (case 5), POD/FAR for storm surge exceedance of warning levels (case 8).

**Exogenous handling.** Three tiers, applied uniformly:

- Tier 0: target only.
- Tier 1: target + calendar features (hour, day-of-week, day-of-year, holiday).
- Tier 2: Tier 1 + domain covariates (rainfall, wind, SST, etc.).

Each method runs at the highest tier it supports. The matrix can be rendered for Tier 1 and Tier 2 separately to expose how much the covariates help each method family.

**Horizons.** Each case has one primary horizon (Section 2 table), but we evaluate at 3 sub-horizons (short / mid / long for that case) so degradation curves can be shown alongside the matrix.

**Compute.** Baselines, classical, and windowed ML on CPU. Modern DL on a single GPU (LightningAI account). Zero-shot FMs on CPU where possible (Chronos-Bolt is laptop-friendly), GPU where required. Fine-tuning round on GPU only. See §5a for the laptop fast-mode that lets the whole pipeline run end-to-end with no GPU.

---

## 4a. Data storage and metadata

Uniform, fast, reliable storage across all 10 cases. The driving constraint is that the same code (model fit/predict, evaluation, plotting) has to run on every case without per-case glue.

**Format: long-format Parquet, one file per case.**

This is the Nixtla convention (`unique_id`, `ds`, `y`, plus exogenous columns) and the most universally consumable tabular timeseries layout. Parquet is columnar, compressed, ~10× faster than CSV to read, splits well, and is natively supported by pandas, polars, pyarrow, duckdb, xarray-via-arrow, and every Nixtla library. Single file per case keeps things simple; if a case grows beyond a few GB we partition by year.

For the cases whose primary data is gridded (chlorophyll, waves, surge reanalysis), the workflow is: download with `copernicusmarine` / `xarray` → extract the point time series we forecast at → write to the same long Parquet layout. The full xarray dataset is kept as a sidecar `.zarr` store under `data/raw/` for traceability, but the benchmark only reads the Parquet.

**Schema (case data file).**

```
unique_id    str        # series identifier (e.g. "bellinge_lvl_F4", "ndbc_44011")
ds           datetime64 # timestamp (UTC, timezone-aware)
y            float64    # target
x_*          float64    # exogenous covariates (rain, wind, etc.)
cov_static_* float64    # static catchment / station attributes (broadcast)
quality      uint8      # optional QC flag (0=ok, 1=interpolated, 2=missing-imputed)
```

Frequencies are pinned per case (no irregular timestamps inside a series — gaps are explicit `NaN` rows so windowed methods don't silently leak across them).

**Metadata: Pydantic `Case` schema, one TOML per case + a Python registry.**

Every case ships a small TOML that the registry validates into a `Case` object:

```toml
# cases/sewerlevel_bellinge.toml
id = "sewerlevel_bellinge"
domain = "urban_drainage"
target = "level_m"
target_unit = "m"
freq = "5min"
horizon_primary = "6h"
horizons_eval = ["1h", "6h", "24h"]
input_window = "7d"
exogenous = ["rain_gauge", "rain_radar_xband", "temp_air"]
exogenous_known_future = ["rain_radar_nowcast"]  # known at prediction time
static_attrs = ["catchment_area_ha", "impervious_frac"]
splits = { train_end = "2018-12-31", val_end = "2019-12-31", test_end = "2020-12-31" }
source = { name = "Bellinge", doi = "10.11583/DTU.c.5029124", license = "CC-BY-4.0" }
data_type = "observed"
notes = "13 level meters in combined sewer; pick F4 as primary series."
```

The `Case` object exposes `.load() -> pl.DataFrame` (polars by default, with a pandas accessor), `.splits()`, `.metric_set()`, etc. The same object is consumed by every method wrapper, by the evaluation loop, and by the matrix renderer.

**Downloads: reproducible, checksummed scripts.**

Each case has `src/tsforecastbench/cases/<id>/download.py` that fetches raw data into `data/raw/<id>/`, validates checksums, and writes the canonical Parquet into `data/processed/<id>.parquet`. Raw is git-ignored; processed Parquets get a SHA-256 written to a manifest that *is* committed, so anyone re-running the download gets a byte-equality check. Downloads are idempotent.

**Why not other formats.**

- *NetCDF / xarray Datasets.* Excellent for gridded + multi-dim — and the natural fit for MIKE IO / ModelSkill — but heavier and slower than Parquet for the long-format tabular usage every forecasting library expects. We keep xarray *upstream* of the canonical Parquet (sidecar `.zarr` for the gridded sources) so the lineage is preserved without forcing each method wrapper to understand n-dim arrays.
- *HDF5.* Older, more painful concurrency, weaker tooling than Parquet/Zarr now.
- *Per-case ad hoc CSVs.* What we're explicitly avoiding.
- *DuckDB / single-file SQLite.* Tempting for the cross-case queries, but Parquet + duckdb-on-demand gives the same SQL surface without locking us into a single database file.

**ModelSkill integration.**

A thin adapter converts a `(case, method, forecast_df)` triple into a ModelSkill `Comparer` so we get the standard DHI plots (scatter, taylor, residual histograms, time-series overlay) "for free". This is also where any DHI-style metric customisation lives.

---

## 5. Deliverables and roadmap

Phased, with a usable artefact at the end of each phase.

**Phase A — Pilot end-to-end (≈ 2 weeks).** Three cases (one hydrology, one urban drainage, one coastal), five methods (mean, persistence, ARIMA, LightGBM, Chronos zero-shot). Working evaluation pipeline, draft matrix figure, ModelSkill integration confirmed. The point is to lock the interfaces before scaling.

**Phase B — Full benchmark (≈ 4–6 weeks).** All 10 cases, all 16 methods, Tier 1 + Tier 2 exogenous. Notebooks per case for transparency; aggregate matrix + per-horizon matrices as the headline figures. Internal report.

**Phase C — Fine-tuning round (≈ 2–3 weeks).** Pick top 1–2 FMs, fine-tune per case (and pooled), add as extra columns. Update report.

**Phase D — Streamlit app on DHI internal hosting (≈ 1–2 weeks).** Interactive matrix — filter by domain, horizon, metric; drill into per-case forecast plots; download forecasts. Reuses the benchmark outputs, no new model runs.

**Phase E — External release.** Clean public repo (uv-managed, MIT or similar), polished figures, ESSD- or JOSS-style paper. Triggered when the internal report is reviewed and the funder communication needs are clear.

---

## 5a. Laptop fast-mode (CPU-only end-to-end)

A first-class run configuration, not a hack. The whole pipeline — download a small slice of every case, fit every laptop-capable method, evaluate, render the matrix — must complete on a developer laptop without a GPU in under ~30 minutes. This is what we use for the pilot, for CI, and for live demos.

**What runs in fast-mode:**

- All 10 cases, but each truncated to its last ~2 years and downsampled where the native rate is sub-minute (Bellinge, WWTP, L-Town go to 15-min or hourly).
- Methods: all baselines, ARIMA, ETS, linear, LightGBM, MLP, DLinear, GRU (small), Chronos-Bolt small zero-shot, TimeGPT (API), conceptual hydro model where applicable, hybrid ML residual where a process baseline is available.
- One horizon per case (the primary one).
- Single time-ordered hold-out instead of rolling origin.

**What is skipped in fast-mode** (added in full-mode on GPU): large LSTM, DeepAR, N-HiTS, PatchTST, iTransformer, TimeMixer, TFT, Chronos-2, TimesFM 2.5, Moirai 2.0, TimeMoE, Lag-Llama, all fine-tuning.

A `--mode={fast,full}` flag at the benchmark entry point switches between them; the matrix renderer shows greyed-out cells for methods not run.


**Phase F (later) - Per-case reports.** A 2-3 page printable report per case with: domain description, target/exogenous explanation, basic series visualisation, splits, methods' MASE/skill on this case, example forecasts at a few origins (with observed overlay), and the case's lead-time degradation curve. Lets a reader 'get a feel' for the case without running anything.

---

## 6. Open questions for Jesper

These shape Phase A and below — no need to answer all at once, but flagging them now.

1. **Hydrology contrast pair (cases 1 & 2).** Are you happy with one DK lowland + one Central European alpine catchment, or would you rather have two DK catchments to keep it Denmark-centric?
2. **CSO vs. urban drainage as separate cases (4 & 5).** Bellinge supports both. If you'd rather free a slot for a different domain (e.g. lake water temperature, reservoir level), say so.
3. **Sediments.** Drop them, or add the 11th case?
4. **WWTP second target.** Influent flow is the obvious one; happy to add effluent NH₄ as a sidebar target on the same pipeline, or leave it out for now?
5. **TimeGPT.** Nixtla API, not open. Include for completeness or skip on principle?
6. **Process-model reference columns.** Adding Copernicus operational forecasts as reference columns (cases 8, 9, 10) is the right thing for client realism but adds data engineering — confirm scope.
7. **Hybrid ML residual correction.** Requires a process baseline per case. Roll it out everywhere it's available (1, 2, 8, 9, 10) or pilot on storm surge only?
8. **Storage: Parquet long-format.** I've proposed Parquet (Nixtla style) as the canonical format with xarray/Zarr sidecars upstream. If you'd rather have NetCDF as the canonical (MIKE IO / ModelSkill home turf), happy to switch — but I'd push back on this; it makes every method wrapper more complex for limited gain on tabular series.
9. **ModelSkill integration.** Anything specific you'd like the benchmark to feed back into ModelSkill (e.g. a new comparer subclass for ML forecasts, or extra metrics)?
10. **Simulated-data follow-on.** You mentioned wanting more simulated cases later. Worth noting now what those should be (MIKE 1D pipe network, MIKE SHE, MIKE 21 wave/surge synthetic, BSM2 WWTP scenarios) so the case loader interface is forward-compatible?

---

## 7. Repository layout (sketch, for Phase A)

```
RK2026-timeseries-forecasting/
├── pyproject.toml              # uv-managed
├── research_plan.md            # this document
├── README.md
├── src/tsforecastbench/
│   ├── cases/                  # one module per case: load(), config()
│   ├── methods/                # one module per method, common fit/predict interface
│   ├── eval/                   # metrics, rolling origin, matrix rendering
│   └── viz/                    # matrix figure, per-case degradation curves
├── notebooks/                  # per-case exploratory + per-method sanity
├── data/                       # gitignored; download scripts in src/tsforecastbench/cases
├── results/                    # per-run parquet of forecasts + metrics
└── app/                        # Phase D Streamlit
```

---

## 8. Key references (verified, May 2026)

- **CAMELS-DK** — Koch et al. (2025), ESSD: 3330 Danish catchments, 304 with streamflow obs, daily, 1989–2019, groundwater-dominated lowland focus. https://essd.copernicus.org/articles/17/1551/2025/ · DOI 10.22008/FK2/AZXSYP
- **Caravan** — Kratzert et al. (2023), Scientific Data: 6830 catchments globally, ERA5-Land forcing, standardised. https://www.nature.com/articles/s41597-023-01975-w · https://github.com/kratzert/Caravan
- **Bellinge** — Nedergaard Pedersen et al. (2021), ESSD: 1.7 km² combined sewer system, Odense DK, 2010–2020, 13 level meters + flow + radar rainfall + MIKE Urban/SWMM models. https://essd.copernicus.org/articles/13/4779/2021/ · DOI 10.11583/DTU.c.5029124
- **BattLeDIM / L-Town** — CCWI/WDSA 2020 competition: real Cyprus network, EPANET-generated SCADA, 2018 train / 2019 test, leak scenarios. https://battledim.ucy.ac.cy/
- **N₂O WWTP dataset** — Mendeley Data: 350 000 PE plant, 2 min sampling, June 2022 – June 2024, influent + effluent + control + N₂O. https://data.mendeley.com/datasets/xmbxhscgpr/1 · arXiv 2407.05959
- **GEMS-GER** — German groundwater dataset, 3000+ wells, 32 years weekly. (To confirm canonical DOI in Phase A.)
- **GESLA-3** — Haigh et al. (2023), GDJ: 5119 tide gauges globally, 15-min / 1-h. https://rmets.onlinelibrary.wiley.com/doi/10.1002/gdj3.174 · https://gesla787883612.wordpress.com/downloads/
- **Copernicus Marine** — chlorophyll OMI, NW Shelf wave hindcast, global biogeochemistry hindcast. https://data.marine.copernicus.eu/products
- **NDBC** — NOAA moored buoys, significant wave height + spectral data, 1970–present. https://www.ndbc.noaa.gov/
- **GIFT-Eval (FM benchmark)** — 97 tasks across horizons/domains; current leaders Chronos-2, Moirai 2.0, TimesFM 2.5. arXiv 2510.13654
- **Chronos-2** — Amazon, encoder-only with group attention, native multivariate.
- **TimesFM 2.5** — Google, 200M params, top GIFT-Eval Sept 2025.
- **Moirai 2.0** — Salesforce, decoder-only, native any-variate.

---

*End of plan. Awaiting Jesper's feedback on the open questions in §6.*
