# Water timeseries forecasting benchmark — research spec

## Goal

Build a **morphology-based taxonomy** of water-domain timeseries, then benchmark an array of forecasting models against each category. The central question is: *what morphological classes of timeseries do we actually find in water domains?*

Domain (hydrology, water quality, etc.) is **metadata**, not a taxonomy axis. Two series from different domains may share the same morphological class — that is a finding, not a problem. Conversely, series from the same domain may scatter across classes — that too is informative.

The structural template is the Monash Time Series Forecasting Archive (Godahewa et al., NeurIPS 2021): dataset inventory → feature characterization → baseline evaluation.

---

## Datasets

| ID | Domain | Variable | Frequency | Horizon | Source |
|----|--------|----------|-----------|---------|--------|
| `discharge_lowland_denmark` | hydrology (groundwater-fed) | river discharge | daily | 7 d | CAMELS-DK |
| `discharge_flashy_catskills` | hydrology (flashy) | river discharge | daily | 7 d | USGS NWIS |
| `discharge_snowmelt_yosemite` | hydrology (snowmelt) | river discharge | daily | 7 d | USGS NWIS |
| `groundwater_yorktown` | groundwater | water-table depth | daily | 30 d | USGS NWIS |
| `watertemp_james` | water quality | river temperature | daily | 7 d | USGS NWIS |
| `sewerlevel_bellinge` | urban drainage | sewer level | 15 min | 6 h | Bellinge (DTU) |
| `demand_ltown` | water distribution | network demand | hourly | 24 h | BattLeDIM L-Town |
| `wwtp_influent_n2o_350k` | wastewater | influent flow | hourly | 24 h | N₂O WWTP (Cembrowicz) |
| `surge_cuxhaven` | storm surge | non-tidal residual | hourly | 48 h | GESLA-3 + ERA5 |
| `waves_oregon` | spectral waves (swell) | significant wave height | hourly | 24 h | NOAA NDBC 46050 |
| `waves_massachusetts` | spectral waves (wind-sea) | significant wave height | hourly | 24 h | NOAA NDBC 44013 |

A small number of water-adjacent series from the Monash archive (e.g. Saugeen River flow) may be added to reach ~15 series total.

---

## Morphological feature vector

Six features computed on the **raw series** (no seasonal adjustment, no detrending). Preprocessing is deliberately avoided — the feature space should reflect the signal as observed, and multivariate features jointly disambiguate what univariate preprocessing would obscure.

| # | Feature | Captures | Implementation |
|---|---------|----------|----------------|
| 1 | ACF1 | Short-range autocorrelation | `tsfeatures` / `statsmodels` |
| 2 | Trend strength | Monotonic drift | `tsfeatures` |
| 3 | Spectral entropy | Predictability / regularity | `tsfeatures` / `antropy` |
| 4 | Seasonal strength | Periodic structure | `tsfeatures` |
| 5 | Box-Cox lambda | Variance–mean relationship | `forecast::BoxCox.lambda` / `scipy` |
| 6 | Kurtosis | Impulsiveness, heavy tails | `scipy.stats.kurtosis` |

Kurtosis is the one addition beyond the Monash 5-feature set, motivated by the prevalence of event-driven spikes in water domains (storm surge, flashy discharge, sewer overflow) that are not captured by the other five features. A series with high kurtosis *and* high seasonal strength is distinguishable from one with high kurtosis and low seasonal strength — no preprocessing needed to separate these cases.

---

## Methodology

### 1. Feature extraction
Compute the 6-feature vector for each series. For multi-seasonal series (sub-daily), use the dominant seasonality period for seasonal strength computation.

### 2. Feature space visualization
Reduce to 2D via PCA. Plot all ~15 series as labeled points. With this sample size, the goal is **characterization and positioning**, not clustering — robust cluster boundaries cannot be established from ~15 points. The contribution is showing that water-domain series occupy distinct, interpretable regions of morphological feature space, and that domain label does not predict position.

### 3. Taxonomy
Interpret the feature space geometry. Name morphological archetypes based on the regions where series concentrate (e.g. smooth-seasonal, impulsive-aperiodic, slow-memory). These labels emerge from the data, not from domain knowledge imposed a priori.

### 4. Baseline evaluation
Run a fixed model suite against each dataset using a consistent rolling-origin evaluation scheme. Models span statistical, ML, and deep learning families:

- **Naive baselines** — persistence (last value), seasonal naive
- **Statistical** — ARIMA / SARIMA, ETS, Theta
- **ML** — LightGBM with lag features
- **Deep learning** — LSTM, N-BEATS, TFT (or similar)
- **Foundation model (zero-shot)** — Chronos or TimesFM as reference

Model suite is a design choice, not a finding. The point is not to crown a winner but to show how relative performance shifts across morphological classes.

---

## Evaluation metrics

Primary: **MASE** (scale-free, handles non-stationarity, avoids division-by-zero issues of MAPE).

Secondary: MAE, RMSE (scale-dependent, useful within-dataset).

Note: MASE can be unreliable when the training period is unusually stable relative to the test period — relevant for event-driven series. Flag this where it occurs.

Separate evaluation of **event periods vs. baseline periods** is desirable for impulsive series (surge, flashy discharge, sewer level).

---

## Open questions

- **Train/test split strategy** — chronological split is standard, but event-driven series need sufficient events in both train and test. Consider stratified splitting by event count.
- **Covariates** — some series (turbidity, stormwater, surge) are nearly unpredictable from history alone and benefit from precipitation/wind forcing. Decision pending: univariate-only, or domain-appropriate covariates allowed?
- **Input window length** — standardize in step-counts or calendar time across datasets?
- **Probabilistic forecasting** — point forecasts only for now (consistent with Monash); probabilistic evaluation is future work.

---

## Key references

- Godahewa et al. (2021) — Monash Time Series Forecasting Archive. NeurIPS Datasets & Benchmarks. https://forecastingdata.org/
- Kratzert et al. (2019) — LSTM for rainfall-runoff. HESS.
- Gauch et al. (2021) — LSTM vs. conceptual models for streamflow. HESS.
- Papacharalampous et al. (2018) — Univariate benchmarks across hydrological series.
- Vrachimis et al. (2022) — BattLeDIM / L-Town competition.
- Palmitessa et al. (2021) — Sewer level forecasting, Bellinge.
