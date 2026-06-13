# Documentation

| Document | Purpose |
|---|---|
| [`RESULTS.md`](RESULTS.md) | Headline cross-case findings: the cases × methods skill matrix and analysis |
| [`cases/`](cases/) | One doc per case — domain context, drivers, exogenous, why this station/sensor, operational use |
| [`reports/`](reports/) | Auto-generated per-case reports (data summary, scores, example forecasts) — `scripts/case_report.py` |
| [`SCOPING.md`](SCOPING.md) | *Historical* — the original scoping-review request to Rocco (see [`review_responses_pr2.md`](review_responses_pr2.md)) |
| [`review_responses_pr2.md`](review_responses_pr2.md) | *Historical* — point-by-point response to Rocco's review (PR #2/#3) |

## Cases

| Case | Domain | Target | Freq | Horizon |
|---|---|---|---|---|
| [`camels_dk_discharge`](cases/camels_dk_discharge.md) | hydrology (groundwater-fed) | river discharge | daily | 7 d |
| [`usgs_river_discharge`](cases/usgs_river_discharge.md) | hydrology (flashy) | river discharge | daily | 7 d |
| [`usgs_snowmelt_discharge`](cases/usgs_snowmelt_discharge.md) | hydrology (snowmelt) | river discharge | daily | 7 d |
| [`usgs_groundwater`](cases/usgs_groundwater.md) | groundwater | depth to water | daily | 30 d |
| [`usgs_stream_temp`](cases/usgs_stream_temp.md) | water quality | river water temperature | daily | 7 d |
| [`bellinge_sewer_level`](cases/bellinge_sewer_level.md) | urban drainage | sewer water level | 15 min | 6 h |
| [`wd_battledim`](cases/wd_battledim.md) | water distribution | network demand | hourly | 24 h |
| [`wwtp_influent`](cases/wwtp_influent.md) | wastewater | influent flow | hourly | 24 h |
| [`gesla_storm_surge`](cases/gesla_storm_surge.md) | storm surge | non-tidal residual | hourly | 48 h |
| [`ndbc_waves`](cases/ndbc_waves.md) | spectral waves (swell) | significant wave height | hourly | 24 h |
| [`ndbc_waves_atlantic`](cases/ndbc_waves_atlantic.md) | spectral waves (wind-sea) | significant wave height | hourly | 24 h |

The three hydrology cases share the same target and horizon by design — a groundwater /
flashy / snowmelt contrast that tests whether method ranking is regime-robust. Likewise the
two wave cases (swell vs wind-sea).

Each case doc covers the domain, the physical drivers, exogenous covariate candidates, why
the specific catchment/sensor/station was chosen, and the operational use of the forecast.

For implementation see the top-level [`README.md`](../README.md), the original design at
[`research_plan.md`](../research_plan.md), and [`CONTRIBUTING.md`](../CONTRIBUTING.md).
