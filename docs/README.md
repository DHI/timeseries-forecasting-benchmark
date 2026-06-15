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
| [`discharge_lowland_denmark`](cases/discharge_lowland_denmark.md) | hydrology (groundwater-fed) | river discharge | daily | 7 d |
| [`discharge_flashy_catskills`](cases/discharge_flashy_catskills.md) | hydrology (flashy) | river discharge | daily | 7 d |
| [`discharge_snowmelt_yosemite`](cases/discharge_snowmelt_yosemite.md) | hydrology (snowmelt) | river discharge | daily | 7 d |
| [`groundwater_yorktown`](cases/groundwater_yorktown.md) | groundwater | depth to water | daily | 30 d |
| [`watertemp_james`](cases/watertemp_james.md) | water quality | river water temperature | daily | 7 d |
| [`sewerlevel_bellinge`](cases/sewerlevel_bellinge.md) | urban drainage | sewer water level | 15 min | 6 h |
| [`demand_ltown`](cases/demand_ltown.md) | water distribution | network demand | hourly | 24 h |
| [`wwtp_influent_n2o_350k`](cases/wwtp_influent_n2o_350k.md) | wastewater | influent flow | hourly | 24 h |
| [`surge_cuxhaven`](cases/surge_cuxhaven.md) | storm surge | non-tidal residual | hourly | 48 h |
| [`waves_oregon`](cases/waves_oregon.md) | spectral waves (swell) | significant wave height | hourly | 24 h |
| [`waves_massachusetts`](cases/waves_massachusetts.md) | spectral waves (wind-sea) | significant wave height | hourly | 24 h |

The three hydrology cases share the same target and horizon by design — a groundwater /
flashy / snowmelt contrast that tests whether method ranking is regime-robust. Likewise the
two wave cases (swell vs wind-sea).

Each case doc covers the domain, the physical drivers, exogenous covariate candidates, why
the specific catchment/sensor/station was chosen, and the operational use of the forecast.

For implementation see the top-level [`README.md`](../README.md), the original design at
[`research_plan.md`](../research_plan.md), and [`CONTRIBUTING.md`](../CONTRIBUTING.md).
