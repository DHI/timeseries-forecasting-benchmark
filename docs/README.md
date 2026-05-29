# Documentation

| Document | Audience | Purpose |
|---|---|---|
| [`SCOPING.md`](SCOPING.md) | Reviewer (Rocco) | Structured review request: cases, targets, horizons, metrics |
| [`cases/camels_dk_discharge.md`](cases/camels_dk_discharge.md) | Both | Daily river discharge, groundwater-dominated DK lowland catchment |
| [`cases/bellinge_sewer_level.md`](cases/bellinge_sewer_level.md) | Both | 15-min combined-sewer water level (Odense, DK) |
| [`cases/wd_battledim.md`](cases/wd_battledim.md) | Both | Hourly inlet flow into the L-Town water distribution network |
| [`cases/wwtp_influent.md`](cases/wwtp_influent.md) | Both | Hourly influent flow into a 350k-PE wastewater treatment plant |
| [`cases/gesla_storm_surge.md`](cases/gesla_storm_surge.md) | Both | Hourly non-tidal sea-level residual at a coastal tide gauge |
| [`cases/ndbc_waves.md`](cases/ndbc_waves.md) | Both | Hourly significant wave height at a NOAA NDBC buoy |

**Both** = aimed at a reader with strong time-series forecasting background (Jaume) but no specific water-domain knowledge, *and* at a domain expert (Rocco) for scoping validation.

Each case doc covers the domain context, the physical drivers, exogenous covariate candidates, why the specific catchment/sensor/station was chosen, and the operational use of forecasts at the chosen horizon. The [`SCOPING.md`](SCOPING.md) document collects the design decisions we want feedback on across cases.

For implementation details see the top-level [`README.md`](../README.md), the full design at [`research_plan.md`](../research_plan.md), and the contributor guide at [`CONTRIBUTING.md`](../CONTRIBUTING.md).
