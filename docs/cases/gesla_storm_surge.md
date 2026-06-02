# `gesla_storm_surge` — non-tidal sea-level residual at a coastal tide gauge

**Domain**: Storm surge · **Target**: non-tidal residual (m) · **Horizon**: 48 hours · **Data**: GESLA-3

## What this is

Coastal sea level = astronomical tide (predictable to centimetres years ahead) + mean sea-level trend + **non-tidal residual**. The residual is dominated by meteorological forcing: low pressure lifts the surface (~1 cm per 1 hPa, the inverse barometer effect) and onshore winds pile water against the coast (wind setup). When a storm coincides with high tide, this is the **storm-surge** flood-risk problem.

We forecast the *residual*, not the raw level: the tide is removed via UTide harmonic analysis before the canonical Parquet is written, so this case is exclusively about the meteorologically driven part. GESLA-3 (Haigh et al., 2023) provides **5,119 stations globally**, sub-hourly resolution, unified to UTC with QC flags.

```mermaid
flowchart LR
    NWP([NWP forecast<br/>MSLP, 10 m wind]) --> O[Ocean / shelf dynamics]
    O --> SL[Observed sea level]
    T([Astronomical tide<br/>UTide]) -->|subtract| SL
    SL -.->|residual| R((y))
```

![EDA](../figures/cases/gesla_storm_surge_eda.png)

## The forecasting problem

Forecast the **non-tidal residual** at one tide gauge, **hourly**, **48 hours** ahead. The horizon matches operational coastal-flood-warning lead times (DMI, Met Office, NOAA).

Station pinned to **Cuxhaven_Steubenhoft (`Cux`)** by `--scan` over the UK + Atlantic NW Europe bounding box: 108 years (1917–2024), shallow German Bight shelf, surge-rich climate (the 1962 Hamburg flood site), operational benchmark for the Elbe estuary and Hamburg port. Long-record + recent + active makes it the highest-leverage single station in the open data.

## Dynamics and drivers

Inverse barometer + wind setup, both highly nonlinear in shallow shelves like the North Sea. Tide-surge interaction modulates the signal nonlinearly with the tide. Far-field forcing (a storm in the Norwegian Sea drives a Kelvin wave to the German Bight days later) adds a long-memory component. Persistence works for a few hours; beyond ~6–12 h, **skill comes almost entirely from the meteorological forecast** driving the surge — this is the canonical case for *known-future* exogenous dominance.

## Exogenous candidates

- **Past forcing**: ERA5 MSLP + 10 m wind components at the station's grid point.
- **Forecast forcing** (known-future): IFS/GFS at the time of forecast issue — what an operational system uses.
- **Process-model reference column**: Copernicus Marine global surge product; the realistic operational baseline to beat. Hybrid ML residual correction on top is the state-of-practice pipeline.

## Operational relevance

Coastal flood warnings (DMI, EFAS), port closures and mooring decisions, sluice/gate operation (Maeslantkering, Thames Barrier), dyke-height design via extreme-value analysis. For DHI specifically this case bridges the benchmark to **MIKE 21** hydrodynamic forecasts — any external publication should be honest that MIKE-21 + bias correction is the realistic operational competition.

## References

- Haigh et al. (2023). *GESLA Version 3.* Geoscience Data Journal 10, 293. [doi:10.1002/gdj3.174](https://doi.org/10.1002/gdj3.174)
- Codiga (2011). *UTide.* Tide harmonic analysis (Python port `utide`).
- Copernicus Marine global surge: [data.marine.copernicus.eu](https://data.marine.copernicus.eu/)
