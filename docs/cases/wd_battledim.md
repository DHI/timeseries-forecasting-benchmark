# `wd_battledim` — inlet flow into a water distribution network

**Domain**: Water distribution · **Target**: total network demand (L/s) · **Horizon**: 24 hours · **Data**: BattLeDIM L-Town (Cyprus)

## What this is

A water distribution network delivers treated drinking water from sources through a pressurised pipe system to consumers. There's almost no storage *inside* the network, so total inlet flow equals total consumer **demand plus leakage** in near-real-time (the leakage component is what BattLeDIM's 2019 leak scenarios perturb). Forecasting demand 24 hours ahead drives pump scheduling, energy contracts, and is the baseline against which leaks are detected. *(`demand` → `demand plus leakage` per Rocco, PR #2.)*

**BattLeDIM 2020** (Vrachimis et al., 2020) models **L-Town**, a real Cyprus network with 2 reservoirs, 782 nodes, 905 pipes, ~10,000 consumers. The data is **EPANET-simulated SCADA at 5-min**, covering 2018 (training) and 2019 (evaluation with introduced leak events).

```mermaid
flowchart LR
    R1[Reservoir 1] --> N[Distribution network]
    R2[Reservoir 2] --> N
    N --> T[Tanks]
    N --> C([10000 consumers])
    L([Leaks in 2019]) -.->|added| N
```

![EDA](../figures/cases/wd_battledim_eda.png)

*EDA panel: **total network demand** = sum of all node demands from `SCADA_Demands.csv`,
resampled to hourly. **To regenerate (Rocco, PR #2):** label the panel so it's clear which
series is plotted (the summed demand, not a single pump).*

## The forecasting problem

Forecast **total network demand** (sum of consumer demand across all nodes from `SCADA_Demands.csv`) at **hourly** resolution, **24 hours** ahead. The hourly cadence matches operational pump scheduling; 24 h spans a full diurnal cycle. *(Rocco, PR #2: worth validating this case with a water-distribution domain expert — e.g. CRCO.)*

We deliberately do **not** use `SCADA_Flows.csv`: its three channels are `PUMP_1` (refill schedule, ON/OFF on tank setpoint — not demand) plus two internal pipe flowmeters at sub-zone boundaries. Demand flows from the *consumer side*, smoothed by the tank.

The data being simulated is a feature: ground truth is exact, no sensor noise, demand drivers are clean. The case is also the standard reference for leak-detection research, so methods have an obvious downstream application.

## Dynamics and drivers

Three highly predictable components dominate: a strong diurnal pattern (morning + evening peaks, deep overnight minimum), a weekly modulation (weekday ~5–10% higher than weekend), and a summer seasonal lift (irrigation, especially in Cyprus). On top of that: small noise, occasional industrial spikes, and (in 2019) leak-induced anomalies. **A small calendar + temperature feature set already explains most of the variance** — seasonal-naive and gradient-boosting baselines are strong here.

## Exogenous candidates

- **Calendar features**: hour, day-of-week, month, holidays (almost all the signal).
- **Temperature** — drives outdoor / summer demand. ERA5 reanalysis at the L-Town location is the natural source.
- **Other district inlet flows / tank levels** — mass-balance constraints if multiple zones forecast jointly.
- **Leak ground truth (2019)** — counterfactual exogenous for anomaly-baseline studies.

## Operational relevance

Pump scheduling and energy optimisation (pumps are the bulk of a utility's electricity bill; off-peak shifting and tank storage exploit the forecast). Energy-market participation. **Leak detection** — sudden, sustained forecast-vs-obs residual = leak signature. Maintenance planning, peak-day capacity decisions. Most utilities today still use simple persistence / day-of-week-average baselines.

## References

- Vrachimis et al. (2020). *BattLeDIM: Battle of the Leakage Detection and Isolation Methods.* CCWI/WDSA. [battledim.ucy.ac.cy](https://battledim.ucy.ac.cy/) · Dataset: [zenodo:4017659](https://zenodo.org/records/4017659)
- Donkor et al. (2014). *Urban water demand forecasting: review.* J. Water Resour. Plann. Manage. 140, 146.
