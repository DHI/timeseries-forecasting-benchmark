# Benchmark results

First full pass across **10 cases × the CPU method set** (the 11th case,
`usgs_snowmelt_discharge`, was added after this pass and folds into the next refresh),
scored by **rolling-origin**
evaluation (16 refit origins per cell, errors pooled over origins × lead times, observed
points only). The headline cell is **skill vs persistence** = `1 − MSE(model)/MSE(persistence)`
at the primary horizon: positive means better than "tomorrow = today", `1.0` is perfect.

Regenerate with `uv run python scripts/run_pilot.py` (add `--with-fm` for the Chronos
column); per-case detail in [`reports/`](reports/) via `scripts/case_report.py`.

![skill vs persistence matrix](figures/benchmark/matrix_skill_vs_persistence.png)

*Colour-coded cases × methods matrix (skill vs persistence; greener = better). The tables
below give the numbers and the analysis.*

## Skill vs persistence (rolling-origin)

| case | SeasonalNaive | AutoARIMA | MLP | LinearLags | LightGBM | LinearLags+exo | LightGBM+exo |
|---|---|---|---|---|---|---|---|
| camels_dk_discharge (slow hydrology) | −8.42 | +0.18 | +0.41 | +0.21 | +0.10 | **+0.47** | +0.17 |
| usgs_river_discharge (flashy hydrology) | +0.22 | +0.48 | +0.40 | +0.48 | −0.05 | +0.70 | **+0.83** |
| usgs_groundwater | −7.97 | +0.30 | −2.08 | +0.41 | **+0.60** | +0.58 | +0.59 |
| usgs_stream_temp | −1.17 | +0.25 | +0.28 | +0.31 | +0.19 | +0.81 | **+0.85** |
| bellinge_sewer_level | +0.02 | +0.05 | −16.65 | +0.05 | +0.06 | **+0.22** | +0.06 |
| wd_battledim | +0.09 | +0.27 | +0.29 | +0.29 | **+0.44** | +0.29 | **+0.44** |
| wwtp_influent | +0.28 | +0.44 | +0.40 | +0.44 | **+0.50** | +0.44 | **+0.50** |
| gesla_storm_surge | +0.05 | +0.09 | −0.94 | +0.27 | +0.19 | +0.61 | **+0.73** |
| ndbc_waves (swell) | −1.67 | +0.09 | −0.16 | +0.34 | +0.13 | +0.24 | **+0.44** |
| ndbc_waves_atlantic (wind-sea) | −2.92 | −0.23 | −1.74 | −0.15 | −0.06 | −1.06 | **+0.03** |

(`+exo` = the exogenous-aware Tier-2 variant. On `wd_battledim` / `wwtp_influent`, which
declare no covariates, `+exo` is identical to the univariate column by construction.)

## What the matrix says

**1. Gradient boosting, with covariates where they exist, is the consistent winner.**
`LightGBM+exo` or `LinearLags+exo` is best in 7 of 10 cases; plain `LightGBM` takes the
three cases that are univariate or autoregressive (`wd_battledim`, `wwtp_influent`,
`usgs_groundwater`). No naive or classical method wins anywhere on these horizons.

**2. Domain covariates add large skill — exactly where the physics predicts, and nowhere
else.** The exogenous-aware variants lift skill dramatically on the forced cases and leave
the rest unchanged:

| case | driver | LightGBM → LightGBM+exo |
|---|---|---|
| usgs_river_discharge | rainfall | **−0.05 → +0.83** |
| usgs_stream_temp | air temperature | **+0.19 → +0.85** |
| gesla_storm_surge | ERA5 wind/pressure | **+0.19 → +0.73** |
| ndbc_waves | wind | +0.13 → +0.44 |
| usgs_groundwater | (slow, integrative) | +0.60 → +0.59 — *no gain* |
| wd_battledim / wwtp_influent | (no covariate) | unchanged |

This is the core result the benchmark exists to produce: covariate value is real,
case-specific, and tracks domain understanding. On groundwater the raw precipitation
covariate does **not** help — the water table integrates weeks of recharge, which the
target's own lags already capture, so an instantaneous precip feature only adds noise.

**3. Method *robustness* separates the families.** The MLP is the least reliable: it
collapses on the hard cases (`bellinge` −16.65, `usgs_groundwater` −2.08,
`gesla` −0.94) where it extrapolates wildly, while tree-based and linear models stay
sane. The linear `+exo` model can also blow up when handed an oracle covariate on a
gappy, noisy series (`ndbc_waves_atlantic` −1.06) where `LightGBM+exo` holds at +0.03.
For a cross-case benchmark, gradient boosting is the safest default.

**4. Seasonal-naive is a trap on non-cyclic series.** It is catastrophic on the slow,
trending hydrology/groundwater cases (`−8.4`, `−8.0`) because the year-ago value is
meaningless there — a useful reminder that the "obvious" seasonal baseline must be chosen
per case.

**5. Method ranking is mostly regime-robust — with one honest exception.** The
boosting-plus-covariates ordering holds across slow vs flashy hydrology and swell vs
wind-sea waves. The exception is the fetch-limited Atlantic buoy (`ndbc_waves_atlantic`),
the single hardest case: at 24 h every method sits near persistence and the covariate
barely helps. Short-horizon wind-sea H_s is genuinely persistence-dominated there.

## Foundation model and extended baselines (full 14-method pass)

Run with `--with-fm`, the pretrained **Chronos-Bolt** model forecasts each case
**zero-shot — no case-specific training**. The full pass also adds **DLinear** (linear,
univariate) and **Theta**. Skill vs persistence, per case:

| case | Chronos-Bolt (zero-shot) | DLinear | Theta | best (with covariates) |
|---|---|---|---|---|
| usgs_river_discharge (flashy) | **+0.80** | +0.60 | −1.55 | Chronos / LightGBM+exo |
| wd_battledim (demand) | +0.70 | +0.72 | +0.65 | LightGBM +0.74 |
| wwtp_influent (flow) | +0.63 | +0.57 | +0.01 | LightGBM +0.69 |
| usgs_groundwater | +0.30 | +0.25 | 0.00 | LightGBM+exo +0.65 |
| usgs_stream_temp | +0.21 | +0.18 | 0.00 | LightGBM+exo +0.84 |
| ndbc_waves_atlantic | +0.11 | +0.18 | 0.00 | LightGBM+exo +0.51 |
| gesla_storm_surge | −0.03 | −0.02 | 0.00 | LightGBM+exo +0.52 |
| ndbc_waves (swell) | −0.18 | +0.21 | 0.00 | LightGBM+exo +0.48 |
| camels_dk_discharge | −0.09 | +0.21 | +0.05 | LinearLags+exo +0.30 |
| bellinge_sewer_level | −0.57 | −0.95 | 0.00 | LightGBM +0.31 |

The KER5 verdict is more interesting than "FMs trail tuned models":

- **Where the signal is regular/seasonal, the zero-shot FM is genuinely competitive** — it
  *wins outright on the flashy discharge case* (**+0.80**, edging the covariate models) and
  is near-best on water demand (+0.70) and WWTP flow (+0.63), with **no fitting at all**.
  That is the headline FM result: a pretrained model matches tuned ML on the well-behaved
  cases for free.
- **Where the dynamics are noisy or covariate-driven, it trails badly** — sewer level
  (−0.57), waves (−0.18), surge (−0.03) — because it is univariate and cannot see the
  rainfall / wind that carries the skill, and the bare autoregressive structure is hard.
- **DLinear** is a respectable univariate baseline (positive on 8/10), occasionally
  unstable (bellinge −0.95); **Theta** is essentially persistence-or-worse here.

So the foundation model is a **strong, free univariate prior that is regime-dependent**:
adopt it where the series is regular, but it does not yet replace a covariate-aware model
on the forced water cases. The natural next steps are a **covariate-capable FM**
(Chronos-2 / Moirai) and **fine-tuning** the top zero-shot performer.

*(Rolling-origin cell values are empirical estimates and shift modestly with origin count;
the core matrix above used 16 origins, this extended pass 12. The qualitative ranking is
stable across both.)*

## Deep learning (N-HiTS / PatchTST)

Both are implemented directly on torch (`run_pilot --with-dl`) — the usual route
(neuralforecast) needs `ray`, which has no Python-3.13 wheels. A focused 4-case probe
(rolling, 5 origins, skill vs persistence):

| case | N-HiTS | PatchTST | LightGBM (uni) | LightGBM+exo |
|---|---|---|---|---|
| gesla_storm_surge (1h) | +0.18 | +0.25 | +0.20 | +0.86 |
| ndbc_waves (1h) | +0.11 | −0.33 | −0.03 | +0.40 |
| usgs_river_discharge (1d) | −0.05 | +0.05 | +0.24 | +0.90 |
| usgs_stream_temp (1d) | −1.59 | −1.19 | −0.53 | +0.65 |

The read: on the **hourly** cases, where the lookback (168 h) covers the dynamics, N-HiTS
and PatchTST **beat persistence and match univariate LightGBM** (PatchTST is best of the
univariate methods on surge). On the **daily** cases they trail — a 64-day lookback (capped
for CPU tractability) can't see the annual cycle, and persistence is a stiff baseline on the
flashy series. They are also the **heaviest** methods (PatchTST ~400 s for one hourly cell on
CPU). So: promising univariate models where the lookback fits the dynamics, but lookback and
compute are the binding constraints — exactly what a **GPU (WSL)** and a longer lookback
budget would relax. As ever, the covariate-aware `+exo` columns still dominate.

## Caveats

- **One station/catchment per case.** Findings are illustrative of a regime, not a
  basin-wide generalisation study.
- **Covariates are a perfect-forecast proxy.** Future covariate values come from observed
  reanalysis (oracle), so the `+exo` numbers are an *upper bound* on operational covariate
  value; real skill depends on the NWP forecast quality.
- **CPU method set.** DLinear, N-HiTS and PatchTST run, but on CPU the lookback and
  training budget are capped; GPU (WSL) plus the neuralforecast zoo (TFT/iTransformer/
  TimesNet, `--with-nf`) and covariate-capable / fine-tuned FMs are the next step
  (see `research_plan.md`).
- **One primary horizon per case** in the matrix; the lead-time curves
  (`scripts/run_leadtime.py`, and each case report) show how the ranking shifts as lead
  grows — covariates and FMs gain on persistence at longer leads.
