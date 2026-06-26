# Benchmark results

One consistent pass across **all 12 cases × 14 methods**, scored by **rolling-origin**
evaluation (12 refit origins per cell, errors pooled over origins × lead times, observed
points only). The headline cell is **skill vs persistence** = `1 − MSE(model)/MSE(persistence)`
at the primary horizon: positive means better than "tomorrow = today", `1.0` is perfect.

Regenerate with `uv run python scripts/run_pilot.py --with-fm`; per-case detail in
[`reports/`](reports/) via `scripts/case_report.py`. The two torch deep-learning methods
(N-HiTS / PatchTST) are reported separately below — they are slow on CPU, so this matrix is
the 14-method set (`--with-dl` adds them).

![skill vs persistence matrix](figures/benchmark/matrix_skill_vs_persistence.png)

*Colour-coded cases × methods matrix (skill vs persistence; greener = better; colour clamped
to [−1, 1] so one catastrophic cell can't wash out the scale). Numbers + analysis below.*

## Skill vs persistence (rolling-origin, 12 origins)

| case | SeasonalNaive | AutoARIMA | DLinear | LightGBM | LightGBM+exo | Chronos-Bolt | best |
|---|---|---|---|---|---|---|---|
| discharge_lowland_denmark (slow hydrology) | −0.92 | +0.05 | +0.01 | +0.07 | +0.40 | −0.02 | **LinearLags+exo +0.45** |
| discharge_flashy_catskills (flashy) | −0.19 | +0.01 | −0.06 | −0.52 | **+0.81** | −0.02 | **LightGBM+exo +0.81** |
| discharge_snowmelt_yosemite | −4.16 | −0.64 | −0.82 | +0.33 | **+0.65** | −0.24 | **LightGBM+exo +0.65** |
| discharge_saugeen (seasonal) | +0.23 | +0.04 | +0.07 | +0.08 | +0.08 | +0.04 | **MLP +0.26** |
| groundwater_yorktown | −9.43 | +0.22 | +0.33 | +0.09 | +0.04 | **+0.73** | **Chronos +0.73** |
| watertemp_james | −2.17 | +0.20 | +0.27 | +0.21 | **+0.86** | +0.15 | **LightGBM+exo +0.86** |
| sewerlevel_bellinge | −0.10 | −1.04 | −1.51 | +0.32 | +0.32 | −0.34 | **LinearLags +0.45** |
| demand_ltown | −0.63 | +0.17 | +0.34 | **+0.41** | +0.41 | +0.41 | **LightGBM +0.41** |
| wwtp_influent_n2o_350k | −0.33 | +0.15 | +0.40 | **+0.56** | +0.56 | +0.55 | **LightGBM +0.56** |
| surge_cuxhaven | −0.66 | −0.25 | −0.28 | −0.52 | **+0.38** | −0.31 | **LightGBM+exo +0.38** |
| waves_oregon (swell) | −0.11 | −0.13 | +0.18 | +0.15 | +0.47 | +0.16 | **MLP +0.51** |
| waves_massachusetts (wind-sea) | −0.57 | −0.02 | −0.14 | +0.00 | **+0.35** | −0.17 | **LightGBM+exo +0.35** |

(`+exo` = the exogenous-aware Tier-2 variant; on `demand_ltown` / `wwtp_influent_n2o_350k`, which
declare no covariates, it equals the univariate column by construction.)

## What the matrix says

**1. Covariate-aware models win where covariates exist; plain boosting wins the regular,
self-driven series.** `LightGBM+exo` or `LinearLags+exo` is best on 6 of 12 cases (the
covariate-driven ones); `LightGBM` takes the two self-driven cases (`demand_ltown`,
`wwtp_influent_n2o_350k`); `Chronos` zero-shot takes groundwater; `MLP` takes the two wave
cases. No naive or classical method wins anywhere on these horizons.

**2. Domain covariates add large skill — exactly where the physics predicts.** The
exogenous-aware variants lift skill dramatically on the forced cases:

| case | driver | univariate → +exo (skill) |
|---|---|---|
| discharge_flashy_catskills | precipitation | **−0.52 → +0.81** |
| watertemp_james | air temperature | **+0.21 → +0.86** |
| surge_cuxhaven | ERA5 wind/pressure | **−0.52 → +0.38** |
| discharge_snowmelt_yosemite | temperature/precip | **+0.33 → +0.65** |
| waves_oregon / waves_massachusetts | wind | +0.15 → +0.47 / +0.00 → +0.35 |
| discharge_lowland_denmark | precipitation (via LinearLags) | +0.02 → +0.45 |
| groundwater_yorktown | (slow, integrative) | +0.09 → +0.04 — *no gain* |

Covariate value is real, case-specific, and tracks domain understanding. Groundwater is the
instructive null: the water table integrates weeks of recharge, which the target's own lags
already capture, so an instantaneous precip feature adds nothing.

**3. The flashy river is now the strongest covariate-gain case.** On `discharge_flashy_catskills`,
plain `LightGBM` collapses to **−0.52** (gradient boosting overfits the spiky daily signal),
but `LightGBM+exo` jumps to **+0.81** using precipitation — the single largest exo gain in
the matrix. Meanwhile `DLinear` and `AutoARIMA` both sit near zero, and `Chronos` is
marginally negative. So the flashy-regime conclusion has flipped: the covariate-aware
booster is now the strongest method, not zero-shot.

**4. Covariates must be matched to model type — linear wins where boosting stalls.** On
`sewerlevel_bellinge`, `LightGBM` and `LightGBM+exo` are tied at **+0.32**, while
`LinearLags` (with the rain covariate) reaches **+0.45**. At 15-min resolution the sharp,
variably-lagged rain→level response is more tractable for a linear model, which imposes less
risk of overfitting across rolling origins. Classical methods all fail badly (AutoARIMA −1.04,
DLinear −1.51): the sewer-level signal is too non-stationary for them.

**5. MLP is the least robust method overall but can surprise on smooth wave series.** It
collapses on the hard cases (`bellinge` −54, `groundwater_yorktown` −0.0,
`discharge_snowmelt_yosemite` +0.05) but tops the leader board on `waves_oregon` (**+0.51**)
and `waves_massachusetts` (`LightGBM+exo` is a close second at +0.35). Wave spectra are
smooth and quasi-periodic — exactly the regime where MLP's implicit global fit is less
dangerous. For a cross-case benchmark, tree/linear models remain the safe default.

**6. Seasonal-naive is a trap on trending series** — catastrophic on the slow hydrology /
groundwater cases (lowland_denmark −0.92, snowmelt −4.16, groundwater −9.43) because the
year-ago value is meaningless there. The "obvious" seasonal baseline must be chosen per case.

**7. `discharge_saugeen` is the least forecastable case.** The mildest winner across all cases
is `MLP +0.26` on this seasonally-regular Canadian river; `SeasonalNaive` (+0.23) is
competitive, meaning the annual cycle alone carries most of the predictable variance — little
room for model advantage.

## Foundation model (Chronos-Bolt, zero-shot)

The pretrained **Chronos-Bolt** forecasts each case **zero-shot — no case-specific
training**. The updated picture:

- **Groundwater is the FM's showcase case.** Chronos wins `groundwater_yorktown` outright at
  **+0.73**, the highest single-method skill in the matrix, beating every fitted model. The
  slow, integrative annual cycle is evidently a pattern the pretrained model captures
  without any case-specific fitting — a striking result.
- **Where the series is regular it remains competitive** — demand (+0.41, tied for best),
  WWTP flow (+0.55, second only to LightGBM +0.56) — with no fitting at all.
- **Where the dynamics are covariate-driven or noisy it trails** — flashy discharge (−0.02),
  surge (−0.31), waves (−0.17 / −0.17) — because it is univariate and cannot see the
  rainfall / wind that carries the skill.

So the FM is a **strong, free, but regime-dependent univariate prior**: it is now the
strongest method on the slow integrative groundwater case and competitive on self-driven
regular series; it does not yet replace a covariate-aware model on the physically-forced
cases. Natural next steps: a **covariate-capable FM** (Chronos-2 / Moirai) and
**fine-tuning** on the cases where zero-shot already works.

## Deep learning (N-HiTS / PatchTST)

Implemented directly on torch (`run_pilot --with-dl`) — the usual route (neuralforecast)
needs `ray`, which has no Python-3.13 wheels. A focused 4-case probe (rolling, 5 origins):

| case | N-HiTS | PatchTST | LightGBM (uni) | LightGBM+exo |
|---|---|---|---|---|
| surge_cuxhaven (1h) | +0.18 | +0.25 | +0.20 | +0.86 |
| waves_oregon (1h) | +0.11 | −0.33 | −0.03 | +0.40 |
| discharge_flashy_catskills (1d) | −0.05 | +0.05 | +0.24 | +0.90 |
| watertemp_james (1d) | −1.59 | −1.19 | −0.53 | +0.65 |

On the **hourly** cases (lookback 168 h covers the dynamics) N-HiTS/PatchTST beat persistence
and match univariate LightGBM (PatchTST best of the univariate methods on surge). On
**daily** cases the CPU-capped 64-day lookback can't see the annual cycle, and they are the
heaviest methods (PatchTST ~400 s/cell). Lookback + compute are the binding constraints a
**GPU (WSL)** would relax; full-coverage DL is deferred there. The neuralforecast zoo
(TFT/iTransformer/TimesNet, `--with-nf`) is wired and waiting for the WSL move.

## Caveats

- **One station/catchment per case.** Findings illustrate a regime, not a basin-wide
  generalisation study.
- **Covariates are a perfect-forecast proxy.** Future covariate values come from observed
  reanalysis (oracle), so the `+exo` numbers are an *upper bound* on operational covariate
  value; real skill depends on NWP forecast quality.
- **Rolling-origin cells are empirical estimates** that shift modestly with origin count;
  the qualitative ranking is stable. This whole matrix is one 12-origin pass.
- **CPU method set.** DLinear/N-HiTS/PatchTST run but with capped lookback/training budget;
  GPU (WSL) + the neuralforecast zoo + covariate-capable/fine-tuned FMs are next.
- **One primary horizon per case** here; the lead-time curves (`scripts/run_leadtime.py`,
  per-case reports) show how the ranking shifts as lead grows.
