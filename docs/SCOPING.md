# Scoping & review

**For**: Rocco (scoping review, urban + catchment hydrology)
**Authors**: Jesper (project lead); Claude (drafting support)
**Estimated review time**: 30–60 minutes

## What we want from this review

We have committed to running a forecasting benchmark across DHI's water and environment domains (research outcome **RK2026 / KER 5**, "next-generation modelling"). The full plan is in [`research_plan.md`](../research_plan.md); the design is now far enough along that the *cases* and the overall *scoping* are the thing to lock in before Jaume takes over the bulk of the engineering.

From you specifically we want decisions on four things:

1. **Cases**. Are these the right six cases for the domains we're claiming to cover? Are the chosen sensors/catchments/stations representative? What's missing that should be added before we go further?
The inland cases look all relevant, can't think of any major subdmain overlooked. Be aware that combined collection system and wwtp inflow can have almost the same dynamics, so ensure they are different enough.
2. **Targets and horizons**. Is each `(target, horizon)` operationally meaningful? Would a different horizon make the case more useful?
see case-specific comments. Overall, not the most critical aspect; make sure instead the exogenous data is always relevant (e.g. when using observed rainfall as replacement of weather forecasts make this explicitly clear to the reader)
3. **Metrics**. We have MAE, RMSE, MASE (vs. seasonal naive), and skill vs. persistence. What domain-specific metrics matter for *your* domains that we're missing? (Event hit rate for CSO, peak-flow bias for floods, POD/FAR at warning thresholds for surge, etc.)
Consider using NSE and KGE for water level and flow predictions inland
4. **What's wrong or missing**. Open ended — anything that would make you uncomfortable showing this to an external partner.
Make it clear the goal is not to find the perfect model for each case but to have a somewhat direct comparison of different models accross cases

The case docs in [`docs/cases/`](cases/) are the primary input; one per case, 2–3 pages, written to explain the domain for a reader (Jaume) with strong TS forecasting but no water background. Please read them in any order, treat them as drafts, and challenge anything that doesn't match operational reality.

If a 20-minute walkthrough first would help, Jesper can set one up.

## Status of the work so far
I've limited my review to the cases, so I haven't looked at the preliminary forecasts yet, but these should be investigated to check the cases where selected meaningfully
- Six cases implemented, five on real data, one (`gesla_storm_surge`) waiting on a manual GESLA archive download.
- Nine methods running: four naive baselines (Mean, Persistence, SeasonalNaive, Climatology), two classical (AutoARIMA, AutoETS), three windowed ML (LinearLags = Ridge, LightGBM, MLP).
- Two output artefacts: the **cases × methods matrix** (MASE + skill vs persistence) and the **error-vs-lead-time curves** averaged over rolling origins.
- Standard pipeline: long-format Parquet on disk, pandas in memory, fraction-or-calendar splits, NaN-masked metrics on gappy series.
- CI runs on GitHub Actions; the repo is private under `DHI-Internal/timeseries-forecasting-benchmark`.

Deliberately **not yet** built (the next things on the queue, which we want to defer until the scoping is clean):

- Exogenous-aware variants of the windowed-ML methods (the current LightGBM etc. are univariate).
- Recurrent nets (LSTM, GRU), modern deep learning (DLinear, N-HiTS, PatchTST, iTransformer, TFT), and time-series foundation models (Chronos, TimesFM, Moirai, ...).
- Hybrid process-model + ML residual correction.
- Per-case publishable reports with example forecasts (Phase F).
- Streamlit UI for browsing the matrix.

## Specific decision points by case

For each case we list the design choices we've made and where your domain input would change things. The case docs justify each in more detail.

### `camels_dk_discharge` — daily discharge, 7-day horizon

We chose CAMELS-DK catchment **`37470707`** by ranking the 304 gauged catchments by **baseflow index** (BFI 0.971, top of the distribution) and **low slope** (0.88, lowland) to make it a defensibly "typical Danish groundwater-dominated lowland" gauge.

- Is this the right *category* of catchment to lead with, or would a more rainfall-runoff-driven catchment make the matrix more informative? not sure. might be validated by domain expert
- Is 7 days the right horizon, or should we evaluate at 1 / 3 / 7 / 14 / 30 (seasonal forecast)? isn't the second objective to create error-vs-lead-time curves?
- Should we include `q_sim` from the DK-model as a process-model reference column from day one? yes if there's no equivalent high quality observations of streamflow
- Does the open question of contrasting with a flashy catchment (Caravan / LamaH-CE) need to happen before we expand methods, or can it wait? it can wait

### `bellinge_sewer_level` — 15-min sewer level, 6-hour horizon
This is incosistent with the case description. Which sensor was actually used?
We chose sensor **`G71F04R_Level1_System2000p1`** as a continuous level record on the combined main; the original Bellinge paper highlights F04R as representative.

- Is F04R the right primary sensor, or would a CSO-chamber level make a more operationally compelling case?
- 6-hour horizon: too long, too short, or right?
- Should the case be the *event-aware* version (CSO event hit rate, peak-level bias) rather than continuous MAE/RMSE? The Bellinge data supports the event framing trivially. Check the bellinge article for inspiration, but yes event-based metrics are often more meaningful.
- Is the X-band radar nowcast as known-future exogenous a realistic operational story, or do we need to be honest that most utilities don't have it? I believe the rain gauge is good enough, check the original paper on this

### `wd_battledim` — hourly inlet flow, 24-hour horizon
need to be validated with domain expert (try CRCO)
L-Town `PUMP_1`, synthetic SCADA from EPANET simulation. The benchmark forecasts the 24-hour demand pattern at the largest inlet pump.

- Is "inlet flow at the largest pump" the right target, or would district demand (sub-zone flow / pressure) be more operationally meaningful?
- 24 hours for pump scheduling — agree?
- Should we add a leak-detection extension? The 2019 evaluation period has leaks; a "demand-forecast residual = leak signal" anomaly metric writes itself.

### `wwtp_influent` — hourly influent flow, 24-hour horizon

Mendeley N₂O dataset; target is `INLET.Q value`, the plant inlet totaliser.

- Is influent flow the right primary target for a wastewater case, or should it be **effluent NH₄** (compliance-relevant) or **N₂O** (the dataset's published focus)? influent discharge or concentration
- 24-hour horizon for aeration optimisation — agree? check case comments
- The same dataset trivially yields two more cases (`wwtp_effluent_nh4`, `wwtp_n2o`). Is one rich-dataset-three-cases the right strategy, or should we add a different plant? i wouldn't mix influent and effluent
- A synthetic BSM2 case alongside the real plant — useful or distracting? not useful

### `gesla_storm_surge` — hourly residual, 48-hour horizon
not checked
Non-tidal residual after UTide harmonic removal. Station not yet chosen (Esbjerg / Hirtshals / Sheerness candidates).

- Which station? Danish coverage is desirable; richness of surge climate matters too.
- 48-hour horizon — matches operational warning systems; agree?
- Process-model reference: Copernicus Marine surge product as a "free" baseline column. Worth pulling in?
- Tide-surge interaction — should we be evaluating *total water level* rather than *residual*, since that's what matters operationally for flood thresholds?

### `ndbc_waves` — hourly H_s, 24-hour horizon
not checked
NDBC station 46050 (Stonewall Bank, Oregon). Open-ocean Pacific buoy with a long record.

- Pacific buoy vs. a North Sea / Danish buoy: which is the right primary case? (Both for DHI relevance and for swell-dominated vs. wind-sea-dominated dynamics.)
- 24-hour horizon — short for ship routing, OK for surf forecasting; what's the right operational story?
- Add peak period T_p as a secondary target?
- Process-model reference: a Copernicus NW Shelf wave hindcast point. Same question as for surge.

## Cross-cutting design choices we'd like your reaction to

1. **Split mode**: calendar dates per case (with explicit train/val/test ends) vs. observed-span fractions (e.g. 70/15/15 of non-NaN obs). We default to calendar; CAMELS-DK uses fractions because gauged catchments have varying record periods. Is this confusing? Should we standardise? not critical if different accross cases.
2. **Gap handling**: training-time linear interpolation + mean-fill, scoring on observed-only test points. Defensible? yes if traning gaps are within a certain limit (depends on resolution of the data, doesn't need to be the same for all cases)
3. **Univariate first**: all method classes can take exogenous inputs eventually, but our pilot is univariate. The right way to introduce exogenous (lagged-only vs known-future-aware) matters for fairness across cases; any preferences from the operational side? to be addressed case specific. in some domains univariate does not make sense (e.g. collection systems)
4. **Process-model reference columns**: we've talked about them but not implemented. Should they be added before Jaume picks this up, or is that his first piece of work? if you mean comparing to simulated data in addition to observed, I don't think it's actually necessary as it sort of complicates the picture
5. **Domain-specific metrics**: which cases need them, and which can we honestly run with generic MAE/RMSE/MASE? Event metrics for CSO (case 4–5), POD/FAR for surge (8), NSE/KGE for hydrology (1), peak-flow ratio for both 1 and 4–5 are the obvious additions. the one mentioned are very good guesses. But is it really useful as it breaks the ability to compare the efficacy of models accross domains? maybe just find some dimensionless score that work accross all cases (e.g. MAPE)

## Data-quality findings surfaced by the EDA panels

Two issues caught by running `scripts/case_eda.py` over the committed processed
Parquets. Both are honest reports we'd rather you see now than discover later:

- **Bellinge** — the chosen sensor instance (`G71F04R_Level1_System2000p1`) has
  continuous level dynamics only in 2010–2012 plus briefly in late 2019; the rest
  of the record reads near zero (sensor not active under that SCADA system).
  Earlier "real-data" Bellinge metrics (e.g. SeasonalNaive MASE 0.32) were
  effectively scored on near-zero data and don't mean what we said they meant.
  Fix in progress via a `--scan` command added to the Bellinge download that
  reports time span and active-row count per sensor instance, so a
  continuously-active sensor can be picked on evidence rather than by guessing.
  Might be because 04R has two sensors level1 and level2 of which only one is active for the most part of the period
- **BattLeDIM** — `PUMP_1` alone alternates with PUMP_2 (operational pump
  scheduling), so the single-pump signal is a square wave, not a demand signal.
  The download is now updated to **sum all PUMP_* columns** as the network-total
  demand target, which is the operationally meaningful one.

These won't bias your case-design review, but they explain any earlier numbers we
showed for those two cases that looked surprisingly easy.

## Open items not for this review (just so you know they exist)

- Process-model reference columns + hybrid ML residual correction (we know this is the operational reality but it's not implemented yet).
- Deep learning + foundation models (deferred until the scoping is locked).
- A second contrasting hydrology case (flashy catchment).
- The remaining datasets in the original plan (GEMS-GER groundwater, Caravan, Copernicus chlorophyll, sediments).
- Streamlit interactive matrix.
- External release. Currently private under `DHI-Internal/timeseries-forecasting-benchmark`.

## How to respond

Either:
- Annotate the case docs and SCOPING directly via a PR / branch, or
- Send back this document with comments in-line, or
- Walk through it with Jesper in a session and we'll capture decisions.

Whatever is fastest for you.
