# Scoping & review

> **⚠️ Historical (June 2026).** This was the original scoping-review request sent to Rocco
> when the benchmark had six cases and nine methods. It is kept for the design rationale
> only — the review is **done** (responses in [`review_responses_pr2.md`](review_responses_pr2.md)),
> and the project has since grown to 11 cases and 16 methods. For the current state see the
> top-level [`README.md`](../README.md) and the results in [`RESULTS.md`](RESULTS.md). The
> counts and "not yet built" lists below are no longer accurate.

**For**: Rocco (scoping review, urban + catchment hydrology)
**Authors**: Jesper (project lead); Claude (drafting support)
**Estimated review time**: 30–60 minutes

## What we want from this review

We have committed to running a forecasting benchmark across DHI's water and environment domains (research outcome **RK2026 / KER 5**, "next-generation modelling"). The full plan is in [`research_plan.md`](../research_plan.md); the design is now far enough along that the *cases* and the overall *scoping* are the thing to lock in before Jaume takes over the bulk of the engineering.

> **Purpose (clarified after Rocco's review).** The goal is **not** to find the single
> best model for each case. It is to enable a **direct, fair comparison of the same
> methods across very different water/environment cases**. That drives two design
> consequences: the headline matrix cell stays a **dimensionless** skill score (so cases
> are comparable), and domain-specific metrics live as **per-case sidebars**, not as
> matrix cells.

From you specifically we want decisions on four things:

1. **Cases**. Are these the right six cases for the domains we're claiming to cover? Are the chosen sensors/catchments/stations representative? What's missing that should be added before we go further?
2. **Targets and horizons**. Is each `(target, horizon)` operationally meaningful? Would a different horizon make the case more useful? *(Per Rocco: horizon is not the most critical aspect — the more important thing is that the **exogenous data is genuinely relevant**, and that where observed rainfall stands in for a weather forecast we **say so explicitly**. The error-vs-lead-time curves already explore sub-horizons, so we do not multiply primary horizons.)*
3. **Metrics**. We have MAE, RMSE, MASE (vs. seasonal naive), and skill vs. persistence. What domain-specific metrics matter for *your* domains that we're missing? (Event hit rate for CSO, peak-flow bias for floods, POD/FAR at warning thresholds for surge, etc.) *(Per Rocco: add **NSE and KGE** for inland water-level/flow cases; but keep the **primary cross-case cell dimensionless** — MASE, and consider MAPE/sMAPE — so domains stay comparable, with NSE/KGE/event metrics as per-case sidebars.)*
4. **What's wrong or missing**. Open ended — anything that would make you uncomfortable showing this to an external partner.

The case docs in [`docs/cases/`](cases/) are the primary input; one per case, 2–3 pages, written to explain the domain for a reader (Jaume) with strong TS forecasting but no water background. Please read them in any order, treat them as drafts, and challenge anything that doesn't match operational reality.

If a 20-minute walkthrough first would help, Jesper can set one up.

## Status of the work so far

> **Reviewer note (Rocco, PR #2).** His review was limited to the *cases* — he did **not**
> look at the preliminary forecasts. Those should be investigated to confirm each case was
> selected meaningfully (full-enough records, exogenous inputs with real predictive skill, a
> sane trivial baseline). This is the motivation for the per-case review tooling now being
> built; see [`review_responses_pr2.md`](review_responses_pr2.md) A5.

- Six cases implemented, five on real data, one (`surge_cuxhaven`) waiting on a manual GESLA archive download.
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

### `discharge_lowland_denmark` — daily discharge, 7-day horizon

We chose CAMELS-DK catchment **`37470707`** by ranking the 304 gauged catchments by **baseflow index** (BFI 0.971, top of the distribution) and **low slope** (0.88, lowland) to make it a defensibly "typical Danish groundwater-dominated lowland" gauge.

- Is this the right *category* of catchment to lead with, or would a more rainfall-runoff-driven catchment make the matrix more informative? *(Per Rocco: validate with a hydrology domain expert.)*
- Is 7 days the right horizon, or should we evaluate at 1 / 3 / 7 / 14 / 30 (seasonal forecast)? *(Per Rocco: the error-vs-lead-time curves already cover this — the primary horizon is just the headline matrix cell, so we don't multiply primary horizons.)*
- Should we include `q_sim` from the DK-model as a process-model reference column from day one? *(Per Rocco: only worthwhile where there's no high-quality streamflow obs — and we **do** have `q_obs`, so it's optional here.)*
- Does the open question of contrasting with a flashy catchment (Caravan / LamaH-CE) need to happen before we expand methods, or can it wait? *(Per Rocco: it can wait.)*

### `sewerlevel_bellinge` — 15-min sewer level, 6-hour horizon

We use sensor **`G71F05R_LevelInlet_System2000p1`**, selected by `download --scan` as the
level instance that is 100% active over the full 2010–2020 span. (An earlier draft of this
document named `G71F04R_Level1_System2000p1` — that was the *original* pick, dropped because
F04R was only ~20% active under the System2000 SCADA system. This section is now consistent
with the case doc and `case.toml`. Rocco flagged the mismatch.)

- Is F05R LevelInlet the right primary sensor — and **what does it physically measure** (pumping-station inlet vs CSO-chamber inlet)? A CSO-chamber level might make a more operationally compelling, event-aware case. *(Per Rocco: confirm the sensor's location in the asset database before finalizing.)*
- 6-hour horizon: too long, too short, or right? *(Per Rocco: 2–6 h is good, typically bounded by the validity of the rainfall forecast.)*
- Should the case be the *event-aware* version (CSO event hit rate, peak-level bias) rather than continuous MAE/RMSE? The Bellinge data supports the event framing trivially. *(Per Rocco: yes — event-based metrics are often more meaningful here; see the original Bellinge paper for inspiration.)*
- Is the X-band radar nowcast as known-future exogenous a realistic operational story, or do we need to be honest that most utilities don't have it? *(Per Rocco: the **rain gauge** is likely good enough — the Bellinge models were calibrated on gauges; the radar's added value is uncertain.)*

### `demand_ltown` — hourly network demand, 24-hour horizon

L-Town, synthetic SCADA from EPANET simulation. The benchmark forecasts **total network
demand** (sum of consumer demand across all nodes from `SCADA_Demands.csv`), resampled to
hourly, 24 h ahead. (An earlier draft said `PUMP_1` / "largest inlet pump" — that was dropped
because the single-pump signal is a refill square-wave, not demand; see the data-quality note
below. This is now consistent with the case doc.)

*(Per Rocco: validate this case with a domain expert — try CRCO, who has water-distribution experience.)*

- Is "total network demand" the right target, or would district demand (sub-zone flow / pressure) be more operationally meaningful?
- 24 hours for pump scheduling — agree?
- Should we add a leak-detection extension? The 2019 evaluation period has leaks; a "demand-forecast residual = leak signal" anomaly metric writes itself.

### `wwtp_influent_n2o_350k` — hourly influent flow, 24-hour horizon

Cembrowicz et al. (2024) N₂O WWTP dataset (hosted on Mendeley Data); target is `INLET.Q value`, the plant inlet totaliser.

- Is influent flow the right primary target for a wastewater case? *(Per Rocco: keep it on the **influent** side — influent flow or, to better differentiate from Bellinge, an influent **concentration**. Effluent NH₄ / N₂O is a different task that belongs in mechanistic tools like WEST, so we **drop** the effluent secondary targets.)*
- 24-hour horizon for aeration optimisation — agree? *(Per Rocco: 6 h, or both 6 h and 24 h. `horizons_eval` is `["6h","24h"]`; 6 h is the more defensible control horizon.)*
- ~~The same dataset trivially yields two more cases (`wwtp_effluent_nh4`, `wwtp_n2o`).~~ Dropped per Rocco — don't mix influent and effluent.
- ~~A synthetic BSM2 case alongside the real plant?~~ Dropped per Rocco — "not useful".

### `surge_cuxhaven` — hourly residual, 48-hour horizon

*(Not yet reviewed by Rocco — flagged "not checked". Best given a second pass by a coastal / met-ocean expert.)*

Non-tidal residual after UTide harmonic removal. Station not yet chosen (Esbjerg / Hirtshals / Sheerness candidates).

- Which station? Danish coverage is desirable; richness of surge climate matters too.
- 48-hour horizon — matches operational warning systems; agree?
- Process-model reference: Copernicus Marine surge product as a "free" baseline column. Worth pulling in?
- Tide-surge interaction — should we be evaluating *total water level* rather than *residual*, since that's what matters operationally for flood thresholds?

### `waves_oregon` — hourly H_s, 24-hour horizon

*(Not yet reviewed by Rocco — flagged "not checked". Best given a second pass by a coastal / met-ocean expert.)*

NDBC station 46050 (Stonewall Bank, Oregon). Open-ocean Pacific buoy with a long record.

- Pacific buoy vs. a North Sea / Danish buoy: which is the right primary case? (Both for DHI relevance and for swell-dominated vs. wind-sea-dominated dynamics.)
- 24-hour horizon — short for ship routing, OK for surf forecasting; what's the right operational story?
- Add peak period T_p as a secondary target?
- Process-model reference: a Copernicus NW Shelf wave hindcast point. Same question as for surge.

## Cross-cutting design choices we'd like your reaction to

1. **Split mode**: calendar dates per case (with explicit train/val/test ends) vs. observed-span fractions (e.g. 70/15/15 of non-NaN obs). We default to calendar; CAMELS-DK uses fractions because gauged catchments have varying record periods. Is this confusing? Should we standardise? *(Per Rocco: **not critical** if it differs across cases.)*
2. **Gap handling**: training-time linear interpolation + mean-fill, scoring on observed-only test points. Defensible? *(Per Rocco: yes, **provided training gaps stay within a limit** — resolution-dependent, and need not be the same for every case. → Action: add a per-case max-gap threshold for fill, above which a window is excluded.)*
3. **Univariate first**: all method classes can take exogenous inputs eventually, but our pilot is univariate. The right way to introduce exogenous (lagged-only vs known-future-aware) matters for fairness across cases; any preferences from the operational side? *(Per Rocco: address case-by-case — in some domains **univariate makes no sense** (e.g. combined collection systems), where the rainfall response is the whole point. So for Bellinge and WWTP, exogenous rainfall is essential, not optional.)*
4. **Process-model reference columns**: we've talked about them but not implemented. Should they be added before Jaume picks this up, or is that his first piece of work? *(Per Rocco: comparing to simulated data in addition to observed is **not necessary — it complicates the picture**. This is a meaningful scope reduction vs. `research_plan.md` §3; flagged for Jesper — see [`review_responses_pr2.md`](review_responses_pr2.md) E4.)*
5. **Domain-specific metrics**: which cases need them, and which can we honestly run with generic MAE/RMSE/MASE? Event metrics for CSO (case 4–5), POD/FAR for surge (8), NSE/KGE for hydrology (1), peak-flow ratio for both 1 and 4–5 are the obvious additions. *(Per Rocco: those are good guesses, but domain metrics **break cross-domain comparability** — so keep a **dimensionless score across all cases** (MASE; consider MAPE) as the matrix cell, and treat domain metrics as per-case sidebars. See the Purpose note at the top.)*

## Data-quality findings surfaced by the EDA panels

Two issues caught by running `scripts/case_eda.py` over the committed processed
Parquets. Both are honest reports we'd rather you see now than discover later:

- **Bellinge** — the *original* sensor instance (`G71F04R_Level1_System2000p1`) had
  continuous level dynamics only in 2010–2012 plus briefly in late 2019; the rest
  of the record read near zero (sensor not active under that SCADA system; likely
  because F04R has two instances, Level1 and Level2, of which only one is active for
  most of the period — Rocco's read). Earlier "real-data" Bellinge metrics (e.g.
  SeasonalNaive MASE 0.32) were effectively scored on near-zero data and didn't mean
  what we said they meant. **Fixed**: a `--scan` command added to the Bellinge download
  reports time span and active-row count per sensor instance; on that evidence we
  switched the case to **`G71F05R_LevelInlet_System2000p1`**, which is 100% active over
  the full 2010–2020 span.
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
