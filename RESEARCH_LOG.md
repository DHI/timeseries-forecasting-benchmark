# Research Log

A running record of the testing and decision-making behind this benchmark — what was tried,
what was decided, and why. Intended to be readable by collaborators without needing to dig
through commit history or notebooks.

Entries are in reverse-chronological order (newest first).

<!-- Add new entries below this line, newest first. -->

## 2026-07-03 — Per-case curated `seasonalities` replace the freq → seasonality map

Seasonal periods were hardcoded per sampling frequency (`_default_season_length` /
`_seasonal_periods`): every hourly case was assumed to carry 24 h/168 h cycles, every daily
case 7 d/365 d. That assumption encodes human-activity calendars and is wrong for much of the
environmental half of the benchmark — waves and de-tided surge have no diurnal cycle, rivers
have no work week — and it wasn't just cosmetic: the map configured SeasonalNaive, TBATS,
AutoTheta, MSTL and the ML feature pipeline, **and set the MASE denominator**, so cross-case
MASE comparisons were distorted by how well each case happened to match the calendar.

**Decision.** Each case now declares `seasonalities` in its case.toml — periods in steps,
dominant first, `[]` = explicitly non-seasonal. The field is *required* and is the single
source of truth during the train/predict cycle; the freq maps are deleted, with no fallback
path. Values are fixed once, at curation time, from train-split evidence; deviating from the
detection proposal requires a justification in the case notes.

**Detection rule** (`data/seasonality.py`, reviewed interactively in
`notebooks/seasonality.py`): candidates come from a fixed physically motivated menu per
frequency (daily {7, 365}; hourly {12, 24, 168}; 15-min {96, 672}), so spectral noise can
never mint an arbitrary period. A candidate is accepted when the train split holds ≥ 2 full
cycles, its periodogram peak exceeds 10× the median power of the surrounding octave band
(robust to the red spectra of environmental series), and the ACF at that lag clears 0.1.
The ACF floor was calibrated down from 0.15 on real-case evidence: demand_ltown's
unmistakable diurnal cycle (spectral ratio ≈ 740) sits at ACF(24) ≈ 0.10 because heavy
high-frequency noise dilutes the ACF while the periodogram concentrates the cycle in one bin.

**Committed values** (train-split evidence: spectral ratio / ACF at lag):

| case | seasonalities | notes on evidence |
|---|---|---|
| demand_ltown (1h) | [24, 168] | 24: 739/0.10; 168: 32/0.12 — classic demand calendar |
| wwtp_influent_n2o_350k (1h) | [24, 12] | 12 is the bimodal daily shape (2nd harmonic), kept as evidence says |
| surge_cuxhaven (1h) | [12] | semidiurnal residual (tide–surge interaction); **no** 24 h cycle (6.9/0.41 rejected) |
| waves_massachusetts (1h) | [12, 24] | tidal modulation dominant + diurnal (sea-breeze) signal |
| waves_oregon (1h) | [12] | tidal modulation only; 24 rejected (5.2) |
| sewerlevel_bellinge (15min) | [96] | daily cycle clear; weekly (672) rejected (2.2) |
| discharge_flashy_catskills (1D) | [] | rain-driven flashy creek: annual rejected (6.0/0.04) — genuinely non-seasonal |
| discharge_lowland_denmark (1D) | [365] | annual only |
| discharge_snowmelt_yosemite (1D) | [365] | annual (snowmelt) only; 7 rejected |
| discharge_saugeen (1D) | [365, 7] | genuine **weekly spectral line** (14.5/0.56 over 2374 cycles) — consistent with weekly dam-regulation |
| groundwater_yorktown (1D) | [365] | annual only |
| watertemp_james (1D) | [365, 7] | annual dominant (546/0.94); weekly line (20/0.95) — plausibly anthropogenic thermal effluent |

Two upstream assumptions did not survive contact with the data: hourly cases are *not*
generally diurnal (3 of 5 have tidal 12 h structure instead of, or ahead of, 24 h), and one
daily case is genuinely non-seasonal. The declared-non-seasonal case makes SeasonalNaive and
MSTL honestly inapplicable there (they raise instead of producing a strawman column), and its
MASE falls back to persistence scaling.

**Going forward.** New cases must declare `seasonalities` (schema enforces it); run the
notebook, review the evidence table, commit the proposal or justify the deviation in the case
notes. MASE values shift on cases where the declared dominant differs from the old freq
default — expected, that is the point.

## 2026-07-03 — Silencing a spurious TBATS Box-Cox warning

`scripts/run_pilot.py` was emitting a `UserWarning: Data contains zero or negative values,
disabling Box-Cox transformation` from `statsforecast`'s TBATS implementation on some cases
(e.g. `watertemp_james`, whose `water_temp` target dips to -0.1°C).

**What was going on.** `TBATSForecaster.fit` already has logic for this: when the raw target
contains non-positive values, it pre-applies a Yeo-Johnson power transform (which, unlike
Box-Cox, is defined for negative/zero inputs) before handing the series to TBATS, and TBATS's
own internal Box-Cox is then expected to be a no-op. The problem is that Yeo-Johnson with
`standardize=False` is not a positivity-guaranteeing transform — it can (and did, for values
already near zero) leave some transformed points at or below zero. TBATS's model-selection
code checks for that itself (`use_boxcox in (True, None) and np.any(y <= 0)`) and, on the
default `use_boxcox=True`, warns and disables Box-Cox internally — redundant with what we'd
already decided, just noisy.

**What was decided and why.** Pass `use_boxcox=False` to `TBATS(...)` explicitly whenever our
own Yeo-Johnson pre-transform was applied (`self._pt is not None`), so TBATS's positivity
check never fires. This produces the identical fit (Box-Cox off either way) without the
warning. When the raw data is already strictly positive, `use_boxcox` is left at its default
(`True`) so TBATS can still decide for itself whether a Box-Cox transform helps.

## 2026-07-03 — Gap handling: one materialized policy (cap + longest-segment trim)

Consolidated how missing target data is handled. Previously there were **two** layers: the
eval loop imputed gaps on the fly (`_complete_for_fit`: linear interpolation, then mean-fill
for whatever remained), and several methods *also* interpolated/dropped NaNs internally
(`dl`, `torchdl`, `nf_models`, and exogenous fills in `ml`). The per-method layer was
redundant (the loop already delivered a filled frame) but non-uniform — `nf_models` *dropped*
NaN rows while the others *filled*, a latent inconsistency — and, more importantly, mean-fill
meant models could train on large stretches of fabricated target (e.g. one 6574-step block =
54% of `discharge_lowland_denmark`, a single interpolated straight line).

**What was decided and why.** Keep a single, honest policy, materialized once rather than
re-derived every run:

1. **Per-case gap cap = decorrelation time.** A gap is interpolatable only if its observed
   endpoints still inform the interior — exactly what autocorrelation measures. The cap is
   the lag at which the target ACF first drops below 0.8, bounded by one dominant seasonal
   cycle. It adapts to dynamics with no hand-tuning: flashy series (demand, flashy discharge)
   get cap=1; smooth ones (groundwater 26, water temperature 31) tolerate larger fills.
2. **Longest-segment trim.** Gaps larger than the cap are hard boundaries, never bridged.
   The record splits into contiguous segments and only the **longest** is kept — "no target =
   nonexistent." We chose *longest* over *most-recent*: the most-recent segment starved
   `waves_massachusetts` to 5 training points (a big gap sits at its train/test boundary),
   whereas the longest segment gives it 21,126. Parameter-averaging across per-segment fits
   (e.g. one ARIMA per segment) was rejected — averaging AR/MA coefficients across different
   orders / a nonlinear parameter space is not a valid estimator, and a pre-gap segment
   cannot forecast a post-gap origin without bridging anyway. Using *all* segments would
   require per-family handling (window-pooling for global learners, missing-data Kalman for
   state-space models), which reintroduces exactly the divergence this change removes; noted
   as a possible future enhancement.
3. **`y_observed` mask.** The curated target is complete (small gaps filled) so any method
   can fit, but the mask records which points were genuinely measured. Scoring, origin
   selection, and the fraction-split all key off the mask, so filled points never enter a
   metric — honesty preserved despite materialization.

**Implementation.** New `data/curate.py` holds the policy; `scripts/curate_data.py` writes
`data/curated/<case>.parquet` (+ a manifest of cap/gap/span per case) once. `Case.load(curated=True)`
reads it (falling back to on-the-fly curation), and the three eval entry points switched to
it; `_complete_for_fit` and all per-method gap handling were removed. `processed/` stays the
raw, gappy canonical for EDA/review/reports. Impact: 7 cases are untouched (no big gaps); the
5 gappy cases keep their longest span (kept fractions 27–44%, all thousands of points in
absolute terms).

**What it means going forward.** Every method now sees byte-identical fitting data per case,
so cross-method cells are no longer confounded by preprocessing. Training-set size still
varies widely *across* cases (≈3k–214k) — this is inherent to the domains and handled by the
relative metrics (MASE, skill), and is worth surfacing as a stratifying variable (data-hungry
vs data-efficient methods) rather than suppressing. Re-run `curate_data.py` after any data
refresh or a change to the ACF threshold / ceiling.

## 2026-07-02 — Seasonal classical baselines: MSTL forecasters + seasonally-adjusted Theta (issue #46)

Closes the "classical methods run non-seasonal by default" flaw from the method-suite audit
below. Previously AutoARIMA/AutoETS/AutoTheta all ran with `season_length=1` and the pilot
never overrode it, while the windowed-ML methods got calendar features and seasonal lags
(168h etc.). So any "classical vs ML" cell was partly an artifact of deliberately
de-seasonalized classical baselines rather than a genuine methodology comparison.

**What was decided and why.** Native seasonal fitting is only tractable for small periods:
seasonal ARIMA cost explodes with the period *m*, and AutoETS's seasonal state grows with
*m*, so *m*=365 (daily) or *m*=96 (15-min) are impractical. Rather than switch
AutoARIMA/AutoETS to seasonal on some cases and not others (which would make the same-named
column behave differently across the matrix), we kept them as the **non-seasonal
short-memory baselines** and added dedicated seasonal classical columns:

- **MSTL-ETS** and **MSTL-ARIMA** (`classical.py`): statsforecast `MSTL` strips one or more
  seasonal components with a fast STL decomposition, fits AutoETS (`model="ZZN"`) or
  AutoARIMA on the seasonally-adjusted series, then re-seasonalizes. This handles large and
  multiple periods (daily+weekly, weekly+yearly) uniformly and cheaply. Periods come from a
  new `_seasonal_periods(freq)` helper (`naive.py`): daily `[7, 365]`, hourly `[24, 168]`,
  15-min `[96, 672]`.
- **Theta** is now seasonal by default. AutoTheta deseasonalizes with a cheap classical
  decomposition (not a costly seasonal state), so wiring the case's period in is tractable
  at any frequency and is standard practice (the M3/M4 seasonally-adjusted Theta).

**Robustness / honest reporting.** STL needs ≥2 full cycles of each period, so MSTL filters
its candidate periods to those the (truncated) training series supports. If *none* survive
— only on pathologically short series; real cases always clear the smallest period's bar,
and a daily case short of ~2 years simply drops the 365 component and runs on weekly-7 — the
method **raises** rather than falling back to a bare non-seasonal trend forecaster. A silent
fallback would have made MSTL-ETS/MSTL-ARIMA byte-identical to the AutoETS/AutoARIMA columns,
i.e. a fake independent data point. Instead the eval loop records the cell as blank/NaN with
an explanatory status — the honest outcome the audit argued for.

**What it means going forward.** The classical-vs-ML comparison is now a genuine methodology
contrast: non-seasonal ARIMA/ETS as short-memory baselines, versus seasonally-adjusted Theta
and MSTL-seasonalized ARIMA/ETS as the seasonal classical family. Verified end-to-end on a
synth pilot (hourly `waves_oregon`, daily `discharge_flashy_catskills`): all three new
columns produce finite, distinct MASE, and the no-surviving-period guard raises cleanly.

## 2026-07-02 — Audit of the method suite (`bench/methods/`): flaws and gaps identified

Did a full review of the method wrappers (`naive.py`, `classical.py`, `ml.py`, `dl.py`,
`fm.py`, `nf_models.py`, `torchdl.py`) plus how the eval loops call them. Nothing here is
fixed yet — this entry records what was found so the fixes can be prioritized deliberately.
Highlights, roughly by severity:

**Validity of cross-family conclusions**
- Classical methods (AutoARIMA/AutoETS/AutoTheta) run non-seasonal (`season_length=1`)
  by default and the pilot script never overrides it per case, while the ML methods get
  calendar features and seasonal lags (168h etc.). The "classical vs ML" comparison is
  therefore partly an artifact of deliberately de-seasonalized classicals. The cheap
  remedy is an MSTL+Auto{ARIMA,ETS} method (statsforecast MSTL is fast and takes
  multiple periods), and/or wiring `season_length` from case frequency where tractable.
- Training budgets are asymmetric: statsforecast auto-selects model orders, while the
  neural methods get fixed tiny budgets (500 steps in `nf_models.py`, 600 in
  `torchdl.py`, no validation/early stopping). "DL underperforms" conclusions need this
  caveat.
- `MLPForecaster` feeds unscaled features (raw lags + dayofyear up to 366) to sklearn's
  `MLPRegressor` with no `StandardScaler` — its scores likely measure missing
  preprocessing rather than the architecture.
- Chronos uses the predictive **median** as the point forecast, but the matrix scores
  RMSE/MSE-based skill, for which the mean (which the pipeline returns and we discard)
  is the optimal point. Slight systematic handicap on squared-error metrics.

**Correctness / silent-failure issues**
- `_complete_for_fit` is applied to the *full* frame (train+test) before slicing at the
  origin, so linear interpolation across a gap spanning the origin, and mean-fill using
  the full-series mean, leak test-period information into training history. Small in
  magnitude but real, and it affects every method.
- DLinear/N-HiTS/PatchTST silently fall back to persistence when the series is too
  short to window, and report status "ok" under the method's name.
- Fixed-horizon methods (DLinear, torch DL, neuralforecast) pad forecasts with the last
  value when asked for a horizon longer than the trained one — relevant to
  `run_leadtime_curve(max_horizon=...)`, where the padded tail would be scored as if the
  model produced it.
- The frequency-dispatch tables (`_default_season_length`, `_lags_for`,
  `_date_features_for`, `_lookback_for`, `Climatology._key`) only know
  {daily, hourly, 15-min, weekly} and fall back to silent generic defaults otherwise —
  e.g. a 30-min case would get season_length 24 (a 12-hour "day", anti-phase seasonal
  naive) and that same wrong season feeds the MASE denominator in the eval loop.
- `Climatology` degenerates to the global mean for weekly data, and for hourly data keys
  on hour-of-day only (no weekday) while 15-min data keys on weekday×hour — inconsistent.

**API/design risks**
- `predict(history=...)` is dead in every non-naive method: statsforecast, mlforecast,
  neuralforecast, Chronos, DLinear, and torch DL all forecast from state captured at
  `fit` time and ignore `history`. Correct today only because both eval loops refit per
  origin; any future fit-once/predict-rolling usage would silently produce forecasts
  anchored at train end. The contract in `base.py` should either be honored or narrowed.
- The exogenous tier merges historical-only and known-future covariates and gives both
  perfect future values, so "oracle exo" and "realistically available exo" cannot be
  separated in results.
- `is_probabilistic` exists but no probabilistic metric does; Chronos quantiles are
  computed and discarded.
- Duplication drift: the tz-strip/truncate boilerplate is copy-pasted 4× in
  `classical.py`; `_lookback_for` exists in three near-copies that already disagree
  (`dl.py` lacks the `max(base, 2*h)` floor the torch version has); `nf_models.py` is
  unseeded while `torchdl.py`/MLP are seeded.

**Suite gaps** — no ensemble-of-baselines column (cheap, usually embarrassingly strong),
no MSTL/STL-based classical, single foundation model (chronos-bolt-small only), and the
mlforecast methods are recursive-only while the DL methods are direct multi-output — a
strategy confound between families.

## 2026-07-02 — Fixed rolling-origin selection collapsing onto the front of the test set

`run_case_rolling` (`src/tsforecastbench/bench/eval/rolling.py`) is supposed to spread
`n_origins` forecast cutoffs across the whole test period so pooled errors reflect
performance throughout test, not just at one point in time. The origin-selection code
built `2 * n_origins` candidates evenly spaced across `[test_start, last_origin]`,
filtered out any whose horizon window was <50% observed, then took `[:n_origins]` —
i.e. truncated the sorted, filtered list to its first `n_origins` entries. Since the
list was sorted ascending, "first n_origins of 2n evenly-spaced points" is the front
~50% of the range, not a spread sample. On a real case
(`discharge_lowland_denmark`, `H=7`, test range 1241–1453) this meant the last origin
used was ~1347 instead of 1453 — only 46% of the usable test range was ever scored,
and the tail of the test set (where any regime shift or trend change would show up)
was silently ignored. This bites hardest exactly when `horizon << len(test)`, since
that's when the true usable origin range is nearly the whole test set. Note
`run_leadtime_curve` in the same file was not affected — it samples origins directly
via `linspace(test_start, last_origin, n_origins)` with no oversample-then-truncate step.

Fixed by extracting the logic into `_select_rolling_origins()`: oversample as before,
filter for the 50%-observed criterion, but then evenly downsample the *filtered*
list back to `n_origins` (via `linspace` over indices into the survivors) instead of
truncating it. Added `tests/test_rolling.py` covering the regression case (small `H`
relative to test length), a case with some candidates dropped for missing data, the
fewer-than-`n_origins`-valid case, and the all-filtered-out fallback. Verified on
`discharge_lowland_denmark` that origins now span the full `[1241, 1453]` range.

This changes which points get scored, so `run_case_rolling` metrics (MAE/RMSE/MASE/
skill scores) will shift for any case where `horizon_primary << len(test)` — the
change should generally make the metric more representative, not just different, but
any published matrix numbers generated before this fix should be treated as based on
only the first half of each case's test period.

## 2026-07-02 — Added missing `discharge_saugeen` case doc and EDA panel

`docs/cases/` had 11 pages for the 12 cases under `src/tsforecastbench/data/cases/` —
`discharge_saugeen` was implemented (case.toml, download.py, synth.py) but never got a
docs page, and was also missing from `scripts/case_eda.py`'s `CASE_HINTS` (so no EDA
figure existed) and from the case table in `docs/README.md`.

Added the `discharge_saugeen` entry to `CASE_HINTS` (no overlay — it's exogenous-free,
so the panel falls back to the 7-day rolling-mean trend line like `demand_ltown`),
regenerated `docs/figures/cases/discharge_saugeen_eda.png`, wrote
`docs/cases/discharge_saugeen.md`, and added the case to the `docs/README.md` table.
Framed the doc around its role in the suite: it's the deliberate univariate control
against the three exogenous hydrology cases (`discharge_lowland_denmark`,
`discharge_flashy_catskills`, `discharge_snowmelt_yosemite`) — same kind of target,
no forcing covariates, so it isolates pure autoregressive/seasonal skill. No case
behavior changed; this is docs/tooling catch-up only.

## 2026-07-02 — Dropped `optional_exogenous` as a redundant stored field

A case's exogenous columns end up described along two independent axes: timing
(`exogenous` vs `exogenous_known_future`) and necessity (`necessary_exogenous` vs
`optional_exogenous`, added earlier today). But given the fixed set of available columns
and which ones are necessary, "optional" is always just the complement — storing it
explicitly in every case.toml was pure duplication, and the validator that checked the
two didn't overlap or leave gaps only existed to police that duplication.

`Case.optional_exogenous` (`data/schema.py`) is now a derived property —
`(exogenous ∪ exogenous_known_future) − necessary_exogenous` — not a TOML field.
`necessary_exogenous` is the only exogenous-necessity field a case.toml declares; removed
the now-unnecessary `optional_exogenous = [...]` line from all 9 case.toml files that had
it. `Case.exo_tier` and everything downstream (`bench/eval/loop.py`'s `case_exo_tier`
column) is unaffected — same values, one fewer place for a case author to get out of
sync with itself.

## 2026-07-02 — Structured exo-necessity fields, persisted as result metadata (not matrix aggregation)

Follow-up to the prose-only necessity judgment recorded earlier today. Two decisions:

**1. Promoted the necessity judgment into schema, not just prose.** Added
`necessary_exogenous` / `optional_exogenous` fields to `Case` (`data/schema.py`), plus a
model validator requiring every column in `exogenous` ∪ `exogenous_known_future` to be
classified as exactly one of the two (no silent gaps, no double-classification). A derived
`Case.exo_tier` property (`"none" | "optional" | "necessary"`) reads off these fields:
`necessary` if any column is necessary, `optional` if only optional columns exist, `none`
if there's no exogenous input at all. All 9 case.toml files with exogenous inputs were
updated with the classification agreed on earlier; the 3 exo-free cases (`demand_ltown`,
`discharge_saugeen`, `wwtp_influent_n2o_350k`) default to `"none"` with no change needed.

**2. Rejected splitting the results matrix by tier, in favor of metadata-only labeling.**
The original motivation (mixing exo-essential and exo-optional cases in one matrix is
misleading) initially suggested computing a separate leaderboard/ranking per case tier,
with univariate methods demoted to a "reference floor" on necessary-exo cases. That was
wrong: this benchmark's matrix (`viz/matrix.py`) never aggregates or ranks methods — every
case × method cell is shown raw, by design — and demoting some methods out of a ranking
on some cases is aggregation by another name, just moved one level up. Partitioning cases
into separate per-tier matrices was also rejected: several cases are deliberately grouped
for cross-regime comparison regardless of tier (e.g. `discharge_snowmelt_yosemite`'s note
that the three hydrology cases test "whether method ranking holds across slow / flashy /
snowmelt dynamics," even though those three land in different exo tiers).

Landed instead: `case.exo_tier` and each method's existing `uses_exogenous` flag
(`bench/methods/base.py`) are now persisted as plain columns on every result row
(`bench/eval/loop.py`), alongside `case_id`/`method`. No computation, no filtering, no
new derived statistic — just metadata attached to the existing flat results table so
downstream consumers can filter or facet by tier/exo-usage if they want to, without the
benchmark itself ever declaring which methods "count" for which cases. The matrix
visualization itself is unchanged for now; visually labeling exo-using method columns in
`viz/matrix.py` is deferred to a later pass.

## 2026-07-02 — Recorded necessary-vs-optional exogenous judgment in case notes

Mixing cases where exogenous input is domain-essential (e.g. forecasting storm surge
without wind/pressure) with cases where it's a nice-to-have enrichment makes a single
results matrix hard to read: a method's ranking on an exo-essential case isn't
comparable to its ranking on a case where a good univariate baseline is plausible on
its own. Decided to separate this in the eventual results view, split by whether a
case's exogenous inputs are *necessary* or merely *optional* — a domain-expert
judgment call, not something inferrable from measured skill deltas alone (a low
measured lift from an exo method could mean "exo isn't needed" or could just mean the
exo methods aren't good yet; those are different claims).

As a first step, before touching schema or pipeline code, recorded that judgment call
in prose in each case's `notes` field in `case.toml`, so it's reviewable independently
of any code change:

- **Extracted from existing notes** (the necessity claim was already implicit or
  explicit in the case's original description): `surge_cuxhaven` (wind+pressure all
  necessary — explicitly called "the physical surge drivers"), `watertemp_james`
  (temp_air necessary, precip optional), `discharge_snowmelt_yosemite` (temp_air
  necessary, precip optional), `groundwater_yorktown` (precip necessary; temp_air
  marked optional as an assumption, since the original note doesn't credit it with
  skill either way).
- **Fresh domain judgment** (no prior necessity claim in the note, so flagged as an
  assumption pending validation): `discharge_flashy_catskills` (precip necessary,
  temp_air optional), `discharge_lowland_denmark` (nothing necessary — the
  groundwater-dominated baseflow is the deliberate exo-available-but-not-essential
  contrast case), `sewerlevel_bellinge` (rain_gauge/rain_radar_nowcast necessary,
  temp_air optional), `waves_massachusetts` and `waves_oregon` (wspd necessary as the
  direct physical driver of wind-sea wave height; pres/atmp/wtmp optional).
- `demand_ltown`, `discharge_saugeen`, and `wwtp_influent_n2o_350k` have no exogenous
  inputs at all, so trivially have no necessary exogenous columns.

This is prose only — no schema or eval-pipeline change yet. The planned next step is
to promote this into structured `necessary_exogenous` / `optional_exogenous` fields in
`schema.py` (derived from the union `exogenous` field methods already consume) and use
`len(necessary_exogenous) > 0` to split the results matrix. Several of the "fresh
judgment" calls above are genuinely arguable (e.g. whether `discharge_lowland_denmark`
should have zero necessary columns, or whether pressure should count as necessary
for the wave cases) and may get revised before that structured version lands.

## 2026-07-02 — Added CatBoost to the windowed-ML method set

Added `CatBoost` and `CatBoost+exo` to the windowed-ML column family
(`bench/methods/ml.py`), alongside the existing `LightGBM`/`LinearLags`/`MLP` methods.
CatBoost slots into the shared `_WindowedMLBase` lag + calendar feature pipeline exactly
like `LightGBMForecaster` — it only supplies its own `_make_model()` (a
`CatBoostRegressor` with `iterations=200`, `learning_rate=0.05`, matching LightGBM's
`n_estimators=200`/`learning_rate=0.05` so the two GBDT methods are compared on
roughly equal training budget). `allow_writing_files=False` is set on the regressor so
runs don't leave `catboost_info/` log directories scattered around the repo.

Registered the two new classes in `bench/methods/__init__.py` and `scripts/run_pilot.py`
(both the univariate and `+exo` variants, following the Tier-1/Tier-2 pattern used for
the other windowed-ML methods), added `catboost>=1.2` to `pyproject.toml` core
dependencies, and mirrored the existing LightGBM smoke tests in
`tests/test_pipeline_synth.py`. Verified with a synthetic pilot run
(`run_pilot.py --synth --cases waves_oregon --fast`) that both new columns produce
`ok` status alongside the rest of the matrix.

This gives the benchmark a second GBDT implementation to compare against LightGBM —
useful since CatBoost's ordered-boosting scheme handles the lag/calendar feature set
differently and can behave better or worse depending on how noisy a given case's
covariates are. No methodology change; this is a straightforward "add another column"
addition to the existing Tier-1/Tier-2 windowed-ML family.

## 2026-07-02 — Backfilling this log from the scoping documents

*Meta note: this entry and the ones below it were written today, reconstructing the
project's decision history from `research_plan.md`, `docs/SCOPING.md`,
`docs/review_responses_pr2.md`, and the git history, since this log was created empty.
Dates reflect when each decision actually happened, not when it was written down here.*

While doing this, we compared the original plan against the current codebase and found
several places where the project quietly diverged from `research_plan.md` without an
explicit decision ever being recorded:

- **Foundation models**: the plan (§3) called for six FMs (Chronos-2, TimesFM 2.5,
  Moirai 2.0, TimeMoE, TimeGPT, Lag-Llama). Only **Chronos-Bolt zero-shot** was ever
  implemented (`bench/methods/fm.py`). Recurrent nets (LSTM, GRU, DeepAR) and the
  fine-tuning round (Phase C) were never started either.
- **Process-model reference columns / hybrid ML residual correction**: proposed as
  important in the plan (§3, "the column ML methods have to beat"), pushed back on by
  Rocco during scoping review (`review_responses_pr2.md` E4 — "comparing to simulated
  data... complicates the picture"), left as an open question for Jesper, and never
  implemented. `grep` across `src/tsforecastbench` today turns up no trace of it. In
  effect Rocco's position won by default.
- **NSE / KGE metrics**: agreed in the review response (A3) as a queued code change,
  still not present in `bench/eval/metrics.py` as of this writing.
- **Library strategy**: the plan's Nixtla-based approach *did* land — `statsforecast`
  drives the classical methods, `mlforecast` the windowed-ML methods, and
  `neuralforecast` the TFT/iTransformer/TimesNet zoo — but wrapped behind a project-own
  `Forecaster` interface rather than used directly, and N-HiTS/PatchTST were
  reimplemented on bare `torch` instead of via `neuralforecast`, specifically to dodge
  `neuralforecast`'s `ray` dependency (no Windows / Python-3.13 wheels). A `modelskill`
  extra (`ms`) is declared in `pyproject.toml` but is not imported anywhere in
  `src/`, `scripts/`, or `notebooks/` — the ModelSkill `Comparer` adapter described in
  the plan (§4a) was never built.
- **Package layout**: the plan sketched one flat `src/tsforecastbench/` tree; the repo
  split into `tsforecastbench.data` and `tsforecastbench.bench` on 2026-06-13
  (`840fcf5`), explicitly so the two halves could become separate repos later.

None of this needed fixing — the codebase and docs (`ONBOARDING.md`, `CONTRIBUTING.md`)
are internally consistent and describe what actually exists. It just means
`research_plan.md` should be read as the original *proposal*, not as a description of
the current system.

---

## 2026-06-25 — Added a second, unforced hydrology contrast case

Added `discharge_saugeen` (daily discharge, Saugeen River, Ontario — Monash Time Series
Forecasting Archive, 65 years of record, no exogenous inputs) alongside the existing
`discharge_flashy_catskills`. This answers open question #1 from `research_plan.md` §6
("are you happy with one DK lowland + one alpine catchment, or two DK catchments?") in
a third direction neither option anticipated: a long-record, exogenous-free univariate
series as a methodological contrast to the exogenous-rich cases, rather than another
geographic contrast. Brings the case count to 12.

---

## 2026-06-12 to 2026-06-13 — Scoping review with Rocco (domain expert), and the response

At this point the benchmark had 6 cases and 9 methods (naive baselines, AutoARIMA,
AutoETS, and three windowed-ML variants). Before scaling up further, Jesper sent a
scoping review (`docs/SCOPING.md`) to Rocco, an internal urban + catchment hydrology
expert, to sanity-check the case selection and design choices against operational
reality before Jaume took over the bulk of the engineering. Rocco's comments and the
responses are fully recorded in `docs/review_responses_pr2.md`; the decisions that
changed the project's direction:

- **Purpose reframed.** The benchmark is not about finding the best model per case —
  it's about a *fair, direct comparison of the same methods across very different
  water/environment domains*. This is why the headline matrix cell must stay a
  **dimensionless** score (MASE, primarily) that means the same thing everywhere, and
  why domain-specific metrics (NSE/KGE for hydrology, event hit-rate for CSOs, POD/FAR
  for surge) live as **per-case sidebars**, never as matrix cells — mixing dimensioned
  domain metrics into the cross-case matrix would break comparability, which was the
  entire point.
- **Process-model reference columns and hybrid ML residual correction — de-scoped.**
  `research_plan.md` treated these as central ("the column ML methods have to beat to
  be commercially interesting"). Rocco's view: comparing against simulated/process-model
  output in addition to observations isn't necessary and complicates the picture. This
  directly contradicts the plan and was left as an open decision for Jesper rather than
  settled unilaterally — but as of today it has never been implemented, so it's the
  de facto scope.
- **Exogenous inputs are not optional everywhere.** For Bellinge and the WWTP case,
  the rainfall response *is* the phenomenon being forecast — a univariate run there is
  a floor, not a real benchmark entry. This ruled out the plan's original idea of
  applying one uniform Tier 0/1/2 exogenous schedule identically to every case; exogenous
  handling is decided case-by-case instead.
- **Horizons: don't multiply the headline horizon.** Rather than evaluating several
  "primary" horizons per case (as `research_plan.md` §4 suggested), each case keeps one
  primary horizon for the matrix; the existing error-vs-lead-time curves already sweep
  sub-horizons, so a second set of primary horizons would just duplicate that.
- **Two data-quality bugs surfaced by the EDA, ahead of scaling method count**: see the
  2026-05-29 entry below — both were driving factors in wanting the case-review tooling
  (`scripts/review_case.py` / `/review-case`) that shipped as a direct follow-on.
- **WWTP vs. Bellinge overlap.** Both are rainfall-driven sewer-catchment signals and
  risked being redundant cases. Resolved by differentiating them explicitly: Bellinge is
  framed as in-pipe water level for real-time control / CSO warning, WWTP as influent
  flow for aeration/energy optimisation. The WWTP case's effluent NH₄/N₂O secondary
  targets and a proposed synthetic BSM2 case were dropped — Rocco: mixing influent and
  effluent doesn't make sense, effluent chemistry belongs in mechanistic tools like WEST.
- **Architecture**: the package split into `tsforecastbench.data` (case-building) and
  `tsforecastbench.bench` (model comparison) landed in this window (`840fcf5`,
  2026-06-13), along with exogenous-aware ("Tier 2") windowed-ML methods and a
  skill-vs-mean metric.

Net effect: the project moved from "one plan, unreviewed" to a design that's been
stress-tested against one domain expert's operational judgement, with the disagreements
(process-model columns, chiefly) explicitly flagged rather than silently resolved.

---

## 2026-05-29 — Two data-quality bugs found before scaling further

Running the early EDA over the first six real-data cases surfaced two bugs serious
enough to invalidate earlier "real-data" numbers for those cases:

- **Bellinge sensor.** The originally chosen level-sensor instance,
  `G71F04R_Level1_System2000p1`, was only active for a small fraction of its nominal
  2010–2020 span (continuous dynamics only in 2010–2012 and briefly in late 2019 — the
  rest read near-zero, likely because F04R has two instances and only one is active at
  a time under the System2000 SCADA system). Earlier metrics computed on this sensor
  (e.g. a suspiciously good SeasonalNaive MASE of 0.32) were effectively scored on
  near-zero data and didn't mean what they appeared to mean. Fixed by adding a `--scan`
  mode to the Bellinge downloader that reports active-row coverage per sensor instance,
  and switching the case to `G71F05R_LevelInlet_System2000p1`, confirmed 100% active
  over the full span.
- **BattLeDIM demand target.** The target had been a single pump's flow (`PUMP_1`),
  which alternates with `PUMP_2` under operational pump scheduling — so the series was
  a square wave reflecting on/off scheduling, not a demand signal. Fixed by summing all
  `PUMP_*` columns to get total network inflow, which is the operationally meaningful
  target.

Lesson carried forward: **don't trust a case's numbers until its EDA has been read**,
which is why a per-case quality gate (`scripts/review_case.py`, the `/review-case`
skill) became a standing requirement for adding any new case, not just these two.

---

## 2026-05-27 to 2026-05-28 — Initial scoping plan and repo scaffold

`research_plan.md` laid out the original proposal: 10 candidate cases (later an
optional 11th for sediments) spanning the water cycle from atmosphere to ocean, a
methods axis of ~16 forecasters grouped from naive baselines up through time-series
foundation models, a phased roadmap (pilot → full benchmark → fine-tuning → Streamlit
app → external release), and a storage design (long-format Parquet, `unique_id`/`ds`/`y`
Nixtla-style schema, one TOML per case validated by a Pydantic `Case` model).

The initial commit (`715b4f8`) scoped this down for a first working pipeline: **6 cases,
9 methods** (four naive baselines, AutoARIMA, AutoETS, and three windowed-ML variants),
with the cases × methods matrix and lead-time evaluation working end-to-end and CI in
place from day one. This was a deliberate "lock the interfaces first" move consistent
with the plan's own Phase A framing, rather than trying to build all 10 cases and 16
methods before anything ran.
