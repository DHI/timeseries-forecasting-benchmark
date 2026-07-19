# Research Log

A running record of the testing and decision-making behind this benchmark — what was tried,
what was decided, and why. Intended to be readable by collaborators without needing to dig
through commit history or notebooks.

Entries are in reverse-chronological order (newest first).

<!-- Add new entries below this line, newest first. -->

## 2026-07-19 — Fixed leap-year bug in the 365-day position-in-cycle path

`_dayofyear_365` (used by `position_in_cycle` for the annual cycle, which backs Climatology,
SeasonalNaive-via-ensemble, and any ML feature keyed on a 365-day period) called
`idx.is_leap_year.to_numpy(...)`, but `DatetimeIndex.is_leap_year` already returns a plain numpy
array rather than a pandas object, so the `.to_numpy()` call raised `AttributeError` on every
call. This broke all baseline/ensemble tests on cases with a declared 365-day seasonality plus
the dedicated leap-year boundary test.

**Decision.** Wrap the result in `np.asarray(...)` instead of calling `.to_numpy()`. No behavior
change beyond fixing the crash — full suite passes again.

## 2026-07-09 — Persist real per-origin forecasts and overlay methods in the results app

The results-matrix app showed seeded placeholder forecasts because the eval pipeline only ever
saved metrics, not the forecast series behind them. `run_case_rolling` now keeps the full-horizon
per-origin forecasts it was already computing and returns them alongside the metrics as a tidy
`(case_id, method, origin_id, origin_ds, ds, y_hat)` frame, written incrementally to
`results/pilot_snapshots.parquet` (single hold-out `--fast` runs don't produce snapshots). Origins
are now chosen once per case, before the method loop, so `origin_id`/`origin_ds` line up across
methods — the app uses this to overlay several methods' forecasts on the same panel by clicking
cells along a matrix row. Actuals are reloaded from the case's curated series rather than
duplicated into the snapshot file, keeping the target series in one place. A fresh pilot run is
needed to regenerate the snapshot artifact before the app shows real forecasts.

## 2026-07-09 — Moved the results-matrix app from Streamlit to Dash

The results-matrix app moved from Streamlit to Dash because Plotly's `Heatmap` trace has no
selection module, so Streamlit's `on_select="rerun"` could never fire on a cell click — the
earlier workaround (an invisible `Scatter` overlay per cell just to catch clicks) worked but was
fragile against reversed axes and cell layout. Dash's heatmap emits `clickData` natively, so the
click-to-snapshot interaction is now a plain callback with no overlay hack. `data.py` and
`snapshot.py` carried over unchanged since they only ever built pandas frames and Plotly figures;
`app.py` was rewritten, and `@st.cache_data` was swapped for stdlib `functools.cache`. Run command
is now `uv run --extra app python app/app.py`. One known gap: the reload button re-reads the
parquet but won't show a brand-new domain or family in the filter widgets until the app restarts.

## 2026-07-09 — Two app bugs found on first real-browser test, both pre-existing

First manual, real-browser test of the results-matrix app (headless checks alone hadn't caught
these) surfaced two pre-existing bugs. `DOMAIN_KEY_DICT` only covered 4 of the 8 real domains, so
`spectral_waves`, `storm_surge`, `wastewater_treatment`, and `water_quality` cases crashed the
matrix's row labels with a `KeyError` — fixed by filling in the missing abbreviations and falling
back to the plain domain name instead of raising. Clicking a matrix cell also did nothing:
Plotly.js's `Heatmap` trace has no `selectPoints` module at all, so it can never emit the click
event Streamlit's `on_select="rerun"` needs, regardless of app-level config — fixed at the time
with an invisible `Scatter` overlay per cell, later replaced outright by the Dash migration above.
Worth clicking through a running app before trusting headless checks whenever a Streamlit+Plotly
interaction is added — headless harnesses can't catch a client-side trace limitation like this one.

## 2026-07-09 — Multi-snapshot forecast panel for the results-matrix app

The results-matrix app's cell-click detail panel showed one flat placeholder line with no time
axis. Replaced it with `app/snapshot.py`, a Plotly panel showing 3 evenly-spaced forecast-origin
snapshots side by side, each with actual values, predicted values, and a dashed line marking where
the forecast starts — following a matplotlib sketch (`app/snapshot_example.py`) worked out first.
Used Plotly rather than matplotlib to stay visually consistent with the rest of the app, and a
fixed count of 3 snapshots rather than a configurable one since nothing yet needed more than a
quick visual check. The time axis was real (`freq` and `horizon_primary` pulled from each case's
case.toml) but the actual/predicted values were still placeholders at this point — resolved by the
snapshot-persistence entry above.

## 2026-07-08 — Interactive results-matrix app (Streamlit + Plotly)

Added `app/app.py`, a local Streamlit app rendering the same case × method grid as the static
`results/pilot_matrix_*.png` plots, but interactive: hover for the metric value plus run
diagnostics, switch metrics from a dropdown, and filter or remove rows and columns by case,
domain, method, or method family. Two supporting changes: every `Forecaster` subclass now declares
a `family` (baseline/classical/ml/deep_learning/foundation_model/ensemble), read by the app via
direct import rather than added to `pilot_metrics.parquet` — family is a property of the method,
not of a run, so attaching it shouldn't force a rerun of the multi-hour pilot; and `matrix.py`'s
per-metric colormap config was made public (`METRIC_CONFIG`) and imported by the app, so the
interactive heatmap and the static PNG can't drift apart on how a metric is colored. Clicking a
cell opens a detail panel with the case's review notes and a forecast-preview chart — the chart
used placeholder data until the eval pipeline was later changed to persist real per-origin
forecasts (see the 2026-07-09 entries above). Run with `uv run --extra app streamlit run app/app.py`.

## 2026-07-08 — Swapped statsforecast's MSTL wrapper for statsmodels' directly

`_mstl_decompose` called `statsforecast.mstl.mstl`, a thin wrapper over `statsmodels.tsa.STL` that
added no behavior this code relied on. Switched to calling `statsmodels.tsa.seasonal.MSTL`
directly. One shape difference for anyone touching this code: statsmodels' `MSTL(...).fit()`
returns a `DecomposeResult` with `.seasonal` as a 1D array for a single period or 2D (columns in
period order) for multiple, rather than statsforecast's named-column DataFrame — `_mstl_strengths`
and the notebook's decomposition plot now index by position instead of name. `statsforecast` stays
a dependency since `classical.py`'s MSTL-ETS/MSTL-ARIMA forecasters still use it for actual
forecasting, a separate use from decomposition-for-evidence.

## 2026-07-08 — Audit of the RobustPeriod implementation: three weaknesses found and fixed

Audited `robust_period_candidates` with synthetic-data probes beyond the existing test suite and
found three weaknesses, each fixed with a regression test. The `prominent` flag wasn't calibrated
for series length — `_spectral_ratio`'s max-over-window approach let long series with many noise
bins clear the threshold by chance (a quarter of white-noise draws at n=100k falsely cleared it);
fixed by raising the bar by log2 of the window's bin count and reporting each candidate's
`prominence_bar` so a reviewer sees the margin. Period-2 (Nyquist) candidates returned garbage
amplitudes because the regression's sine term is numerically zero at that frequency; fixed by
skipping periods below 3, since a period that short is below anything this benchmark declares
anyway. `_huber_clean` silently did nothing when the MAD was zero (common on flat or
zero-inflated series), letting spikes pass straight into the spectrum; fixed with a
mean-absolute-deviation fallback. The core MODWT/periodogram/dedup machinery itself checked out
fine across all probes.

## 2026-07-08 — RobustPeriod: data-driven seasonality discovery for curation

The seasonality curation notebook scored a fixed menu of physically-motivated periods and, for
off-menu cycles, ranked raw periodogram peaks — fragile on environmental data, since red-noise
variance and blurred-together cycles fool raw peaks and leave the off-menu half resting on a
human reading scattered plots. Implemented RobustPeriod (Wen et al., SIGMOD 2021) as a new
discovery pass, `robust_period_candidates` in `data/seasonality.py`: the series is detrended and
Huber-winsorised, decomposed by MODWT into dyadic frequency bands so coexisting cycles land in
separate levels, and each level's dominant period is scored by spectral prominence and a robust
Huber-periodogram amplitude. Two deliberate deviations from the paper: the O(n²) per-frequency
Huber periodogram is evaluated only at each level's candidate peak (not every frequency) to stay
fast on 100k+ point cases, and Fisher's g-test was dropped in favor of the spectral-prominence
gate since the g-test flagged every wavelet level significant even on white noise. Validated by
checking whether the tool honestly surfaces known structure (demand_ltown ranks 24/12/168
correctly; surge_cuxhaven finds a plausible off-menu 4h tidal overtide) rather than against each
case's already-declared seasonalities, since that list came from the old tooling plus human
judgment and matching it would be circular. Discovery is advisory: a reviewer still reads the
ranked evidence and decides what to commit to case.toml.

## 2026-07-08 — Non-seasonal MSTL test now builds its own non-seasonal case

The test asserting MSTL refuses to run on a non-seasonal case used `discharge_flashy_catskills`
as its example — but since curation (see 2026-07-03 below) gave every committed case a declared
`seasonalities` list, that case is no longer non-seasonal and the test was silently exercising the
wrong path. Fixed by constructing the non-seasonal case explicitly via `model_copy` with
`seasonalities` overridden to `[]`, since the synthetic data generator doesn't read that field
anyway. Worth flagging: with all committed cases now seasonal, MSTL's inapplicability guard and
the SeasonalNaive non-seasonal branch are only ever exercised by synthetic overrides, never by a
real case in the suite.

## 2026-07-08 — Dominant period is now chosen at fit time by variance explained

`dominant_period` used to return the first entry of a case's declared `seasonalities`, trusting
curation-time ordering — but in a rolling evaluation, the globally dominant cycle isn't always
what a given fold's window shows most strongly. Changed it to pick, at fit time, whichever
declared period's sinusoid explains the most variance of the detrended training target in that
fold's own window. Chose a single-sinusoid R² over a per-cycle-position seasonal mean or an ANOVA
F-statistic because declared periods are almost always multiples of each other (12|24, 24|168),
which would bias raw explained-variance toward always picking the longest period, while the
sinusoid's fixed two-parameter cost compares periods fairly even with few cycles in a fold. This
affects SeasonalNaive, Climatology, the ML sin/cos feature, and TBATS/Theta's default
`season_length` — predictions can shift for any case with more than one declared seasonality, and
`docs/RESULTS.md` needs regenerating on the next benchmark run for those cases.

## 2026-07-07 — Climatology baseline now keys on the declared dominant seasonality (issue #54)

`Climatology`'s lookup key came from a frequency-dispatch table (day-of-year for daily,
hour-of-day for hourly, weekday×hour for sub-daily), so how hard a baseline it was changed with
sampling rate rather than the series' actual seasonal structure — weekly data collapsed to the
global mean, hourly data ignored weekday effects. Changed it to key on position within the case's
dominant declared period (`case.seasonalities`), the same source SeasonalNaive and the ML sin/cos
features already use, so every seasonal component of the suite now keys the same way at every
frequency. A non-seasonal case (`seasonalities = []`) has no cycle to key on, so it falls back to
one global bucket — deliberately matching the Mean baseline there rather than raising, so the
naive ensemble stays intact. Only Climatology's own predictions change; it isn't used as a
skill-score reference, so the Climatology row and the naive-ensemble median need regenerating on
the next benchmark run, but no other method's numbers move.

## 2026-07-07 — Added a PACF panel to the seasonality notebook, alongside ACF

The seasonality notebook's ACF plot shows peaks at every multiple of a candidate period, which is
ambiguous given this project's candidate menu — within a frequency, longer menu periods are exact
multiples of shorter ones, so a strong 12h cycle mechanically inflates ACF(24) whether or not an
independent 24h effect exists. Added a PACF panel alongside ACF, since PACF nets out shorter lags
already regressed out, so a candidate that's a multiple of a shorter one only spikes if it carries
independent information. This is reviewer-facing only — the acceptance gate in
`detect_candidates()` still uses ACF plus MSTL strength, since PACF's max lag is capped below half
the sample size and its cutoff interpretation isn't guaranteed for arbitrary periodic shapes.

## 2026-07-07 — Interactive periodogram + find_peaks in the seasonality notebook

The seasonality notebook's periodogram was buried near the bottom, and off-menu peaks appeared
only as a static, non-interactive text list. Reworked it to sit under the case dropdown with a
fully interactive `find_peaks` overlay. The key fix: `scipy.signal.periodogram` is uniform in
frequency but seasonality lives in period on a log axis, so the raw spectrum is wildly
over-resolved at short periods and under-resolved at long ones — a daily case's annual peak can
fall between two bins. Resampling log-power onto a uniform log-frequency grid (aggregated by max,
not mean, to avoid averaging a sharp seasonal line into the noise floor) fixes this and makes
`find_peaks` parameters finally comparable across the spectrum and across cases. Peak-detection
parameters are now exposed as live text inputs, and three marker colors distinguish the fixed
candidate menu, detected peaks, and manually-entered candidates. This is discovery/inspection
tooling only — nothing enters a case's declared seasonalities without the full ACF/power-ratio
evidence and written justification `detect_candidates()` already requires.

## 2026-07-03 — Periodogram-peak suggestions for off-menu seasonalities

The seasonality notebook could only evidence periods a human already thought to test — its fixed
candidate menu is deliberately narrow, so a case with a different physical driver (e.g.
`waves_oregon`'s 17h swell) had no way to surface unless someone already suspected it. Added
`suggest_peak_candidates()`: finds periodogram local maxima, applies the same prominence bar as
`detect_candidates`, and drops anything within 10% of a menu/excluded period as spectral leakage.
It's advisory only — up to 5 ranked suggestions the reviewer can copy into the extra-candidates
box to get full evidence, not an acceptance path in itself. Superseded in practice by the
interactive periodogram added 2026-07-07 above, but the underlying function is still used by the
registry and tests.

## 2026-07-03 — Per-case curated `seasonalities` replace the freq → seasonality map

Seasonal periods were hardcoded per sampling frequency, assuming every hourly case carries
24h/168h cycles and every daily case 7d/365d — wrong for much of the environmental half of the
benchmark (waves and de-tided surge have no diurnal cycle; rivers have no work week) and
consequential, since that map also set the MASE denominator, distorting cross-case MASE
comparisons. Each case now declares `seasonalities` in its case.toml — required, dominant-first,
`[]` meaning explicitly non-seasonal — as the single source of truth for SeasonalNaive, TBATS,
AutoTheta, MSTL, and the ML feature pipeline; the old frequency maps were deleted with no
fallback. Candidates come from a fixed physically-motivated menu per frequency and are accepted
only with ≥2 full train-split cycles, a periodogram peak exceeding 10× the surrounding octave
band's median power, and ACF ≥ 0.1 at that lag (the ACF floor was calibrated down from 0.15 after
demand_ltown's unmistakable diurnal cycle tested at only 0.10 due to noise dilution). Two upstream
assumptions didn't survive: hourly cases are mostly tidal (12h), not diurnal, and one daily case
(`discharge_flashy_catskills`) is genuinely non-seasonal. New cases must declare `seasonalities`
(schema-enforced); per-case evidence and rationale live in each case's case.toml notes rather than
being duplicated here, and MASE values shift on any case where the declared dominant period
differs from the old frequency default — expected, that's the point.

## 2026-07-03 — Silencing a spurious TBATS Box-Cox warning

`TBATSForecaster.fit` pre-applies a Yeo-Johnson transform for cases with non-positive targets
(e.g. `watertemp_james`), so TBATS's own internal Box-Cox is expected to be a no-op — but
Yeo-Johnson with `standardize=False` isn't positivity-guaranteeing, so some transformed points
still landed at or below zero and triggered TBATS's own warning-and-disable logic. Fixed by
passing `use_boxcox=False` explicitly whenever the pre-transform was applied, producing an
identical fit without the noise; `use_boxcox` stays at its default when the raw data is already
positive, so TBATS can still decide for itself whether a Box-Cox transform helps.

## 2026-07-03 — Gap handling: one materialized policy (cap + longest-segment trim)

Missing target data was handled in two places — the eval loop imputed gaps on the fly (linear
interpolation then mean-fill), and several methods also interpolated or dropped NaNs internally,
inconsistently (`nf_models` dropped rows, others filled). Worse, mean-fill let models train on
large fabricated stretches — one 6574-step block was 54% of `discharge_lowland_denmark`, a single
interpolated straight line. Replaced both layers with one materialized policy in `data/curate.py`:
gaps are interpolatable only up to a per-case cap (the lag at which the target's ACF first drops
below 0.8, bounded by one seasonal cycle), gaps larger than the cap are treated as hard
boundaries, and only the longest contiguous segment is kept — chosen over most-recent because a
big gap near `waves_massachusetts`'s train/test boundary would have starved it to 5 training
points. A `y_observed` mask on the curated data tracks which points were genuinely measured, so
metrics and origin selection never score a filled point as real. `scripts/curate_data.py` writes
`data/curated/<case>.parquet` once; re-run it after any data refresh or a change to the ACF
threshold.

## 2026-07-03 — Ensemble-of-baselines columns (issue #59)

Added a generic `EnsembleForecaster` that composes existing Forecasters and combines their
per-step forecasts by median (robust to one member going badly off, e.g. Persistence during a
spike), closing the 2026-07-02 method-suite audit's "no ensemble-of-baselines column" gap. Two
columns are registered: `EnsembleBaselines` (median of the four naive baselines) and
`EnsembleBaselinesTheta` (the same four plus Theta), bracketing a pure zero-cost reference and a
naive+classical blend. A member that raises at predict time is dropped for that origin; the
ensemble only raises if every member fails. An early smoke test on `discharge_lowland_denmark`
showed the intended calibration signal — EnsembleBaselinesTheta beat persistence while the
naive-only ensemble trailed it on that flashy series.

## 2026-07-02 — Seasonal classical baselines: MSTL forecasters + seasonally-adjusted Theta (issue #46)

AutoARIMA/AutoETS/AutoTheta all ran non-seasonal (`season_length=1`) while the windowed-ML
methods got calendar features and seasonal lags, so any "classical vs ML" comparison was partly
an artifact of deliberately de-seasonalized classical baselines. Since seasonal ARIMA/ETS cost
explodes with large periods (365 daily, 96 for 15-min data), kept AutoARIMA/AutoETS as
non-seasonal short-memory baselines and added dedicated seasonal columns instead: MSTL-ETS and
MSTL-ARIMA (statsforecast's MSTL strips seasonal components via STL, fits on the residual, then
re-seasonalizes) and made Theta seasonal by default, since its classical deseasonalization is
cheap at any frequency. If a training window doesn't support even the smallest declared period
(rare — real cases clear it), MSTL raises rather than silently falling back to a bare non-seasonal
forecaster, which would have made the new columns byte-identical fakes of the existing ones. This
makes "classical vs ML" a genuine methodology contrast rather than an artifact of an unfair setup.

## 2026-07-02 — Audit of the method suite (`bench/methods/`): flaws and gaps identified

Reviewed all method wrappers (`naive.py`, `classical.py`, `ml.py`, `dl.py`, `fm.py`,
`nf_models.py`, `torchdl.py`) and how the eval loops call them, recording findings to prioritize
rather than fixing them immediately. Biggest validity concern: classical methods ran non-seasonal
by default while ML methods got seasonal features, an asymmetry fixed by the seasonal-classical-
baselines entry above; also flagged were asymmetric training budgets between statsforecast's
auto-selection and the neural methods' fixed tiny step counts, unscaled MLP features, and Chronos
scoring its median against a mean-optimal metric. Correctness issues found: `_complete_for_fit`
ran on the full train+test frame before slicing, leaking test-period information into training
gap-fills (fixed by the 2026-07-03 gap-handling rewrite above); DLinear/N-HiTS/PatchTST silently
fell back to persistence on short series while reporting status "ok"; and several
frequency-dispatch tables only handled {daily, hourly, 15-min, weekly} with silent generic
defaults otherwise. Design risks noted: `predict(history=...)` is dead in every non-naive method
(each refits from fit-time state, correct today only because both eval loops refit per origin),
and the exogenous tier conflated historical-only and known-future covariates. Suite gaps flagged
included the missing ensemble-of-baselines and MSTL columns (both closed above), a single
foundation model, and a strategy confound between recursive mlforecast methods and direct-
multi-output DL methods.

## 2026-07-02 — Fixed rolling-origin selection collapsing onto the front of the test set

`run_case_rolling`'s origin selection was supposed to spread `n_origins` cutoffs across the whole
test period, but it oversampled 2×n candidates, filtered for ≥50% observed horizon, then
truncated to the first `n_origins` of the sorted list — since the list was sorted ascending,
that's the front ~50% of the range, not a spread sample. On `discharge_lowland_denmark` (H=7,
test range 1241–1453) the last origin used was ~1347 instead of 1453, so only 46% of the usable
test range was ever scored and any regime shift near the end was silently missed. This bites
hardest when horizon is small relative to test length. Fixed by extracting
`_select_rolling_origins()`, which now evenly downsamples the filtered survivor list via
`linspace` over indices rather than truncating it; `run_leadtime_curve` in the same file wasn't
affected, since it already samples origins directly via `linspace`. This changes which points get
scored, so any published matrix numbers generated before this fix should be treated as based on
only the first half of each case's test period — re-running the pilot is the fix.

## 2026-07-02 — Added missing `discharge_saugeen` case doc and EDA panel

`discharge_saugeen` had case code (case.toml, download.py, synth.py) but no docs page, EDA
figure, or entry in `docs/README.md`'s case table — added all three, framing the case's role as
the deliberate univariate control against the three exogenous hydrology cases, isolating pure
autoregressive/seasonal skill. No case behavior changed.

## 2026-07-02 — Dropped `optional_exogenous` as a redundant stored field

`Case.optional_exogenous` was a stored case.toml field, but given the fixed set of available
columns and which are necessary, "optional" is always just the complement — pure duplication that
only needed a validator to police it. Made it a derived property instead
(`(exogenous ∪ exogenous_known_future) − necessary_exogenous`) and removed the field from all 9
case.toml files that had it; `Case.exo_tier` and everything downstream is unaffected.

## 2026-07-02 — Structured exo-necessity fields, persisted as result metadata (not matrix aggregation)

Promoted the exogenous necessity judgment (recorded in prose the same day, see below) into
schema: added `necessary_exogenous`/`optional_exogenous` fields to `Case` with a validator
requiring every exogenous column to be classified as exactly one, plus a derived `exo_tier`
property (`none`/`optional`/`necessary`). Considered splitting the results matrix by tier or
demoting univariate methods to a "reference floor" on necessary-exo cases, but rejected both —
the matrix never aggregates or ranks methods by design, and demoting some methods on some cases is
aggregation by another name; some cases are also deliberately grouped for cross-regime comparison
regardless of tier. Landed instead on metadata-only labeling: `case.exo_tier` and each method's
`uses_exogenous` flag are now persisted as plain columns on every result row, so downstream
consumers can filter or facet by tier without the benchmark itself declaring which methods "count"
for which cases.

## 2026-07-02 — Recorded necessary-vs-optional exogenous judgment in case notes

Mixing cases where exogenous input is domain-essential (e.g. storm surge needs wind/pressure)
with cases where it's a nice-to-have enrichment makes a single results matrix hard to read, since
a method's ranking on one isn't comparable to its ranking on the other — and this necessity
judgment is a domain call, not something inferable from measured skill deltas alone. As a first
step, recorded that judgment in prose in each case's `notes` field: extracted from existing case
descriptions where the necessity claim was already implicit (`surge_cuxhaven`, `watertemp_james`,
`discharge_snowmelt_yosemite`, `groundwater_yorktown`), and flagged as a fresh, arguable
assumption where it wasn't (`discharge_flashy_catskills`, `discharge_lowland_denmark`,
`sewerlevel_bellinge`, the two wave cases). This is prose only — no schema or pipeline change yet;
promoting it to structured fields followed the same day, see the entry above.

## 2026-07-02 — Added CatBoost to the windowed-ML method set

Added `CatBoost`/`CatBoost+exo` to the windowed-ML column family, slotting into the same shared
lag+calendar feature pipeline as LightGBM with a matched training budget (200 iterations, 0.05
learning rate) so the two GBDT methods are compared on roughly equal footing. Gives the benchmark
a second GBDT implementation — useful since CatBoost's ordered-boosting scheme handles the same
lag/calendar features differently and may behave better or worse depending on how noisy a case's
covariates are. No methodology change, straightforward column addition.

## 2026-07-02 — Backfilling this log from the scoping documents

This entry and those below it were written on this date, reconstructing the project's decision
history from `research_plan.md`, `docs/SCOPING.md`, `docs/review_responses_pr2.md`, and git
history, since this log started empty — dates reflect when decisions actually happened, not when
they were logged. Comparing the original plan to the codebase surfaced several undocumented
drifts: only Chronos-Bolt zero-shot was ever implemented of the six foundation models the plan
called for, and recurrent nets plus the fine-tuning phase were never started; process-model
reference columns and hybrid ML residual correction, called out in the plan as central, were
pushed back on by domain expert Rocco during scoping review and never built, so his position won
by default; NSE/KGE metrics were agreed as a queued change but still aren't in
`bench/eval/metrics.py`; the Nixtla-based library strategy landed but wrapped behind a
project-own `Forecaster` interface, and N-HiTS/PatchTST were reimplemented on bare torch to avoid
neuralforecast's `ray` dependency; and the package split into `tsforecastbench.data`/
`tsforecastbench.bench` on 2026-06-13, deliberately so the two halves could become separate repos
later. None of this needed fixing — the codebase and docs are internally consistent with what
actually exists; `research_plan.md` should just be read as the original proposal, not a
description of the current system.

---

## 2026-06-25 — Added a second, unforced hydrology contrast case

Added `discharge_saugeen` (daily discharge, Saugeen River, Ontario, 65 years of record, no
exogenous inputs) alongside `discharge_flashy_catskills`, answering `research_plan.md`'s open
question about whether to add a second Danish catchment or an alpine one — with a third option
instead: a long-record, exogenous-free univariate series as a methodological contrast to the
exogenous-rich cases, rather than another geographic contrast. Brings the case count to 12.

---

## 2026-06-12 to 2026-06-13 — Scoping review with Rocco (domain expert), and the response

Before scaling past 6 cases and 9 methods, sent a scoping review to Rocco (internal urban +
catchment hydrology expert) to sanity-check case selection and design choices against operational
reality; full comments and responses are in `docs/review_responses_pr2.md`. The purpose got
reframed as a result: this benchmark isn't about finding the best model per case, but about a
fair comparison of the same methods across very different water/environment domains — which is
why the matrix must stay a dimensionless score (MASE) and domain-specific metrics (NSE/KGE, event
hit-rate) live as per-case sidebars, never matrix cells. Process-model reference columns and
hybrid ML residual correction, central to the original plan, were de-scoped on Rocco's view that
comparing against simulated output complicates the picture without being necessary — left as an
open decision for Jesper but never implemented since. Exogenous inputs were also decided to not be
optional everywhere: for Bellinge and the WWTP case the rainfall response is the phenomenon being
forecast, so a univariate run there is a floor, not a real entry, ruling out one uniform
exogenous schedule across all cases. The architecture split into
`tsforecastbench.data`/`tsforecastbench.bench` also landed in this window, along with
exogenous-aware windowed-ML methods and a skill-vs-mean metric.

---

## 2026-05-29 — Two data-quality bugs found before scaling further

Early EDA over the first six real-data cases surfaced two bugs serious enough to invalidate
earlier real-data numbers. The originally chosen Bellinge level sensor was only active for a
small fraction of its nominal 2010–2020 span (near-zero the rest of the time), so an earlier
suspiciously good SeasonalNaive MASE of 0.32 was effectively scored on near-zero data — fixed by
adding a `--scan` coverage mode to the downloader and switching to a sensor confirmed 100% active.
The BattLeDIM demand target had been a single pump's flow, which alternates on/off with a second
pump under scheduling, producing a square wave rather than a demand signal — fixed by summing all
pump columns for total network inflow. This is why a per-case quality gate
(`scripts/review_case.py`, the `/review-case` skill) became a standing requirement for adding any
new case.

---

## 2026-05-27 to 2026-05-28 — Initial scoping plan and repo scaffold

`research_plan.md` laid out the original proposal: 10-11 candidate cases spanning the water
cycle, ~16 forecasters from naive baselines to foundation models, a phased roadmap, and a
Parquet/Pydantic storage design. The initial commit scoped this down to a first working
pipeline — 6 cases, 9 methods — with the matrix and lead-time evaluation working end-to-end and
CI in place from day one, a deliberate "lock the interfaces first" move rather than building out
the full scope before anything ran.
