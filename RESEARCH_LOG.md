# Research Log

A running record of the testing and decision-making behind this benchmark — what was tried,
what was decided, and why. Intended to be readable by collaborators without needing to dig
through commit history or notebooks.

Entries are in reverse-chronological order (newest first).

<!-- Add new entries below this line, newest first. -->

## 2026-07-09 — Moved the results-matrix app from Streamlit to Dash

The results-matrix app was built on Streamlit + Plotly, and cell-click-to-snapshot was the
sore spot. The root cause is structural, not a config mistake: Plotly's `Heatmap` trace has
no selection module, so Streamlit's `on_select="rerun"` never fires for a heatmap click. The
earlier workaround (overlay an invisible `Scatter` per cell just to catch clicks) worked but
was fragile — it stacked on top of reversed axes, square-cell anchoring, and paper-referenced
annotations, all of which are places where the click round-trip can stop lining up.

Rather than keep patching that layer, moved the app to **Dash**, where a heatmap emits
`clickData` natively and a callback reads the clicked cell directly — the pattern Dash is
built for. This also fits where the app is heading: as more linked panels get added (real
forecast snapshots, richer per-cell detail), Dash's explicit input→output callbacks handle
interdependent panels more cleanly than Streamlit's top-to-bottom rerun model.

What changed:
- `app.py` rewritten as a Dash app. The invisible-scatter click hack is gone; the matrix is
  a plain `dcc.Graph` heatmap, and a `clickData` callback drives the snapshot panel and notes.
- `data.py` and `snapshot.py` carried over almost untouched — they only ever built pandas
  frames and `go.Figure` objects, which are framework-agnostic. The one Streamlit coupling was
  `@st.cache_data` on `load_metrics`; swapped it for stdlib `functools.cache`, keeping the same
  `version`-based cache-busting for the reload button.
- Dependencies: the `app` extra now pulls `dash` instead of `streamlit`.
- Run command is now `uv run --extra app python app/app.py` (was `streamlit run app/app.py`).

Verified end-to-end: the server boots and serves, the matrix renders as a single heatmap
trace (no scatter overlay), and a simulated cell click populates the snapshot figure and
case notes while toggling the panel's visibility. The snapshot data is still the seeded
placeholder from the prior work — the pipeline doesn't persist per-origin forecasts yet — so
that stays scaffolding, but the click path it hangs off is now solid.

One known limit of the reload button: it re-reads the parquet and redraws with the current
filters, so new cases within existing domains/families show up, but a brand-new domain or
family won't appear in the filter widgets until the app is restarted.

## 2026-07-09 — Two app bugs found on first real-browser test, both pre-existing

First manual, real-browser test of the results-matrix app surfaced two bugs, neither
introduced by the same-day snapshot-panel work (see below) — both were already there and
just hadn't been exercised in a browser yet (the 2026-07-08 entry flagged this gap directly:
only headless checks had run so far).

- **`DOMAIN_KEY_DICT` only covered 4 of the 8 real domains.** `spectral_waves`, `storm_surge`,
  `wastewater_treatment`, and `water_quality` cases (added earlier, well before this app
  existed) crashed the matrix's row-label annotation with a `KeyError`. Filled in the missing
  abbreviations and made the lookup fall back to the plain domain name instead of raising, so
  a future new domain degrades gracefully rather than crashing the app.
- **Clicking a matrix cell did nothing at all.** Traced this to Plotly.js itself: a `Heatmap`
  trace has no `selectPoints` module (checked directly in the bundled `plotly.js` source —
  `bar`, `scatter`, `box`, and others each register one; `heatmap` doesn't), so it can never
  emit the selection event Streamlit's `on_select="rerun"` depends on. No amount of
  app-level configuration would have fixed this — the trace type itself can't be selected.
  Fixed by overlaying an invisible `Scatter` marker on each cell, aligned to the same
  categorical grid, purely to catch clicks; `Scatter` does support selection, and the marker
  is fully transparent so nothing changes visually.

Both were caught by having the user click through the running app rather than relying on
headless checks — worth doing that earlier next time a Streamlit+Plotly interaction is added,
since headless test harnesses can confirm the Python side is wired correctly without ever
catching a client-side limitation like the heatmap one above.

## 2026-07-09 — Multi-snapshot forecast panel for the results-matrix app
## 2026-07-09 — Multi-snapshot forecast panel for the results-matrix app

The results-matrix app's cell-click detail panel (added 2026-07-08, see below) showed one
flat placeholder line with no time axis and no sense of where a forecast started. Replaced it
with `app/snapshot.py`, a Plotly panel of 3 forecast-origin snapshots side by side — each one
a short window of actual values with the predicted values overlaid and a dashed line marking
where the forecast starts. The layout follows `app/snapshot_example.py`, a matplotlib sketch
the design was worked out in first.

Two decisions worth recording:

- **Plotly, not matplotlib.** The sketch used matplotlib, but the rest of the app (the matrix
  heatmap, and the chart this replaces) is Plotly, so Plotly keeps the panel interactive and
  visually consistent with everything else on the page.
- **Fixed 3 evenly-spaced snapshots**, not a user-configurable count. Nothing yet needs more
  than a quick visual sense of forecast behavior at a few points, so a slider or input box
  would be a control with no real use.

The actual/predicted values are still placeholders (per-origin forecasts aren't persisted by
the eval pipeline yet — same gap noted in the 2026-07-08 entry), but the time axis is real:
`freq` and `horizon_primary` are pulled from each case's `case.toml` via the case registry, so
an hourly case shows an hourly axis and the right horizon length, and a daily case shows a
daily one. Swapping in real forecasts later should only mean rewriting the two data-generating
functions in `app/snapshot.py`; the panel layout and the app wiring shouldn't need to change.

Verification note: same limitation as the 2026-07-08 entry — this sandbox can't screenshot a
real browser. Checked instead by rendering the figure directly and inspecting its structure
(trace count and colors, one dashed vline and "Prediction start" annotation per subplot,
correct x-axis range for both an hourly case and a daily case) and confirming repeated calls
with the same case/method produce byte-identical output. Worth a manual look in a browser
before relying on the visual layout.

## 2026-07-08 — Interactive results-matrix app (Streamlit + Plotly)

Added `app/app.py`, a local Streamlit app that renders the same case x method grid as the
static `results/pilot_matrix_*.png` plots, but interactively: hover a tile for the metric value
plus run diagnostics (status, n_origins, frac_finite_test, runtime, horizon), switch between
MASE / Skill_vs_Persistence / Skill_vs_Mean / MAE / RMSE from a sidebar dropdown, and filter or
remove rows and columns by case, domain, method, or method family. Rows are grouped by domain
and columns by method family, each with a separator line and label, rather than using tile
color for domain — tile color is already the metric value, and a second color channel for
domain would fight it.

Two pipeline-adjacent changes support this:

- **Every `Forecaster` subclass now declares a `family`** (baseline / classical / ml /
  deep_learning / foundation_model / ensemble) in `methods/*.py`. The app builds a
  `method name -> family` lookup by importing the method classes directly rather than adding a
  `family` column to `pilot_metrics.parquet` — the pilot run takes hours, and family is a
  property of the method, not of a specific run, so there's no reason to force a rerun just to
  attach it to old data.
- **`matrix.py`'s `_METRIC_CONFIG` (colormap, center, clamp range per metric) is now public
  (`METRIC_CONFIG`)** and imported by the app, so the interactive heatmap and the static PNG
  always agree on how each metric is colored — one definition instead of two that could drift.

Clicking a cell opens a detail panel with that case's `results/case_review/<case_id>.md` notes
(when they exist) and a forecast-preview chart. The chart is placeholder data — the eval
pipeline (`bench/eval/rolling.py`) pools actual-vs-predicted arrays into metrics and discards
them, so there's currently no persisted per-case, per-method forecast to show. The panel layout
is real; wiring it to real forecasts means persisting them during the pilot run first, which is
a separate, larger change deferred for now.

Run with `uv run --extra app streamlit run app/app.py` (added `streamlit` + `plotly` under an
optional `app` dependency group, kept out of the default install since only this viewer needs
them).

Verification note: this sandbox can't launch a real browser (headless Chromium needs system
libraries that require root, and there's no passwordless sudo here), so the app was verified
with Streamlit's own headless `AppTest` harness — it runs the real app script and checks the
widget tree and state transitions (metric switch, family-filter cascade, reload) — plus direct
calls to the figure-building and formatting functions. Visual appearance (colors, separator
placement, tooltip rendering) has not been screenshot-verified; worth a manual look before
relying on it.

## 2026-07-08 — Swapped statsforecast's MSTL wrapper for statsmodels' directly

`_mstl_decompose` (used by `detect_candidates`'s MSTL-strength corroboration and by the
seasonality curation notebook's decomposition plot) called `statsforecast.mstl.mstl`, a thin
wrapper over `statsmodels.tsa.STL`. Since statsmodels exposes its own `MSTL` class directly and
the wrapper added no behavior this code relied on, switched to calling
`statsmodels.tsa.seasonal.MSTL` directly and dropped the indirection. `statsforecast` stays a
dependency — `classical.py`'s MSTL-ETS/MSTL-ARIMA forecasters still use it for the actual
forecasting models, which is a separate use from decomposition-for-evidence.

One shape difference to note for anyone touching this code: statsforecast's wrapper returned a
DataFrame with named columns (`seasonal`, or `seasonal{period}` per period, plus `remainder`);
statsmodels' `MSTL(...).fit()` returns a `DecomposeResult` with `.trend`/`.seasonal`/`.resid`
attributes, where `.seasonal` is a 1D array for a single period and a 2D array (columns in the
order `periods` was passed) for multiple. `_mstl_strengths` and the notebook's decomposition
plot were updated to index by position instead of by name. Behavior is otherwise unchanged —
existing `test_seasonality.py` coverage of MSTL corroboration passes unmodified.

## 2026-07-08 — Audit of the RobustPeriod implementation: three weaknesses found and fixed

Audited `robust_period_candidates` and its helpers with synthetic-data probes beyond the
existing test suite. The core machinery checks out — MODWT energy preservation and band
mapping, the biweight level ranking, the IRLS Huber periodogram, multi-cycle discovery under
trend and spikes, harmonic flagging, and dedup all behave as designed. Three weaknesses were
confirmed empirically and fixed in the same pass (each with a regression test):

- **The `prominent` flag is not calibrated for series length.** `_spectral_ratio` takes the
  *max* power over a ±2% window around the candidate frequency; that window holds
  `0.04·n/period` Fourier bins, so on long series the max over many noise bins inflates the
  ratio. On pure white noise at n=100k, period 24, a quarter of random draws cleared
  POWER_RATIO_MIN=10; AR(1) noise (φ=0.99, n=20k) produced falsely prominent periods on 3 of
  3 seeds. The discovery path is hit harder than the menu path because the tested period was
  itself picked from the data's ACF (post-selection bias) and prominence is the only gate —
  there is no ACF/MSTL corroboration like `detect_candidates` has. **Fix:** the bar is raised
  by `log2` of the window's bin count (the max of k exponential-ish noise bins grows like
  `ln k`, so this holds the noise clearance probability at its single-bin level), and each
  candidate now reports its `prominence_bar` so the reviewer sees the margin, not just the
  flag. An ACF second gate was considered and rejected: band-pass-filtered noise has in-band
  ACF peaks by construction, so it would not discriminate. After the fix, white noise at
  n=100k produces no prominent candidates on any seed tried, and AR(1) red noise is clean on
  5 of 6 seeds — the survivor is a marginal 12.5 against a bar of 11.6 at period 240, driven
  by the red-noise spectral slope itself rather than window width, and visibly marginal in
  the evidence table. Genuine cycles sit orders of magnitude above the bar either way.
- **Period-2 (Nyquist) candidates return garbage amplitudes.** At frequency 0.5 the sine
  column of the Huber regression design is numerically zero, the normal equations are
  near-singular, and `robust_amplitude` comes out around 10^11 for a unit-amplitude signal.
  Reached through the full pipeline in 8 of 30 seeded runs on an alternating series. **Fix:**
  discovery skips periods below 3. A period-2 cycle is below anything this benchmark declares,
  and detection at the Nyquist edge is unreliable regardless (its ACF peak sits at a dyadic
  band edge, so period 2 was also read as 4 or 6 depending on noise).
- **`_huber_clean` silently does nothing when the MAD is zero** (more than half the values
  identical — flat, censored, or zero-inflated series), returning the data uncleaned, so
  spikes pass straight into the spectrum. **Fix:** the scale falls back to the mean absolute
  deviation (Gaussian-consistent via `sqrt(pi/2)`), which is nonzero unless the series is
  constant, restoring the winsorisation the method is named for.

Minor notes, left as-is: subharmonics (integer multiples of a strong cycle, e.g. 48 under a pure 24)
surface unflagged by `harmonic_of`, but come out non-prominent in practice; the
`robust_period_candidates` docstring says "Huber-ACF" where `_band_period` computes a plain
ACF of the Huber-cleaned series' coefficients; a period at exactly n/2 in a power-of-two
band is caught by the level below (verified — found as 1023 for a true 1024), so the
truncated top band self-heals.

## 2026-07-08 — RobustPeriod: data-driven seasonality discovery for curation

The seasonality curation notebook (`notebooks/seasonality.py`) scored a fixed menu of
physically motivated periods per frequency and, for off-menu cycles, ranked raw periodogram
peaks (`suggest_peak_candidates`). Raw peaks are fragile on environmental data: red-noise
variance, occasional spikes, and two blurred-together cycles all fool them, so the off-menu
half leaned heavily on a human reading scattered plots. We wanted a principled discovery step.

**Decision.** Implement RobustPeriod (Wen et al., SIGMOD 2021) in-repo as a discovery pass in
`data/seasonality.py` (`robust_period_candidates`). The series is linearly detrended and
Huber-winsorised (robust to trend and spikes), decomposed by an FFT-based MODWT into dyadic
frequency bands so coexisting cycles land in separate levels, and each energised level
contributes its dominant in-band period (read off a Wiener-Khinchin autocorrelation). Every
candidate is scored by spectral prominence (the same red-noise-robust ratio the menu evidence
uses) and by a robust amplitude from the exact Huber periodogram, then returned ranked, with
flags for harmonics, resolution-limited long periods, and the nearest menu period.

**Two deviations from the paper, both deliberate:**
- *Huber periodogram.* The paper's Huber periodogram is a per-frequency robust regression
  (O(n²) over all Fourier frequencies) — infeasible on our 100k–150k-point cases. We compute
  the cheap FFT periodogram of the Huber-cleaned signal for peak-finding, and evaluate the
  exact Huber M-estimate (solved by IRLS, same optimum as the paper's ADMM for this convex
  two-parameter fit) only at each level's candidate peak frequency. This uses the real robust
  estimator where it decides the outcome and runs in under a second on the largest case.
- *Fisher's g-test dropped.* The paper applies Fisher's g-test per wavelet level. Empirically
  it flags every level significant, even on white noise: a band-pass-filtered level is not
  white and its periodogram ordinates are correlated, so the test's white-noise null is
  grossly violated. We gate on spectral prominence instead (and rank by it); wavelet variance
  ranking plus the prominence bar cleanly separate real cycles from noise and harmonics.

**Validation, and a correction to how we judge it.** We do *not* validate against the
`seasonalities` already declared in each case.toml — that list came from the old tooling plus
human judgment, so matching it is circular. Judged instead by whether the tool honestly
surfaces the periodic structure in the data: on `demand_ltown` it ranks 24, 12 (flagged a
harmonic of 24), and ~168 far above everything else; on `surge_cuxhaven` it finds a strong
off-menu 4 h cycle (a plausible shallow-water M6 tidal overtide); on `waves_oregon` it reports
only weak, broad structure — the honest answer for broadband swell. Notably the declared list
is not gospel: `demand_ltown`'s 12 h cycle (spectral ratio ~1200) is strongly present but
undeclared, while its declared 96 has a ratio of ~5.

**Going forward.** Discovery is advisory and semi-automated: the reviewer reads the ranked
evidence and decides what to log. `suggest_peak_candidates` is left in place for now; retiring
it is a possible follow-up. Confirmation tests (Friedman/QS on position-in-cycle) were scoped
out of this pass and remain a candidate next step to turn "prominent" into a formal per-period
verdict.

## 2026-07-08 — Non-seasonal MSTL test now builds its own non-seasonal case

`test_mstl_is_honestly_inapplicable_on_declared_nonseasonal_case` checks that MSTL refuses to
run when a case declares no seasonalities, rather than quietly collapsing into the plain
non-seasonal trend model (which would just duplicate the AutoARIMA/AutoETS column). It used
`discharge_flashy_catskills` as its non-seasonal example.

That premise no longer holds. Since curation gave every one of the 12 cases a declared
`seasonalities` list, no committed case is non-seasonal, so the test was exercising a seasonal
case and MSTL correctly returned `ok`.

**Decision.** Construct the non-seasonal case explicitly: `model_copy` the real case with
`seasonalities` overridden to `[]`. The synthetic data is identical (the catskills generator
does not read `seasonalities`), so the test still runs end-to-end through `run_case` and now
reaches MSTL's `if not case.seasonalities` guard as intended.

Worth flagging for collaborators: with all committed cases seasonal, the non-seasonal code
paths — MSTL's inapplicability guard and the SeasonalNaive branch in `test_baselines_run` — are
only ever hit by synthetic overrides, never by a real case in the suite.

## 2026-07-08 — Dominant period is now chosen at fit time by variance explained

`dominant_period` previously returned the first entry of a case's declared `seasonalities`,
trusting the curation-time order (most-prominent first). In a rolling evaluation each fold
fits on a different window, and the globally dominant cycle is not always the one a given
window shows most strongly. So the single period the seasonal methods key on — SeasonalNaive's
tile length, Climatology's bucketing, the ML sin/cos feature, and the default `season_length`
for TBATS and Theta — was fixed offline rather than matched to the data actually in hand.

**Decision.** Pick the dominant period at fit time: among the declared periods, choose the one
whose sinusoid explains the most variance of the detrended training target, measured on the
fold's own window. Options considered:

- *How to score a period.* Chose the **R² of a single sinusoid** (sin + cos at the period) over
  two alternatives. A per-cycle-position seasonal mean (the "arbitrary shape" that Climatology
  forecasts with) is degenerate for ranking here: our declared periods are almost always
  multiples of each other (12|24, 24|168, 96|672), so the longer period's buckets refine the
  shorter one's and mechanically explain at least as much variance — raw explained variance would
  always pick the longest period. An ANOVA F-statistic corrects that length bias but still needs
  many cycles to estimate one mean per bucket, and the long periods are exactly where a fold holds
  fewest cycles. The single sinusoid uses two parameters regardless of period, so it compares
  periods fairly and stays stable with few cycles, and it is phase-invariant (both sin and cos are
  fit), so it needs only the target values — no timestamps or frequency. It also matches what the
  ML feature then encodes: one harmonic at the dominant period.
- *Support.* Assumed, not enforced. A period a case cannot support must not be declared in its
  case.toml; `dominant_period` does not filter on cycle count. It falls back to the first declared
  period only when the target is too degenerate to detrend (fewer than three finite points).
- *Scope.* Climatology's forecast is unchanged in shape — it still keys on the per-position mean
  (arbitrary shape) at whatever period is chosen. The sinusoid is only the yardstick for picking
  the period, never the model of the cycle. `mase_season_for` (the MASE denominator) still walks
  the committed order and is left as-is; the case.toml order still records curation prominence and
  feeds that function.

**Impact.** Predictions can move for any case with more than one declared seasonality —
SeasonalNaive, Climatology, the ML methods, TBATS, and Theta may key on a different period per
fold than the committed first entry. Single-seasonality and non-seasonal cases are unaffected
(the lone period, or 1, is returned without scoring). `docs/RESULTS.md` needs regenerating on the
next benchmark run for the multi-seasonality cases.

## 2026-07-07 — Climatology baseline now keys on the declared dominant seasonality (issue #54)

The `Climatology` baseline (mean target by seasonal position) chose its lookup key from a
frequency-dispatch table: day-of-year for daily data, hour-of-day for hourly, weekday x hour
for sub-daily. The consequence was that how hard a baseline it is changed with the sampling
rate rather than with the series' actual seasonal structure — weekly data collapsed to the
global mean, hourly data ignored the weekday, and only 15-minute data used both weekday and
hour. Because skill scores are usually read against a climatology- or seasonal-naive-style
reference, that made cross-frequency comparisons of skill inconsistent for a reason unrelated
to the series.

**Decision.** Key `Climatology` on the position within the case's dominant declared period
(`case.seasonalities`), the same single source of truth the seasonal methods (SeasonalNaive,
MSTL, the ML sin/cos features) already read. The rule is now identical at every frequency:
one bucket per position in the dominant cycle. Options considered and their outcomes:

- *Keying granularity.* Chose the **dominant** declared period over the longest declared one.
  The dominant period is what SeasonalNaive and the sin/cos features already use, so all
  seasonal components of the suite now key the same way. The longest-period alternative would
  have preserved the old sub-daily weekday x hour richness but diverged from the rest of the
  suite and produced sparser per-bucket means.
- *Non-seasonal cases.* A case declared non-seasonal (`seasonalities = []`) has no cycle to
  key on, so every timestamp shares one bucket and the baseline equals the global training
  mean. Chose this over raising (as SeasonalNaive does) so the naive ensemble stays intact;
  Climatology simply coincides with the Mean baseline there, which is the honest answer.

**Implementation.** The position-in-cycle logic (canonical (freq, period) -> calendar-field
positions, with an epoch-anchored step-count fallback for non-canonical periods) was extracted
from the ML feature code into `seasonality.position_in_cycle` and is now shared by both the ML
sin/cos features and `Climatology`; the ML methods are unchanged in behaviour. Unit tests cover
the helper (canonical pairs, epoch fallback, bounds, determinism) and the two Climatology paths
(periodic forecast in the dominant period; global mean when non-seasonal).

**Impact.** Only `Climatology`'s own predictions change — it is scored as a method but is not
used as a skill-score reference, so no other method's skill numbers move. The Climatology rows
and the naive-ensemble median in `docs/RESULTS.md` need regenerating on the next benchmark run.

## 2026-07-07 — Added a PACF panel to the seasonality notebook, alongside ACF

The seasonality curation notebook's ACF plot showed peaks repeating at every multiple of a
candidate period (e.g. lag 12, 24, 36... for a 12-step cycle) — expected behaviour for any
persistent seasonal signal, but a real source of ambiguity given this project's
`CANDIDATE_MENU`: within a frequency, the longer menu periods are exact integer multiples of
the shorter ones (1h: 24 = 2x12, 168 = 14x12; 15min: 672 = 7x96). A strong semidiurnal (12h)
cycle mechanically inflates ACF(24) whether or not an independent diurnal effect exists, so
ACF alone can't tell a reviewer whether the longer candidate is corroborated or is just
harmonic bleed-through from the shorter one.

**Decision.** Added a second plot cell computing PACF (`statsmodels.tsa.stattools.pacf`,
Yule-Walker, unadjusted — new `_pacf` helper in `seasonality.py`) over the same lag range,
plotted the same way as the existing ACF cell (candidate lags marked, `±2/sqrt(n)`
significance band). Unlike ACF, PACF at a given lag nets out the shorter lags already
regressed out, so a candidate that's a multiple of a shorter one only spikes if it carries
information beyond what the shorter cycle already explains. This makes the notebook's ACF/PACF
pair diagnostic for exactly the multiple-of-menu-period case, without changing what
`detect_candidates()` accepts — PACF is reviewer-facing only; the acceptance gate still uses
ACF (paired with MSTL strength) as documented in `seasonality.py`, since PACF's max lag is
capped at under half the sample size (Yule-Walker requirement) and its "cutoff" interpretation
is only clean for AR-type dependence, not guaranteed for arbitrary periodic shapes.

Added `statsmodels` as an explicit dependency (`pyproject.toml`) — it was previously only
pulled in transitively via `statsforecast`'s MSTL wrapper, but is now imported directly by
`tsforecastbench.data.seasonality._pacf`.

## 2026-07-07 — Interactive periodogram + find_peaks in the seasonality notebook

The seasonality curation notebook (`notebooks/seasonality.py`) previously buried the periodogram
near the bottom and surfaced off-menu peaks only as a static text list (from
`suggest_peak_candidates()`) with no way to interrogate the spectrum. A reviewer who disagreed
with the suggested peaks had no knobs to turn.

**Decision.** Reworked the notebook so the periodogram sits immediately under the case dropdown
and is fully interactive:

- **Log-frequency resampling (the key idea).** `scipy.signal.periodogram` is uniform in
  frequency, but seasonality lives in *period* on a log axis, where `p = 1/f` makes a single
  frequency bin span `≈ p²/N` in period. The spectrum is therefore hugely over-resolved at
  short periods (all the noise clutter) and under-resolved at long ones (for daily Catskills,
  N≈1662, the annual peak at 365 falls *between* two bins that map to 415 and 332 — there is
  no bin at 365). Worse, `find_peaks` works on array index, so `distance`/`width` mean
  completely different things at each end of the spectrum. Fix: resample log-power onto a
  **uniform log-frequency grid** and run detection there, where one bin = a fixed frequency
  ratio (so `find_peaks` params finally transfer across the spectrum *and* across cases). The
  single smoothing knob is now the grid resolution `M` (default 300; fewer bins = coarser =
  only major peaks). Aggregation per log-bin **must be `max`, not mean/interpolation**: a mean
  averages a sharp seasonal line (one raw bin among hundreds at the high-frequency end) into
  the noise floor — tested, and it surfaced only broad red-noise humps, never the diurnal
  line. `max` preserves the lines. Empty low-frequency bins (grid finer than the raw spectrum
  there) are filled by interpolation.
- **`find_peaks` on the resampled curve.** Peak detection runs `scipy.signal.find_peaks` on
  the log-resampled curve, with a box below the plot exposing *every* `find_peaks` parameter
  (`height`, `threshold`, `distance`, `prominence`, `width`, `wlen`, `rel_height`,
  `plateau_size`) as text inputs (blank = unset). Editing any of them repaints the detected
  peaks live. `distance` and `prominence` are pre-filled conservatively (`5`, `0.7`) so only
  major peaks appear out of the box. Bad hyperparameters (e.g. `distance < 1`) are caught and
  shown as a note instead of breaking the cell.
- **Three marker colors.** Grey verticals = the fixed physically-motivated `CANDIDATE_MENU`
  for the case frequency; yellow = `find_peaks` peaks on the resampled curve; orange = the
  manually-entered extra candidates. An info callout above the plot explains why the grey
  menu periods are the fixed acceptance set (spectral noise can't mint a meaningless period),
  how to read a yellow peak with/without a grey line beside it, and warns that long-period
  peaks are approximate (see limitation below).

The static peak-suggestion text cell was removed (the interactive plot supersedes it). The
library function `suggest_peak_candidates()` is left untouched — the registry/tests still use
it, and it stays the non-interactive path.

**Dead ends.** Two smoothing approaches were tried and dropped before log-resampling: a
Gaussian `sigma`-in-bins filter and a zero-phase Butterworth low-pass. Both smooth in the
*frequency* domain, so their one parameter (bins / normalised cutoff) is not comparable across
cases whose spectra range ~1.3k→75k bins — no single default transferred, and heavy smoothing
washed sharp seasonal lines into the red-noise decay. Log-resampling fixes the root cause
(non-uniform period resolution) instead of masking it, and makes the resolution knob `M`
scale-consistent by construction.

**Limitation.** Resampling cannot manufacture resolution the periodogram never had: at low
frequency the raw grid is sparse, so a long-period peak is only located to within a few
percent (annual reads as ~360–375). Fine as a discovery aid; the exact period still comes from
the menu / declared value, not the plot. The info callout states this.

**Verified** across all cases: the resample + find_peaks pipeline runs on every train split
(fast even on the 75k-bin `surge_cuxhaven` spectrum). At the default `M=300`, `distance=5`,
`prominence=0.7`, the menu periods surface cleanly where they exist — `demand_ltown` → 12, 24,
168; `discharge_saugeen` / `discharge_flashy_catskills` → ~370 (annual); `waves_oregon` → 17
(its known swell period) — with per-case counts in the ~10–20 range.

**Going forward.** This is curation tooling only — the interactive peaks are a discovery/
inspection aid. Nothing enters a case's declared `seasonalities` without going through
`detect_candidates`' ACF + power-ratio evidence and a written justification, exactly as before.

## 2026-07-03 — Periodogram-peak suggestions for off-menu seasonalities

The seasonality curation notebook (`notebooks/seasonality.py`) could only evidence periods a
human already thought to ask about: the fixed `CANDIDATE_MENU` per frequency, plus whatever the
reviewer typed into the extra-candidates box. That menu is deliberately narrow (calendar-driven
cycles), so a case with a genuinely different physical driver — e.g. `waves_oregon`'s swell,
which periodograms as a peak at 17 h, distinct from its declared 12 h tidal cycle — had no way
to surface unless someone happened to already suspect 17 h and type it in.

**Decision.** Added `suggest_peak_candidates()` to `data/seasonality.py`: it finds local maxima
in the train-split periodogram, applies the same prominence bar as `detect_candidates`
(spectral ratio ≥ 10×, ≥ 2 full cycles), and drops anything within 10% of a menu/excluded
period as spectral leakage rather than an independent cycle. It deliberately skips the ACF
check and returns up to 5 ranked suggestions — advisory only, not run through full acceptance.
The notebook now displays these suggestions next to the evidence table; the reviewer copies a
promising one into the extra-candidates box to get its full ACF/power-ratio evidence before
deciding whether to commit it. This keeps the single-source-of-truth model intact (`seasonalities`
in case.toml is still only ever set from full accepted evidence) while removing the requirement
that a human pre-guess every candidate period.

**Verified against `waves_oregon`**: the fixed menu evidence (12 accepted, 24/168 rejected)
matches its already-committed `[12]`. Peak suggestions on the same train split return
`[17, 57, 38, 14, 50]` — 17 was the case that prompted this, and 11/13 (leakage of the on-menu
12 h cycle) are correctly excluded rather than double-counted.

**Going forward.** Suggestions are a discovery aid, not an acceptance path — nothing gets
added to a case's declared `seasonalities` without running through `detect_candidates`'
ACF + power-ratio thresholds and a written justification per the existing curation process.

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

## 2026-07-03 — Ensemble-of-baselines columns (issue #59)

Closes the "no ensemble-of-baselines column" gap from the 2026-07-02 method-suite audit. A
cheap combination of the naive baselines is often an embarrassingly strong reference; without
it the matrix had no "hard to beat cheaply" column to calibrate how much the sophisticated
methods actually buy.

**What was added.** A generic `EnsembleForecaster` (`bench/methods/ensemble.py`) that
composes existing `Forecaster` members — each already shares the fit/predict contract and
emits the same future index — and combines their per-step forecasts. Two factory-built
columns are registered in `run_pilot.py`, placed next to the naive baselines they draw from:

- **EnsembleBaselines** — median of the four naive baselines (Mean, Persistence,
  SeasonalNaive, Climatology).
- **EnsembleBaselinesTheta** — the same four plus the fast classical Theta.

**What was decided and why.** *Median, not mean*, as the combination rule: it is robust to a
single member being badly off (e.g. Persistence during a spike) and is NaN-safe via
`np.nanmedian`. *Two columns rather than one* so the matrix shows both the pure zero-cost
reference and the naive+classical blend — the issue explicitly floated "baselines plus a
classical method" as a variant, and the two bracket a useful range. A member that raises at
predict time is dropped for that origin (the ensemble still forecasts from the survivors);
only if all members fail does it raise, so the eval loop records it like any other failure.
Placed in a new module rather than `naive.py` to keep that file dependency-light — the
Theta blend imports from `classical.py`.

**What it means going forward.** Early smoke run on `discharge_lowland_denmark` (synthetic,
single hold-out) already shows the intended calibration signal: EnsembleBaselinesTheta beat
persistence (skill ≈ 0.35) while the naive-only ensemble trailed it (≈ −1.6) on that flashy
series — the naive median pulls toward the mean/climatology where persistence is strong. The
column is doing its job; full rolling-origin numbers will come from the next pilot run.

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
