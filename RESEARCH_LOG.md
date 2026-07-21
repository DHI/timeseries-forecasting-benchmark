# Research Log

A running record of the testing and decision-making behind this benchmark — what was tried,
what was decided, and why. Intended to be readable by collaborators without needing to dig
through commit history or notebooks.

Entries are in reverse-chronological order (newest first).

<!-- Add new entries below this line, newest first. -->

## 2026-07-21 — Fixed the matrix heatmap going stale when leaving NSE/KGE for another metric

Selecting NSE or KGE, then switching to a `Skill_vs_*` metric, left the heatmap showing the old
NSE/KGE cells instead of updating. Same root cause as the training-time entry just below: Plotly.js
fails to repaint a heatmap via `react()` once the matrix is down to 3 rows or fewer, and NSE/KGE
always narrow the matrix to exactly that many rows (the flow/level cases that declare them).
Rather than another disabled-control workaround, `MatrixGraph` now gets a `key` in `App.tsx` that
changes whenever the matrix is in that danger zone — tied to the metric while short, a fixed value
otherwise — forcing React to unmount and remount the component instead of asking Plotly to update
in place. This only remounts on a metric change that enters, leaves, or moves within the short-row
zone; switching between two ordinary (4+ row) metrics is untouched.

## 2026-07-21 — Training-time overlay disabled for short matrices (Plotly bug, no in-code fix found)

Holding "show training time" doesn't repaint the matrix for a short enough matrix: the
colorbar switches to the training-time scale, but the cell colors and numbers stay on the
score view. Traced this to a genuine Plotly.js rendering bug, not an app bug or an NSE/KGE
issue specifically — it reproduces with any metric once the matrix narrows to 3 rows
(confirmed via domain-checkbox narrowing alone, and with NSE's real values relabeled as a
non-restricted metric), and 4 rows renders fine, so row count is the real trigger, not
metric identity or the domain-restriction code path. Every attempted fix — `redraw()`, a
full remount, a second `react()` call, a from-scratch `Plotly.newPlot()`, a two-Plot-instance
architecture with lazy mounting — either did nothing or turned the failure from
stale-but-visible into a blank render, which is worse. Rather than keep chasing an in-code
fix, `TRAINING_TIME_MIN_ROWS` (`web/src/data/constants.ts`) now gates the control itself:
below 4 matrix rows, "Show training time" is disabled with an explanatory message pointing
users toward a broader domain/metric selection. Worth revisiting if the pinned
`plotly.js-cartesian-dist-min` version ever changes.

## 2026-07-21 — Domain controls now follow the matrix's own NSE/KGE narrowing

The matrix already narrowed its rows to only the cases eligible for NSE/KGE (flow/level domains
that declare them in `case.toml`), but two other parts of the sidebar didn't follow suit: the
domain checkboxes stayed checkable for domains with no eligible case, and an open case panel
stayed open even after its case's row disappeared from the matrix. Both now track the same
eligibility check — ineligible domain checkboxes grey out and uncheck (without erasing the user's
underlying selection, so switching back to an unrestricted metric restores it), and the case panel
closes if its case stops being eligible when the metric changes. While fixing this, also found
that the web app's test setup was missing Testing Library's per-test cleanup (`globals: false` in
`vitest.config.ts` means its auto-cleanup never registers), which only mattered once a second test
in `App.test.tsx` needed a full successful render — added an explicit `afterEach(cleanup)` so this
doesn't bite the next multi-render test either.

## 2026-07-21 — Added a seasonal-pattern figure to the case panel

The case panel's Data overview section now shows one subplot per declared seasonal period,
scattering the training tail against its position in the cycle with a smooth curve over the
pattern. The curve is a LOESS fit rather than a single-harmonic sinusoid, so it can show
non-sinusoidal shapes a plain sine wave would flatten out, like a demand series with separate
morning and evening peaks in one day. Since a cycle wraps around (the last position is adjacent
to the first), points are mirrored across each boundary before fitting so the curve doesn't taper
or jump at the seam. The training tail is also detrended (mean + linear trend removed) before
plotting, reusing the same `fill_and_detrend` convention `dominant_period` already uses for
detection — without it, two cycles with the same shape but sitting at different points along a
case's long-run drift would smear into a misleadingly noisy scatter; the y-axis now reads as
deviation from trend rather than the raw target value, so the caption calls this out explicitly.
Everything is computed once in Python and shipped as plain arrays in `cases.json`, matching how
the rest of the panel's plots already work — the LOESS bandwidth and padding fraction are new
tunable constants in `webdata.py` (`SEASONAL_PATTERN_LOESS_FRAC`, `SEASONAL_PATTERN_PAD_FRAC`)
worth adjusting if the rendered curves look under- or over-smoothed once reviewed.

## 2026-07-21 — Moved app instructions from a root markdown file into the app itself

`APP_INSTRUCTIONS.md` used to sit at the repo root, disconnected from the live site — nobody
browsing the app would think to look for it there. Its content now lives in
`web/src/data/appInstructions.ts` and shows up behind a grey info icon next to the page title,
in a scrollable modal styled like the app's other panels. The root file is gone; anyone updating
the instructions should edit the new module instead.

## 2026-07-21 — Removed the legacy Dash app now that `web/` covers everything it did

Deleted `app/` (the original Dash results-matrix viewer), now that `web/` has replicated all of
its functionality. `scripts/export_web_data.py`, which generates `web/`'s static JSON, had been
importing its data-loading and forecast-curation code straight out of `app/data.py` and
`app/snapshot.py`; that code moved to a new `src/tsforecastbench/webdata.py` module so the export
script keeps one shared source of truth without depending on deleted files. The Plotly/Dash
rendering code in the old app had no other callers and was dropped along with it. The `dash`
dependency (and the now-unused `app` extra in `pyproject.toml`) is gone; `plotly` stays, since
`notebooks` still needs it. Tagged the last commit with `app/` intact as `legacy-dash-app` in case
anyone wants to run the old version.

## 2026-07-20 — Fixed heatmap bleeding over the sidebar when the case panel is dragged wide

Dragging the right-hand case detail panel to its maximum width could make the heatmap paint over
the left control sidebar instead of shrinking to fit. The heatmap has a fixed pixel width based on
the number of methods shown, and the center column that holds it had no `overflow` set, so once
that column got squeezed narrower than the heatmap, the excess width spilled left and covered the
sidebar (later elements in the page paint over earlier ones by default). Added `overflow-x: auto`
to the center column so the heatmap now scrolls within its own column instead of bleeding over
neighboring panels.

## 2026-07-20 — Sidebar closed by default, added a hint for first-time visitors

The control sidebar used to open on load and its collapse/expand button was only half
visible once collapsed — it was anchored to the sidebar's own width, which shrinks to
zero when collapsed, so the button's centering offset pushed half of it past the left
edge of the page. The sidebar now starts collapsed, and the button's position is
computed directly from the sidebar's open/closed state instead of relying on that width.
Also added a floating "Click a cell to inspect a case" prompt with an arrow pointing at
the heatmap, shown in the empty space to its right until a cell is selected, since that
space was otherwise blank with no cue that clicking does anything. For the same reason,
added a second prompt, "Reveal controls panel," pointing at the collapse/expand button
itself, since a collapsed sidebar with no explanation looked like missing UI rather than
a closed panel. Both prompts disappear once their target state changes (a case gets
selected, or the sidebar opens).

## 2026-07-20 — Moved training time from the right panel to a press-and-hold heatmap overlay

Training time used to show as one text line per selected method in the right-side case panel,
so comparing training cost across the benchmark meant clicking into each case one at a time.
Added a "Training time" button to the sidebar that, while held down, swaps every heatmap cell
from its metric score to that case/method's measured training time, then reverts the instant the
mouse is released. The right panel keeps its unrelated sample-count line but drops the old
per-method time text. Fastest-method bolding is skipped during the overlay since it's meant as a
quick glance, not a ranking.

Cell values show as whole seconds rather than two decimals. The overlay's colorbar title was
also shrinking the heatmap's plot area relative to the normal metric view — Plotly lets a wide
colorbar title grow past a fixed margin by default, so a short title plus an explicit
`margin.autoexpand: false` now guarantee the grid occupies identical pixels either way. Training
time is heavy-tailed (most methods are fast, a few much slower), so cell color now maps to
`log10(seconds + 1)` instead of raw seconds — a linear scale left almost every cell looking the
same shade; the colorbar ticks and cell text still label real seconds.

## 2026-07-19 — Fixed leap-year bug in the 365-day position-in-cycle path

`_dayofyear_365` (used by `position_in_cycle` for the annual cycle, which backs Climatology,
SeasonalNaive-via-ensemble, and any ML feature keyed on a 365-day period) called
`idx.is_leap_year.to_numpy(...)`, but `DatetimeIndex.is_leap_year` already returns a plain numpy
array rather than a pandas object, so the `.to_numpy()` call raised `AttributeError` on every
call. This broke all baseline/ensemble tests on cases with a declared 365-day seasonality plus
the dedicated leap-year boundary test.

**Decision.** Wrap the result in `np.asarray(...)` instead of calling `.to_numpy()`. No behavior
change beyond fixing the crash — full suite passes again.

## 2026-07-19 — Interactive decision graph for the research log

This log had grown to enough entries that cross-references only existing as prose made related
decisions hard to trace without reading the whole file. Added `docs/research_graph.html`, a
self-contained D3 force-directed graph — one node per log entry, hover for a summary, click for
full detail plus a list of connections, drag to pin nodes, pan/zoom. Node and edge data (relation
types: supersedes/fixes-bug-in/depends-on/relates-to) were hand-extracted from the log's own
cross-reference prose in one pass and embedded directly in the file, so it opens with no server
or fetch; CLAUDE.md's research-log section now asks for a matching node whenever a new entry is
added. Categories are capped at four (methodology, app & tooling, bug fix, project & data) rather
than a finer split, since a force layout can place any two categories next to each other and the
project's palette validator fails a 6-color categorical set under that all-pairs condition.

## 2026-07-19 — Combined the EDA and React Pages workflows into one

GitHub Pages only lets one workflow deploy per repo, so the existing EDA-only
`pages.yml` couldn't run alongside a separate workflow for the new React results-matrix
app in `web/` without one clobbering the other. Merged them into a single workflow that
builds both sites and copies the React build to the site root with the EDA browser under
`/eda/`. Pushing the change surfaced a separate problem: the git credential in use came
from the `gh` CLI's login, which lacked the `workflow` scope GitHub requires for editing
files under `.github/workflows/`, so `gh auth refresh -h github.com -s workflow` was
needed before the push could go through.

## 2026-07-19 — Replaced marker descriptions with a visual swatch

The "Global methods" and "Exogenous inputs" sidebar sections described their corner-triangle
markers in words ("marked with a corner triangle", "marks it with a teal corner"). Added a
`MarkerSwatch` component that draws a small square with the actual colored triangle in the
matching corner, reusing the same `cornerTrianglePath` geometry the matrix overlay uses, so the
preview matches the real marker exactly. The prose was trimmed to drop the now-redundant
shape/color description, keeping only the part explaining when each marker appears.

## 2026-07-19 — Moved the method-info card title to the top when expanded

The method-info cards showed their method name in a title bar that stayed at the bottom of the
card in both states, so an expanded card read top-to-bottom with the resize handle first and the
name last, below the content. Reworked `MethodInfoCard` to render the same title bar first when
expanded, so the method name sits at the top of the open card, with the resize handle and content
below it. Collapsed cards are unaffected — the title bar is still the only thing rendered, in the
same position as before.

## 2026-07-16 — Fixed method-info card height-drag stopping after one pixel

Dragging a method-info card's resize handle up barely moved it. Root cause: `useEdgeResize`'s
`handleMouseMove` closed over the `onResize` callback, so its identity — and the identity of the
`stopDragging`/cleanup-effect chain built on top of it — changed whenever `onResize` did.
`MethodInfoCard` passes an inline callback, recreated on every render, so the first `onResize`
call during a drag triggered a re-render whose new callback identity tore down the drag's own
mouse listeners via React's effect-cleanup-on-dependency-change. The case panel's width-resize
never hit this because its callback happens to come from a stable `useCallback`. Fixed by routing
`onResize` through a ref inside the hook, so `handleMouseMove`'s identity no longer depends on the
caller's callback identity; added a regression test that simulates a mid-drag re-render with a new
callback to catch a recurrence.

## 2026-07-16 — Split the method-info strip into per-method collapsible, resizable cards

The bottom method-info strip (up to `MAX_OVERLAY_METHODS` = 3 cards) used to be one flex-wrap
container where cards grew to fill space, which read as a single blended block rather than
distinct per-method panels. Reworked it into fixed-width (320px) cards docked side by side, each
independently collapsible down to just its title bar and resizable in height (120–520px) by
dragging a handle along its top edge — reusing the `useEdgeResize` hook generalized (from an
x-only `useResizablePanelWidth`) for the case panel's width-resize. Collapse state and height are
local to each card, keyed by method name, so they reset if a method is deselected and reselected
or the case changes, rather than persisting — same call as the case panel's width.

## 2026-07-16 — Made the React case detail panel resizable

The right-hand case detail panel had a fixed 460px width, which cramped the training/test plots
on smaller screens and wasted space on wide ones. Added a drag handle on the panel's left border
so users can resize it, clamped to 320–900px. The width lives in the app-state reducer alongside
the existing `sidebarCollapsed` toggle and resets to the 460px default on reload rather than
persisting via localStorage, matching how the sidebar toggle already behaves — simplest option,
and nothing in the app currently persists UI state across sessions.

## 2026-07-15 — Fixed the React case panel's training/test charts showing timestamps on both axes

The training-data chart in the React app's case panel plotted dates on both x and y instead of
dates on x and values on y (Dash was unaffected). Root cause: `LineSnapshot.tsx`'s shared
`AXIS_STYLE` object was passed to Plotly's `xaxis` by reference rather than copied, and Plotly.js
auto-detects an axis's type from its data and writes that type back onto the layout object it was
given — so the first date-valued x-axis it saw permanently stamped `type: 'date'` onto the shared
`AXIS_STYLE` singleton, which then leaked into every later plot's y-axis (already built as
`{...AXIS_STYLE}`, so it inherited the mutation instead of a clean object). Fixed by always
spreading `AXIS_STYLE` into a fresh object before handing it to Plotly, for both axes; applied the
same defensive copy in `DistributionPlot.tsx`, which has the identical pattern but had stayed
harmless only because its x-data happens to be numeric. Lesson for any future Plotly-based
component here: never pass a shared style object literal straight into a `layout` prop — Plotly.js
treats it as mutable state, not a read-only style constant.

## 2026-07-15 — Closed two React/Dash parity gaps: method-info panel placement, missing benchmark-design text

Found two spots where the React port, despite being logged as feature-complete, didn't actually
match `app/app.py`. First, the method-info cards (family, description, exogenous usage,
implementation detail per selected method) were rendered inline at the bottom of the snapshot
panel, inside the right-hand detail column — unlike Dash, where they sit in their own strip docked
to the bottom of the viewport, spanning the width of the matrix column. Moved that block into a new
`MethodInfoPanel` component, rendered as the last child of the matrix column with `position:
sticky; bottom: 0`, matching the Dash layout; `SnapshotPanel` no longer needs the method registry.
Second, the static "Benchmark Design" writeup below the matrix (splits, curation, model
parameters, case description, exogenous inputs) was missing from React entirely — ported verbatim
into a new `BenchmarkDesign` component. Both are a reminder that "closes out the migration" in the
earlier entry meant functional parity for the interactive panels, not a line-by-line diff against
`app.py`'s layout.

## 2026-07-15 — Started a React port of the results-matrix app for GitHub Pages

`app/app.py` only runs as a live Dash server, so it can't be hosted on GitHub Pages. Decided to
port it to React + TypeScript in a new `web/` directory, built as static files and deployed via
GitHub Actions, with `app/app.py` staying as-is. Key design call: rather than reimplementing
`app/data.py`'s statistics client-side (KDE curves, per-origin MASE/skill-score, method-family
introspection), a new Python export script will precompute everything into static JSON — the React
app is a pure presentation layer, same as `docs/figures/benchmark/matrix_*.png` is already a
precomputed artifact of the same data. This also means the forecast-snapshot panel, whose source
parquet (`results/pilot_snapshots.parquet`) isn't committed today, will ship as a new bounded and
committed JSON artifact. Since GitHub Pages only serves one site per repo and the existing
`pages.yml` already deploys a separate marimo EDA browser there, the React app will take the root
Pages URL and the EDA site moves to `/eda/` under one combined deploy workflow — anyone relying on
the EDA site's current root URL should expect it to move. Scaffolding (Vite+React+TS, oxlint,
Prettier, Vitest) landed first, then `scripts/export_web_data.py` (importing `app/data.py`
directly rather than reimplementing its statistics) and a working static matrix: the sidebar's
metric/domain/family filters and the case x method heatmap render from the exported JSON with the
same colours, cell text, best-cell bolding, and domain/family group boundaries as `render_matrix`.
Next, the hover tooltip (same pixel-geometry crosshair, now driven by onHover instead of a Dash
clientside callback), click-to-select, and the global-history/exogenous toggles all landed, along
with their corner-triangle overlays and selection outlines. Along the way, testing the toggle swap
logic surfaced a real bug in the first matrix pass: a case/method with both a local and an
extended-history row (5 of the 12 cases have one) silently picked whichever row came last in array
order instead of always defaulting to the local row — fixed by explicitly filtering to the local
row before pivoting, then overlaying the extended value only when the toggle is on. The case detail
panel landed next: `cases.json` now carries each case's training tail, full test split, split
counts, and train-vs-test KDE curves (local and extended-history variants), and the React panel
renders them plus the case-summary/training-time markdown, matching `render_case_panel`. One
side effect worth knowing: `surge_cuxhaven`'s test split alone is ~32k points, pushing `cases.json`
to ~5MB — still fine for a static site (gzip compresses numeric JSON well, and the Dash app already
pays this same "no decimation" cost when rendering it server-side), but worth revisiting if more
cases at that scale get added. The forecast-snapshot panel landed last: a new
`snapshots/<case_id>.json` per case (2.8MB total) carries every rolling origin's actual-series
window and every method's prediction, plus curation (best/median/worst by whatever metric/method
is anchored) precomputed via `snapshot.curate_origins` so the browser never re-derives MASE/
skill-score. Methods with a real extended-history fit or a "+exo" sibling get both variants, and
the client resolves which one to show the same way the matrix does (`showGlobal`/`showExogenous`
toggles). This closes the last functional gap from `app/app.py` — matrix, case detail, and
forecast snapshots are all ported. Last, `.github/workflows/pages.yml` now builds and deploys both
static sites in one run — the React app into the repo's root Pages URL, the existing marimo EDA
browser into `/eda/` — since `actions/deploy-pages` only publishes one site per repo; verified
locally by assembling the combined output and serving it under the real base path before trusting
the workflow. This closes out the migration: `web/` now covers everything `app/app.py` does, deploys
as static files, and `app/app.py` itself is untouched for anyone who still wants the live version.

## 2026-07-15 — Gave Chronos-2, TimesFM 2.5 and Moirai-2.0 `+exo` variants

Closed the covariate follow-up flagged in the 2026-07-14 FM entry: each of the three
foundation models that natively supports covariates (Chronos-2 via `predict_df`'s
`future_df`, TimesFM via `forecast_with_covariates`, Moirai via `feat_dynamic_real`) now
has a `+exo` sibling, mirroring the LightGBM/CatBoost/LinearLags `+exo` pattern — future
covariate values are the eval loop's perfect-forecast proxy, not a real forecast of those
covariates. Chronos-Bolt stays univariate-only; its architecture has no covariate
mechanism to expose. One real gotcha: TimesFM's `forecast_with_covariates` requires
`return_backcast=True` at compile time, but that flag silently changes what plain
`forecast()` returns (backcast prepended ahead of the horizon rather than just the
horizon) — so the covariate path now compiles and caches a second model instance,
keeping the univariate path's compiled model and output shape untouched. All three
variants were smoke-tested end-to-end against real cached model weights (not just
synthetic placeholders) on a covariate-bearing case and a covariate-free one before
committing.

## 2026-07-14 — Fixed matrix cell clicks crashing with `KeyError: 'customdata'`

Every click on a matrix cell in `app/app.py` raised `KeyError: 'customdata'`. Root cause: Dash's
`clickData` only reconstructs a trace's `customdata` for points indexed by a flat
`pointNumber`/`pointNumbers` (how 1D traces like scatter work) — a Heatmap addresses cells by
`x`/`y` instead, so Dash silently omits `customdata` from every heatmap click regardless of what
the figure sets it to. The exo-toggle's effective-method lookup (added in the entry below) had
relied on reading it from `customdata[5]`. Fixed by resolving the effective method server-side
from the clicked `x` (base method) plus the `show-exogenous` toggle state and the case's
`case_exo_tier`, the same swap condition `render_matrix` already uses — no more reliance on
`customdata` for heatmap clicks. Any future click-driven feature on this matrix should read
`x`/`y` off the click point, not `customdata`.

## 2026-07-14 — Added three zero-shot foundation models (Chronos-2, TimesFM 2.5, Moirai-2.0)

Chronos-Bolt was the only foundation model implemented of the six the plan called for; added
Chronos-2, TimesFM 2.5 and Moirai-2.0 alongside it, each as a univariate zero-shot wrapper
behind the common `Forecaster` interface and added to `run_pilot --with-fm`. They're
univariate for now (matching the existing Chronos-Bolt precedent) — the models' covariate
modes are a possible follow-up rather than part of this change. All three install through a
single `fm` extra (`uv sync --extra fm`), so one command sets up everything `--with-fm`
scores. Moirai's `uni2ts` is the binding dependency: it hard-pins scipy~=1.11.3 / numpy~=1.26
/ torch<2.5, none of which have Python-3.13 wheels, so the whole project is now capped at
**Python 3.11–3.12** (`requires-python = ">=3.11,<3.13"`) and its base stack drops to
numpy 1.26 / scipy 1.11 / torch 2.4 to keep every model installable in one environment.
Chronos and TimesFM both fit comfortably inside those bounds. TimeMoE, TimeGPT and Lag-Llama
were dropped from the near-term plan (TimeGPT is API-only; all three have been overtaken in
recent comparisons). No benchmark numbers yet — scoring these needs a `--with-fm` pilot run.

## 2026-07-14 — Folded +exo method variants into an app-level toggle; retired +calendar

Replaced the separate `LightGBM+exo`/`CatBoost+exo`/`LinearLags+exo` matrix columns with one
column per algorithm plus a sidebar checkbox that swaps in the exogenous-aware score (marked
with a teal corner) for cases with real exogenous data — the same swap-and-mark mechanism the
app already used for the extended-history toggle. This also settled an open question about
calendar/seasonal features: unlike genuine exogenous covariates, they're deterministic and free
for any future timestamp, so they no longer sit behind a separate `+calendar` variant (only
LightGBM had one) and are now always included for every windowed-ML method. While wiring this
up, found that `run_case_rolling` — the default pilot path — never emitted the
`case_exo_tier`/`uses_exogenous` fields the toggle depends on (only the `--fast` path did), so
the committed `docs/results/scores.parquet` doesn't have them yet: the pilot needs a re-run
before the checkbox works, and the method list is three columns shorter as a result.

## 2026-07-14 — Committed distilled scores to docs/ for onboarding (issue #19)

`run_pilot.py` now writes its distilled metrics to `docs/results/scores.parquet` and the
three main matrix PNGs to `docs/figures/benchmark/`, in addition to the full working output
in gitignored `results/`. The results-matrix app (`app/data.py`) now reads from that
committed file too, so a fresh clone can open the app and see real scores without running
the pilot first; per-origin snapshots (the click-through forecast detail) still need a
local pilot run, since those stay in `results/` and aren't committed. While wiring this up,
found that `.gitignore`'s `results/` rule was unanchored and silently swallowed
`docs/results/` as well — fixed to `/results/` so only the top-level directory is ignored.
`docs/results/scores.parquet` and the three matrix figures are now populated from the most
recent pilot run and should be regenerated (`uv run python scripts/run_pilot.py --with-fm`)
and committed after future runs.

## 2026-07-13 — Added MAPE, sMAPE, NSE, and KGE (issue #34)

Both eval paths (single hold-out and rolling-origin) now also compute MAPE, sMAPE, NSE, and
KGE for every case and method, alongside the existing five metrics. MAPE/sMAPE are
dimensionless like MASE, so they're a plain addition to the matrix's metric dropdown. NSE
and KGE are hydrology-standard but only meaningful for flow/level cases, so each case now
declares eligibility in `case.toml` via a new `domain_metrics` field; the app still lists
NSE/KGE in the same dropdown as everything else, but narrows the matrix rows to only the
cases that declare them, rather than adding a separate sidebar UI. Eligibility is set for
the 3 discharge cases (confirmed by Rocco's review) plus `sewerlevel_bellinge` and
`wwtp_influent_n2o_350k`, flagged in their `case.toml` notes as an unconfirmed extension.
`results/pilot_metrics.parquet` needs a full re-run before the new metrics show real
values in the app.

## 2026-07-13 — Windowed ML/DL methods are now fit once per case, not refit at every rolling origin

The rolling-origin eval used to refit every method at each of `n_origins` cutoffs, so a
learned method (LightGBM, CatBoost, MLP, LinearLags, DLinear, and the optional
torch/neuralforecast models) paid a full retrain per origin and its cost grew with
`--origins`. Those methods keep a trained model separate from its input, so the repeat
training was never needed to stay leak-free — the train/val/test boundary already does
that. They are now fit once, on data strictly before the test period, and scored across
every origin by inference: each origin's forecast conditions on the observed history up to
that cutoff without re-estimating parameters. The existing `is_global` flag already draws
this line, so no new flag was needed — global methods (the windowed ML/DL models plus
Chronos) keep a trained model apart from their input and are fit once, while local methods
(AutoARIMA, AutoETS, Theta, TBATS, MSTL, the naive baselines) keep refitting, since for
them the fit *is* the forecast and statsforecast cannot forecast from a later origin
without one. Raising `--origins` is now cheap, but this shifts the windowed ML/DL scores
(they no
longer see test-period data in training), so earlier pilot numbers for those methods are
stale and need a re-run; baseline, classical, and Chronos scores are unchanged. The torch
and neuralforecast paths were not exercised locally (neither package is installed here) —
verify them on a WSL run before trusting those columns.

## 2026-07-13 — Snapshot panel now shows curated best/median/worst origins, ranked by whichever metric the matrix is showing

The snapshot panel used to show every rolling-origin forecast stored in
`pilot_snapshots.parquet`, in whatever order the eval pipeline happened to produce them.
It now picks up to three: the best, median, and worst performing origin for the
first-clicked method, so a viewer sees a representative spread instead of an arbitrary
sequence. Ranking follows whichever metric is selected on the matrix (MAE, RMSE, MASE,
or one of the two skill scores), rather than a fixed MAE, so switching the matrix's
metric dropdown also reorders the snapshots. No per-origin error existed anywhere in the
eval pipeline before this, so the app recomputes each metric itself from the stored
predictions and the case's actual series — replicating `rolling.py`'s own per-origin
math (MASE against the case-level train+val naive scale, skill scores against a
persistence/mean reference built from history strictly before the origin, everything
scored against observed-only actuals rather than the gap-filled series used for
plotting). Spot-checked against the matrix's pooled scores for one case/method: every
per-origin value falls in the expected range around the pooled figure, and a method
that predicts the historical mean scores exactly 0 `Skill_vs_Mean` at every origin, as
it should. When several methods are overlaid, they all plot on the same three origins
(fixed by the first-clicked method) so the comparison stays apples-to-apples. The
current pilot data has only 3 origins per case, so today this mostly relabels the
existing panels rather than filtering anything out — the value grows if a future run
stores more origins per case (`TSFB_ORIGINS`).


## 2026-07-13 — Method-info panel now shows library, compute cost, limitations, and a code snippet

Grew the method-info panel beyond family/description/exogenous-inputs to answer the question a
non-ML user actually needs answered when comparing methods: what library does the work, how
expensive is it to run, when should you not trust it, and what does the pipeline actually look
like in code. The new fields live as plain class attributes directly on each `Forecaster`
subclass (`library`, `compute_profile`, `limitations`, `code_snippet` in `base.py`), the same way
`family` already does, rather than in a separate metadata file — so a method's description of
itself can't drift out of sync with its implementation. Several method families share a base
class that holds the real pipeline (e.g. the windowed-ML lag-feature pipeline, the neuralforecast
wrapper), so the shared fields — especially the code snippet — are set once on the base and only
`library` gets overridden per leaf, to avoid repeating the same snippet 4-8 times. The detail is
collapsed by default behind an "Implementation detail" toggle on each card, so the panel's default
view is unchanged; only opening it reveals the new fields. Couldn't get a headless-browser
screenshot of the panel in this environment (no root to install Chromium's system libraries), so
verification here relies on checking every registered method has all four fields populated and on
inspecting the rendered markdown directly — worth a manual look in a real browser before trusting
the layout at a glance.


## 2026-07-10 — Added a bottom method-info panel, capped the method overlay at 3

Clicking a cell now also opens a panel docked to the bottom of the screen with a brief
description of that method and which of the case's exogenous inputs (if any) it used —
placeholder content while we figure out what's actually worth showing there. The
description comes straight from each Forecaster subclass's own docstring, so it's free as
methods gain or lose docstrings rather than a second copy to keep in sync; methods with no
docstring anywhere in their class hierarchy just show a "no description yet" placeholder.
Since both the snapshot graph and this new panel lay per-method content out side by side,
the same-row method overlay is now capped at 3: clicking a 4th method evicts the oldest
instead of piling on indefinitely.


## 2026-07-10 — Merged the case-info and snapshot panels into one collapsible column

The results viewer used to open two separate side-by-side panels on a cell click: case
info on the left, forecast snapshots on the right. Merged them into a single right-hand
column, case info on top and snapshots below, matching the natural reading order. The
case-info block (the four graphs plus the case summary text) is now wrapped in a native
HTML `<details>`/`<summary>` element so it can be collapsed to make more room for the
snapshots, defaulting to open so nothing changes for someone who never touches it.


## 2026-07-10 — Moved the snapshot panel into a vertical column next to case info

The forecast-snapshot panel used to sit below the matrix, full width, with a
"case review notes" column next to the forecast graph. Moved it into a sticky
column of its own between the matrix and the existing case-info panel, and
dropped the notes column — that content already lives in the case-info panel,
so it was redundant. Since a narrow column can't fit the old three-wide grid of
forecast-origin snapshots, `render_snapshots` now takes a `max_cols` argument
and the app passes `max_cols=1` so origins stack one per row. Selecting
methods, the chip row, and the clear button all work the same as before.

## 2026-07-10 — Results viewer can show the extended-history fit for gappy cases

The interactive viewer (`app/app.py`) previously pivoted on every row in
`pilot_metrics.parquet` without splitting out `used_extended_history`, so once a case had
both a local and an extended-history row for an `is_global` method (see the entry above)
the matrix pivot would hit duplicate `(case_id, method)` pairs. Fixed the viewer to plot
the local fit by default, and added a sidebar checkbox that swaps in the extended fit for
`is_global` methods on the five cases with data beyond the curated span, marking those
cells with a thin hotpink diagonal hash so they read as not directly comparable to the
rest of the row.

The checkbox originally only changed the matrix's metric, not the forecast-snapshot
panel, because `run_case_rolling` computed the extended fit's predictions but never saved
them — snapshots came from the local fit only. Fixed by tagging every snapshot row with
`used_extended_history` too, and having the snapshot panel pick a method's extended fit
when the toggle is on and one exists for the case (falling back to the local fit
otherwise), labelling the swapped trace "(global fit)" in the legend. This needs a rerun
of `scripts/run_pilot.py` to regenerate `pilot_snapshots.parquet` with the new column —
older snapshot artifacts won't have it. Checked end-to-end on `discharge_lowland_denmark`:
toggling swaps exactly the ten `is_global` methods' matrix values and their snapshot
forecasts, everything else unchanged.

## 2026-07-10 — Made the extended-history fit live via per-segment training (supersedes the "inert" entry below)

The extended-history path added earlier today was inert because it refused any extended
frame containing a gap, and a gap is exactly where the extra history lives. Replaced that
gap-free-or-fallback rule with segment-aware training: `_segment_extended` (in
`bench/eval/loop.py`) splits the extended frame at its big gaps into contiguous segments,
each carrying its own `unique_id`, and the `is_global` methods now build windows per
segment so no training sample ever straddles a gap (`segment_arrays` in `methods/base.py`;
mlforecast/neuralforecast get this for free, the manual DLinear/torch builders were
changed to loop over segments). The tail segment equals the shared curated span, so the
local and extended fits are scored against identical persistence/mean/seasonal-naive
references and the same held-out window — the extended row isolates the gain from more
training data, nothing else. Each `is_global`+`pools_segments` method now emits two result
rows per gappy case (`used_extended_history` False/True); the main matrix shows the local
fit, the `_global` matrix the extended fit. A quick 3-origin check on
`discharge_lowland_denmark` already shows the gain is real (LinearLags MASE 0.24→0.20,
DLinear 0.24→0.20). Five of the twelve cases have history beyond the curated span
(`discharge_lowland_denmark`, `sewerlevel_bellinge`, `waves_massachusetts`, `waves_oregon`,
`wwtp_influent_n2o_350k`); the other seven are gap-free and produce the local row only.
Zero-shot Chronos and the fixed-`n_series` iTransformer keep `pools_segments=False` (they
can't absorb extra segments), and the unused cross-series `fit_pooled` seam was removed.
`results/pilot_metrics.parquet` and the matrix PNGs are still stale and must be regenerated
with `uv run python scripts/run_pilot.py` (add `--with-dl --with-nf --with-fm` for full
method coverage) before the numbers reflect this.

## 2026-07-10 — Added an is_global capability flag and an extended-history training path (currently inert)

Added `Forecaster.is_global` (`base.py`) marking methods whose architecture could use more
data than the shared curated span — on for the ml/deep_learning/foundation_model families,
off for baseline/classical/ensemble. Alongside it, curation now anchors the held-out tail
segment to the record's most recent observation instead of picking the longest gap-bounded
segment anywhere in the series — the old rule could put the test window in the middle of a
case's history with no room to extend backward, and it changed the curated span for
`discharge_lowland_denmark`, `sewerlevel_bellinge`, `waves_massachusetts`, and
`wwtp_influent_n2o_350k` (their previous "longest" segment wasn't the tail one). A new
`min_tail_length` floor makes curation fail loudly instead of silently accepting a span too
short to use. `Case.load_extended()` and a new `data/curated_extended/` cache give
`is_global` methods the full single-series history back to the tail segment's `val_end`,
wired into both eval paths (`eval.loop.run_case`, and the rolling-origin path
`run_pilot.py` actually uses by default).

The finding that matters most: the tail segment is always bounded by the nearest preceding
gap, so any case where extended history would be meaningfully longer necessarily crosses
that same gap. Checked across all 12 cases: every one either has no gap at all (extended
history comes out identical to what curation already used) or falls back to the shared
curated data, because bridging a gap bigger than the per-case cap would mean either
fabricating a value across it or letting a windowed model treat the rows on either side as
adjacent — no method's `fit` handles gaps today. So right now `is_global` routing changes
nothing measurable; each result row's new `used_extended_history` column reads `False`
everywhere. That's deliberate, not a bug — it's the honest baseline for the follow-up that
would actually deliver value: gap-aware or cross-series-pooled training, which is why
`Forecaster.fit_pooled` exists already as an unimplemented hook. Pooling across series won't
fix the gap problem on its own; it needs the same boundary-aware sample construction the gap
problem needs, generalized to series boundaries too, so building that properly for pooling
should close both at once.

Nothing has been re-run yet: `results/pilot_metrics.parquet` and the matrix PNGs still
reflect the old curation and don't have the `is_global`/`used_extended_history` columns.
Curation itself is already regenerated and committed; `uv run python scripts/run_pilot.py`
will refresh the results (add `--with-dl --with-nf --with-fm` for full method coverage, or
`--fast` for a quick single hold-out check) and produce the new `pilot_matrix_*_global.png`
files alongside the existing ones.

## 2026-07-10 — Closed out EDA figure-quality review comments (issue #36)

Worked through Rocco's PR #2 figure comments (B5/B6/C3 + the WWTP figures): missing
axis units, missing captions, and a stale/spiky Bellinge panel. Two `docs/figures/cases/*_eda.png`
had never been regenerated since the case-id rename and still showed the old titles
(`camels_dk_discharge`, `bellinge_sewer_level`) — regenerating them, plus labelling every
exogenous-overlay axis with its unit in `case_eda.py`, resolved most of what looked "off."
Added captions to every case doc and redrew the CAMELS-DK mermaid diagram, and gave
`notebooks/eda.py` the same treatment (axis labels on the ACF/CCF panels, plus a
raw-vs-curated overlay on the full-series plot so it's visible how much of the record the
benchmark actually uses). That overlay initially thinned the raw and curated series to 3000
points each for plotting, but computed the stride independently per series — since the two
have different lengths, the strides diverged and the two lines ended up sampling different
timestamps, making spikes in the raw line (e.g. `demand_ltown` in Jan–Feb 2018) look like
they'd been smoothed away in the curated line even though the underlying values were
identical everywhere outside the small gap-filled stretches. Fixed by dropping the
downsampling entirely and plotting both series at full resolution, so they trace exactly on
top of each other except where curation actually filled a gap. That broke `waves_oregon` and
`waves_massachusetts` (hourly data over 2014–2023, ~79k rows), whose full-resolution plot
exceeds marimo's 8MB cell-output cap. Fixed by only downsampling when a series is too large
to render, using min/max-per-bin binning on a single shared set of absolute time edges
across raw and curated (rather than each series computing its own stride), so a spike always
survives and the two traces still match exactly at every plotted point outside a filled gap.
One finding didn't get fixed:
`sewerlevel_bellinge`'s level series
has two ~67-hour flatline plateaus at a fixed value (2.059/2.013 m against a 0.2–0.6 m
baseline) — a stuck-sensor/logger fault that neither `data/processed` nor `curate.py`'s
gap-fill removes. Despiking it needs a filter in `sewerlevel_bellinge/download.py`
(mirroring the existing rain-sentinel filter) calibrated against the raw 1-min CSV, which
is still open.

## 2026-07-10 — Scaled features ahead of MLPRegressor (issue #48)

`MLPForecaster` (`ml.py`) was feeding raw lags and calendar features straight into
`sklearn`'s `MLPRegressor` with no scaling, even though those features span very different
ranges (lags near the target's scale, `dayofyear` up to 366). MLPs are sensitive to input
scale, so its scores were likely reflecting that missing preprocessing step rather than
anything about the architecture, which would undermine any MLP-vs-GBDT comparison in the
matrix. Fixed by wrapping the regressor in an `sklearn` `Pipeline` with a `StandardScaler`
step first. That exposed a latent bug: the code that reads back the prediction column name
assumed the fitted model object's class name matched mlforecast's internal alias, which
breaks for any model wrapped in a `Pipeline` (mlforecast unwraps to the last step's class
name for the alias, but the object itself is still the `Pipeline`). Fixed that first, in its
own commit, by reading the alias directly from mlforecast's model dict instead of
re-deriving it from the object's type — a fix that's also correct for the unwrapped models,
so it carries no risk for `LightGBM`/`CatBoost`/`Ridge`. Any MLP results generated before
this fix should be treated as unreliable and re-run.

## 2026-07-10 — Cleaned up duplicated helper logic across method wrappers (issue #58)

Fixed three cases of drifted duplicate code found in the 2026-07-02 method-suite audit.
`classical.py`'s five AutoARIMA/AutoETS/TBATS/Theta/MSTL forecasters each repeated the same
tz-strip-and-truncate block; that's now one `_prepare_train` helper. `_lookback_for` existed
in three near-identical copies (`dl.py`, `torchdl.py`, and `nf_models.py`'s differently-named
`_input_size`), and `dl.py`'s had silently drifted to drop the `max(base, 2*h)` floor the other
two had — merged into one `lookback_for` in `base.py` with the floor applied everywhere.
`nf_models.py`'s TFT/iTransformer/TimesNet models were also unseeded, unlike the other DL
families, so `fit()` now calls `pytorch_lightning.seed_everything(0)` for reproducibility.
No test asserted the pre-fix behavior, so nothing needed updating beyond the three files.

## 2026-07-09 — Added a case-context side panel to the results-matrix app

Clicking a matrix cell now opens a right-hand panel next to the matrix, separate from the
existing forecast-snapshot panel below it, showing the last 1,000 points of that case's
training split and a summary of the case's static config (domain, target, frequency, horizon,
exogenous inputs, source). The point cap keeps the plot fast on long series; there's no
existing formatter for `Case` metadata elsewhere in the codebase, so `app/data.py` gained a
small `case_summary()` that reads straight off the `Case` model rather than any generated
report. The panel only depends on the clicked case, so it stays open across method-chip clicks
that change the row's overlay selection.

## 2026-07-09 — Locked down the results-matrix heatmap to click-and-download only

The matrix heatmap is the app's centerpiece, so it's now locked against pan, zoom, box/lasso
select, and axis dragging — visitors can click a cell to open its snapshot panel or use the
mode bar's camera icon to download a PNG, and nothing else. This is a `dcc.Graph` `config` plus
`fixedrange`/`dragmode=False` change in `app/app.py`; the click-to-select callback needed no
changes since `clickData` still fires normally.

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
