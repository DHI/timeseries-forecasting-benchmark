# Research Log Index

Title-only index of every entry in `details.md`, grouped by date (newest first). Scan this to
find which entries are relevant, then open `details.md` (or search for the id) to read the full
entry and its commits. See `schema.md` for the format both files follow.

<!-- Add new entries below this line, newest first, matching details.md. -->

### 2026-07-30
- 88 — Fixed MLP's divergence at the source, and added a Forecast strategy card to every method

### 2026-07-29
- 87 — MLP+exo existed but was never in the pilot run
- 86 — Method cards get a Gaps capability, derived rather than hand-authored
- 85 — Gave MLP a `+exo` variant
- 84 — TBATS now fits once too, via genuine AIC selection instead of a fixed config
- 83 — Method cards now carry a capability set and an operational scorecard
- 82 — Converted the classical methods to fit once and roll forward, except TBATS

### 2026-07-28
- 81 — Flagged local methods in the case panel as refit at every origin
- 80 — Fixed FM context to a single pre-test window, no longer growing per origin
- 79 — Reverted foundation-model context to full history, capped at model max
- 78 — Split training-time measurement into fit and predict, averaged per rolling origin
- 77 — Foundation-model context length now tracks seasonal periods, not model max

### 2026-07-27
- 76 — Stopped hardcoding foundation-model context lengths; read them from the models instead
- 75 — Flagged foundation models in the case panel as not using the train/val split
- 74 — Bounded MLP's recursive forecast to stop it diverging on smooth series
- 73 — One color scheme for every metric in the matrix heatmap, and it now adapts to filtering

### 2026-07-23
- 72 — Split the pilot run across machines: tagged partial runs + a merge step

### 2026-07-21
- 71 — Fixed the matrix heatmap going stale when leaving NSE/KGE for another metric
- 70 — Training-time overlay disabled for short matrices (Plotly bug, no in-code fix found)
- 69 — Domain controls now follow the matrix's own NSE/KGE narrowing
- 68 — Added a seasonal-pattern figure to the case panel
- 67 — Moved app instructions from a root markdown file into the app itself
- 66 — Removed the legacy Dash app now that `web/` covers everything it did

### 2026-07-20
- 65 — Fixed heatmap bleeding over the sidebar when the case panel is dragged wide
- 64 — Sidebar closed by default, added a hint for first-time visitors
- 63 — Moved training time from the right panel to a press-and-hold heatmap overlay

### 2026-07-19
- 62 — Fixed leap-year bug in the 365-day position-in-cycle path
- 61 — Interactive decision graph for the research log
- 60 — Combined the EDA and React Pages workflows into one
- 59 — Replaced marker descriptions with a visual swatch
- 58 — Moved the method-info card title to the top when expanded

### 2026-07-16
- 57 — Fixed method-info card height-drag stopping after one pixel
- 56 — Split the method-info strip into per-method collapsible, resizable cards
- 55 — Made the React case detail panel resizable

### 2026-07-15
- 54 — Fixed the React case panel's training/test charts showing timestamps on both axes
- 53 — Closed two React/Dash parity gaps: method-info panel placement, missing benchmark-design text
- 52 — Started a React port of the results-matrix app for GitHub Pages
- 51 — Gave Chronos-2, TimesFM 2.5 and Moirai-2.0 `+exo` variants

### 2026-07-14
- 50 — Fixed matrix cell clicks crashing with `KeyError: 'customdata'`
- 49 — Added three zero-shot foundation models (Chronos-2, TimesFM 2.5, Moirai-2.0)
- 48 — Folded +exo method variants into an app-level toggle; retired +calendar
- 47 — Committed distilled scores to docs/ for onboarding (issue #19)

### 2026-07-13
- 46 — Added MAPE, sMAPE, NSE, and KGE (issue #34)
- 45 — Windowed ML/DL methods are now fit once per case, not refit at every rolling origin
- 44 — Snapshot panel now shows curated best/median/worst origins, ranked by whichever metric the matrix is showing
- 43 — Method-info panel now shows library, compute cost, limitations, and a code snippet

### 2026-07-10
- 42 — Added a bottom method-info panel, capped the method overlay at 3
- 41 — Merged the case-info and snapshot panels into one collapsible column
- 40 — Moved the snapshot panel into a vertical column next to case info
- 39 — Results viewer can show the extended-history fit for gappy cases
- 38 — Made the extended-history fit live via per-segment training (supersedes the "inert" entry below)
- 37 — Added an is_global capability flag and an extended-history training path (currently inert)
- 36 — Closed out EDA figure-quality review comments (issue #36)
- 35 — Scaled features ahead of MLPRegressor (issue #48)
- 34 — Cleaned up duplicated helper logic across method wrappers (issue #58)

### 2026-07-09
- 33 — Added a case-context side panel to the results-matrix app
- 32 — Locked down the results-matrix heatmap to click-and-download only
- 31 — Persist real per-origin forecasts and overlay methods in the results app
- 30 — Moved the results-matrix app from Streamlit to Dash
- 29 — Two app bugs found on first real-browser test, both pre-existing
- 28 — Multi-snapshot forecast panel for the results-matrix app

### 2026-07-08
- 27 — Interactive results-matrix app (Streamlit + Plotly)
- 26 — Swapped statsforecast's MSTL wrapper for statsmodels' directly
- 25 — Audit of the RobustPeriod implementation: three weaknesses found and fixed
- 24 — RobustPeriod: data-driven seasonality discovery for curation
- 23 — Non-seasonal MSTL test now builds its own non-seasonal case
- 22 — Dominant period is now chosen at fit time by variance explained

### 2026-07-07
- 21 — Climatology baseline now keys on the declared dominant seasonality (issue #54)
- 20 — Added a PACF panel to the seasonality notebook, alongside ACF
- 19 — Interactive periodogram + find_peaks in the seasonality notebook

### 2026-07-03
- 18 — Periodogram-peak suggestions for off-menu seasonalities
- 17 — Per-case curated `seasonalities` replace the freq → seasonality map
- 16 — Silencing a spurious TBATS Box-Cox warning
- 15 — Gap handling: one materialized policy (cap + longest-segment trim)
- 14 — Ensemble-of-baselines columns (issue #59)

### 2026-07-02
- 13 — Seasonal classical baselines: MSTL forecasters + seasonally-adjusted Theta (issue #46)
- 12 — Audit of the method suite (`bench/methods/`): flaws and gaps identified
- 11 — Fixed rolling-origin selection collapsing onto the front of the test set
- 10 — Added missing `discharge_saugeen` case doc and EDA panel
- 9 — Dropped `optional_exogenous` as a redundant stored field
- 8 — Structured exo-necessity fields, persisted as result metadata (not matrix aggregation)
- 7 — Recorded necessary-vs-optional exogenous judgment in case notes
- 6 — Added CatBoost to the windowed-ML method set
- 5 — Backfilling this log from the scoping documents

### 2026-06-25
- 4 — Added a second, unforced hydrology contrast case

### 2026-06-12 to 2026-06-13
- 3 — Scoping review with Rocco (domain expert), and the response

### 2026-05-29
- 2 — Two data-quality bugs found before scaling further

### 2026-05-27 to 2026-05-28
- 1 — Initial scoping plan and repo scaffold
