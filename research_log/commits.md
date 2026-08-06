# Research Log Commits

Git commits behind each choice in `details.md`, in the same order and matching headings.
Short SHA + subject line — run `git show <sha>` for the full diff. Commits within an entry are
listed oldest first (the order the work happened). Entries reconstructed after the fact (see
`details.md`'s "Backfilling this log..." entry) are marked approximate: their commit lists were
matched by date and topic rather than by a research-log commit that names the entry directly.

<!-- Add new entries below this line, newest first, matching details.md. -->

## 2026-07-30 — Fixed MLP's divergence at the source, and added a Forecast strategy card to every method

- `7107040` Switch MLP to mlforecast's direct multi-horizon strategy
- `7d08804` Add ForecastStrategy metadata field alongside Capabilities/Scorecard
- `ee39fcb` Declare forecast_strategy on every concrete Forecaster
- `b6bb9cc` Export forecast_strategy through the web data pipeline
- `dc779e4` Add a Forecast strategy block to the method info card
- `961a732` Extend method-scorecard guard tests to cover forecast_strategy
- `4bd6535` Log the MLP direct-horizon fix and forecast_strategy taxonomy

## 2026-07-29 — MLP+exo existed but was never in the pilot run

- `1d553b0` Including exo to MLP

## 2026-07-29 — Method cards get a Gaps capability, derived rather than hand-authored

- `c7ac4a4` Derive a gaps capability from pools_segments, not a hand-authored one
- `2b02844` Add Gaps chip to the method card's Handles row
- `6122294` Regenerate methods.json with the new gaps capability
- `cd9a7e7` Log the gaps capability decision in the research log and its graph

## 2026-07-29 — Gave MLP a `+exo` variant

- `9d6bea2` Add MLP+exo variant (Tier 1 of exogenous expansion)

## 2026-07-29 — TBATS now fits once too, via genuine AIC selection instead of a fixed config

- `c684594` Make TBATS fit once and roll forward, using genuine AIC model selection
- `c0cf951` Add TBATS to the fit-once test suite, replacing the stays-local pin
- `98da600` Correct app copy now that TBATS fits once like the rest of classical
- `38492a1` Log the TBATS fit-once fix in the research log and its graph
- `fe1a5c6` Correct TBATS fix numbers to the real eval pipeline's own run

## 2026-07-29 — Method cards now carry a capability set and an operational scorecard

- `aae01bd` Add Capabilities and Scorecard models for method self-description
- `a5d1431` Declare capabilities and scorecards for the baselines and ensembles
- `ceba177` Declare capabilities and scorecards for the classical methods
- `3aef62d` Declare capabilities and scorecards for the windowed ML methods
- `30909bc` Declare capabilities and scorecards for the deep-learning methods
- `0033f47` Declare capabilities and scorecards for the foundation models
- `b7570e7` Export capabilities and scorecard to methods.json, with guard tests
- `93f3a1d` Add web types, display config and level colour tokens for the scorecard
- `e36954a` Render the Handles row and scorecard meters on the method cards
- `c1457be` Log the method scorecard work and add its research-graph node
- `2d1250e` Drop the level text from the scorecard rows
- `e8b29cb` Updated jsons. Fixing integer number test.
- `9bce6fa` Removing family from method card.

## 2026-07-29 — Converted the classical methods to fit once and roll forward, except TBATS

- `86aa455` Convert AutoARIMA and AutoETS to fit-once, is_global forecasters
- `719b4dd` Convert Theta to fit-once, is_global forecaster
- `d1867d6` Reimplement TBATS as a custom fit-once state-space model
- `46221fb` Convert MSTL-ETS/MSTL-ARIMA to fit-once, is_global forecasters
- `9595914` Update docs/UI text for classical methods now fitting once
- `e2de2e4` Add tests for classical methods' fit-once conversion
- `54e0f86` Log the classical fit-once conversion in RESEARCH_LOG and the graph
- `53be293` Choose TBATS/AutoETS positivity-dependent specs from the whole record
- `6750b8e` Test TBATS/AutoETS specs against history before the search window
- `e813708` Roll the classical methods forward with statsforecast's own forward()
- `0f24e6d` Read local-vs-global off is_global instead of the method's family
- `9d7b400` Revise today's log entry for the final fit-once scope
- `4e68f76` Document the fit-once protocol in the Benchmark Design modal

## 2026-07-28 — Flagged local methods in the case panel as refit at every origin

- `cd68a41` Extract shared method-to-family lookup in CasePanel
- `433793c` Warn when a selected method is refit at every origin
- `9826363` Log the local-method case-panel caveat

## 2026-07-28 — Fixed FM context to a single pre-test window, no longer growing per origin

- `2c8ba13` Fix FM context to a pre-test window recorded at fit time
- `1a94b00` Log the fixed pre-test window FM context change

## 2026-07-28 — Reverted foundation-model context to full history, capped at model max

- `fbc5750` Condition zero-shot FMs on full history capped at model max
- `5b2d969` Log the revert to full-history FM context

## 2026-07-28 — Split training-time measurement into fit and predict, averaged per rolling origin

- `c2b3f05` Split eval-loop timing into fit_seconds and predict_seconds
- `cf28aab` Update print_results.py for the fit_seconds/predict_seconds split
- `3f2b011` Rename ScoreRow.seconds to fit_seconds/predict_seconds; show cost per forecast
- `3c55482` Show fit/predict breakdown on hover in the training-time overlay
- `090ca41` Explain fit/predict semantics in training-time help copy
- `0359dd9` Log the fit/predict timing split decision
- `52c3141` run with fixed fm (TimesFM is still buggy)
- `f0ae975` Export new run after timesfm fix
- `b1d5487` exporting json files

## 2026-07-28 — Foundation-model context length now tracks seasonal periods, not model max

- `4428aff` Size zero-shot FM context to 2 seasonal periods, not model max
- `cb0e7dd` Restructure TimesFM loading to fix NaN forecasts on short-history cases
- `65b7db7` Remove dead input_window field from Case schema
- `b094a7e` Log the seasonal-period context-length change and TimesFM NaN fix

## 2026-07-27 — Stopped hardcoding foundation-model context lengths; read them from the models instead

- `da2ae3c` Read Chronos-2's real context length instead of hardcoding it
- `b1972d1` Read TimesFM-2.5's real context limit instead of hardcoding it
- `a57a10f` Use full history for Moirai-2.0 context, capped at cache size
- `0035f68` Log the foundation-model context-length fix

## 2026-07-27 — Flagged foundation models in the case panel as not using the train/val split

- `3fb251d` Including warning sign for train/test split and fm models.

## 2026-07-27 — Bounded MLP's recursive forecast to stop it diverging on smooth series

- `f7552df` Bound MLPForecaster's recursive predictions to stop runaway divergence
- `122bf8f` Fix _BoundedRegressor's sklearn Pipeline fitted-check
- `361f732` Add MLP forecaster coverage, incl. a regression test for the bounded fix
- `d793a25` Log the MLP recursive-divergence fix in the research log

## 2026-07-27 — One color scheme for every metric in the matrix heatmap, and it now adapts to filtering

- `2b5562a` Unify heatmap color scheme: green good, red bad for all metrics
- `19d33da` Log the heatmap color-scheme unification in the research log
- `1260cef` Adapt heatmap colour range to filtered/visible cells
- `3826b3d` Update research log entry to cover the colour-range adaptation fix

## 2026-07-23 — Split the pilot run across machines: tagged partial runs + a merge step

- `7b9333c` fast run with dl
- `2f74f7c` Run of 40 origins with gpu models
- `aaaed04` Run of 40 origins with baselines
- `18ed673` commit combined scores
- `04588c9` Commit json files
- `16032a6` Changes to split running pilot by methods

## 2026-07-21 — Fixed the matrix heatmap going stale when leaving NSE/KGE for another metric

- `dd0472b` Fixing bug for metric switch in app.

## 2026-07-21 — Training-time overlay disabled for short matrices (Plotly bug, no in-code fix found)

- `551a9d9` Log training-time + NSE/KGE overlay as a known Plotly.js limitation
- `d49056f` Disable training-time overlay for matrices too short to render it
- `3e678b1` Update training-time overlay log entry with the row-count gating fix

## 2026-07-21 — Domain controls now follow the matrix's own NSE/KGE narrowing

- `a836c53` Add domainsEligibleForMetric selector
- `22d1625` Add disabled-domain support to DomainChecklist
- `341885f` Run RTL cleanup after each test
- `e06f178` Gate domain checkboxes and close case panel on domain-specific metric
- `07107df` Log domain-checkbox/case-panel gating and RTL cleanup fix

## 2026-07-21 — Added a seasonal-pattern figure to the case panel

- `6b347f4` Add seasonal-pattern data to the case export
- `d5e5349` Add SeasonalPatternPlot component and SeasonalPattern type
- `db313b6` Wire seasonal-pattern figure into the case panel
- `4de609d` Log research entry for the seasonal-pattern figure
- `a433f3c` Detrend the seasonal-pattern figure before plotting
- `af251dd` Note detrending in the seasonal-pattern caption
- `48deb77` Fold detrending decision into today's seasonal-pattern log entry
- `ac6791d` Give subplot x-ticks room in the seasonal-pattern figure
- `83788f1` Explain the seasonal-pattern figure's smoothed line and encourage zooming
- `94c342a` Add subcaption
- `c541ec0` Label seasonal-pattern subplot x-axes with "# of steps"
- `10f8821` Adding xlabel to seasonality plots

## 2026-07-21 — Moved app instructions from a root markdown file into the app itself

- `51f0ba8` Move app instructions from repo root into web/src/data
- `121d413` Add InstructionsModal component
- `ad2951c` Show the instructions modal next to the page title
- `bf1c1ff` Log research entry for the app instructions move
- `c525c3c` Thin out the info icon and title the instructions modal

## 2026-07-21 — Removed the legacy Dash app now that `web/` covers everything it did

- `a9eb46c` Extract shared data/curation logic into tsforecastbench.webdata
- `4c32615` Drop the app extra now that the legacy Dash app is gone
- `c5fa176` Update docs to stop referencing the removed app/
- `6b7ce9c` Log research entry for legacy Dash app removal

## 2026-07-20 — Fixed heatmap bleeding over the sidebar when the case panel is dragged wide

- `e34c339` Clipping heatmap under controls

## 2026-07-20 — Sidebar closed by default, added a hint for first-time visitors

- `d0b4a4b` Close control panel by default and fix toggle button clipping
- `6eee149` Add empty-state hint pointing at the heatmap
- `858196e` Log research entry for sidebar/hint UX changes
- `c151332` Add hint pointing at the controls-panel toggle button

## 2026-07-20 — Moved training time from the right panel to a press-and-hold heatmap overlay

- `1e5a9a6` Fixing drag handle right panel
- `e272313` Remove training time from right panel, keep sample counts
- `c7f89b4` Add press-and-hold hook and training-time button
- `05e4265` Add trainingTimeOverlay to app state
- `b371bcf` Generalize pivotValues with an optional value accessor
- `ebe182b` Add TRAINING_TIME_CONFIG display config
- `ed7bc8b` Wire the training-time overlay into the heatmap and control panel
- `0ada92c` Log the training-time overlay decision
- `411ac60` Fix training-time overlay: integers, stable grid, log color scale
- `40264de` Fold overlay fixes into today's research log entry
- `56c88f7` Drop the "s" suffix from training-time cell text
- `c160e2e` Drop the "s" suffix from colorbar tick labels too
- `6ddd715` Improve hold button description.
- `c71a332` Cleaning right panel
- `9cc435d` Allow holding space bar to overlay training time
- `82842ca` Prevent space-release from clicking the focused button
- `b990b3b` Stop space-hold from scrolling the page on OS key-repeat
- `035bdb4` Add unit/scale/log-color fields to metric config
- `0bdf54c` Fix colorbar title line breaks and add log color scale for heavy-tailed metrics
- `17243bf` Add tests for colorbar titles and log color scale
- `084d88d` Correct Skill vs * colorbar title
- `17e941a` Fix clipped colorbar unit line and adapt log-tick decimals to magnitude
- `23446c5` Rename training-time colorbar title to "Train time [seconds]"
- `ffac5ff` Fix train time colorbar title
- `915d906` Switch training-time overlay to greyscale, with light text on dark cells
- `e633079` Keep hold-space-for-training-time working with Controls panel hidden

## 2026-07-19 — Fixed leap-year bug in the 365-day position-in-cycle path

- `ed101c2` Accounting for leap years in climatology models
- `a5e1fc0` Including test for 365 day cycles
- `9e72704` Fix tests

## 2026-07-19 — Interactive decision graph for the research log

- `ad1bf7f` Add interactive research-log decision graph
- `791c944` Link the new decision graph from docs and CLAUDE.md
- `dfe373c` Use directional relation labels in the connections panel
- `2b01966` Log the new decision-graph tool in the research log
- `33a46fa` Rebuild the decision graph against this branch's actual log
- `2dcda67` Improve research graph
- `aaee2fd` Including list view in research graph
- `4d115c5` Fix research graph and commiting latest results.
- `cf4a580` Fix research graph

## 2026-07-19 — Combined the EDA and React Pages workflows into one

- `fe22848` Consolidate EDA and React Pages workflows into one deployment
- `1605e84` Log the Pages workflow consolidation and auth fix

## 2026-07-19 — Replaced marker descriptions with a visual swatch

- `189a69c` Show marker swatches instead of describing them in words

## 2026-07-19 — Moved the method-info card title to the top when expanded

- `e3b3477` Improving method cards

## 2026-07-16 — Fixed method-info card height-drag stopping after one pixel

- `178bffc` Fix method-info card drag stopping after one pixel of movement
- `30117de` Log the drag-resize bug fix in the research log

## 2026-07-16 — Split the method-info strip into per-method collapsible, resizable cards

- `381f494` Turn method-info cards into collapsible, height-resizable panels
- `a322f3c` Add bottom padding to the method-info dock
- `40ddf25` Make method-info cards fill the width between the two panels
- `b12aa19` Log the method-info card redesign in the research log

## 2026-07-16 — Made the React case detail panel resizable

- `6058cea` Add resizable width state for the case detail panel
- `54f4b34` Add useResizablePanelWidth hook for drag-to-resize panels
- `4a44a60` Make the case detail panel resizable by dragging its left border
- `ef34133` Log the case panel resize feature in the research log
- `3c6f804` Generalize the resize hook to support both axes

## 2026-07-15 — Fixed the React case panel's training/test charts showing timestamps on both axes

- `f1818d3` Fix train plot bug

## 2026-07-15 — Closed two React/Dash parity gaps: method-info panel placement, missing benchmark-design text

- `a435ad6` Fixes after migration
- `d9bcaca` Show Benchmark Design content in the instructions modal

## 2026-07-15 — Started a React port of the results-matrix app for GitHub Pages

- `5b0a024` Scaffold web/ React+TS+Vite app for the GitHub Pages migration
- `81fa37b` Log the React/GitHub Pages migration decision
- `2cc8579` Add scripts/export_web_data.py to export benchmark data as static JSON
- `985c1b1` Add matrix data layer: metric config, colour scale, and selectors
- `f1f80f4` Wire up the static results matrix with sidebar filters
- `a406fe5` Update the React migration log entry with M1 progress
- `a65168d` Add selection reducer and wire toggle/selection state
- `5aac3c6` Wire hover tooltip, click-to-select, and global/exogenous overlays
- `9f464d1` Update the React migration log entry with M2 progress
- `b396b59` Export bounded per-case derived series for the case detail panel
- `aff2bfe` Add case-summary/training-stats text ports and extended-history selector
- `b2e0422` Wire up the case detail panel
- `1a4f83d` Update the React migration log entry with M3 progress
- `f8ae49a` Export per-case snapshot bundles for the forecast panel
- `37d2e46` Add snapshot-panel data layer: types, freq parsing, method-info text
- `20721f8` Wire up the forecast-snapshot panel
- `10b6abe` Update the React migration log entry with M4 progress
- `d4be80a` Combine the React app and EDA site into one Pages deploy
- `096df16` Document the web app in the READMEs
- `42c429d` Log the completed React/GitHub Pages migration

## 2026-07-15 — Gave Chronos-2, TimesFM 2.5 and Moirai-2.0 `+exo` variants

- `a72c5b6` Add +exo variants for Chronos-2, TimesFM-2.5, Moirai-2.0
- `a0e65b4` Wire FM +exo classes into the methods package and pilot runner
- `3ef9b85` Add tests for FM +exo variants
- `309d05d` Log the FM +exo research entry

## 2026-07-14 — Fixed matrix cell clicks crashing with `KeyError: 'customdata'`

- `d1e9686` Fix app so cells are interactive
- `749add2` Fix exo bug for clicked cell

## 2026-07-14 — Added three zero-shot foundation models (Chronos-2, TimesFM 2.5, Moirai-2.0)

- `afb528d` Add zero-shot Chronos-2 foundation-model method
- `a78b426` Add zero-shot TimesFM 2.5 foundation-model method
- `ce9dc8c` Add zero-shot Moirai-2.0 foundation-model method
- `ed6b72e` Unify foundation-model deps into one fm extra; cap project at py<3.13
- `3d088c9` Document the new foundation-model methods and their install extras

## 2026-07-14 — Folded +exo method variants into an app-level toggle; retired +calendar

- `47c2195` Emit case_exo_tier/uses_exogenous from the rolling-origin eval path
- `acfb00f` Fold +exo method columns into a toggle in the results matrix
- `39d1672` Document the exogenous-toggle methodology in the app
- `760374f` Retire +calendar method variants, always include seasonal features
- `b6cfe54` Log the exogenous-toggle and +calendar-retirement decision

## 2026-07-14 — Committed distilled scores to docs/ for onboarding (issue #19)

- `7538ca3` Anchor .gitignore results/ rule to repo root
- `8ec325a` Commit distilled benchmark scores to docs/ for onboarding
- `f5e7003` Publish current pilot scores and matrix figures to docs/

## 2026-07-13 — Added MAPE, sMAPE, NSE, and KGE (issue #34)

- `ba95a91` Add MAPE, sMAPE, NSE, KGE to the eval loops
- `77dc3f3` Declare per-case eligibility for domain-restricted metrics
- `e550b0b` Surface MAPE, sMAPE, NSE, KGE in the matrix metric dropdown
- `84d65a0` Add unit tests for MAPE, sMAPE, NSE, KGE and domain_metrics validation
- `a3d03b4` Log the new domain/dimensionless metrics decision (issue #34)
- `51a7db6` Fix metric direction for new metrics.
- `f3e8f97` Take new metrics into account for snapshots.
- `9340874` Removing domain metrics from Bellinge (until confirmed)
- `047861a` Remove domain specific metrics for wwtp (until confirmed)

## 2026-07-13 — Windowed ML/DL methods are now fit once per case, not refit at every rolling origin

- `8d70c56` Add refit_per_origin capability flag to Forecaster
- `90df331` Condition windowed ML/DL predict on the history argument, not fit-time state
- `f41d5a2` Fit windowed ML/DL once per case in run_case_rolling, infer across origins
- `3b73f02` Fit windowed ML/DL once in run_leadtime_curve too
- `bcdff15` Log the fit-once rolling-origin decision in the research log
- `6336a1b` Condition Chronos predict on history instead of fit-time context
- `a6e0fd6` Key rolling fit-cadence off is_global; drop the refit_per_origin flag
- `366dc60` Update test_rolling_matrix_runs for the tagged snapshot schema

## 2026-07-13 — Snapshot panel now shows curated best/median/worst origins, ranked by whichever metric the matrix is showing

- `3406c64` Curate snapshot panel to best/median/worst origins by MAE
- `1fa26cd` Explain curated snapshot origins in the app's copy
- `86dcddc` Log the curated-snapshot decision in the research log
- `5e479ff` Curate snapshot origins by the matrix's selected metric, not fixed MAE
- `a03c6d0` Update research log entry for metric-generic snapshot curation
- `83f410b` Simplifying title. De-separate y-axis in snapshot.
- `a2809f0` Remove legend in snapshot

## 2026-07-13 — Method-info panel now shows library, compute cost, limitations, and a code snippet

- `1f90e2d` Add method-metadata attributes to Forecaster base class
- `4bf99e6` Populate method metadata for naive and classical forecasters
- `95ed89c` Populate method metadata for windowed ML forecasters
- `a6509b6` Populate method metadata for DL, torch, neuralforecast, FM, ensemble
- `13e7773` Extend method registry and add method_info_detail()
- `701ac5c` Add collapsible implementation-detail section to method-info cards
- `b0eb162` Log the method-info panel's new library/compute/limitations/snippet fields
- `c96dedb` Add a method-metadata registry and method_info() markdown builder
- `cd2542d` Simplify family names

## 2026-07-10 — Added a bottom method-info panel, capped the method overlay at 3

- `08b1534` Add a bottom method-info panel on cell click
- `f653aad` Log the bottom method-info panel and overlay cap

## 2026-07-10 — Merged the case-info and snapshot panels into one collapsible column

- `0a9c6cc` Merge case-info and snapshot panels into one collapsible column
- `70f92ea` Log the case-info/snapshot panel merge

## 2026-07-10 — Moved the snapshot panel into a vertical column next to case info

- `eb10211` Improving right pannel with case information
- `ed40356` Move test snapshots into a vertical right-hand panel
- `0fe54d4` Log the snapshot-panel layout move

## 2026-07-10 — Results viewer can show the extended-history fit for gappy cases

- `60b0caa` Add global-fit toggle to the results matrix
- `8e4cc20` Wire the global-fit toggle into forecast snapshots
- `b68ec61` Save extended-history fit snapshots, not just local-fit ones
- `c8356b1` Replace hash with triangle
- `ffce616` Log the global-fit toggle and snapshot-panel fix

## 2026-07-10 — Made the extended-history fit live via per-segment training (supersedes the "inert" entry below)

- `a3c37c6` Make is_global methods segment-aware via unique_id
- `fbc3da4` Score segment-pooling methods on the full series as a second fit
- `f7aea8d` Split local and extended-history matrices in the results scripts
- `a58ecce` Log the segment-pooled extended-history fit

## 2026-07-10 — Added an is_global capability flag and an extended-history training path (currently inert)

- `1210689` Add is_global capability flag and a fit_pooled seam for future pooling
- `fb57472` Add extended-history frame for is_global methods
- `df2d01c` Route is_global methods onto extended history, with a gap-free fallback
- `ae6d125` Render a second, is_global-filtered matrix for each metric
- `6818a04` Log the is_global feature and its currently-inert extended-history path

## 2026-07-10 — Closed out EDA figure-quality review comments (issue #36)

- `1aaf14f` Label exogenous overlay axes with units in case_eda.py
- `cc9ce0a` Add EDA captions to all case docs, redraw CAMELS-DK diagram
- `eff7899` Overlay raw vs curated data and add axis labels/captions to eda.py
- `2352fbf` Log EDA figure-quality work and Bellinge follow-up in RESEARCH_LOG.md
- `97920bf` Including raw background span.

## 2026-07-10 — Scaled features ahead of MLPRegressor (issue #48)

- `9994a60` Scale features ahead of MLPRegressor in MLPForecaster
- `0715b25` Log MLPRegressor scaling fix in research log

## 2026-07-10 — Cleaned up duplicated helper logic across method wrappers (issue #58)

- `964ba63` Derive ML model prediction column from mlforecast's alias key
- `903cad9` Remove boilerplate code

## 2026-07-09 — Added a case-context side panel to the results-matrix app

- `d965610` Including training data snapshot pannel
- `7f2641a` Making right panel scrollable

## 2026-07-09 — Locked down the results-matrix heatmap to click-and-download only

- `7837e30` Include frame at clicked cell
- `6c00170` Fixing heatmap

## 2026-07-09 — Persist real per-origin forecasts and overlay methods in the results app

- `9f03c94` Persist per-origin forecasts from the rolling eval
- `0f14ddb` Show real forecasts and overlay methods in the results viewer
- `841464b` Update rolling test for the (metrics, snapshots) return
- `a875aa8` Log the real-snapshots + method-overlay work

## 2026-07-09 — Moved the results-matrix app from Streamlit to Dash

- `fc725b7` Migrate results-matrix app from Streamlit to Dash
- `073e327` Clean app
- `36f752f` Log the Streamlit-to-Dash app migration

## 2026-07-09 — Two app bugs found on first real-browser test, both pre-existing

- `46bd055` Fix KeyError for domains missing from DOMAIN_KEY_DICT
- `4469803` Fix cell clicks never registering on the matrix heatmap
- `2e6b732` Log the domain-dict and heatmap-click bugs

## 2026-07-09 — Multi-snapshot forecast panel for the results-matrix app

- `917b324` Add dummy forecast-snapshot panel to results-matrix app
- `a7fd76b` Log the forecast-snapshot panel work

## 2026-07-08 — Interactive results-matrix app (Streamlit + Plotly)

- `4b29132` Add optional app dependency group (streamlit + plotly)
- `b3e26c8` Add interactive results-matrix Streamlit app
- `b778b8b` Log the results-matrix app work
- `5d553e4` Simplifying frontend
- `8b5f9b9` Domains as checkboxes

## 2026-07-08 — Swapped statsforecast's MSTL wrapper for statsmodels' directly

- `7f39f91` Implementing mstl using statsmodels

## 2026-07-08 — Audit of the RobustPeriod implementation: three weaknesses found and fixed

- `20daeb6` Drop unused FISHER_ALPHA constant
- `4800f7e` Fall back to the mean absolute deviation when the MAD is zero in _huber_clean
- `4a9c990` Skip Nyquist (period-2) candidates in RobustPeriod discovery
- `562a309` Size-adjust the RobustPeriod prominence bar to the peak window's bin count
- `17e3683` Log the RobustPeriod audit and the three fixes

## 2026-07-08 — RobustPeriod: data-driven seasonality discovery for curation

- `f847fd1` Add FFT-based MODWT and robust wavelet variance for period discovery
- `1e2fc03` Add Huber cleaning, Huber periodogram, and band-period helpers
- `c62b2c5` Add robust_period_candidates: data-driven multi-period discovery
- `1bb84f6` Add RobustPeriod discovery cell to the seasonality notebook
- `07b1a08` Allowing extra candidates if RobustPEriod does not find any
- `8de893d` Log RobustPeriod seasonality-discovery work

## 2026-07-08 — Non-seasonal MSTL test now builds its own non-seasonal case

- `8f70ee1` Fix missing test

## 2026-07-08 — Dominant period is now chosen at fit time by variance explained

- `5f5794a` Compute dominiant_seasonality instead of getting it by convention.

## 2026-07-07 — Climatology baseline now keys on the declared dominant seasonality (issue #54)

- `ff6a3a7` Extract shared position_in_cycle helper into seasonality module
- `2834505` Key Climatology on the declared dominant seasonality
- `74c743d` Log Climatology seasonality-keying change (issue #54)

## 2026-07-07 — Added a PACF panel to the seasonality notebook, alongside ACF

- `3d5c229` Including pacf in seasonality notebook
- `7241748` Commit new seasonalities, based on enhanced study

## 2026-07-07 — Interactive periodogram + find_peaks in the seasonality notebook

- `e28a677` Improving peak finding in seasonality notebook
- `66b8f4a` Improving tooltip
- `ca742f3` Hidding clutter info
- `2ca0fb4` Improving plot styling

## 2026-07-03 — Periodogram-peak suggestions for off-menu seasonalities

- `1887763` Suggest off-menu seasonalities from periodogram peaks

## 2026-07-03 — Per-case curated `seasonalities` replace the freq → seasonality map

- `d1a3cfe` Add seasonality module: resolution helpers + detection evidence
- `2d562d9` Add marimo notebook for one-time seasonality curation
- `f410038` Calibrate ACF floor to 0.1 on real-case evidence
- `c78afe3` Require Case.seasonalities; curate all 12 cases from train-split evidence
- `56a387a` Switch methods to case.seasonalities
- `b21e320` Resolve MASE season from declared seasonalities; delete freq maps
- `ae01228` Add advisory seasonality drift check to review_case
- `2263648` Document required seasonalities field in the case checklist
- `77e2bc2` LAtest changes of notebook. Revisiting declared seasonalities per case.
- `2d6c54b` Clarify that seasonalities ordering is by prominence, not length

## 2026-07-03 — Silencing a spurious TBATS Box-Cox warning

- `e0e5324` Guarding boxcox error for negative serues after boxcox.
- `2795599` suppress overflow warning from tbats

## 2026-07-03 — Gap handling: one materialized policy (cap + longest-segment trim)

- `c71d6f0` Creating curation pipeline for data sets. Minimal gap imputation + trimming to longest contiguous run. Cases are now based on curated data.
- `f7cc7ae` Anchor curation to the tail segment, not the longest one
- `80c97d9` Fix stale 'longest span' docstring wording after tail-segment change

## 2026-07-03 — Ensemble-of-baselines columns (issue #59)

- `6fa6403` Include naive ensembles to module suite

## 2026-07-02 — Seasonal classical baselines: MSTL forecasters + seasonally-adjusted Theta (issue #46)

- `0a40378` Add seasonal classical baselines: MSTL-ETS/MSTL-ARIMA + seasonal Theta (#46)

## 2026-07-02 — Audit of the method suite (`bench/methods/`): flaws and gaps identified

- `8e76b47` Logging fable findings

## 2026-07-02 — Fixed rolling-origin selection collapsing onto the front of the test set

- `8e508f4` Fixing origins bug in rolling metric

## 2026-07-02 — Added missing `discharge_saugeen` case doc and EDA panel

- `8f8c61d` Adding saugeen to docs

## 2026-07-02 — Dropped `optional_exogenous` as a redundant stored field

- `d65b85e` Including necessary_exog in schema

## 2026-07-02 — Structured exo-necessity fields, persisted as result metadata (not matrix aggregation)

- `d65b85e` Including necessary_exog in schema

## 2026-07-02 — Recorded necessary-vs-optional exogenous judgment in case notes

- `b7c0a93` Including Catboost. Updating notes with exogenous assumptions.

## 2026-07-02 — Added CatBoost to the windowed-ML method set

- `b7c0a93` Including Catboost. Updating notes with exogenous assumptions.

## 2026-07-02 — Backfilling this log from the scoping documents

- `70cdbaf` Commit research log

_(approximate — reconstructed after the fact)_

## 2026-06-25 — Added a second, unforced hydrology contrast case

- `aa0e90c` Include discharge_saugeen

_(approximate — reconstructed after the fact)_

## 2026-06-12 to 2026-06-13 — Scoping review with Rocco (domain expert), and the response

- `0cdc9cb` docs: respond to Rocco's scoping review (PR #2)
- `840fcf5` refactor: split package into data/ and bench/ namespaces
- `25a6ef9` feat(bench): exogenous-aware windowed ML (Tier 2) + skill-vs-mean
- `c5d44a1` fix(cases): populate real ERA5 exogenous for GESLA + Bellinge

_(approximate — reconstructed after the fact)_

## 2026-05-29 — Two data-quality bugs found before scaling further

- `0f348ea` fix(bellinge): pin G71F05R LevelInlet (100% active); fix(battledim): sum pumps
- `2de8c8c` feat(skill): add /review-case case-quality review

_(approximate — reconstructed after the fact)_

## 2026-05-27 to 2026-05-28 — Initial scoping plan and repo scaffold

- `715b4f8` Initial commit: 6 cases, 9 methods, matrix + lead-time eval, CI

_(approximate — reconstructed after the fact)_

