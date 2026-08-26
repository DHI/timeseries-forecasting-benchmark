---
id: 97
date: "2026-08-24"
title: "The ensembles are fit once per cell instead of refit at every rolling origin"
description: |
  `EnsembleForecaster` declared itself local, so the eval loop refit the whole composite at every
  rolling origin — re-running `AutoTheta`'s parameter search 100 times per cell for a forecast
  change of at most a fraction of a percent. On `sewerlevel_bellinge` that is ~39,500s of measured
  cost against ~700s fitting once (one 208s fit plus ~4.9s per origin), a 57x difference, and it is
  why `surge_cuxhaven` — 214,000 points, ~110s per Theta origin — had never completed a full run.
  The composite is now `is_global` with two cadences inside it: members holding a trained model are
  fit once on the declared train split and rolled forward, while members whose parameters are a
  closed-form statistic of whatever frame they are shown (the naive baselines) are re-derived inside
  `predict` from that origin's history, so they keep tracking the origin exactly as the refit did.
  Fitting once is also what keeps the trained member off the validation split, the entry-92 problem
  (gh #99): a model reused across origins must be fit on the declared train split alone, and the
  only frame the local path ever hands `fit` is the history up to an origin, which runs through val.
  `EnsembleBaselines`' forecasts are provably unchanged (max absolute difference `0.000e+00` against
  the median of its four naive columns on all seven recorded cases), though its recorded
  `is_global`, `train_points`, `val_points` and timings all move; `EnsembleBaselinesTheta` is the
  only column that moves numerically, and **its published numbers are stale** until the
  `ensembles_fitonce` and `surge_full` re-runs land.
commits:
  - bd41ef1: "Fit ensemble members once instead of at every origin"
  - 7f2af0e: "Add --methods to run_pilot for refreshing single columns"
  - a009f5d: "Describe the ensembles' two fit cadences in the app copy"
  - 3db6ef8: "Re-export the web method cards"
---
id: 96
date: "2026-08-18"
title: "TFT, iTransformer and TimesNet were crashing at the end of every fit"
description: |
  The pilot's `--with-nf` methods failed on every case and every seed with `RuntimeError:
  <Class> is not attached to a 'Trainer'`, each failure only surfacing after a full training run
  had already completed. The cause was `last_fit_steps` reading `model.trainer.global_step` off
  the pre-fit model object — `NeuralForecast` deep-copies `models_init` internally and only
  attaches a trainer to the copy it actually trains, so the original's `.trainer` property raised
  instead of returning `None`. Fixed by reading the step count from the fitted copy's own
  `train_trajectories` instead of a live trainer reference. This was a version-drift bug (the
  code was written against an older neuralforecast that left the trainer attached differently),
  not a logic error in the wrapper's intent, so no other `_NFBase` behavior changed.
commits:
  - 85e89be: "Fix TFT/iTransformer/TimesNet crashing at the end of every fit"
---
id: 95
date: "2026-08-14"
title: "Stochastic methods are scored over five seeds, reporting the median run whole"
description: |
  Every cell in the results matrix was a single fit at a single seed, so for a method whose
  forecast moves with its initialisation the cell mixed the method's skill with the luck of one
  draw — and nothing in the output separated the two. Each such method is now run five times per
  cell and the **middle run by pooled MSE is kept whole**: its nine metrics, its snapshots, its
  timings, its `train_steps`. Keeping the run whole is the point. Averaging the forecasts across
  seeds would have made the number better rather than merely steadier, benchmarking a five-member
  deep ensemble against non-ensembled classical and foundation methods; medianing each metric
  separately would have left a row whose `MASE` and `Skill_vs_Persistence` came from different
  runs and whose drawn forecast matched neither. The app is unchanged — still one number per cell,
  no uncertainty display — because selection happens inside the eval loop, so there is still
  exactly one row per `(case_id, method, used_extended_history, metric)`.

  The sweep covers nine methods, not the seven first identified: N-HiTS and PatchTST
  (`torch.manual_seed`), TFT, iTransformer and TimesNet (`pl.seed_everything`), MLP and MLP+exo
  (`MLPRegressor(random_state=...)`), and — newly — CatBoost and CatBoost+exo. CatBoost's default
  MVS bootstrap resamples rows at every iteration, so two fits differing only in `random_seed`
  give different forecasts; it had looked deterministic only because the library's own default
  seed is 0. Reproducible is not the same as deterministic, and by the test that puts MLP in the
  list CatBoost belongs there too. Everything else still runs once: LightGBM's sampling defaults
  are off, Ridge is closed-form, and the classical, baseline and foundation families draw no
  random numbers, so repeating them would be pure cost for identical results.

  `DEFAULT_SEEDS` defaults to 5 (override with `TSFB_SEEDS` or `--seeds`, in the same style as
  `DEFAULT_ORIGINS`), costing roughly seven extra hours on a full pilot. Every stochastic method's
  existing scores are now stale, so this wants to land before the next full pilot rather than
  after one. Left open: `run_leadtime_curve` still runs a single seed, and gh #104 (MLP
  early-stopping on a shuffled holdout rather than the declared val split) also restales MLP.
commits:
  - a5ba0f8: "Route every method's randomness through a settable seed"
  - 4b0c9e9: "Add DEFAULT_SEEDS, the runs-per-stochastic-cell knob"
  - 1dbdef9: "Score stochastic methods over several seeds, keep the median run"
  - a11b82b: "Apply the same seed sweep to the single-holdout path"
  - 3e28ba0: "Add --seeds to run_pilot and pin CatBoost's seed sensitivity"
---
id: 94
date: "2026-08-14"
title: "Auditing the method set for comparability turned up two more picked constants, both in the ML family"
description: |
  Swept every method, both eval loops and `eval.budget` before the next full pilot, looking for
  anything that varies by case or by method without being derived from the data. The classical,
  baseline, foundation-model and deep-learning families came back clean: season lengths and lag
  ladders resolve from `case.seasonalities`, MSTL drops periods the record cannot support,
  AutoARIMA and AutoETS are non-seasonal by one uniform declared choice, and `TRAIN_POINT_CEILING`
  does not bind anywhere. Two constants in `methods/ml.py` did not hold up. LightGBM's 200
  boosting rounds at learning rate 0.05 and CatBoost's 200 iterations at 0.05 were picked rather
  than taken from either library, and both are now each library's own defaults — this benchmark
  does not tune, so the default is the honest configuration to compare, even though it leaves the
  two an order of magnitude apart in round count.

  MLP's `max_iter = 300` was the more interesting one, and it bound. Lifting it and reading back
  `n_iter_` showed where each per-horizon model actually converges: 586 epochs at worst, on
  `wwtp_influent_n2o_350k`, against a median of 87 on `sewerlevel_bellinge`. The pattern is the
  opposite of entry 89's — scikit-learn's `max_iter` counts epochs, and a larger record packs more
  updates into each one, so big cases converge in *fewer* epochs and the cap bit the small, noisy
  cases instead. It bound on 4 of the 11 cases measured, which made those columns reflect the
  constant rather than the data. `MLP_MAX_EPOCHS = 2_000` replaces it in `eval.budget`, above
  every measured convergence point and kept separate from `DL_MAX_EPOCHS` because the two stopping
  rules are not comparable: `torchdl` gives up after 250 gradient steps, a fraction of one epoch
  on a large case, where MLPRegressor gives up after 10 whole epochs.

  Three asymmetries were checked and left alone, all deliberate: extended history reaches only the
  11 `pools_segments` methods (a classical fit takes a 1-D array and a foundation model conditions
  on one contiguous window, neither of which can span a gap); local methods refit per origin on the
  growing history, val included, because for them the fit *is* the forecast; and the foundation
  models pin context to the train length so it stays a fixed width across origins. One real
  inconsistency is left open as gh #104: MLP early-stops on a shuffled 10% of its training rows
  rather than the declared val split, because scikit-learn exposes no way to pass an explicit
  holdout. Every ML-family score is stale until the next pilot.
commits:
  - d14b492: "Use each boosting library's own defaults"
  - e5b7e15: "Lift MLP's iteration cap above where it actually converges"
  - b75f3da: "Correct a stale comment about what a global fit is handed"
---
id: 93
date: "2026-08-14"
title: "The last two training caps are gone, and the step budget is now counted in epochs"
description: |
  `MAX_TRAIN_WINDOWS = 15_000` and `DL_MAX_STEPS = 2_000` were the two hand-picked constants left
  in `eval.budget` after #95 and #97 removed every training-history cap, and they only made sense
  as a pair: dropping the window cap alone would have spread a fixed number of gradient steps over
  a pool up to 24x larger, reinstating entry 89's imbalance on a different axis. Both are removed
  (gh #101). The window cap bound on 5 of 17 (frame, case) pairs and never applied to the
  neuralforecast zoo at all, so removing it makes `torchdl` and `dl` consistent with `nf_models`
  rather than introducing a new regime. The step ceiling is replaced by `DL_MAX_EPOCHS = 150`
  passes over whichever window pool a fit actually has (`budget.dl_step_ceiling`), which is what
  makes it derived: views-per-window are now equal across cases instead of varying inversely with
  case size. The value follows the long-horizon literature with headroom — PatchTST (Nie et al.
  2023) trains for 100 epochs with early stopping; N-HiTS (Challu et al. 2023, Table A3) instead
  fixes 1,000 steps at the batch 256 and learning rate 1e-3 `torchdl` already uses, which is
  exactly the flat-step-count shape this benchmark rejected.

  Measured afterwards, the ceiling does not bind anywhere near current cases: N-HiTS early-stops
  at 300-450 steps and PatchTST at 300-650, against ceilings of 900 (`discharge_lowland_denmark`)
  to 87,900 (`surge_cuxhaven`). Fit cost stays flat rather than growing with the record — N-HiTS
  ~3s and PatchTST 6-22s across a 100x range in window count — because early stopping decides the
  step count and the per-step cost does not depend on pool size. That contradicts the expectation
  in #101 that DL fit cost would become proportional to record length, so no `fit_cost` scorecard
  rating changes on account of this work. Window building moved to a shared
  `methods.base.build_windows` that allocates float32 and fills it through a strided view: the
  largest pair (`sewerlevel_bellinge` extended, 244,565 windows) builds in 0.04s against the 1.36s
  a per-window Python loop took at that size, and holds one 188 MB copy instead of three.
commits:
  - 5b7fab2: "Add a shared window builder that keeps every window"
  - c8f9e4a: "Remove the last two training caps and budget training in epochs"
---
id: 92
date: "2026-08-13"
title: "The declared val split is now a real holdout instead of extra training data"
description: |
  The benchmark displayed a three-way train/val/test split while `eval.loop` concatenated train
  and val into one fit frame, so every method estimated parameters on train+val and the val
  boundary survived only as the offset locating where test begins; the methods that needed an
  early-stopping holdout each carved their own from the tail of that frame (20% of points in
  `nf_models`, 20% of windows in `torchdl`, none in DLinear). Nothing leaked into test, but the
  near-agreement between the declared val (17.6% of train+val) and `DL_VAL_FRACTION` (20%) rested
  on two unrelated constants, and a case declaring `split_fraction = [0.6, 0.9]` would have put a
  third of the frame the UI labels validation into training. `fit` now takes the val split as a
  required third argument and the eval loops separate three frames that used to be one:
  parameters come from train, val is the holdout the iteratively trained methods stop on, and
  predict still conditions on the whole pre-origin history — reading val is not estimating on it.
  The reference scales for MASE and the skill scores deliberately stay on that history, so the
  yardstick did not move with the fit frame. Every method's parameters now come from 70% of the
  record rather than 85%, which restales the pilot along with the rest of #98/#101.

  The declared val was left at its full 15% rather than trimmed to what early stopping strictly
  needs: a long validation period is the point, since it lets the stopping decision average over
  a span of months to years (`surge_cuxhaven` 670 disjoint 48-step origins over 3.7 years) instead
  of one narrow slice. The one place that cannot be expressed directly is a gap-segmented
  neuralforecast panel: `NeuralForecast.fit(df, val_size=V)` takes a single V, cuts the last V
  steps of *every* series and drops those left below `lookback + h`, with no per-series cut and no
  way to pass an explicit validation frame. So `budget.nf_val_size` picks the V leaving the most
  training windows, never above the declared val and never below four horizons: it takes the whole
  val split where the panel affords it (`discharge_lowland_denmark`, `waves_massachusetts`) and
  less where that buys segments, which keeps every history segment in all five multi-segment cases
  — `waves_oregon` trains on 9/9 segments and 96.9% of its windows against 5/9 and 68.0% under the
  old expression. Where V lands below the declared val, the remainder of that split is left out of
  the panel: unused, and trained on by nobody.
commits:
  - 522b229: "Pin the val split as a real holdout (failing test)"
  - e1c6931: "Reserve the declared val split from every method's fit"
  - fe6d18c: "Say what each split is actually used for"
---
id: 91
date: "2026-08-13"
title: "MLP's 609s fit was BLAS thread oversubscription, not the cost of direct_horizon"
description: |
  Issue #97 measured MLP's fit at 608.5s for 2,000 training points on a 48-step horizon and
  attributed it to `direct_horizon`, which trains one `MLPRegressor` per horizon step at an
  apparent ~12s each. The real cause was thread oversubscription: a minibatch matmul here is
  200 x ~15 features into a 64-wide layer, roughly 0.4 MFLOP, so spreading it across a dozen
  cores costs more in synchronisation than the arithmetic it saves — `direct_horizon` merely
  multiplied an already-inflated constant by H. One fit of identical data, seed and iteration
  count takes 37.3s unrestricted against 0.50s on one thread, and MLP's fit now runs under a
  one-thread BLAS limit (`surge_cuxhaven` 608.5s -> 29.8s). A shape sweep put the crossover
  where multi-threading starts paying at roughly batch >= 2,000 *and* a 1,024-wide layer *and*
  ~500 features simultaneously, so none of the three redesigns #97 proposed was needed. The cap
  also stabilises the measurement — four repeats of one fit ranged 2.72s to 0.38s unrestricted
  as the thread pool warmed, while one thread held 0.14s, which matters because `fit_cost` is
  meant to rest on measured `fit_seconds` (entry 89). With that fixed, MLP was remeasured up
  the ladder: it now reaches the full 42,073-point `sewerlevel_bellinge` record in 3.3 min, and
  on `surge_cuxhaven`'s 48-step horizon reaches 20,000 points in 124s, extrapolating to roughly
  20 min over that case's full 182,304 points. That is under Theta's 47-66 min, which entry 90
  already accepted as affordable for a once-per-case fit, so MLP's 20,000-point cap was removed
  too and no method carries one now. The whole calibration CSV was regenerated in the same
  pass, which also cleared a second defect: the previous file predated the script's
  throwaway-warmup fit, so each method's smallest rung was inflated by library import cost
  (the `LinearLags` 2,000-point rung read 1.53s against 0.05s at the same size, and AutoARIMA's
  first three rungs *decreased* with more data). Every MLP and MLP+exo number in the pilot is
  still stale pending a re-run. Two questions came out of this and are tracked separately:
  MLP's batch size is left at scikit-learn's `min(200, n)` default even though 32 scored better
  on held-out R2 at equal wall time (#98, changing it changes scores), and the declared val
  split is displayed as a split but folded into training by every method while the DL methods
  carve their own 20% holdout (#99).
commits:
  - c565e6b: "Cap BLAS threads during the MLP fit"
  - f342a87: "Add a --methods filter to the cap-calibration script"
  - 4b33f4d: "Remove MLP's training cap, the last one left after #95"
  - 069244c: "Commit compute time results."
---
id: 90
date: "2026-08-12"
title: "Removed the per-method training caps outright rather than equalising them"
description: |
  Measured how fit cost grows with training-history length for every method whose cost depends
  on it (`scripts/calibrate_train_caps.py`; curves in `docs/results/train_cap_calibration.csv`),
  to decide how to fix the unequal training budgets of entry 89. A shared cap turned out to be
  unnecessary: six methods read the entire 182,304-point `surge_cuxhaven` record in under 8
  seconds each, and the four expensive classical ones reach it in an extrapolated 7 min
  (AutoARIMA) to 47-66 min (Theta) per fit — affordable, since a fit happens once per case, not
  once per origin. So the caps were removed rather than equalised, leaving one ceiling in the
  new `bench/eval/budget.py` set far above any current record: a guard against a future
  oversized case, not a budget. An equal-fit-compute rule was considered first and rejected,
  because fixing the time budget would have pinned `fit_seconds` at that budget for every
  expensive method and destroyed the cost measurement the scorecards are meant to rest on.
  The deep-learning methods now train to convergence — a chronological 20% validation tail,
  patience of 5 checks, a 2,000-step ceiling as a safety net — instead of the fixed 500/600
  gradient steps that made a large case see each training window ~12x less often than a small
  one; every result row now records `train_points` and `train_steps`. MLP is the one exception,
  keeping its 20,000-point cap until #97 brings down its per-horizon-step cost, and every
  number in the current pilot is stale until the re-run.
commits:
  - 924f395: "Create scripts for calibrating training caps"
  - fb8f263: "Add measured fit-cost curves for the training-cap decision"
  - caa7e04: "Remove the classical and ML training-history caps"
  - 43210c9: "Train the DL methods to convergence instead of a fixed step count"
  - df0c40f: "Update the cap-calibration script for the post-#95 world"
---
id: 89
date: "2026-08-12"
title: "Auditing the method cards surfaced a training-budget problem that blocks the card fixes"
description: |
  Audited every method's `Handles` and `Scorecard` against the shipped results to check the
  rubric's claim that `fit_cost`/`inference_cost` are "anchored to the measured
  `fit_seconds`/`predict_seconds`" — they aren't: N-HiTS (5.8s) and iTransformer (6.0s) are rated
  High fit while the slower Theta (8.6s) is rated Low, and Moirai-2.0 is rated High inference at
  34ms per origin while methods 3x slower are rated Low. Chasing *why* fit time doesn't track
  training size showed that for the neuralforecast/torch methods it is a deterministic function
  of `lookback + h` (correlation 1.000) and completely independent of the number of training
  samples: `_MAX_STEPS = 500` (`nf_models.py`) and `max_steps = 600` (`torchdl.py`) fix the
  gradient-step count, so `wwtp_influent_n2o_350k` (2,594 points) and `waves_oregon` (20,735)
  both fit in ~140.7s and each available training window is seen roughly 12x more often on the
  small cases than the large ones. Separately, 13 methods truncate training history to a
  per-method cap, so on one case the methods are compared at very unequal data budgets — on
  `surge_cuxhaven` (182,304 points) TBATS sees 1.6% of the history, AutoARIMA 2.7%, the DL family
  11-16%, and the mlforecast family 100%. **Decision: fix the step budgets and sample caps and
  re-run the pilot before touching the scorecards**, since making a card consistent with a matrix
  whose columns were produced under unequal budgets would only lock in the distortion. Two
  smaller decisions came out of the same discussion and are deferred to that later card work: the
  cost fields become explicitly qualitative (drop the anchoring claim, keep the levels as
  declared judgement, guard them with a per-field monotonicity test rather than numeric bands),
  and a "scaling with history" card field was rejected — it discards the cost *level* an engineer
  asks for first, and for the 13 capped methods its slope is zero by construction.
commits: []
---
id: 88
date: "2026-07-30"
title: "Fixed MLP's divergence at the source, and added a Forecast strategy card to every method"
description: |
  Switched `MLPForecaster` from mlforecast's recursive strategy to its direct multi-horizon
  strategy (one model per horizon step, no feedback loop) — the `_BoundedRegressor` output clip
  added on 2026-07-27 treated the symptom (a miscalibrated step compounding through recursion),
  not the cause, and cost MLP the ability to extrapolate a trend. With the feedback loop gone,
  the clip is removed entirely and `capabilities.trend` reverts to `True`; fit cost moves to
  High, since this now trains `H` independent `MLPRegressor`s instead of one. Working through
  *why* this fixes it corrected a mental model: a classical ARIMA/ETS/TBATS forecast is
  recursive in the exact same sense as mlforecast's lag-feedback strategy (both substitute the
  model's own prior forecast for an unobserved future value) — they just don't diverge because
  their fitted coefficients are constrained to a stable region, the way a tree's leaf bound or
  Ridge's linearity does for those regressors under the same recursive strategy. That correction
  became a new `ForecastStrategy` field (Recursive/Direct/N-A), hand-declared per concrete
  method rather than hoisted onto shared base classes, and surfaced as a new block on every
  method's info card; foundation models are labeled Direct per this benchmark's one-call
  interface only, since their pretrained decoders' internal generation mechanism isn't verified
  against the pinned library versions. Existing MLP pilot results are stale pending a re-run.
commits:
  - "7107040": "Switch MLP to mlforecast's direct multi-horizon strategy"
  - "7d08804": "Add ForecastStrategy metadata field alongside Capabilities/Scorecard"
  - "ee39fcb": "Declare forecast_strategy on every concrete Forecaster"
  - "b6bb9cc": "Export forecast_strategy through the web data pipeline"
  - "dc779e4": "Add a Forecast strategy block to the method info card"
  - "961a732": "Extend method-scorecard guard tests to cover forecast_strategy"
  - "4bd6535": "Log the MLP direct-horizon fix and forecast_strategy taxonomy"
---
id: 87
date: "2026-07-29"
title: "MLP+exo existed but was never in the pilot run"
description: |
  `MLPExoForecaster` was added earlier today (see below) but `scripts/run_pilot.py`'s
  `build_methods()` never picked it up — the other three Tier 2 exo siblings
  (LinearLags, LightGBM, CatBoost) were added to that list, MLP was not. `methods.json`
  still showed an MLP+exo card because that generator reads the method class registry, not
  the results, so the card looked live even with zero backing scores. Added
  `MLPExoForecaster()` to `build_methods()`; a pilot re-run is needed before MLP+exo scores
  show up anywhere.
commits:
  - "1d553b0": "Including exo to MLP"
---
id: 86
date: "2026-07-29"
title: "Method cards get a Gaps capability, derived rather than hand-authored"
description: |
  Added a fifth Handles chip answering whether a method can use history from before a gap in
  the record. Tracing how gaps actually get handled showed it's centralized infrastructure
  (`curate.py` fills small ones, `_segment_extended`/`segment_arrays` split the record at big
  ones before any `fit` call), not something each method implements — the one place a real
  per-method distinction already exists is the existing `pools_segments` flag. Rather than
  hand-copying that value into each of the ~19 `Capabilities(...)` declarations (risking drift
  the moment a subclass overrides `pools_segments` without touching `capabilities`, as
  `ITransformerForecaster` already does), `gaps` is computed from `pools_segments` once in
  `webdata.py`'s export step, guarded by a new test that would catch exactly that drift.
commits:
  - "c7ac4a4": "Derive a gaps capability from pools_segments, not a hand-authored one"
  - "2b02844": "Add Gaps chip to the method card's Handles row"
  - "6122294": "Regenerate methods.json with the new gaps capability"
  - "cd9a7e7": "Log the gaps capability decision in the research log and its graph"
---
id: 85
date: "2026-07-29"
title: "Gave MLP a `+exo` variant"
description: |
  MLP already had the full covariate pipeline through `_WindowedMLBase` (the same lag +
  calendar feature machinery LightGBM/LinearLags/CatBoost use), but no `+exo` subclass had
  been registered — an oversight from when those three got theirs. Added
  `MLPExoForecaster`, following the same one-line pattern (flip `use_exogenous = True`,
  inherit everything else). This was Tier 1 of a larger plan to extend covariate support to
  DLinear, N-HiTS, and PatchTST, which are standalone implementations and need real wiring
  work rather than a flag flip — those stay univariate for now.
commits:
  - "9d6bea2": "Add MLP+exo variant (Tier 1 of exogenous expansion)"
---
id: 84
date: "2026-07-29"
title: "TBATS now fits once too, via genuine AIC selection instead of a fixed config"
description: |
  TBATS's scores on `discharge_saugeen` were far worse than every other classical method
  (MASE 8.76, RMSE 1105.9) — traced to the fixed-config `statsforecast.models.TBATS` class
  this benchmark used, which hardcodes an undamped trend and never searches Box-Cox/trend/
  damped-trend/ARMA-error combinations. On a log-space fit (near-universal for heavy-tailed
  discharge data), that forced undamped trend extrapolates a rising origin's local trend
  indefinitely; one origin (1973-03-09) forecast up to 16,761 against an actual peak of ~210.
  Switched TBATS to genuine AIC-based selection via `tbats_selection` (what `AutoTBATS.fit()`
  calls internally), fit once per the `is_global` protocol and rolled forward via
  `calcTBATSFaster`/`tbats_forecast` — the same internal recursion `AutoTBATS.predict()` uses
  — verified bit-exact against `AutoTBATS`'s own output over the same window. This differs
  from the fit-once TBATS rejected in the entry below: that attempt substituted a simplified
  model under the TBATS name, while this one runs the library's own genuinely selected model,
  just rolled forward by hand since statsforecast never wired up a `forward()` for it.
  Re-running `discharge_saugeen`'s 40-origin backtest through the real eval pipeline: MASE
  8.76 → 0.97, RMSE 1105.9 → 46.75, no more per-origin blowups. TBATS's existing results
  across every case are now stale and need a pilot re-run.
commits:
  - "c684594": "Make TBATS fit once and roll forward, using genuine AIC model selection"
  - "c0cf951": "Add TBATS to the fit-once test suite, replacing the stays-local pin"
  - "98da600": "Correct app copy now that TBATS fits once like the rest of classical"
  - "38492a1": "Log the TBATS fit-once fix in the research log and its graph"
  - "fe1a5c6": "Correct TBATS fix numbers to the real eval pipeline's own run"
---
id: 83
date: "2026-07-29"
title: "Method cards now carry a capability set and an operational scorecard"
description: |
  The method cards described each method in prose only, so nothing on them let you compare two
  methods at a glance. Every method now declares two Pydantic objects beside its existing
  metadata: `Capabilities`, four booleans for what structure it can represent (trend,
  seasonality, multiple seasonality, exogenous support), and `Scorecard`, six Low/Medium/High
  ratings for what it costs to use (history needed, interpretability, fit cost, inference cost,
  tuning effort, robustness to noise). Splitting them matters: a graded rating answers "how
  much" and a boolean answers "can it at all", and "Medium seasonality" is not a real thing.

  Three fields from the first sketch were reshaped for the same reason. Implementation effort
  was dropped — all 32 methods are thin library wrappers, so the rating would have been
  fiction — and replaced with tuning effort, which genuinely varies. A single computational
  cost hid the fit-once versus refit-per-origin distinction, so it was split into fit cost and
  inference cost, both anchored to the `fit_seconds`/`predict_seconds` the eval loop already
  records; that immediately separates Theta and MSTL (cheap to fit, expensive per origin) from
  TBATS (the mirror image). Small/large dataset support collapsed into one history-need field,
  since whether a method can exploit a long record is already *measured* by the
  extended-history matrix.

  The ratings are hand-authored judgement, not derived from results, and the app says so in an
  info modal on each card. A method and its `+exo` variant share one declaration, since `+exo`
  records which configuration a column ran rather than a different estimator.
commits:
  - "aae01bd": "Add Capabilities and Scorecard models for method self-description"
  - "a5d1431": "Declare capabilities and scorecards for the baselines and ensembles"
  - "ceba177": "Declare capabilities and scorecards for the classical methods"
  - "3aef62d": "Declare capabilities and scorecards for the windowed ML methods"
  - "30909bc": "Declare capabilities and scorecards for the deep-learning methods"
  - "0033f47": "Declare capabilities and scorecards for the foundation models"
  - "b7570e7": "Export capabilities and scorecard to methods.json, with guard tests"
  - "93f3a1d": "Add web types, display config and level colour tokens for the scorecard"
  - "e36954a": "Render the Handles row and scorecard meters on the method cards"
  - "c1457be": "Log the method scorecard work and add its research-graph node"
  - "2d1250e": "Drop the level text from the scorecard rows"
  - "e8b29cb": "Updated jsons. Fixing integer number test."
  - "9bce6fa": "Removing family from method card."
---
id: 82
date: "2026-07-29"
title: "Converted the classical methods to fit once and roll forward, except TBATS"
description: |
  The classical methods used to refit from scratch at every rolling origin while the ML, DL and
  foundation models were fit once and rolled forward, which confounds method quality with how
  often a method is allowed to re-estimate. AutoARIMA, AutoETS, Theta and MSTL-ETS/MSTL-ARIMA are
  now `is_global=True` via statsforecast's own `forward(y, h)`, which applies an already-fitted
  model to a new series without calling an optimizer — `AutoARIMA.forward` reproduces
  statsforecast's native forecast exactly, so these are still the library's estimators rather than
  reimplementations of them. Two deterministic pieces are re-derived per origin (Theta's
  seasonality test and decomposition, MSTL's STL), which is no different from the ML methods
  recomputing their calendar features and leaves the seasonal shape current.

  TBATS stays local: statsforecast couples its automatic component search to every `.fit()` and
  exposes no `forward`, so the only fit-once option was to substitute a simplified custom
  state-space model under the TBATS name — which was tried, and rejected as making the column stop
  meaning what it says. Its protocol difference is now flagged in the app and in the method's own
  `limitations`. The roll-forward is deliberately **not** capped to the fit window: `predict`
  conditions on the whole history it is handed, since truncating would substitute an arbitrary
  re-initialization for the state the fitted model implies. That has a real cost, which the
  benchmark reports rather than engineers around — on a 200,000-point history one origin costs
  ~0.03s for AutoARIMA but ~110s for Theta (superlinear) and ~28s for MSTL, so a rolling run over
  the longest records is dominated by their `predict_seconds`, not their fit. Existing classical
  results are stale and need a pilot re-run, and that re-run will be substantially slower than the
  last one for `surge_cuxhaven` in particular.
commits:
  - "86aa455": "Convert AutoARIMA and AutoETS to fit-once, is_global forecasters"
  - "719b4dd": "Convert Theta to fit-once, is_global forecaster"
  - "d1867d6": "Reimplement TBATS as a custom fit-once state-space model"
  - "46221fb": "Convert MSTL-ETS/MSTL-ARIMA to fit-once, is_global forecasters"
  - "9595914": "Update docs/UI text for classical methods now fitting once"
  - "e2de2e4": "Add tests for classical methods' fit-once conversion"
  - "54e0f86": "Log the classical fit-once conversion in RESEARCH_LOG and the graph"
  - "53be293": "Choose TBATS/AutoETS positivity-dependent specs from the whole record"
  - "6750b8e": "Test TBATS/AutoETS specs against history before the search window"
  - "e813708": "Roll the classical methods forward with statsforecast's own forward()"
  - "0f24e6d": "Read local-vs-global off is_global instead of the method's family"
  - "9d7b400": "Revise today's log entry for the final fit-once scope"
  - "4e68f76": "Document the fit-once protocol in the Benchmark Design modal"
---
id: 81
date: "2026-07-28"
title: "Flagged local methods in the case panel as refit at every origin"
description: |
  The case panel already warned when a selected method is a zero-shot foundation model, since it
  never fits the train/val split shown in the split bar — but gave no equivalent warning for local
  methods (baseline, classical, ensemble), whose training routine is just as different from what a
  user familiar with global-model workflows would expect: they have no single trained model at all,
  refitting from scratch at every rolling forecast origin instead of fitting once on train/val like
  the ML/DL families. Added a second inline note, styled and worded to match the existing
  foundation-model one, that appears whenever a selected method's family is baseline, classical, or
  ensemble. Kept it as a separate banner rather than merging with the foundation-model note, since
  the two families fail the "trained once on train/val" assumption for different reasons.
commits:
  - "cd68a41": "Extract shared method-to-family lookup in CasePanel"
  - "433793c": "Warn when a selected method is refit at every origin"
  - "9826363": "Log the local-method case-panel caveat"
---
id: 80
date: "2026-07-28"
title: "Fixed FM context to a single pre-test window, no longer growing per origin"
description: |
  The revert earlier today put the FMs back on full history capped at the model max, but because
  the rolling-origin history grows at each origin, that context grew with it — harmless for the
  three eager models, but it forced TimesFM (whose `compile()` bakes a static input shape) to
  recompile a variant at every origin for any case with less pre-test history than its cap.
  Fixed all four to a single window whose width is set once from the pre-test history (train+val)
  at fit time and capped per model; predict still slices the most recent points, so the window
  slides forward to each origin without growing. This is safe because the eval loop fits global
  methods once on train+val and the first origin conditions on exactly that block, so the fixed
  width never exceeds any origin's available history — TimesFM stays NaN-free and compiles once
  per case. Only short series (train+val below the cap) change from the revert; long series
  already sat at the cap. FM results are stale and need a re-run.
commits:
  - "2c8ba13": "Fix FM context to a pre-test window recorded at fit time"
  - "1a94b00": "Log the fixed pre-test window FM context change"
---
id: 79
date: "2026-07-28"
title: "Reverted foundation-model context to full history, capped at model max"
description: |
  The seasonal-period context rule from earlier today (2 x the case's longest declared seasonal
  cycle) drastically hurt foundation-model accuracy: `2 x max(seasonalities)` is only tens to a
  few hundred points for most cases, far below the 1024–2048 of history these models were built
  to use. "Two cycles" is a rule of thumb for detecting a period, not for how much history a
  pretrained model should condition on, so it threw away the FMs' main strength. Switched all
  four zero-shot models (Chronos-Bolt, Chronos-2, TimesFM-2.5, Moirai-2.0) back to the full
  available history capped at each model's real limit, and deleted the seasonal-sizing helpers
  rather than leave a dead fallback. This is safe for TimesFM now: it sizes its per-origin
  compiled shape to the available history, so the all-NaN bug that motivated the seasonal rule
  can't recur — which also means that earlier "model max hurt 21 of 28 comparisons" verdict was
  contaminated by the NaN bug and is no longer trustworthy. Prior FM results are stale; re-run
  the pilot for a clean comparison against the original moderate caps.
commits:
  - "fbc5750": "Condition zero-shot FMs on full history capped at model max"
  - "5b2d969": "Log the revert to full-history FM context"
---
id: 78
date: "2026-07-28"
title: "Split training-time measurement into fit and predict, averaged per rolling origin"
description: |
  Audited why the web app's training-time heatmap showed classical methods (AutoARIMA, Theta,
  TBATS) as far more expensive than zero-shot foundation models, which read as counterintuitive.
  The single `seconds` field turned out to conflate fit and predict time, and — because
  classical/naive methods refit at every rolling origin while global methods (including zero-shot
  FMs, whose fit is a no-op) fit once — implicitly multiplied only the classical numbers by
  `n_origins`. Split the eval loop's timing into `fit_seconds`/`predict_seconds`, each averaged per
  origin so the number reads as "cost of one origin's cycle" regardless of how many origins a run
  used; the heatmap now shows their sum (cost to produce one forecast) by default, with the split
  available on hover. TimesFM-2.5's current ~149s average is entirely predict time despite
  `fit=pass` — likely the known TimesFM slowness noted in the fixed-fm commits — so a fit-only
  metric would have hidden it as free, which is why both numbers are kept rather than fit alone.
  `results/pilot_metrics.parquet`/`docs/results/scores.parquet` need a re-run before the new fields
  are populated; the old `seconds` column no longer exists.
commits:
  - "c2b3f05": "Split eval-loop timing into fit_seconds and predict_seconds"
  - "cf28aab": "Update print_results.py for the fit_seconds/predict_seconds split"
  - "3f2b011": "Rename ScoreRow.seconds to fit_seconds/predict_seconds; show cost per forecast"
  - "3c55482": "Show fit/predict breakdown on hover in the training-time overlay"
  - "090ca41": "Explain fit/predict semantics in training-time help copy"
  - "0359dd9": "Log the fit/predict timing split decision"
  - "52c3141": "run with fixed fm (TimesFM is still buggy)"
  - "f0ae975": "Export new run after timesfm fix"
  - "b1d5487": "exporting json files"
---
id: 77
date: "2026-07-28"
title: "Foundation-model context length now tracks seasonal periods, not model max"
description: |
  Re-scored the merged pilot results from yesterday's context-length fix and found it wasn't
  clearly better: of 28 case/method comparisons across Chronos-2, TimesFM-2.5, and Moirai-2.0,
  only 7 improved and 21 got worse. Worse, TimesFM-2.5 came back all-NaN on 8 of 12 cases — its
  `compile()` step bakes in a fixed input shape, and the `timesfm` library returns NaN whenever
  the actual input is shorter than that shape by a full patch (32 steps) or more, confirmed by
  bisection and independent of the data itself; yesterday's fix made this reachable by pushing
  the compiled shape up to the model's full 16,128-step capacity, well past what most cases'
  early rolling origins have available. Switched all four zero-shot models (Chronos-Bolt,
  Chronos-2, TimesFM-2.5, Moirai-2.0) from "as much history as the model can take" to "2 periods
  of the case's longest declared seasonal cycle, capped at the model's real limit" — a standard
  rule of thumb rather than a tuned value, and it also clears the TimesFM bug since every case's
  seasonal window now sits well below its available history at every rolling origin. Also removed
  the `input_window` field from the `Case` schema and every `case.toml`: it turned out to be dead,
  never read anywhere outside the schema declaration itself. Prior fm results (including
  yesterday's) are stale and need re-running.
commits:
  - "4428aff": "Size zero-shot FM context to 2 seasonal periods, not model max"
  - "cb0e7dd": "Restructure TimesFM loading to fix NaN forecasts on short-history cases"
  - "65b7db7": "Remove dead input_window field from Case schema"
  - "b094a7e": "Log the seasonal-period context-length change and TimesFM NaN fix"
---
id: 76
date: "2026-07-27"
title: "Stopped hardcoding foundation-model context lengths; read them from the models instead"
description: |
  Chronos-2 and TimesFM-2.5's context caps in `fm.py` turned out to be far below what the
  checkpoints actually support: Chronos-2 was capped at 2048 when `amazon/chronos-2` supports
  8192, and TimesFM-2.5 was capped at 1024 when `google/timesfm-2.5-200m-pytorch` supports 16384.
  Both wrappers now read the real value off the loaded model (`pipe.model_context_length`,
  `model.model.config.context_limit`) instead of a guessed constant, so a checkpoint swap can't
  silently go stale again. Moirai-2.0's old cap (1680) turned out to have no basis at all — it
  was just the length of an unrelated demo dataset in Salesforce's example notebook — and
  Moirai-2.0 has no published trained max context length to read instead, so it now uses the full
  available history capped at the loaded module's position-embedding cache size (`max_seq_len *
  patch_size` = 8192), the one real architectural limit that exists for this checkpoint.
  Chronos-Bolt was already correct (2048) and untouched. Verified with a pilot spot-check on
  `discharge_saugeen` (23,741 rows, well over the old caps) — all seven fm methods, including the
  `+exo` variants, ran and returned finite forecasts; scores are expected to shift for Chronos-2,
  TimesFM-2.5, and Moirai-2.0 since they now condition on meaningfully more history, so prior fm
  results are stale and worth re-running.
commits:
  - "da2ae3c": "Read Chronos-2's real context length instead of hardcoding it"
  - "b1972d1": "Read TimesFM-2.5's real context limit instead of hardcoding it"
  - "a57a10f": "Use full history for Moirai-2.0 context, capped at cache size"
  - "0035f68": "Log the foundation-model context-length fix"
---
id: 75
date: "2026-07-27"
title: "Flagged foundation models in the case panel as not using the train/val split"
description: |
  The case panel's train/val/test split bar and training-tail plot assume every method fits on
  the train split, which is false for foundation models (Chronos-Bolt, Chronos-2, TimesFM,
  Moirai-2): they forecast zero-shot from recent context, with no case-specific fit at all.
  That caveat already existed as prose in each method's collapsed "Limitations" card, but
  nothing tied it to the split visualization itself, so selecting a foundation model and
  looking at the split bar was misleading. Added an inline note to `CasePanel` that appears
  only when a selected method's `family` is `fm`, reusing that existing classification rather
  than adding a new flag. If a fine-tuned FM variant is added later (per the plan's "fine-tuning
  round"), this note's `family === 'fm'` check will need revisiting, since it would then be
  wrong to say that variant skips the train split.
commits:
  - "3fb251d": "Including warning sign for train/test split and fm models."
---
id: 74
date: "2026-07-27"
title: "Bounded MLP's recursive forecast to stop it diverging on smooth series"
description: |
  MLP's MASE on `groundwater_yorktown` had blown up to 314.65 (RMSE 1383 m against a target
  that only spans 1.45–5.47 m). Traced it to `mlforecast`'s recursive multi-step strategy,
  shared by MLP with LightGBM/CatBoost/Ridge: each step's prediction feeds straight back in as
  the next step's lag input, and `MLPRegressor`'s unbounded output layer let one early
  miscalibration compound over the 30-step horizon — worst on groundwater because it's an
  unusually smooth, slow-moving series with long lags (90, 365) baked into the features, giving
  the recursion many steps to drift. Fixed by wrapping only the MLP regressor in a small
  estimator (`_BoundedRegressor` in `ml.py`) that clips each step's output to the training
  window's own range plus 4 standard deviations — applied on every recursive step, so drift
  can't build up. LightGBM/CatBoost/Ridge are untouched. `MLPForecaster` had no test coverage at
  all before this; added a regression test on `groundwater_yorktown` plus a smoke test. Scores
  need re-running to confirm the corrected numbers, and `docs/RESULTS.md`'s "MLP is the least
  robust method" write-up cites older, milder numbers that will need updating once they are.
commits:
  - "f7552df": "Bound MLPForecaster's recursive predictions to stop runaway divergence"
  - "122bf8f": "Fix _BoundedRegressor's sklearn Pipeline fitted-check"
  - "361f732": "Add MLP forecaster coverage, incl. a regression test for the bounded fix"
  - "d793a25": "Log the MLP recursive-divergence fix in the research log"
---
id: 73
date: "2026-07-27"
title: "One color scheme for every metric in the matrix heatmap, and it now adapts to filtering"
description: |
  The matrix heatmap used two different color schemes depending on the metric: a green-to-red
  diverging scale for MASE/skill/NSE/KGE, but a plain white-to-red scale for MAE/RMSE/MAPE/sMAPE
  that never reached green even at their best value in view. Switched those four metrics (and the
  fallback used for any unrecognized metric) to the same green-to-red scale, auto-ranged over
  whatever's currently on screen rather than pinned to a fixed center — unlike MASE or the skill
  metrics, MAE/RMSE/MAPE/sMAPE have no fixed "no better than X" reference point, so there's no
  absolute value to center on. Separately, MASE/skill/NSE/KGE's color range turned out to be pinned
  to their fixed `[vlo, vhi]` cap regardless of what was actually filtered into view — now the range
  shrinks to the visible cells' own min/max, still clamped into `[vlo, vhi]` so a catastrophic
  outlier left on screen still saturates to the edge color instead of stretching the scale for
  everything else. `web/src/data/metricConfig.ts`, `web/src/components/Matrix/useMatrixFigure.ts`,
  and the Python twin `src/tsforecastbench/bench/viz/matrix.py` were updated to keep the color logic
  in sync (the range-adaptation fix is web-only, since the Python script renders one static dataset
  per image rather than reflecting interactive filtering).
commits:
  - "2b5562a": "Unify heatmap color scheme: green good, red bad for all metrics"
  - "19d33da": "Log the heatmap color-scheme unification in the research log"
  - "1260cef": "Adapt heatmap colour range to filtered/visible cells"
  - "3826b3d": "Update research log entry to cover the colour-range adaptation fix"
---
id: 72
date: "2026-07-23"
title: "Split the pilot run across machines: tagged partial runs + a merge step"
description: |
  `torchdl.py` (N-HiTS, PatchTST) and `nf_models.py` (TFT, iTransformer, TimesNet) are the only
  method families that actually benefit from a GPU — `dl.py`'s `DLinearForecaster` turned out to
  be a plain NumPy/scikit-learn Ridge fit with no GPU involvement, so it stays in the CPU-only
  baseline run. The plan is to run those two families on a Lightning AI Studio's GPU tier and
  everything else on a local machine, then combine the results — but `run_pilot.py` always
  overwrote the single `results/pilot_metrics.parquet` (and its `docs/results/scores.parquet`
  mirror), which would have clobbered whichever run finished last instead of merging with it.
  Added `--tag` to `run_pilot.py` (writes a partial result to the git-tracked
  `docs/results/partial/` instead of the gitignored `results/`, so it survives a push/pull between
  machines) and `--skip-baseline` (skip the hardcoded baseline/ML methods, so a GPU-only machine
  doesn't waste time re-fitting methods that gain nothing from a GPU), extracted the
  final-artifact-writing logic into `src/tsforecastbench/bench/report.py`, and added
  `scripts/merge_pilot_results.py` to recombine tagged files (deduping on case/method/metric) back
  into the merged parquet the web export already reads. Next: run
  `--tag gpu_dl_nf --skip-baseline --with-dl --with-nf` on the Lightning Studio, then merge with a
  local `--tag local` run that uses the same `--cases`/`--origins`.
commits:
  - "7b9333c": "fast run with dl"
  - "2f74f7c": "Run of 40 origins with gpu models"
  - "aaaed04": "Run of 40 origins with baselines"
  - "18ed673": "commit combined scores"
  - "04588c9": "Commit json files"
  - "16032a6": "Changes to split running pilot by methods"
---
id: 71
date: "2026-07-21"
title: "Fixed the matrix heatmap going stale when leaving NSE/KGE for another metric"
description: |
  Selecting NSE or KGE, then switching to a `Skill_vs_*` metric, left the heatmap showing the old
  NSE/KGE cells instead of updating. Same root cause as the training-time entry just below: Plotly.js
  fails to repaint a heatmap via `react()` once the matrix is down to 3 rows or fewer, and NSE/KGE
  always narrow the matrix to exactly that many rows (the flow/level cases that declare them).
  Rather than another disabled-control workaround, `MatrixGraph` now gets a `key` in `App.tsx` that
  changes whenever the matrix is in that danger zone — tied to the metric while short, a fixed value
  otherwise — forcing React to unmount and remount the component instead of asking Plotly to update
  in place. This only remounts on a metric change that enters, leaves, or moves within the short-row
  zone; switching between two ordinary (4+ row) metrics is untouched.
commits:
  - "dd0472b": "Fixing bug for metric switch in app."
---
id: 70
date: "2026-07-21"
title: "Training-time overlay disabled for short matrices (Plotly bug, no in-code fix found)"
description: |
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
commits:
  - "551a9d9": "Log training-time + NSE/KGE overlay as a known Plotly.js limitation"
  - "d49056f": "Disable training-time overlay for matrices too short to render it"
  - "3e678b1": "Update training-time overlay log entry with the row-count gating fix"
---
id: 69
date: "2026-07-21"
title: "Domain controls now follow the matrix's own NSE/KGE narrowing"
description: |
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
commits:
  - "a836c53": "Add domainsEligibleForMetric selector"
  - "22d1625": "Add disabled-domain support to DomainChecklist"
  - "341885f": "Run RTL cleanup after each test"
  - "e06f178": "Gate domain checkboxes and close case panel on domain-specific metric"
  - "07107df": "Log domain-checkbox/case-panel gating and RTL cleanup fix"
---
id: 68
date: "2026-07-21"
title: "Added a seasonal-pattern figure to the case panel"
description: |
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
commits:
  - "6b347f4": "Add seasonal-pattern data to the case export"
  - "d5e5349": "Add SeasonalPatternPlot component and SeasonalPattern type"
  - "db313b6": "Wire seasonal-pattern figure into the case panel"
  - "4de609d": "Log research entry for the seasonal-pattern figure"
  - "a433f3c": "Detrend the seasonal-pattern figure before plotting"
  - "af251dd": "Note detrending in the seasonal-pattern caption"
  - "48deb77": "Fold detrending decision into today's seasonal-pattern log entry"
  - "ac6791d": "Give subplot x-ticks room in the seasonal-pattern figure"
  - "83788f1": "Explain the seasonal-pattern figure's smoothed line and encourage zooming"
  - "94c342a": "Add subcaption"
  - "c541ec0": "Label seasonal-pattern subplot x-axes with \"# of steps\""
  - "10f8821": "Adding xlabel to seasonality plots"
---
id: 67
date: "2026-07-21"
title: "Moved app instructions from a root markdown file into the app itself"
description: |
  `APP_INSTRUCTIONS.md` used to sit at the repo root, disconnected from the live site — nobody
  browsing the app would think to look for it there. Its content now lives in
  `web/src/data/appInstructions.ts` and shows up behind a grey info icon next to the page title,
  in a scrollable modal styled like the app's other panels. The root file is gone; anyone updating
  the instructions should edit the new module instead.
commits:
  - "51f0ba8": "Move app instructions from repo root into web/src/data"
  - "121d413": "Add InstructionsModal component"
  - "ad2951c": "Show the instructions modal next to the page title"
  - "bf1c1ff": "Log research entry for the app instructions move"
  - "c525c3c": "Thin out the info icon and title the instructions modal"
---
id: 66
date: "2026-07-21"
title: "Removed the legacy Dash app now that `web/` covers everything it did"
description: |
  Deleted `app/` (the original Dash results-matrix viewer), now that `web/` has replicated all of
  its functionality. `scripts/export_web_data.py`, which generates `web/`'s static JSON, had been
  importing its data-loading and forecast-curation code straight out of `app/data.py` and
  `app/snapshot.py`; that code moved to a new `src/tsforecastbench/webdata.py` module so the export
  script keeps one shared source of truth without depending on deleted files. The Plotly/Dash
  rendering code in the old app had no other callers and was dropped along with it. The `dash`
  dependency (and the now-unused `app` extra in `pyproject.toml`) is gone; `plotly` stays, since
  `notebooks` still needs it. Tagged the last commit with `app/` intact as `legacy-dash-app` in case
  anyone wants to run the old version.
commits:
  - "a9eb46c": "Extract shared data/curation logic into tsforecastbench.webdata"
  - "4c32615": "Drop the app extra now that the legacy Dash app is gone"
  - "c5fa176": "Update docs to stop referencing the removed app/"
  - "6b7ce9c": "Log research entry for legacy Dash app removal"
---
id: 65
date: "2026-07-20"
title: "Fixed heatmap bleeding over the sidebar when the case panel is dragged wide"
description: |
  Dragging the right-hand case detail panel to its maximum width could make the heatmap paint over
  the left control sidebar instead of shrinking to fit. The heatmap has a fixed pixel width based on
  the number of methods shown, and the center column that holds it had no `overflow` set, so once
  that column got squeezed narrower than the heatmap, the excess width spilled left and covered the
  sidebar (later elements in the page paint over earlier ones by default). Added `overflow-x: auto`
  to the center column so the heatmap now scrolls within its own column instead of bleeding over
  neighboring panels.
commits:
  - "e34c339": "Clipping heatmap under controls"
---
id: 64
date: "2026-07-20"
title: "Sidebar closed by default, added a hint for first-time visitors"
description: |
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
commits:
  - "d0b4a4b": "Close control panel by default and fix toggle button clipping"
  - "6eee149": "Add empty-state hint pointing at the heatmap"
  - "858196e": "Log research entry for sidebar/hint UX changes"
  - "c151332": "Add hint pointing at the controls-panel toggle button"
---
id: 63
date: "2026-07-20"
title: "Moved training time from the right panel to a press-and-hold heatmap overlay"
description: |
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
commits:
  - "1e5a9a6": "Fixing drag handle right panel"
  - "e272313": "Remove training time from right panel, keep sample counts"
  - "c7f89b4": "Add press-and-hold hook and training-time button"
  - "05e4265": "Add trainingTimeOverlay to app state"
  - "b371bcf": "Generalize pivotValues with an optional value accessor"
  - "ebe182b": "Add TRAINING_TIME_CONFIG display config"
  - "ed7bc8b": "Wire the training-time overlay into the heatmap and control panel"
  - "0ada92c": "Log the training-time overlay decision"
  - "411ac60": "Fix training-time overlay: integers, stable grid, log color scale"
  - "40264de": "Fold overlay fixes into today's research log entry"
  - "56c88f7": "Drop the \"s\" suffix from training-time cell text"
  - "c160e2e": "Drop the \"s\" suffix from colorbar tick labels too"
  - "6ddd715": "Improve hold button description."
  - "c71a332": "Cleaning right panel"
  - "9cc435d": "Allow holding space bar to overlay training time"
  - "82842ca": "Prevent space-release from clicking the focused button"
  - "b990b3b": "Stop space-hold from scrolling the page on OS key-repeat"
  - "035bdb4": "Add unit/scale/log-color fields to metric config"
  - "0bdf54c": "Fix colorbar title line breaks and add log color scale for heavy-tailed metrics"
  - "17243bf": "Add tests for colorbar titles and log color scale"
  - "084d88d": "Correct Skill vs * colorbar title"
  - "17e941a": "Fix clipped colorbar unit line and adapt log-tick decimals to magnitude"
  - "23446c5": "Rename training-time colorbar title to \"Train time [seconds]\""
  - "ffac5ff": "Fix train time colorbar title"
  - "915d906": "Switch training-time overlay to greyscale, with light text on dark cells"
  - "e633079": "Keep hold-space-for-training-time working with Controls panel hidden"
---
id: 62
date: "2026-07-19"
title: "Fixed leap-year bug in the 365-day position-in-cycle path"
description: |
  `_dayofyear_365` (used by `position_in_cycle` for the annual cycle, which backs Climatology,
  SeasonalNaive-via-ensemble, and any ML feature keyed on a 365-day period) called
  `idx.is_leap_year.to_numpy(...)`, but `DatetimeIndex.is_leap_year` already returns a plain numpy
  array rather than a pandas object, so the `.to_numpy()` call raised `AttributeError` on every
  call. This broke all baseline/ensemble tests on cases with a declared 365-day seasonality plus
  the dedicated leap-year boundary test.

  **Decision.** Wrap the result in `np.asarray(...)` instead of calling `.to_numpy()`. No behavior
  change beyond fixing the crash — full suite passes again.
commits:
  - "ed101c2": "Accounting for leap years in climatology models"
  - "a5e1fc0": "Including test for 365 day cycles"
  - "9e72704": "Fix tests"
---
id: 61
date: "2026-07-19"
title: "Interactive decision graph for the research log"
description: |
  This log had grown to enough entries that cross-references only existing as prose made related
  decisions hard to trace without reading the whole file. Added `research_log/resources/research_graph.html`, a
  self-contained D3 force-directed graph — one node per log entry, hover for a summary, click for
  full detail plus a list of connections, drag to pin nodes, pan/zoom. Node and edge data (relation
  types: supersedes/fixes-bug-in/depends-on/relates-to) were hand-extracted from the log's own
  cross-reference prose in one pass and embedded directly in the file, so it opens with no server
  or fetch; CLAUDE.md's research-log section now asks for a matching node whenever a new entry is
  added. Categories are capped at four (methodology, app & tooling, bug fix, project & data) rather
  than a finer split, since a force layout can place any two categories next to each other and the
  project's palette validator fails a 6-color categorical set under that all-pairs condition.
commits:
  - "ad1bf7f": "Add interactive research-log decision graph"
  - "791c944": "Link the new decision graph from docs and CLAUDE.md"
  - "dfe373c": "Use directional relation labels in the connections panel"
  - "2b01966": "Log the new decision-graph tool in the research log"
  - "33a46fa": "Rebuild the decision graph against this branch's actual log"
  - "2dcda67": "Improve research graph"
  - "aaee2fd": "Including list view in research graph"
  - "4d115c5": "Fix research graph and commiting latest results."
  - "cf4a580": "Fix research graph"
---
id: 60
date: "2026-07-19"
title: "Combined the EDA and React Pages workflows into one"
description: |
  GitHub Pages only lets one workflow deploy per repo, so the existing EDA-only
  `pages.yml` couldn't run alongside a separate workflow for the new React results-matrix
  app in `web/` without one clobbering the other. Merged them into a single workflow that
  builds both sites and copies the React build to the site root with the EDA browser under
  `/eda/`. Pushing the change surfaced a separate problem: the git credential in use came
  from the `gh` CLI's login, which lacked the `workflow` scope GitHub requires for editing
  files under `.github/workflows/`, so `gh auth refresh -h github.com -s workflow` was
  needed before the push could go through.
commits:
  - "fe22848": "Consolidate EDA and React Pages workflows into one deployment"
  - "1605e84": "Log the Pages workflow consolidation and auth fix"
---
id: 59
date: "2026-07-19"
title: "Replaced marker descriptions with a visual swatch"
description: |
  The "Global methods" and "Exogenous inputs" sidebar sections described their corner-triangle
  markers in words ("marked with a corner triangle", "marks it with a teal corner"). Added a
  `MarkerSwatch` component that draws a small square with the actual colored triangle in the
  matching corner, reusing the same `cornerTrianglePath` geometry the matrix overlay uses, so the
  preview matches the real marker exactly. The prose was trimmed to drop the now-redundant
  shape/color description, keeping only the part explaining when each marker appears.
commits:
  - "189a69c": "Show marker swatches instead of describing them in words"
---
id: 58
date: "2026-07-19"
title: "Moved the method-info card title to the top when expanded"
description: |
  The method-info cards showed their method name in a title bar that stayed at the bottom of the
  card in both states, so an expanded card read top-to-bottom with the resize handle first and the
  name last, below the content. Reworked `MethodInfoCard` to render the same title bar first when
  expanded, so the method name sits at the top of the open card, with the resize handle and content
  below it. Collapsed cards are unaffected — the title bar is still the only thing rendered, in the
  same position as before.
commits:
  - "e3b3477": "Improving method cards"
---
id: 57
date: "2026-07-16"
title: "Fixed method-info card height-drag stopping after one pixel"
description: |
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
commits:
  - "178bffc": "Fix method-info card drag stopping after one pixel of movement"
  - "30117de": "Log the drag-resize bug fix in the research log"
---
id: 56
date: "2026-07-16"
title: "Split the method-info strip into per-method collapsible, resizable cards"
description: |
  The bottom method-info strip (up to `MAX_OVERLAY_METHODS` = 3 cards) used to be one flex-wrap
  container where cards grew to fill space, which read as a single blended block rather than
  distinct per-method panels. Reworked it into fixed-width (320px) cards docked side by side, each
  independently collapsible down to just its title bar and resizable in height (120–520px) by
  dragging a handle along its top edge — reusing the `useEdgeResize` hook generalized (from an
  x-only `useResizablePanelWidth`) for the case panel's width-resize. Collapse state and height are
  local to each card, keyed by method name, so they reset if a method is deselected and reselected
  or the case changes, rather than persisting — same call as the case panel's width.
commits:
  - "381f494": "Turn method-info cards into collapsible, height-resizable panels"
  - "a322f3c": "Add bottom padding to the method-info dock"
  - "40ddf25": "Make method-info cards fill the width between the two panels"
  - "b12aa19": "Log the method-info card redesign in the research log"
---
id: 55
date: "2026-07-16"
title: "Made the React case detail panel resizable"
description: |
  The right-hand case detail panel had a fixed 460px width, which cramped the training/test plots
  on smaller screens and wasted space on wide ones. Added a drag handle on the panel's left border
  so users can resize it, clamped to 320–900px. The width lives in the app-state reducer alongside
  the existing `sidebarCollapsed` toggle and resets to the 460px default on reload rather than
  persisting via localStorage, matching how the sidebar toggle already behaves — simplest option,
  and nothing in the app currently persists UI state across sessions.
commits:
  - "6058cea": "Add resizable width state for the case detail panel"
  - "54f4b34": "Add useResizablePanelWidth hook for drag-to-resize panels"
  - "4a44a60": "Make the case detail panel resizable by dragging its left border"
  - "ef34133": "Log the case panel resize feature in the research log"
  - "3c6f804": "Generalize the resize hook to support both axes"
---
id: 54
date: "2026-07-15"
title: "Fixed the React case panel's training/test charts showing timestamps on both axes"
description: |
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
commits:
  - "f1818d3": "Fix train plot bug"
---
id: 53
date: "2026-07-15"
title: "Closed two React/Dash parity gaps: method-info panel placement, missing benchmark-design text"
description: |
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
commits:
  - "a435ad6": "Fixes after migration"
  - "d9bcaca": "Show Benchmark Design content in the instructions modal"
---
id: 52
date: "2026-07-15"
title: "Started a React port of the results-matrix app for GitHub Pages"
description: |
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
commits:
  - "5b0a024": "Scaffold web/ React+TS+Vite app for the GitHub Pages migration"
  - "81fa37b": "Log the React/GitHub Pages migration decision"
  - "2cc8579": "Add scripts/export_web_data.py to export benchmark data as static JSON"
  - "985c1b1": "Add matrix data layer: metric config, colour scale, and selectors"
  - "f1f80f4": "Wire up the static results matrix with sidebar filters"
  - "a406fe5": "Update the React migration log entry with M1 progress"
  - "a65168d": "Add selection reducer and wire toggle/selection state"
  - "5aac3c6": "Wire hover tooltip, click-to-select, and global/exogenous overlays"
  - "9f464d1": "Update the React migration log entry with M2 progress"
  - "b396b59": "Export bounded per-case derived series for the case detail panel"
  - "aff2bfe": "Add case-summary/training-stats text ports and extended-history selector"
  - "b2e0422": "Wire up the case detail panel"
  - "1a4f83d": "Update the React migration log entry with M3 progress"
  - "f8ae49a": "Export per-case snapshot bundles for the forecast panel"
  - "37d2e46": "Add snapshot-panel data layer: types, freq parsing, method-info text"
  - "20721f8": "Wire up the forecast-snapshot panel"
  - "10b6abe": "Update the React migration log entry with M4 progress"
  - "d4be80a": "Combine the React app and EDA site into one Pages deploy"
  - "096df16": "Document the web app in the READMEs"
  - "42c429d": "Log the completed React/GitHub Pages migration"
---
id: 51
date: "2026-07-15"
title: "Gave Chronos-2, TimesFM 2.5 and Moirai-2.0 `+exo` variants"
description: |
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
commits:
  - "a72c5b6": "Add +exo variants for Chronos-2, TimesFM-2.5, Moirai-2.0"
  - "a0e65b4": "Wire FM +exo classes into the methods package and pilot runner"
  - "3ef9b85": "Add tests for FM +exo variants"
  - "309d05d": "Log the FM +exo research entry"
---
id: 50
date: "2026-07-14"
title: "Fixed matrix cell clicks crashing with `KeyError: 'customdata'`"
description: |
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
commits:
  - "d1e9686": "Fix app so cells are interactive"
  - "749add2": "Fix exo bug for clicked cell"
---
id: 49
date: "2026-07-14"
title: "Added three zero-shot foundation models (Chronos-2, TimesFM 2.5, Moirai-2.0)"
description: |
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
commits:
  - "afb528d": "Add zero-shot Chronos-2 foundation-model method"
  - "a78b426": "Add zero-shot TimesFM 2.5 foundation-model method"
  - "ce9dc8c": "Add zero-shot Moirai-2.0 foundation-model method"
  - "ed6b72e": "Unify foundation-model deps into one fm extra; cap project at py<3.13"
  - "3d088c9": "Document the new foundation-model methods and their install extras"
---
id: 48
date: "2026-07-14"
title: "Folded +exo method variants into an app-level toggle; retired +calendar"
description: |
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
commits:
  - "47c2195": "Emit case_exo_tier/uses_exogenous from the rolling-origin eval path"
  - "acfb00f": "Fold +exo method columns into a toggle in the results matrix"
  - "39d1672": "Document the exogenous-toggle methodology in the app"
  - "760374f": "Retire +calendar method variants, always include seasonal features"
  - "b6cfe54": "Log the exogenous-toggle and +calendar-retirement decision"
---
id: 47
date: "2026-07-14"
title: "Committed distilled scores to docs/ for onboarding (issue #19)"
description: |
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
commits:
  - "7538ca3": "Anchor .gitignore results/ rule to repo root"
  - "8ec325a": "Commit distilled benchmark scores to docs/ for onboarding"
  - "f5e7003": "Publish current pilot scores and matrix figures to docs/"
---
id: 46
date: "2026-07-13"
title: "Added MAPE, sMAPE, NSE, and KGE (issue #34)"
description: |
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
commits:
  - "ba95a91": "Add MAPE, sMAPE, NSE, KGE to the eval loops"
  - "77dc3f3": "Declare per-case eligibility for domain-restricted metrics"
  - "e550b0b": "Surface MAPE, sMAPE, NSE, KGE in the matrix metric dropdown"
  - "84d65a0": "Add unit tests for MAPE, sMAPE, NSE, KGE and domain_metrics validation"
  - "a3d03b4": "Log the new domain/dimensionless metrics decision (issue #34)"
  - "51a7db6": "Fix metric direction for new metrics."
  - "f3e8f97": "Take new metrics into account for snapshots."
  - "9340874": "Removing domain metrics from Bellinge (until confirmed)"
  - "047861a": "Remove domain specific metrics for wwtp (until confirmed)"
---
id: 45
date: "2026-07-13"
title: "Windowed ML/DL methods are now fit once per case, not refit at every rolling origin"
description: |
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
commits:
  - "8d70c56": "Add refit_per_origin capability flag to Forecaster"
  - "90df331": "Condition windowed ML/DL predict on the history argument, not fit-time state"
  - "f41d5a2": "Fit windowed ML/DL once per case in run_case_rolling, infer across origins"
  - "3b73f02": "Fit windowed ML/DL once in run_leadtime_curve too"
  - "bcdff15": "Log the fit-once rolling-origin decision in the research log"
  - "6336a1b": "Condition Chronos predict on history instead of fit-time context"
  - "a6e0fd6": "Key rolling fit-cadence off is_global; drop the refit_per_origin flag"
  - "366dc60": "Update test_rolling_matrix_runs for the tagged snapshot schema"
---
id: 44
date: "2026-07-13"
title: "Snapshot panel now shows curated best/median/worst origins, ranked by whichever metric the matrix is showing"
description: |
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
commits:
  - "3406c64": "Curate snapshot panel to best/median/worst origins by MAE"
  - "1fa26cd": "Explain curated snapshot origins in the app's copy"
  - "86dcddc": "Log the curated-snapshot decision in the research log"
  - "5e479ff": "Curate snapshot origins by the matrix's selected metric, not fixed MAE"
  - "a03c6d0": "Update research log entry for metric-generic snapshot curation"
  - "83f410b": "Simplifying title. De-separate y-axis in snapshot."
  - "a2809f0": "Remove legend in snapshot"
---
id: 43
date: "2026-07-13"
title: "Method-info panel now shows library, compute cost, limitations, and a code snippet"
description: |
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
commits:
  - "1f90e2d": "Add method-metadata attributes to Forecaster base class"
  - "4bf99e6": "Populate method metadata for naive and classical forecasters"
  - "95ed89c": "Populate method metadata for windowed ML forecasters"
  - "a6509b6": "Populate method metadata for DL, torch, neuralforecast, FM, ensemble"
  - "13e7773": "Extend method registry and add method_info_detail()"
  - "701ac5c": "Add collapsible implementation-detail section to method-info cards"
  - "b0eb162": "Log the method-info panel's new library/compute/limitations/snippet fields"
  - "c96dedb": "Add a method-metadata registry and method_info() markdown builder"
  - "cd2542d": "Simplify family names"
---
id: 42
date: "2026-07-10"
title: "Added a bottom method-info panel, capped the method overlay at 3"
description: |
  Clicking a cell now also opens a panel docked to the bottom of the screen with a brief
  description of that method and which of the case's exogenous inputs (if any) it used —
  placeholder content while we figure out what's actually worth showing there. The
  description comes straight from each Forecaster subclass's own docstring, so it's free as
  methods gain or lose docstrings rather than a second copy to keep in sync; methods with no
  docstring anywhere in their class hierarchy just show a "no description yet" placeholder.
  Since both the snapshot graph and this new panel lay per-method content out side by side,
  the same-row method overlay is now capped at 3: clicking a 4th method evicts the oldest
  instead of piling on indefinitely.
commits:
  - "08b1534": "Add a bottom method-info panel on cell click"
  - "f653aad": "Log the bottom method-info panel and overlay cap"
---
id: 41
date: "2026-07-10"
title: "Merged the case-info and snapshot panels into one collapsible column"
description: |
  The results viewer used to open two separate side-by-side panels on a cell click: case
  info on the left, forecast snapshots on the right. Merged them into a single right-hand
  column, case info on top and snapshots below, matching the natural reading order. The
  case-info block (the four graphs plus the case summary text) is now wrapped in a native
  HTML `<details>`/`<summary>` element so it can be collapsed to make more room for the
  snapshots, defaulting to open so nothing changes for someone who never touches it.
commits:
  - "0a9c6cc": "Merge case-info and snapshot panels into one collapsible column"
  - "70f92ea": "Log the case-info/snapshot panel merge"
---
id: 40
date: "2026-07-10"
title: "Moved the snapshot panel into a vertical column next to case info"
description: |
  The forecast-snapshot panel used to sit below the matrix, full width, with a
  "case review notes" column next to the forecast graph. Moved it into a sticky
  column of its own between the matrix and the existing case-info panel, and
  dropped the notes column — that content already lives in the case-info panel,
  so it was redundant. Since a narrow column can't fit the old three-wide grid of
  forecast-origin snapshots, `render_snapshots` now takes a `max_cols` argument
  and the app passes `max_cols=1` so origins stack one per row. Selecting
  methods, the chip row, and the clear button all work the same as before.
commits:
  - "eb10211": "Improving right pannel with case information"
  - "ed40356": "Move test snapshots into a vertical right-hand panel"
  - "0fe54d4": "Log the snapshot-panel layout move"
---
id: 39
date: "2026-07-10"
title: "Results viewer can show the extended-history fit for gappy cases"
description: |
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
commits:
  - "60b0caa": "Add global-fit toggle to the results matrix"
  - "8e4cc20": "Wire the global-fit toggle into forecast snapshots"
  - "b68ec61": "Save extended-history fit snapshots, not just local-fit ones"
  - "c8356b1": "Replace hash with triangle"
  - "ffce616": "Log the global-fit toggle and snapshot-panel fix"
---
id: 38
date: "2026-07-10"
title: "Made the extended-history fit live via per-segment training (supersedes the \"inert\" entry below)"
description: |
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
commits:
  - "a3c37c6": "Make is_global methods segment-aware via unique_id"
  - "fbc3da4": "Score segment-pooling methods on the full series as a second fit"
  - "f7aea8d": "Split local and extended-history matrices in the results scripts"
  - "a58ecce": "Log the segment-pooled extended-history fit"
---
id: 37
date: "2026-07-10"
title: "Added an is_global capability flag and an extended-history training path (currently inert)"
description: |
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
commits:
  - "1210689": "Add is_global capability flag and a fit_pooled seam for future pooling"
  - "fb57472": "Add extended-history frame for is_global methods"
  - "df2d01c": "Route is_global methods onto extended history, with a gap-free fallback"
  - "ae6d125": "Render a second, is_global-filtered matrix for each metric"
  - "6818a04": "Log the is_global feature and its currently-inert extended-history path"
---
id: 36
date: "2026-07-10"
title: "Closed out EDA figure-quality review comments (issue #36)"
description: |
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
commits:
  - "1aaf14f": "Label exogenous overlay axes with units in case_eda.py"
  - "cc9ce0a": "Add EDA captions to all case docs, redraw CAMELS-DK diagram"
  - "eff7899": "Overlay raw vs curated data and add axis labels/captions to eda.py"
  - "2352fbf": "Log EDA figure-quality work and Bellinge follow-up in RESEARCH_LOG.md"
  - "97920bf": "Including raw background span."
---
id: 35
date: "2026-07-10"
title: "Scaled features ahead of MLPRegressor (issue #48)"
description: |
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
commits:
  - "9994a60": "Scale features ahead of MLPRegressor in MLPForecaster"
  - "0715b25": "Log MLPRegressor scaling fix in research log"
---
id: 34
date: "2026-07-10"
title: "Cleaned up duplicated helper logic across method wrappers (issue #58)"
description: |
  Fixed three cases of drifted duplicate code found in the 2026-07-02 method-suite audit.
  `classical.py`'s five AutoARIMA/AutoETS/TBATS/Theta/MSTL forecasters each repeated the same
  tz-strip-and-truncate block; that's now one `_prepare_train` helper. `_lookback_for` existed
  in three near-identical copies (`dl.py`, `torchdl.py`, and `nf_models.py`'s differently-named
  `_input_size`), and `dl.py`'s had silently drifted to drop the `max(base, 2*h)` floor the other
  two had — merged into one `lookback_for` in `base.py` with the floor applied everywhere.
  `nf_models.py`'s TFT/iTransformer/TimesNet models were also unseeded, unlike the other DL
  families, so `fit()` now calls `pytorch_lightning.seed_everything(0)` for reproducibility.
  No test asserted the pre-fix behavior, so nothing needed updating beyond the three files.
commits:
  - "964ba63": "Derive ML model prediction column from mlforecast's alias key"
  - "903cad9": "Remove boilerplate code"
---
id: 33
date: "2026-07-09"
title: "Added a case-context side panel to the results-matrix app"
description: |
  Clicking a matrix cell now opens a right-hand panel next to the matrix, separate from the
  existing forecast-snapshot panel below it, showing the last 1,000 points of that case's
  training split and a summary of the case's static config (domain, target, frequency, horizon,
  exogenous inputs, source). The point cap keeps the plot fast on long series; there's no
  existing formatter for `Case` metadata elsewhere in the codebase, so `app/data.py` gained a
  small `case_summary()` that reads straight off the `Case` model rather than any generated
  report. The panel only depends on the clicked case, so it stays open across method-chip clicks
  that change the row's overlay selection.
commits:
  - "d965610": "Including training data snapshot pannel"
  - "7f2641a": "Making right panel scrollable"
---
id: 32
date: "2026-07-09"
title: "Locked down the results-matrix heatmap to click-and-download only"
description: |
  The matrix heatmap is the app's centerpiece, so it's now locked against pan, zoom, box/lasso
  select, and axis dragging — visitors can click a cell to open its snapshot panel or use the
  mode bar's camera icon to download a PNG, and nothing else. This is a `dcc.Graph` `config` plus
  `fixedrange`/`dragmode=False` change in `app/app.py`; the click-to-select callback needed no
  changes since `clickData` still fires normally.
commits:
  - "7837e30": "Include frame at clicked cell"
  - "6c00170": "Fixing heatmap"
---
id: 31
date: "2026-07-09"
title: "Persist real per-origin forecasts and overlay methods in the results app"
description: |
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
commits:
  - "9f03c94": "Persist per-origin forecasts from the rolling eval"
  - "0f14ddb": "Show real forecasts and overlay methods in the results viewer"
  - "841464b": "Update rolling test for the (metrics, snapshots) return"
  - "a875aa8": "Log the real-snapshots + method-overlay work"
---
id: 30
date: "2026-07-09"
title: "Moved the results-matrix app from Streamlit to Dash"
description: |
  The results-matrix app moved from Streamlit to Dash because Plotly's `Heatmap` trace has no
  selection module, so Streamlit's `on_select="rerun"` could never fire on a cell click — the
  earlier workaround (an invisible `Scatter` overlay per cell just to catch clicks) worked but was
  fragile against reversed axes and cell layout. Dash's heatmap emits `clickData` natively, so the
  click-to-snapshot interaction is now a plain callback with no overlay hack. `data.py` and
  `snapshot.py` carried over unchanged since they only ever built pandas frames and Plotly figures;
  `app.py` was rewritten, and `@st.cache_data` was swapped for stdlib `functools.cache`. Run command
  is now `uv run --extra app python app/app.py`. One known gap: the reload button re-reads the
  parquet but won't show a brand-new domain or family in the filter widgets until the app restarts.
commits:
  - "fc725b7": "Migrate results-matrix app from Streamlit to Dash"
  - "073e327": "Clean app"
  - "36f752f": "Log the Streamlit-to-Dash app migration"
---
id: 29
date: "2026-07-09"
title: "Two app bugs found on first real-browser test, both pre-existing"
description: |
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
commits:
  - "46bd055": "Fix KeyError for domains missing from DOMAIN_KEY_DICT"
  - "4469803": "Fix cell clicks never registering on the matrix heatmap"
  - "2e6b732": "Log the domain-dict and heatmap-click bugs"
---
id: 28
date: "2026-07-09"
title: "Multi-snapshot forecast panel for the results-matrix app"
description: |
  The results-matrix app's cell-click detail panel showed one flat placeholder line with no time
  axis. Replaced it with `app/snapshot.py`, a Plotly panel showing 3 evenly-spaced forecast-origin
  snapshots side by side, each with actual values, predicted values, and a dashed line marking where
  the forecast starts — following a matplotlib sketch (`app/snapshot_example.py`) worked out first.
  Used Plotly rather than matplotlib to stay visually consistent with the rest of the app, and a
  fixed count of 3 snapshots rather than a configurable one since nothing yet needed more than a
  quick visual check. The time axis was real (`freq` and `horizon_primary` pulled from each case's
  case.toml) but the actual/predicted values were still placeholders at this point — resolved by the
  snapshot-persistence entry above.
commits:
  - "917b324": "Add dummy forecast-snapshot panel to results-matrix app"
  - "a7fd76b": "Log the forecast-snapshot panel work"
---
id: 27
date: "2026-07-08"
title: "Interactive results-matrix app (Streamlit + Plotly)"
description: |
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
commits:
  - "4b29132": "Add optional app dependency group (streamlit + plotly)"
  - "b3e26c8": "Add interactive results-matrix Streamlit app"
  - "b778b8b": "Log the results-matrix app work"
  - "5d553e4": "Simplifying frontend"
  - "8b5f9b9": "Domains as checkboxes"
---
id: 26
date: "2026-07-08"
title: "Swapped statsforecast's MSTL wrapper for statsmodels' directly"
description: |
  `_mstl_decompose` called `statsforecast.mstl.mstl`, a thin wrapper over `statsmodels.tsa.STL` that
  added no behavior this code relied on. Switched to calling `statsmodels.tsa.seasonal.MSTL`
  directly. One shape difference for anyone touching this code: statsmodels' `MSTL(...).fit()`
  returns a `DecomposeResult` with `.seasonal` as a 1D array for a single period or 2D (columns in
  period order) for multiple, rather than statsforecast's named-column DataFrame — `_mstl_strengths`
  and the notebook's decomposition plot now index by position instead of name. `statsforecast` stays
  a dependency since `classical.py`'s MSTL-ETS/MSTL-ARIMA forecasters still use it for actual
  forecasting, a separate use from decomposition-for-evidence.
commits:
  - "7f39f91": "Implementing mstl using statsmodels"
---
id: 25
date: "2026-07-08"
title: "Audit of the RobustPeriod implementation: three weaknesses found and fixed"
description: |
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
commits:
  - "20daeb6": "Drop unused FISHER_ALPHA constant"
  - "4800f7e": "Fall back to the mean absolute deviation when the MAD is zero in _huber_clean"
  - "4a9c990": "Skip Nyquist (period-2) candidates in RobustPeriod discovery"
  - "562a309": "Size-adjust the RobustPeriod prominence bar to the peak window's bin count"
  - "17e3683": "Log the RobustPeriod audit and the three fixes"
---
id: 24
date: "2026-07-08"
title: "RobustPeriod: data-driven seasonality discovery for curation"
description: |
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
commits:
  - "f847fd1": "Add FFT-based MODWT and robust wavelet variance for period discovery"
  - "1e2fc03": "Add Huber cleaning, Huber periodogram, and band-period helpers"
  - "c62b2c5": "Add robust_period_candidates: data-driven multi-period discovery"
  - "1bb84f6": "Add RobustPeriod discovery cell to the seasonality notebook"
  - "07b1a08": "Allowing extra candidates if RobustPEriod does not find any"
  - "8de893d": "Log RobustPeriod seasonality-discovery work"
---
id: 23
date: "2026-07-08"
title: "Non-seasonal MSTL test now builds its own non-seasonal case"
description: |
  The test asserting MSTL refuses to run on a non-seasonal case used `discharge_flashy_catskills`
  as its example — but since curation (see 2026-07-03 below) gave every committed case a declared
  `seasonalities` list, that case is no longer non-seasonal and the test was silently exercising the
  wrong path. Fixed by constructing the non-seasonal case explicitly via `model_copy` with
  `seasonalities` overridden to `[]`, since the synthetic data generator doesn't read that field
  anyway. Worth flagging: with all committed cases now seasonal, MSTL's inapplicability guard and
  the SeasonalNaive non-seasonal branch are only ever exercised by synthetic overrides, never by a
  real case in the suite.
commits:
  - "8f70ee1": "Fix missing test"
---
id: 22
date: "2026-07-08"
title: "Dominant period is now chosen at fit time by variance explained"
description: |
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
commits:
  - "5f5794a": "Compute dominiant_seasonality instead of getting it by convention."
---
id: 21
date: "2026-07-07"
title: "Climatology baseline now keys on the declared dominant seasonality (issue #54)"
description: |
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
commits:
  - "ff6a3a7": "Extract shared position_in_cycle helper into seasonality module"
  - "2834505": "Key Climatology on the declared dominant seasonality"
  - "74c743d": "Log Climatology seasonality-keying change (issue #54)"
---
id: 20
date: "2026-07-07"
title: "Added a PACF panel to the seasonality notebook, alongside ACF"
description: |
  The seasonality notebook's ACF plot shows peaks at every multiple of a candidate period, which is
  ambiguous given this project's candidate menu — within a frequency, longer menu periods are exact
  multiples of shorter ones, so a strong 12h cycle mechanically inflates ACF(24) whether or not an
  independent 24h effect exists. Added a PACF panel alongside ACF, since PACF nets out shorter lags
  already regressed out, so a candidate that's a multiple of a shorter one only spikes if it carries
  independent information. This is reviewer-facing only — the acceptance gate in
  `detect_candidates()` still uses ACF plus MSTL strength, since PACF's max lag is capped below half
  the sample size and its cutoff interpretation isn't guaranteed for arbitrary periodic shapes.
commits:
  - "3d5c229": "Including pacf in seasonality notebook"
  - "7241748": "Commit new seasonalities, based on enhanced study"
---
id: 19
date: "2026-07-07"
title: "Interactive periodogram + find_peaks in the seasonality notebook"
description: |
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
commits:
  - "e28a677": "Improving peak finding in seasonality notebook"
  - "66b8f4a": "Improving tooltip"
  - "ca742f3": "Hidding clutter info"
  - "2ca0fb4": "Improving plot styling"
---
id: 18
date: "2026-07-03"
title: "Periodogram-peak suggestions for off-menu seasonalities"
description: |
  The seasonality notebook could only evidence periods a human already thought to test — its fixed
  candidate menu is deliberately narrow, so a case with a different physical driver (e.g.
  `waves_oregon`'s 17h swell) had no way to surface unless someone already suspected it. Added
  `suggest_peak_candidates()`: finds periodogram local maxima, applies the same prominence bar as
  `detect_candidates`, and drops anything within 10% of a menu/excluded period as spectral leakage.
  It's advisory only — up to 5 ranked suggestions the reviewer can copy into the extra-candidates
  box to get full evidence, not an acceptance path in itself. Superseded in practice by the
  interactive periodogram added 2026-07-07 above, but the underlying function is still used by the
  registry and tests.
commits:
  - "1887763": "Suggest off-menu seasonalities from periodogram peaks"
---
id: 17
date: "2026-07-03"
title: "Per-case curated `seasonalities` replace the freq → seasonality map"
description: |
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
commits:
  - "d1a3cfe": "Add seasonality module: resolution helpers + detection evidence"
  - "2d562d9": "Add marimo notebook for one-time seasonality curation"
  - "f410038": "Calibrate ACF floor to 0.1 on real-case evidence"
  - "c78afe3": "Require Case.seasonalities; curate all 12 cases from train-split evidence"
  - "56a387a": "Switch methods to case.seasonalities"
  - "b21e320": "Resolve MASE season from declared seasonalities; delete freq maps"
  - "ae01228": "Add advisory seasonality drift check to review_case"
  - "2263648": "Document required seasonalities field in the case checklist"
  - "77e2bc2": "LAtest changes of notebook. Revisiting declared seasonalities per case."
  - "2d6c54b": "Clarify that seasonalities ordering is by prominence, not length"
---
id: 16
date: "2026-07-03"
title: "Silencing a spurious TBATS Box-Cox warning"
description: |
  `TBATSForecaster.fit` pre-applies a Yeo-Johnson transform for cases with non-positive targets
  (e.g. `watertemp_james`), so TBATS's own internal Box-Cox is expected to be a no-op — but
  Yeo-Johnson with `standardize=False` isn't positivity-guaranteeing, so some transformed points
  still landed at or below zero and triggered TBATS's own warning-and-disable logic. Fixed by
  passing `use_boxcox=False` explicitly whenever the pre-transform was applied, producing an
  identical fit without the noise; `use_boxcox` stays at its default when the raw data is already
  positive, so TBATS can still decide for itself whether a Box-Cox transform helps.
commits:
  - "e0e5324": "Guarding boxcox error for negative serues after boxcox."
  - "2795599": "suppress overflow warning from tbats"
---
id: 15
date: "2026-07-03"
title: "Gap handling: one materialized policy (cap + longest-segment trim)"
description: |
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
commits:
  - "c71d6f0": "Creating curation pipeline for data sets. Minimal gap imputation + trimming to longest contiguous run. Cases are now based on curated data."
  - "f7cc7ae": "Anchor curation to the tail segment, not the longest one"
  - "80c97d9": "Fix stale 'longest span' docstring wording after tail-segment change"
---
id: 14
date: "2026-07-03"
title: "Ensemble-of-baselines columns (issue #59)"
description: |
  Added a generic `EnsembleForecaster` that composes existing Forecasters and combines their
  per-step forecasts by median (robust to one member going badly off, e.g. Persistence during a
  spike), closing the 2026-07-02 method-suite audit's "no ensemble-of-baselines column" gap. Two
  columns are registered: `EnsembleBaselines` (median of the four naive baselines) and
  `EnsembleBaselinesTheta` (the same four plus Theta), bracketing a pure zero-cost reference and a
  naive+classical blend. A member that raises at predict time is dropped for that origin; the
  ensemble only raises if every member fails. An early smoke test on `discharge_lowland_denmark`
  showed the intended calibration signal — EnsembleBaselinesTheta beat persistence while the
  naive-only ensemble trailed it on that flashy series.
commits:
  - "6fa6403": "Include naive ensembles to module suite"
---
id: 13
date: "2026-07-02"
title: "Seasonal classical baselines: MSTL forecasters + seasonally-adjusted Theta (issue #46)"
description: |
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
commits:
  - "0a40378": "Add seasonal classical baselines: MSTL-ETS/MSTL-ARIMA + seasonal Theta (#46)"
---
id: 12
date: "2026-07-02"
title: "Audit of the method suite (`bench/methods/`): flaws and gaps identified"
description: |
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
commits:
  - "8e76b47": "Logging fable findings"
---
id: 11
date: "2026-07-02"
title: "Fixed rolling-origin selection collapsing onto the front of the test set"
description: |
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
commits:
  - "8e508f4": "Fixing origins bug in rolling metric"
---
id: 10
date: "2026-07-02"
title: "Added missing `discharge_saugeen` case doc and EDA panel"
description: |
  `discharge_saugeen` had case code (case.toml, download.py, synth.py) but no docs page, EDA
  figure, or entry in `docs/README.md`'s case table — added all three, framing the case's role as
  the deliberate univariate control against the three exogenous hydrology cases, isolating pure
  autoregressive/seasonal skill. No case behavior changed.
commits:
  - "8f8c61d": "Adding saugeen to docs"
---
id: 9
date: "2026-07-02"
title: "Dropped `optional_exogenous` as a redundant stored field"
description: |
  `Case.optional_exogenous` was a stored case.toml field, but given the fixed set of available
  columns and which are necessary, "optional" is always just the complement — pure duplication that
  only needed a validator to police it. Made it a derived property instead
  (`(exogenous ∪ exogenous_known_future) − necessary_exogenous`) and removed the field from all 9
  case.toml files that had it; `Case.exo_tier` and everything downstream is unaffected.
commits:
  - "d65b85e": "Including necessary_exog in schema"
---
id: 8
date: "2026-07-02"
title: "Structured exo-necessity fields, persisted as result metadata (not matrix aggregation)"
description: |
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
commits:
  - "d65b85e": "Including necessary_exog in schema"
---
id: 7
date: "2026-07-02"
title: "Recorded necessary-vs-optional exogenous judgment in case notes"
description: |
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
commits:
  - "b7c0a93": "Including Catboost. Updating notes with exogenous assumptions."
---
id: 6
date: "2026-07-02"
title: "Added CatBoost to the windowed-ML method set"
description: |
  Added `CatBoost`/`CatBoost+exo` to the windowed-ML column family, slotting into the same shared
  lag+calendar feature pipeline as LightGBM with a matched training budget (200 iterations, 0.05
  learning rate) so the two GBDT methods are compared on roughly equal footing. Gives the benchmark
  a second GBDT implementation — useful since CatBoost's ordered-boosting scheme handles the same
  lag/calendar features differently and may behave better or worse depending on how noisy a case's
  covariates are. No methodology change, straightforward column addition.
commits:
  - "b7c0a93": "Including Catboost. Updating notes with exogenous assumptions."
---
id: 5
date: "2026-07-02"
title: "Backfilling this log from the scoping documents"
description: |
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
commits:
  - "70cdbaf": "Commit research log"
commits_note: "approximate — reconstructed after the fact"
---
id: 4
date: "2026-06-25"
title: "Added a second, unforced hydrology contrast case"
description: |
  Added `discharge_saugeen` (daily discharge, Saugeen River, Ontario, 65 years of record, no
  exogenous inputs) alongside `discharge_flashy_catskills`, answering `research_plan.md`'s open
  question about whether to add a second Danish catchment or an alpine one — with a third option
  instead: a long-record, exogenous-free univariate series as a methodological contrast to the
  exogenous-rich cases, rather than another geographic contrast. Brings the case count to 12.
commits:
  - "aa0e90c": "Include discharge_saugeen"
commits_note: "approximate — reconstructed after the fact"
---
id: 3
date: "2026-06-12 to 2026-06-13"
title: "Scoping review with Rocco (domain expert), and the response"
description: |
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
commits:
  - "0cdc9cb": "docs: respond to Rocco's scoping review (PR #2)"
  - "840fcf5": "refactor: split package into data/ and bench/ namespaces"
  - "25a6ef9": "feat(bench): exogenous-aware windowed ML (Tier 2) + skill-vs-mean"
  - "c5d44a1": "fix(cases): populate real ERA5 exogenous for GESLA + Bellinge"
commits_note: "approximate — reconstructed after the fact"
---
id: 2
date: "2026-05-29"
title: "Two data-quality bugs found before scaling further"
description: |
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
commits:
  - "0f348ea": "fix(bellinge): pin G71F05R LevelInlet (100% active); fix(battledim): sum pumps"
  - "2de8c8c": "feat(skill): add /review-case case-quality review"
commits_note: "approximate — reconstructed after the fact"
---
id: 1
date: "2026-05-27 to 2026-05-28"
title: "Initial scoping plan and repo scaffold"
description: |
  `research_plan.md` laid out the original proposal: 10-11 candidate cases spanning the water
  cycle, ~16 forecasters from naive baselines to foundation models, a phased roadmap, and a
  Parquet/Pydantic storage design. The initial commit scoped this down to a first working
  pipeline — 6 cases, 9 methods — with the matrix and lead-time evaluation working end-to-end and
  CI in place from day one, a deliberate "lock the interfaces first" move rather than building out
  the full scope before anything ran.
commits:
  - "715b4f8": "Initial commit: 6 cases, 9 methods, matrix + lead-time eval, CI"
commits_note: "approximate — reconstructed after the fact"
