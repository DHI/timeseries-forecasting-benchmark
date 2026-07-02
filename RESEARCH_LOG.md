# Research Log

A running record of the testing and decision-making behind this benchmark — what was tried,
what was decided, and why. Intended to be readable by collaborators without needing to dig
through commit history or notebooks.

Entries are in reverse-chronological order (newest first).

<!-- Add new entries below this line, newest first. -->

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
