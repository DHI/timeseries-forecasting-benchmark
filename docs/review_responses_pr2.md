# Responses to Rocco's scoping review (PR #2)

Rocco (`rpalDHI`) reviewed the scoping text and four of the six cases in
[PR #2](https://github.com/DHI-Internal/timeseries-forecasting-benchmark/pull/2),
adding his comments inline as text in the markdown. That PR is **not for merging**
("only for tracking comments"). This document captures every comment, our response,
and a status so the decisions are auditable. The actual doc fixes live alongside this
file in the same PR.

**Status legend:** ✅ Fixed in this PR · 🔧 Action queued (code/data/figure work) ·
❓ Needs Jesper decision · 👍 Acknowledged, no change needed.

---

## A. Scoping framing (the four headline questions)

| # | Rocco's comment | Response | Status |
|---|---|---|---|
| A1 | Inland cases all relevant; no major subdomain overlooked. **But combined collection (Bellinge) and WWTP influent can have almost the same dynamics — ensure they're different enough.** | Genuine risk: both are rainfall-driven sewer-catchment signals. We now differentiate explicitly in both case docs — Bellinge is **in-pipe water level** for real-time control / CSO warning; WWTP is **influent flow** for aeration/energy with RDII as the hard part. Rocco's stronger suggestion (D-wwtp) is to push WWTP toward **influent concentrations** to separate it further. | ✅ / ❓ |
| A2 | Horizons not the most critical aspect; instead **make sure exogenous data is always relevant, and when observed rainfall stands in for a weather forecast, say so explicitly to the reader.** | Added an explicit "observed rainfall is a *perfect-forecast* proxy, not an operational forecast" caveat to Bellinge, WWTP and CAMELS-DK. This is an important honesty point for an external reader. | ✅ |
| A3 | **Add NSE and KGE for inland water-level/flow predictions.** | Added to the metrics plan (SCOPING metrics section). NSE/KGE are the hydrology-standard skill scores and belong on CAMELS-DK and arguably Bellinge/WWTP. Implementation lands with the domain-metrics work in `eval/metrics.py`. | ✅ doc / 🔧 code |
| A4 | **Make clear the goal is not the perfect model per case, but a direct comparison of models across cases.** | This reframes the whole benchmark and we agree. Added an explicit "Purpose" statement to SCOPING (and it should go in `research_plan.md` §1). It also drives the metric choice — see A3 / E5. | ✅ |
| A5 (status) | He reviewed only the cases, not the preliminary forecasts; **the forecasts should be investigated to confirm the cases were selected meaningfully.** | Exactly right, and this is the motivation for the **case-review skill** Jesper asked for next: a per-case check that the data is full enough, the exogenous inputs carry predictive skill, and a trivial baseline produces a sane forecast. Tracked as the next deliverable. | 👍 / 🔧 |

## B. CAMELS-DK (`camels_dk_discharge`)

| # | Rocco's comment | Response | Status |
|---|---|---|---|
| B1 | Right *category* of catchment? "not sure, might be validated by domain expert." | Agreed — flagged for a hydrology domain expert. The groundwater-dominated pick is deliberate (a regime where persistence is hard to beat); a contrasting flashy catchment is the planned second hydrology case. | ❓ |
| B2 | 7-day vs 1/3/7/14/30 horizon — "isn't the second objective to create error-vs-lead-time curves?" | Correct. The single "primary horizon" is just the headline matrix cell; the **lead-time curves** (`run_leadtime.py`) already sweep sub-horizons, so we don't need to multiply primary horizons. Clarified this in SCOPING so the two artefacts aren't confused. | ✅ |
| B3 | Include `q_sim` (DK-model) as a reference column? "yes if there's no equivalent high-quality observations of streamflow." | We *do* have observed streamflow (`q_obs`) as the target, so a process-model reference column is **optional** here, not required. This aligns with his cross-cutting view (E4) that process-model columns complicate the picture. | ✅ doc / ❓ |
| B4 | Contrast with a flashy catchment "can wait." | Agreed — deferred, as planned. | 👍 |
| B5 | Mermaid illustration "not very clear." | 🔧 Diagram is a candidate for a redraw; logged. Minor clarity edit applied. | 🔧 |
| B6 | "The units on the axis seem off" (EDA figure). | 🔧 Real issue — the EDA panel y-axis/units need checking and the figure regenerating. This is precisely a case-review-skill check. Logged, not done in this docs PR. | 🔧 |
| B7 | Operational pipeline "(not sure about how the post-processing works here)." | Softened the wording so the NWP→DK-model→ML-post-processing line is presented as the *aspiration/hybrid question*, not an established pipeline. | ✅ |

## C. Bellinge (`bellinge_sewer_level`)

| # | Rocco's comment | Response | Status |
|---|---|---|---|
| C1 | SCOPING text "inconsistent with the case description. Which sensor was actually used?" | **Confirmed bug in SCOPING.md** — it still named `G71F04R_Level1_System2000p1`, the *old* pick. The case (and `case.toml`) actually use **`G71F05R_LevelInlet_System2000p1`**, chosen after the `--scan` work because F04R was only ~20% active. Fixed SCOPING (both the case section and the data-quality section) to match. | ✅ |
| C2 | Diagram: "remove Level sensor F05R (unless shown to be at the inlet of the CSO chamber); connect CSO chamber to WWTP as most of the flow goes there (except the overflow)." | The CSO-chamber → WWTP flow path was missing and is hydraulically correct — **added** to the diagram. On removing F05R: it is the *chosen target sensor*, so it can't simply be dropped; the real question is **what `G71F05R_LevelInlet` physically measures** (pumping-station inlet vs CSO-chamber inlet). Flagged to confirm against the Bellinge asset database before finalizing the case framing. | ✅ diagram / ❓ sensor location |
| C3 | EDA figure: "add caption, set y-axis title to water level, use the cleaned version of the data instead of raw (visible spikes suggest raw); rainfall and water level don't match — are you using the right rain gauge?" | Rain gauge: **already addressed** — commit `b05481f` switched rainfall to SVK gauge 5425. Cleaned-vs-raw data, y-axis title and caption: 🔧 figure-regeneration work, logged. The cleaned `2_cleaned_data` members are what the download already targets; the spikes warrant a re-check. | 🔧 / partly ✅ |
| C4 | "15 minutes might be too coarse — you might miss or smoothen a spike." | Real tension between spike fidelity and tractability across the 10-year record. Documented the tradeoff in the case; the resampling rule (max vs mean within the 15-min bin) matters for peaks and is worth revisiting for the event-aware framing. | ✅ doc / ❓ |
| C5 | "2–6 hours is a good horizon, usually constrained by the validity of weather forecasts." | Confirms the 6-h primary horizon. Added the "horizon is bounded by rainfall-forecast validity" point to the doc — it ties to A2 (observed-rain-as-proxy caveat). | ✅ |
| C6 | Exogenous: "the model was calibrated on rain gauges only; not sure how helpful the radar observations are here." | Reworked the exogenous list to **lead with the rain gauge** as the primary driver and mark the X-band radar nowcast as *uncertain added value* rather than canonical. Matches the original Bellinge paper. | ✅ |
| C7 | Data-quality note: "might be because 04R has two sensors, level1 and level2, of which only one is active for most of the period." | Plausible explanation for the F04R inactivity, now moot since we moved to F05R. Noted in SCOPING. | 👍 |

## D. WWTP influent (`wwtp_influent`)

| # | Rocco's comment | Response | Status |
|---|---|---|---|
| D1 | "I wouldn't call it Mendeley — that's just the hosting platform; refer to the plant name or something else." | Reworded throughout to call it the **Cembrowicz et al. (2024) N₂O WWTP dataset** (hosted on Mendeley Data), not "the Mendeley dataset". The plant itself is anonymized in the source, so we name it by author/content. `case.toml`'s `source.name` could follow suit. | ✅ doc / 🔧 toml |
| D2 | "I agree with the resolution. Maybe limit the horizon to 6 hours, or both 6 and 24." | `horizons_eval` is already `["6h", "24h"]`; promoted 6 h to equal billing with 24 h in the doc and noted 6 h is the more defensible aeration-control horizon. | ✅ |
| D3 | Right primary target? "influent discharge or concentration." | Keeping **influent flow (discharge)** as primary; raised influent **concentration** as the differentiator from Bellinge (see A1 / D5). | ✅ / ❓ |
| D4 | "I wouldn't mix influent and effluent." | **Dropped** the effluent-NH₄ and N₂O secondary-target framing from the case doc. Effluent is a different task (mechanistic/WEST territory per Rocco). Keeps the case clean. | ✅ |
| D5 | "To differentiate from Bellinge, focus on influent **concentrations** (if actually measured in the dataset)." | Strong idea — the dataset has NH₄/NO₃/PO₄/SS at the influent. Added as the recommended differentiator. Needs a column-availability check (`download --list`). | ❓ / 🔧 |
| D6 | "A synthetic BSM2 case — not useful." | **Removed** BSM2 as a planned addition from the case doc and SCOPING. | ✅ |

## E. Cross-cutting design choices

| # | Rocco's comment | Response | Status |
|---|---|---|---|
| E1 | Split mode: "not critical if different across cases." | Good — keeps calendar/fraction per-case freedom. No standardization forced. | 👍 |
| E2 | Gap handling: "yes, if training gaps are within a certain limit (resolution-dependent, need not be the same for all cases)." | Agreed. We'll add a **per-case max-gap threshold** for training-time fill, below which a series/window is excluded. Logged for `eval`/case config. | ✅ doc / 🔧 code |
| E3 | Univariate first: "address case-specific; in some domains univariate makes no sense (e.g. collection systems)." | Right — for Bellinge and WWTP the wet-weather skill *is* the rainfall response, so univariate is only a floor. Documented that exogenous rainfall is essential (not optional) for those two cases. | ✅ |
| E4 | Process-model reference columns: "if you mean comparing to simulated data in addition to observed, I don't think it's necessary — it complicates the picture." | Notable **scope reduction**. Rocco would de-prioritize process-model reference columns and hybrid residual correction. This conflicts with the emphasis in `research_plan.md` §3, so flagged for Jesper rather than unilaterally cut. Reflected as "optional / de-scoped pending decision" in SCOPING. | ❓ |
| E5 | Domain metrics "break the ability to compare across domains — maybe find a dimensionless score that works across all cases (e.g. MAPE)." | Aligns with A4. **Primary cross-case cell value stays dimensionless** (MASE; consider adding MAPE/sMAPE). Domain metrics (NSE/KGE, event hit-rate, POD/FAR) become **per-case sidebars**, not matrix cells. Documented in SCOPING metrics section. | ✅ |

## F. Items not yet reviewed by Rocco

`gesla_storm_surge` and `ndbc_waves` were marked "not checked." No action; they remain
open for a second-pass review (ideally a coastal/met-ocean expert rather than the
urban+catchment hydrology lens of this review).

---

## Summary of what changed in this PR

- **Fixed two stale facts in SCOPING.md** that Rocco caught: the Bellinge sensor
  (`G71F04R` → `G71F05R_LevelInlet`) and the BattLeDIM target (`PUMP_1` → summed
  network demand).
- **Reframed the benchmark purpose** (cross-case comparison, dimensionless primary
  metric, domain metrics as sidebars).
- **Differentiated Bellinge vs WWTP**, dropped the WWTP effluent/N₂O/BSM2 extensions,
  renamed the WWTP dataset away from "Mendeley", corrected `demand` → `demand + leakage`
  for BattLeDIM, added the CSO→WWTP flow path to the Bellinge diagram, and added the
  observed-rainfall-as-forecast-proxy caveat where relevant.

## Open decisions for Jesper (the ❓ items)

1. **Process-model reference columns / hybrid residual correction** — keep (per
   `research_plan.md`) or de-scope (per Rocco, E4)?
2. **WWTP differentiation** — add an influent *concentration* target to separate it
   from Bellinge (D3/D5)?
3. **Bellinge sensor location** — confirm what `G71F05R_LevelInlet` physically
   measures (C2), and whether the case should pivot to a CSO-chamber level / event
   framing (C4).
4. **Figure regeneration** — the units/caption/cleaned-data fixes (B6, C3, D-figs)
   are the first job for the case-review skill.
