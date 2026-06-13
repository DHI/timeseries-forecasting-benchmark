---
name: review-case
description: Review a forecasting benchmark case for quality before trusting or committing it. Use when adding a new case, after changing a case's download/case.toml, or when asked to "review a case", "check a case", verify a case is meaningful, has full-enough data, has exogenous inputs with predictive skill, or is actually forecastable. Runs scripts/review_case.py and adds domain judgement.
argument-hint: <case_id> [--strict] [--max-gap 24h]
---

# Review a benchmark case

Every time a new case is built here, the same problems recur: a declared exogenous
column that the download never populated, a target record that is mostly gaps, a
target that no method can beat persistence on, or a `case.toml` that drifts out of sync
with the data and the case doc. This skill catches those **before** the case is trusted
or committed.

It answers four questions:

1. **Is the data record sufficiently full?** — coverage within the observed span, gap
   sizes vs the horizon, grid regularity, per-split coverage.
2. **Is the exogenous data present and informative?** — each declared exogenous column is
   non-empty/non-constant, correlates with the target (lagged Spearman), and a probe
   measures whether adding it actually reduces out-of-sample error.
3. **Is the case forecastable?** — trivial baselines + one ML method are run; at least one
   should beat the naive floor and produce a sane, finite forecast.
4. **Is the config internally consistent?** — `case.toml` vs the data vs the splits vs the
   doc.

The script does everything **measurable**. You add the **judgement** the script can't:
is the target operationally meaningful, is the horizon natural for the domain, does the
exogenous set match the physical story, is the `data_type` honest. Don't skip that part —
the script gives evidence, not a domain opinion.

## Steps

1. **Run the diagnostic** for the requested case (use the user's `$ARGUMENTS` as the case
   id; pass `--strict` to also fail on WARN):

   ```bash
   uv run python scripts/review_case.py --case <case_id>
   ```

   It prints a report and writes `results/case_review/<case_id>.json` and `.md`. Read the
   JSON if you need the structured detail (exact correlations, gap lengths, metrics).

2. **Read the case's own files** and judge what the script can't:
   - `src/tsforecastbench/cases/<case_id>/case.toml` — target, horizon, exogenous, splits,
     `data_type`, `notes`.
   - `src/tsforecastbench/cases/<case_id>/download.py` — does it actually write every
     declared exogenous column? (An empty exogenous in the report usually means the
     download silently skipped it, OR it needs credentials / a manual archive — check the
     docstring to tell which.)
   - `docs/cases/<case_id>.md` — does the prose match the toml and the data? Is the target
     operationally meaningful and the horizon natural? (This is also where Rocco's review
     criteria live — see `docs/SCOPING.md` and `docs/review_responses_pr2.md`.)

3. **Interpret the verdict** (don't just relay it):
   - **FAIL on an empty exogenous** → decide: is the column a *bug* (download should
     populate it — fix the download) or *pending* (needs ERA5/Copernicus credentials or a
     manual archive — note it and, if it'll stay empty, remove it from `exogenous` so the
     case isn't advertising data it doesn't have)?
   - **FAIL on observed_fraction / split_coverage** → the target record is too sparse to
     score; the case needs a different sensor/station/catchment, a different span, or
     `split_mode="fraction"`.
   - **WARN on target_gaps** → gaps exceed one horizon. Confirm the gaps don't sit inside
     the test window; consider a per-case `--max-gap` that reflects the domain (Rocco: the
     tolerable gap is resolution-dependent and need not be uniform across cases).
   - **WARN on forecastable** (nothing beats the naive floor) → the target may be wrong
     (e.g. a near-constant or square-wave signal), or the exogenous it actually needs is
     missing. Look at the baseline metrics table.
   - **exo_skill_probe** is an *upper-bound information probe* (it uses contemporaneous
     exogenous), not the operational model — read it as "is there any signal here", not as
     a forecast result.

4. **Write the review.** Lead with the verdict, then group findings as **must-fix**
   (FAILs), **should-look-at** (WARNs), and **judgement** (the domain points). For each
   FAIL/WARN give a concrete next action (which file, what change). If you spot a fix the
   user clearly wants, offer to make it; otherwise leave it as a recommendation. Keep it
   short — this is a checklist result, not an essay.

## Notes

- Runs on whatever data is present: the committed Parquet if it exists, otherwise the
  case's synthetic generator (the report says which under `data source`). A review on
  **synthetic** data only validates the *plumbing* — rerun after the real download before
  trusting the data-quality findings.
- Cheap to run (seconds per case); safe to run repeatedly while iterating on a download.
- For the whole suite, loop over the case ids from
  `src/tsforecastbench/cases/*/case.toml`.
