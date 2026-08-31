# Audit: what the app can be read wrong

Findings from going through the viewer, the exported data, and the method code, ordered by how
likely an audience is to walk away with a wrong belief. Each one gives the line to say.

Numbers below are from the published data in `web/public/data/` (12 cases, 26 displayed
columns, 312 cells), after the dominant-period fix of research log entry 103.

---

## Tier 1 — say these out loud, unprompted

### 1. The `+exo` scores are an oracle

The exogenous-aware methods get the **observed** rainfall, wind, pressure and temperature over
the forecast horizon, taken from reanalysis. In production a weather forecast sits there, with
its own error.

> "When a model here uses rainfall, we hand it the rain that actually fell. Treat every
> exogenous number as a ceiling — it's what the driver is worth if you nail the driver."

### 2. MASE removes the units but not the reference — so a cross-row difference isn't a quality difference

MASE is the one metric in the app you *can* say across rows at all: dividing by a naive error
strips the target's units, so a number from the sewer row and a number from the wave row belong
in the same sentence, which is not true of MAE or RMSE. That much it delivers, and the
implementation is textbook seasonal MASE.

What the division does not remove is the reference forecast. MASE reads

    (this method's error over the horizon) / (a naive forecast's error at lag m, on the training data)

and `m` is the period pinned as dominant in the case's `case.toml`. Across these 12 cases `m`
spans 12 steps to 365, while the forecast horizon spans 7 steps to 48. So the two halves of the
fraction are measured over horizons that differ by row, and the ratio between them spans a factor
of 209:

| case | lag m | horizon H | m/H | MASE of Persistence | MASE of SeasonalNaive |
|---|---|---|---|---|---|
| surge_cuxhaven | 12 | 48 | **0.25** | 1.49 | 1.50 |
| waves_massachusetts | 12 | 24 | 0.50 | 0.99 | 1.32 |
| waves_oregon | 14 | 24 | 0.58 | 0.98 | 1.23 |
| demand_ltown | 24 | 24 | 1.00 | 0.99 | 0.90 |
| wwtp_influent_n2o_350k | 24 | 24 | 1.00 | 1.32 | 1.04 |
| sewerlevel_bellinge | 96 | 24 | 4.00 | 0.49 | 0.55 |
| discharge_saugeen | 365 | 30 | 12.2 | 1.14 | 1.02 |
| groundwater_yorktown | 365 | 30 | 12.2 | 0.26 | 0.93 |
| discharge_flashy_catskills | 365 | 7 | **52.1** | 0.57 | 0.97 |
| discharge_lowland_denmark | 365 | 7 | 52.1 | 0.23 | 0.57 |
| discharge_snowmelt_yosemite | 365 | 7 | 52.1 | 0.35 | 1.23 |
| watertemp_james | 365 | 7 | 52.1 | 0.67 | 1.06 |

On the daily rivers the denominator is a **year-ahead** naive forecast and the numerator a
7-day-ahead one, so the whole row is deflated. On surge it is the other way round: a 12-hour-lag
naive against a 48-hour forecast, so the row is inflated.

**The check that settles it:** a fixed algorithm should land on the same MASE in every row if
MASE were one shared yardstick. It doesn't.

| fixed algorithm | MASE range across the 12 rows | spread |
|---|---|---|
| Persistence | 0.23 – 1.49 | 6.5x |
| Mean | 0.59 – 2.84 | 4.8x |
| Climatology | 0.58 – 1.87 | 3.2x |
| SeasonalNaive | 0.55 – 1.50 | 2.7x |

Part of that spread is real (persistence genuinely is a better forecast on a slow groundwater
series than on a storm-surge residual). That is exactly the difficulty: MASE folds "how good is
this method" together with "how good is this row's naive reference", and looking at MASE alone
can't separate them.

**What it costs in practice.** Rank the cases by their best cell and the two metrics disagree at
both ends:

| by best MASE | | by best skill vs. persistence | |
|---|---|---|---|
| 1. groundwater_yorktown | 0.13 | 1. watertemp_james | +0.84 |
| 5. discharge_snowmelt_yosemite | 0.30 | 3. surge_cuxhaven | +0.76 |
| 12. surge_cuxhaven | 0.77 | 12. discharge_snowmelt_yosemite | +0.03 |

Snowmelt discharge reads 0.30 — fifth best in the table, and against a "1.0 = naive" label that
looks like a triumph. Its best method beats plain persistence by 3%. Storm surge reads 0.77 —
last in the table — and its best method beats persistence by 76%. Read across rows, MASE ranks
those two cases in the opposite order from how well they are actually forecast.

**So what is safe to say.** Within a row MASE is exact: all 26 columns share the identical
denominator, so the ranking and the ratios between cells are clean, and "above 1.0 means worse
than this case's own seasonal naive" is true row by row. Across rows it answers only "which
method beat its own row's naive reference by more", which is not the same question as "which
forecast is better".

For cross-row reading use **skill vs. persistence** or **skill vs. mean**: their reference is
computed at the same origins and over the same horizon as the method being scored, so 0 means the
same thing in every row by construction.

> "MASE is scale-free, so unlike MAE you can at least say these numbers side by side. What it
> doesn't fix is the reference: each row is divided by its own naive forecast, at that case's
> seasonal lag. On the daily rivers that lag is a year, against a seven-day forecast — so those
> rows come out flattered. Snowmelt reads 0.30 and looks like one of the best-forecast cases here;
> its best method beats holding the last value by 3%. Surge reads 0.77 and looks like the worst;
> its best method beats persistence by 76%. For comparing rows I'll switch to skill vs.
> persistence, where zero means the same thing everywhere."


#### Why the SeasonalNaive column isn't 1.0 on any row

Two reasons, both inherent to textbook MASE. Neither is a bug — say the first one and move on.

**a. The denominator is measured on different data (all 12 rows).** It is the naive error over
train+val; the column is scored on the test window at 100 origins. Same algorithm at the same lag,
different years. On the nine rows where cause (b) doesn't apply, this alone spans **0.55 to
1.23** — Bellinge reads 0.55 and lowland Denmark 0.57 because their test years are calmer than
their training record, snowmelt 1.23 because its are harder.

**b. The horizon outruns the cycle (3 rows).** Surge forecasts 48 steps on a 12-step cycle,
waves_massachusetts 24 on 12, waves_oregon 24 on 14. The method tiles its last observed cycle
forward, so leads past the first cycle reuse it at a larger effective lag and score worse than a
plain lag-m naive. These are the three rows reading above 1.23.

> "The SeasonalNaive column isn't pinned at 1.0, and it shouldn't be. The denominator is that
> same method's error on the training years; the column is its error on the test years. Where the
> test window is calmer than the record, it lands below 1."

**A third reason existed until this week.** The denominator and the `SeasonalNaive` column used
two different rules to pick the case's season, so on 4 of 12 rows the "1.0 = seasonal naive"
anchor measured a naive forecast at a different lag than the column named SeasonalNaive. The
period is now a pinned `case.toml` field that both read, and the affected rows were rescaled from
their stored MAE without re-fitting anything (research log entry 103). If anyone asks why
groundwater, the WWTP or Bellinge differ from a number they saw earlier, that is the reason.

**Still worth an issue in the repo.** The denominator could instead be the SeasonalNaive column's
own pooled error, measured under the benchmark's protocol at the same origins and horizon. Both
numbers are already published. Within-row rankings wouldn't move (it's a row constant), 1.0 would
literally mean "as good as the SeasonalNaive column", and both causes above would be gone — at the
price of no longer being the MASE anyone can look up.

### 3. No feature engineering — and here is exactly how little

Every windowed ML method gets the same fixed features on every case:

- target lags `[1, 2, 4, 8, 16, 32]` plus the case's declared seasonal lags
- date parts (hour / day-of-week / day-of-year / month, by frequency)
- sin/cos of **one** dominant period

No rolling means, no differencing, no log transform on heavy-tailed discharge, no lagged
covariates — the exogenous methods see the covariate at its own timestamp only. No tuning, no
grid search, one hard-coded config per method across all 12 cases.

> "There is no feature engineering here, and no tuning. Six lags, the seasonal lags, and a
> calendar encoding — the same on all twelve cases. A domain expert would beat several of these
> columns in an afternoon by adding a rainfall lag or a log transform. Read a weak column as
> 'weak out of the box', not 'weak method'."

### 4. AutoARIMA and AutoETS run non-seasonal here

Both are configured with `season_length=1` to keep the search affordable on 200k-point series.
The classical family looks worse than it is, and it's a config choice, not a property of ARIMA.

> "Fair warning on the classical block: ARIMA and ETS run non-seasonally in this benchmark. If
> you know your series has a strong daily cycle, that column is not the ARIMA you'd fit."

MSTL and TBATS do model seasonality — point at those instead if someone objects.

### 5. "Skill vs. persistence" compares against a flat line

The persistence reference is the last observed value, held constant across the whole horizon —
30 days on Saugeen and groundwater, 48 hours on surge.

> "Persistence here means the last value carried flat to the end of the horizon. At 30 days
> that's nearly a straight line, so beating it isn't much of a claim. Skill vs. mean is the
> harsher of the two on the seasonal cases."

---

## Tier 2 — things the live demo will visibly do that look like bugs

### 6. Both checkboxes look inert most of the time

- **Use exogenous inputs**: 11 of 26 columns have an `+exo` sibling, and 3 of 12 cases declare
  no covariates (`demand_ltown`, `discharge_saugeen`, `wwtp_influent_n2o_350k`). It can move
  **99 of 312 cells**.
- **Score on full training history**: only ML/DL columns, and only the 5 cases with history
  beyond the curated span (lowland Denmark, Bellinge, both wave buoys, the WWTP). It can move
  **45 of 312 cells**.

> "Two thirds of the grid won't move when I tick this. That's expected — the method needs a
> covariate-aware twin and the case needs covariates. Watch the corner triangles: that's how you
> tell which cells actually changed."

### 7. Cell colours are relative to what's on screen

The colour range shrinks to the values currently displayed, so filtering domains or families
recolours cells that didn't change value. Say it before you filter, or it reads as a bug.

For MAE, RMSE, MAPE and sMAPE the colour is log-scaled across the whole visible grid — and those
metrics carry the target's own units, which differ per row (mm/d, m³/s, L/s, m³/h, °C, m).

> "One thing about the colours: they rescale to whatever is on screen, so cells change shade when
> I filter even though the number is the same. And on MAE or RMSE, don't compare colour between
> rows at all — those are different units. Litres per second against degrees Celsius."

### 8. MAPE and sMAPE are meaningless on some rows

The Cuxhaven target is a non-tidal residual: 53% of values are at or below zero. Its MAPE runs
**103% to 609%** across methods, best to worst. Snowmelt discharge reaches 2028%.

> "Skip MAPE on storm surge. The target is a residual that crosses zero, so a percentage error
> has nothing to divide by — every method reads above 100%, which tells you nothing about any of
> them."

### 9. The bolded row winner can be a cell that saw more data

With the full-history checkbox on, the per-row "best" bolding compares swapped cells against
curated-span cells.

> "With this on, the bolded winner in a row may be the one method that got extra history. The
> pink triangle is the tell."

### 10. Snapshot origins belong to the first method you clicked

The best / median / worst origins come from the **reference** method — the first cell selected.
Every other method is drawn at those same origins, so an overlaid method can look unfairly bad
in the "worst" panel. Selection colours are assigned by position, so they shift when the
selection changes.

> "These three windows are the best, median and worst for the first method I clicked. The others
> are plotted on those same dates, which is the fair comparison — but it's not their own worst
> case."

### 11. The TFT column is incomplete, and the app doesn't say so

`TFT` is published at the older, narrower lookback on the eleven cases whose lookback changed,
and `TFT+exo` isn't published at all (CUDA out-of-memory, issue #109). Nothing in the app marks
either. With the exogenous checkbox on, TFT silently keeps its univariate score and grows no
teal triangle, which is indistinguishable from "TFT has no exogenous variant".

> "One column to ignore for now: TFT. It ran at an older setting than its neighbours and its
> exogenous variant failed on GPU memory, so it isn't a like-for-like comparison. That's not
> visible in the app yet."

### 12. Some surge cells were fitted at the old seasonal period

`surge_cuxhaven`'s pinned dominant period was corrected from 24 hours to 12 (research log entry
103), which invalidates the cells fitted before it: the six linear/tree columns, `MLP` and
`MLP+exo`, `TBATS`, `Theta`, and the two ensembles carrying those members. A 9-cell re-run
measured the difference — worst case `CatBoost` at 3.1% MAE, everything else under 1.3% — so the
cells that stay at 24 are wrong by about the width of the seed noise, not by a rank.

Nothing in the app marks them. The re-run cells land in the published data as they merge; whatever
is left at 24 is a footnote, not a caveat that changes a reading.

> "One case, storm surge, has a handful of cells fitted at a 24-hour cycle where the case now
> declares 12. Re-running the cheap ones moved them by under 3%, so it doesn't change any ranking
> — but the number is on my list."

### 13. The cost view is indicative, not a benchmark

Timings come from whichever machine each run happened on — some on GPU, most on CPU. Naive
baselines refit every origin, so their fit time is one refit. Theta and MSTL redo a seasonal
decomposition at every origin, so their predict cost dominates.

> "Compare cost within a family, not across the grid — these ran on different machines, some on
> GPU. The ranking inside a family is real; the ratios across families aren't."

---

## Tier 3 — one line each, if it comes up

- **One station per case.** A cell is one gauge, one well, one buoy. "The flashy-river row tells
  you about flashy rivers with a sample size of one."
- **Point forecasts only.** All four foundation models produce quantiles; the benchmark scores
  the central forecast and discards the rest. Nothing here answers "probability of exceeding a
  threshold".
- **The 100 origins overlap on two cases.** Groundwater has origins 7 days apart with a 30-day
  horizon, water temperature 5.5 days apart with a 7-day horizon. Those aren't 100 independent
  tests.
- **One number covers all lead times.** Errors are pooled over origins and lead times, so day 1
  and day 30 are averaged together. There's a lead-time script in the repo, but no lead-time view
  in the app.
- **"Necessary" exogenous is a domain judgment**, declared per case (8 of 12), not measured from
  skill.
- **Method-card ratings are opinions.** The Low/Medium/High meters are the authors' judgment, by
  their own admission in the code — except running cost, which is read off the results.
- **Only 12 of 37 methods are scored over 5 seeds** (CatBoost, MLP, and the neural ones),
  reporting the median run. The rest are single-run.
- **Full-history cells aren't comparable within their row.** That's what the pink triangle means,
  and it's easy to miss on a projector.
- **Nothing has a confidence interval.** No error bars anywhere in the app; two cells 0.02 apart
  should not be read as a ranking.
