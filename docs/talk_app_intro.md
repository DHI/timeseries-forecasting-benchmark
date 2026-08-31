# Intro talk: the results viewer

10 minutes of slides, then ~15 minutes of live demo.
Audience: engineers with basic ML knowledge, no forecasting background assumed.

Timings per slide are speaking time. Bullets are what goes on the slide;
prose under them is what you say.

---

## Slide 1 — One grid, 12 cases, 37 methods (1 min)

- A web page: forecasting cases down the side, methods across the top, one score per cell
- 12 water and environment cases x 37 methods, every cell filled
- Static page, no server. It reads JSON files exported from the benchmark runs

Say:

> This is a viewer for one experiment. We took 12 forecasting problems from our own
> domains, ran 37 forecasting methods on every one of them, and put the results in a grid.
> Green cells are methods that worked, red cells are methods that failed. The page has no
> backend — it's a static site reading exported result files, so it loads anywhere and
> keeps working when nobody is maintaining a server.

---

## Slide 2 — Why we built it (1.5 min)

- The question: on a new water forecasting job, which method do you reach for first?
- Usual answers: habit, or one paper on one dataset
- Second question: where do the new time series foundation models actually pay off?
- Part of RK2026 / KER 5, next-generation modelling

Say:

> When a project needs a forecast of river flow, or sewer level, or wave height, the choice
> of method usually comes from habit or from one paper someone read. Those papers each use
> their own dataset, their own split, and their own tuning, so you can't line them up.
> We wanted one consistent pass: same splits, same scoring, same effort per method, across
> problems that differ on purpose. The other reason is the foundation models — the
> pretrained models that forecast a series they've never seen. We wanted a number for where
> they help and where they don't, rather than a vendor claim.

---

## Slide 3 — The cases (1 min)

- Atmosphere to ocean: river discharge, groundwater, river temperature, sewer level,
  water demand, wastewater inflow, storm surge, wave height
- 15-minute to daily data; horizons from 6 hours to 30 days
- All open data (USGS, CAMELS-DK, GESLA, NOAA buoys, Bellinge, BattLeDIM, ERA5)
- One station per case, one target series

Say:

> The rows span the water cycle. Three river discharge cases picked to be different from
> each other — a slow lowland catchment, a flashy one, a snowmelt one — plus groundwater,
> river temperature, a sewer pipe, drinking water demand, a treatment plant inlet, storm
> surge at a tide gauge, and two wave buoys. Frequencies run from 15 minutes to daily and
> horizons from six hours to a month, each chosen to be what someone in that domain would
> actually ask for. Everything is open data, so anyone can rerun it.

---

## Slide 4 — The methods (1.5 min)

- Naive baselines: mean, persistence, seasonal naive, climatology
- Classical: AutoARIMA, AutoETS, Theta, MSTL, TBATS
- Windowed ML on lag features: ridge, LightGBM, CatBoost, MLP
- Deep learning: DLinear, N-HiTS, PatchTST, TimesNet, iTransformer, TFT
- Foundation models, zero-shot: Chronos-Bolt, Chronos-2, TimesFM 2.5, Moirai 2.0
- Ensembles of the baselines
- Most methods also have an `+exo` variant that reads rainfall, wind, temperature
- One fairness note: ARIMA and ETS run **non-seasonally** here, to keep their search
  affordable on 200,000-point series

Say:

> Six families, from the trivial to the pretrained. The baselines matter more than they
> look: "tomorrow equals today" is hard to beat at short horizons, and a method that can't
> beat it has told you something. Then classical statistics, then ordinary supervised
> learning on lagged windows, then the deep nets, then the foundation models that do no
> case-specific training at all. Most methods come in two versions: one that sees only the
> target's own history, and one that also sees the drivers — rain, wind, air pressure,
> temperature. That pair is the point of the whole exercise for us, because it puts a number
> on what the physics buys you.
>
> One thing to flag before you see the grid: ARIMA and ETS run non-seasonally in this
> benchmark. Turning the seasonal search on costs hours on a series with 200,000 points, so we
> didn't. The classical block therefore looks worse than the classical methods are — if you'd
> fit a seasonal ARIMA yourself, that column isn't your ARIMA. MSTL and TBATS in the same
> block do model seasonality.

---

## Slide 5 — How one cell is made (2 min)

- Split each series in time: train / validation / test. No shuffling
- Fit once on data strictly before the test period
- Then 100 forecast origins walk through the test window; errors pooled over all of them
- Gaps stay as gaps: filled for training, masked out of scoring
- No per-case tuning and no feature engineering. Target lags [1, 2, 4, 8, 16, 32] plus the
  case's seasonal lags, date parts, sin/cos of one period — the same on all 12 cases
- No rolling statistics, no differencing, no log transform, no lagged covariates
- Nine metrics to choose from: MASE, RMSE, MAE, MAPE, sMAPE, skill vs. persistence,
  skill vs. mean, and NSE/KGE on the flow and level cases

Say:

> A cell is not one forecast. We split the series by time, fit on the early part only, then
> walk 100 forecast origins through the held-out end and pool the errors. That's deliberate:
> a single test window rewards luck, and with 100 origins a method has to be right in
> several regimes to score well. Missing data is kept as missing — models get filled values
> to train on, but scoring only counts real observations.
>
> The part to remember: there is no tuning and no feature engineering. Every method runs at
> fixed defaults on all 12 cases, and both gradient boosters get the same iterations and
> learning rate. The features are six target lags, the case's seasonal lags, and a calendar
> encoding — that's it. No rolling means, no log transform on discharge that spans three orders
> of magnitude, and the exogenous methods see rainfall at its own timestamp with no lag. Any of
> you could beat several of these columns in an afternoon on a case you know. So read a weak
> column as "weak out of the box", not "weak method".
>
> On the metric: the grid opens on MASE, which divides each error by the error of a naive
> forecast on that case's own training data. That makes it unit-free, so unlike MAE you can say
> two rows' numbers in the same sentence. What it doesn't remove is the reference — each row is
> divided by its own naive forecast, at that case's seasonal lag, and that lag is 12 hours on
> surge and a full year on the daily rivers. Against a seven-day forecast, a year-ahead yardstick
> flatters the row. Concretely: snowmelt discharge reads 0.30 here and looks like one of the
> best-forecast cases in the table, but its best method only beats holding the last value by 3%.
> Storm surge reads 0.77 and looks like the worst, and its best method beats persistence by 76%.
> So MASE inside a row is exact — every column shares the same denominator — and across rows I'll
> switch to skill vs. persistence, where zero means the same thing everywhere.
>
> One detail that will come up: the SeasonalNaive column doesn't read 1.00. The denominator is
> that same method's error on the training years; the column is its error on the test years. Where
> the test window is calmer than the record it lands below 1, and it runs from 0.55 to 1.50 here.

---

## Slide 6 — What you can ask it (1 min)

- Filter rows by domain, columns by method family
- Switch the metric; the colours follow
- Two swaps: score with the drivers, score on the full history
- Hold a button to see cost per forecast instead of accuracy
- Click a cell: what the method is, what the case looks like, and the actual forecasts at
  the best, median and worst origins

Say:

> Four things you can do, and I'll show all of them in a minute. Filter down to the cases
> and families you care about. Change the metric. Flip the two checkboxes that swap in a
> different fit of the same method. And hold a key to turn the grid from accuracy into cost,
> because a method that's 5% better and 200 times slower is often the wrong answer. Clicking
> a cell is where it gets useful: you get the method's own card, the case's data and split,
> and the real forecast curves — including the origin where that method did worst.

---

## Slide 7 — What it does not tell you (2.5 min)

- The `+exo` scores use observed drivers, not forecast drivers. Upper bound, not
  operational skill
- No tuning, so a weak column may mean "weak defaults", not "weak method"
- One station per case. A cell is one site, not a domain
- Point forecasts only. No uncertainty, no exceedance probability
- Timings are indicative: different machines, some on GPU, some on CPU
- MAPE is meaningless where the target crosses zero (storm surge reads 103–609%)
- Cell colours rescale to whatever is on screen; MAE and RMSE carry each case's own units
- Two columns are incomplete: TFT runs at an older setting, TFT+exo is missing (GPU memory,
  issue #109)

Say:

> This is the slide I'd like you to remember.
>
> First and biggest: when a method uses rainfall or wind, we feed it the observed rainfall
> and wind over the forecast horizon. In production you'd have a weather forecast there,
> with its own errors. So every `+exo` number is a ceiling — it tells you how much the
> driver could be worth if you predicted it perfectly. The gap between the univariate and
> the `+exo` column is real information; the absolute value is optimistic.
>
> Second: no tuning, by design. A column that looks bad may just have bad defaults for that
> case. Read the grid as a shortlist generator, not a verdict.
>
> Third: each case is one gauge, one well, one buoy. The lowland river tells you about
> lowland rivers with the confidence of a sample size of one.
>
> Fourth: these are single-number forecasts. Three of the four foundation models actually
> produce a full distribution and we throw it away to score a point error. If your decision
> needs a probability of exceeding a threshold, this benchmark doesn't answer it yet.
>
> Fifth, two traps in the display itself. MAPE on storm surge reads between 103% and 609% for
> every method, because the target is a residual that crosses zero and there's nothing to divide
> by — the metric is broken there, not the models. And the colours rescale to whatever is on
> screen, so a cell changes shade when I filter even though the number is the same.
>
> And the honest bookkeeping: the cost numbers come from whatever machine each run happened
> on, so compare within a family rather than across the grid. And ignore the TFT column — it
> ran at an older setting than its neighbours and its exogenous version ran out of GPU memory,
> so it isn't a like-for-like comparison. Nothing in the app marks that yet, which is why I'm
> saying it here.

---

## Slide 8 — How to use it (1 min)

- Use it to pick two or three methods to try, and a baseline you must beat
- Check whether your case's drivers carry skill before building a driver pipeline
- Check the cost column before promising a model in an operational system
- Don't quote a single cell as "method X is best for Y"

Say:

> The way to get value out of this is narrow. If you're starting a forecasting job in one of
> these domains, the grid gives you a shortlist and a floor — the baseline your model has to
> beat before anyone should care about it. It also tells you, roughly, whether feeding in
> rain or wind is worth the pipeline work, and what the method costs per forecast. What it
> won't support is a sentence like "LightGBM is the best method for storm surge". One
> station, no tuning, perfect drivers — the grid isn't strong enough to carry that claim, and
> saying it out loud is the fastest way to lose the audience's trust in the rest of it.

---

## Slide 9 — Live (hand-off)

- Now the app

Say:

> Let me open it and walk through a case.
