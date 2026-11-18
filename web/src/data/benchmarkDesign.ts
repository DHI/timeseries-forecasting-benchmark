/** Markdown shown in the instructions modal opened from the info icon next to the page title.
 * Benchmark methodology: splits, curation, fitting protocol, model parameters, case
 * description, and exogenous-input handling. */
export const BENCHMARK_DESIGN = `
## Splits

Each case declares its train/val/test split in case.toml, either as calendar dates
(train_end, val_end, test_end) or as fractions (default 70/15/15) of the observed
timestamps, so series with gaps still split cleanly. Splits are strictly chronological —
no shuffling.

Every method estimates its parameters on train and val together. The deep-learning methods
(N-HiTS, PatchTST, TFT, TimesNet, iTransformer) first train on train alone and stop once
their loss on val stops improving, then train a fresh model on train and val for that many
steps. Test is scored, and nothing else is.

## Curation

Curation enforces minimum lengths: train must cover at least two full seasonal cycles, and
val+test together must cover at least one forecast horizon. Evaluation is rolling-origin
by default (errors pooled across many forecast origins in the test window), with a --fast
single-holdout mode for quick checks. Missing data is kept as explicit NaN: imputed for
training, masked out during scoring so only real observations count.

## Fitting protocol

One rule covers every method: parameters are estimated once, before the test period, and
frozen; every forecast reads all observations up to its origin. Classical methods update
their state from those observations, ML and DL methods read them as their input window,
and foundation models read them as context, as far back as each model allows. The naive
baselines have no parameters, so for them reading the history is the whole forecast. No
method is retrained during the test period, so a score never depends on how often a method
was allowed to re-estimate. Predict cost isn't uniformly small either — Theta and MSTL, for
instance, re-derive a seasonal decomposition over the whole history at every origin — so the
benchmark records fit time and predict time separately rather than folding them into one
number.

## Model parameters

There's no per-case tuning or grid search. Each method uses fixed, hard-coded
defaults — e.g., LightGBM and CatBoost both use 200 iterations at a 0.05 learning rate, so
the two gradient-boosting methods compete on equal footing. Classical methods (AutoARIMA,
AutoETS, etc.) run their own internal search. Every method reads all of train and val: the
per-method training caps, which once differed by two orders of magnitude and made a column of
the matrix incomparable, were removed once their cost was measured. Any deviation from a
method's default is manual and explicit, not automated.

## Case description

Cases carry structured metadata in case.toml: domain, target, frequency, horizons,
exogenous variables, source citation, and declared seasonalities. Seasonalities are the
single source of truth for every downstream method — a candidate only qualifies with real
evidence (enough training cycles, a clear periodogram peak, sufficient autocorrelation).
There's no ground-truth seasonality to validate against; any override needs a written
justification. The same evidence-based standard applies to marking exogenous variables as
necessary, since that's a domain judgment, not something a model's error alone can tell you.

## Exogenous inputs

Each case declares its exogenous tier — none, optional, or necessary — a domain judgment,
not something inferable from measured skill deltas. LinearLags, LightGBM, and CatBoost each
have an exogenous-aware sibling; the matrix shows one column per algorithm, and the "Use
exogenous inputs where available" checkbox swaps in that sibling's score for cases with
real exogenous data, marking the cell with a teal corner. Toggling it changes which of two
already-computed scores is shown per cell — it never adds columns, drops cases, or
aggregates across them, matching the matrix's design elsewhere. Calendar and seasonal
features (day-of-week, sin/cos-of-period) sit outside this toggle entirely: they're
deterministic and free for any future timestamp, so every windowed-ML method always
includes them. Genuine exogenous covariates don't have that guarantee — future values fed
to the exogenous-aware methods here come from observed reanalysis data (a perfect-forecast
proxy), so their scores are an upper bound on real operational skill, which depends on
actual forecast quality for those covariates.
`.trim();
