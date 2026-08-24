/** Markdown shown in the instructions modal opened from the info icon next to the page title.
 * Benchmark methodology: splits, curation, fitting protocol, model parameters, case
 * description, and exogenous-input handling. */
export const BENCHMARK_DESIGN = `
## Splits

Each case declares its train/val/test split in case.toml, either as calendar dates
(train_end, val_end, test_end) or as fractions (default 70/15/15) of the observed
timestamps, so series with gaps still split cleanly. Splits are strictly chronological —
no shuffling.

Every method estimates its parameters on train alone. Val is the held-out period the
iteratively trained deep-learning methods (N-HiTS, PatchTST, TFT, TimesNet) stop on once
their validation loss stops improving; no method trains on it. Test is scored, and nothing
else is.

Fitting on train is not the same as never reading val: a fit-once method's forecast
conditions on the observed history running through val and into the test period up to each
origin. It reads that data, it just doesn't estimate parameters from it.

## Curation

Curation enforces minimum lengths: train must cover at least two full seasonal cycles, and
val+test together must cover at least one forecast horizon. Evaluation is rolling-origin
by default (errors pooled across many forecast origins in the test window), with a --fast
single-holdout mode for quick checks. Missing data is kept as explicit NaN: imputed for
training, masked out during scoring so only real observations count.

## Fitting protocol

Methods split into two groups by how often they re-estimate. Global methods — every ML, DL,
and foundation-model family, plus all of the classical ones (AutoARIMA, AutoETS, Theta, MSTL,
TBATS) — fit once on data strictly before the test period, then roll forward across every
origin by inference alone, with no further re-estimation. The naive baselines refit from
scratch at each origin instead, since for them the fit is the forecast. The ensembles sit
across the split: whichever members hold a trained model are fit once, while the naive
members they combine are re-derived from the history at each origin. This split keeps
a method's score from depending on how often it happens to be allowed to re-estimate, and
fitting strictly before the test period keeps the fit-once methods leak-free without repeated
retraining. Predict cost isn't uniformly small for the fit-once methods either — Theta and
MSTL, for instance, re-derive a seasonal decomposition over the whole history at every origin
— so the benchmark records fit time and predict time separately rather than folding them into
one number.

## Model parameters

There's no per-case tuning or grid search. Each method uses fixed, hard-coded
defaults — e.g., LightGBM and CatBoost both use 200 iterations at a 0.05 learning rate, so
the two gradient-boosting methods compete on equal footing. Classical methods (AutoARIMA,
AutoETS, etc.) run their own internal search. Every method reads the whole train split: the
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
