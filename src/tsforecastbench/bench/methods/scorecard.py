"""What each forecasting method says about itself, beyond its prose description.

Three small pieces of metadata, all assigned as class attributes on a `Forecaster` subclass
and read only by the results viewer's method-info panel (via `tsforecastbench.webdata`).
None has any effect on fit/predict.

`Capabilities` is a set of booleans: what structure the method can represent *at all*.
`Scorecard` is six ordinal ratings: what the method costs to use. `ForecastStrategy` is a
one-of-three category: how the method turns one fit into a multi-step forecast. The split
is deliberate — a graded rating answers "how much", a boolean answers "can it", a category
answers "which", and forcing one of these into another's shape reads as precision that
isn't there ("Medium seasonality" is not a thing, and neither is "70% recursive").

**These ratings are the benchmark authors' judgement, not measurements.** They are meant to
narrow a shortlist before you read the matrix, not to stand in for it. The one place judgement
and measurement could contradict each other is cost, so the two cost fields are anchored to the
`fit_seconds` / `predict_seconds` the eval loop actually records (see the rubric below).

Rubric — assign these consistently across methods, or the cards stop being comparable.

Capabilities (True/False):
    trend                 A mechanism that extrapolates level drift, rather than reverting to
                          the mean of the history it saw.
    seasonality           A mechanism for a repeating within-period shape, whether explicit
                          (a seasonal term, Fourier features) or learned (a windowed model
                          that sees a full cycle).
    multiple_seasonality  Handles several periods at once (e.g. daily *and* weekly). Implies
                          `seasonality`.
    exogenous             Can consume covariates *in principle*. This describes the method,
                          not the benchmark column: plain `LightGBM` is True even though its
                          column feeds no covariates — `LightGBM+exo` is the same method with
                          them switched on. False only where there is no covariate pathway at
                          all.

Scorecard (Low/Medium/High; the direction that is *better* is noted for each):

    history_requirements  How much history before the method is worth using. Lower is better.
                          Low: usable from a handful of points, or needs no training data at
                          all (a zero-shot foundation model, Persistence). Medium: needs a few
                          full seasonal cycles. High: needs thousands of points / many training
                          windows before it beats a cheap baseline.

    interpretability      How readable the forecast's reasoning is. Higher is better.
                          High: the forecast follows from a rule or a handful of named
                          parameters. Medium: inspectable structure — a decomposition, feature
                          importances, attention you can plot. Low: a black box.

    fit_cost              Cost of the one-off fit, anchored to the measured `fit_seconds`.
                          Lower is better. Low: sub-second to a few seconds on CPU. Medium:
                          tens of seconds to a couple of minutes on CPU. High: many minutes,
                          or needs a GPU / large pretrained weights loaded to be practical.

    inference_cost        Cost per forecast origin, anchored to the measured `predict_seconds`.
                          Lower is better. Low: milliseconds. Medium: ~0.1-1s. High: seconds or
                          more per origin — which for a `is_global` method rolled over many
                          origins dominates its total time.

    tuning_effort         How many choices you must get right before it works. Lower is better.
                          Low: nothing that matters, or auto-selected. Medium: a couple of
                          settings with obvious defaults (season_length, lag count). High:
                          several interacting hyperparameters (lookback, learning rate, epochs,
                          batch size, architecture) where a bad pick quietly costs accuracy.

    robustness_to_noise   Tolerance to noise, outliers and gaps — the data-quality problems
                          these cases actually have, not just white noise. Higher is better.
                          High: an aggregate or robust statistic that barely moves for one bad
                          point. Medium: a parametric fit that absorbs some of it. Low: fits
                          the noise, or lets a single outlier propagate through the horizon.

ForecastStrategy (Recursive / Direct / N/A):

    Recursive   The method's own forecast feeds forward into computing a later part of the
                horizon: mlforecast's default lag-feedback strategy (LightGBM, CatBoost,
                LinearLags), where a predicted value is fed back in as the next step's lag
                input, and classical state-space/ARIMA-family recursions (AutoARIMA, AutoETS,
                TBATS, Theta, the MSTL variants) alike, where an h-step forecast for h>1
                substitutes the model's own prior forecast for the unobserved future
                observation and reapplies the fitted recursion. These are the same mechanism,
                not two different ones — whether a given method's recursion is well-behaved is
                a separate question already covered by `robustness_to_noise`, not this field.

    Direct      One model call produces the whole horizon at once; nothing computed for one
                step feeds into computing another. Covers mlforecast's direct multi-horizon
                strategy (MLP), the DL methods (DLinear, N-HiTS, PatchTST, TFT, iTransformer,
                TimesNet), and the foundation models (Chronos, Chronos2, TimesFM, Moirai2).
                For the foundation models this describes only this benchmark's calling
                interface — one call, full horizon requested — not whether the pretrained
                decoder itself generates internally autoregressively, which this repo doesn't
                control and hasn't verified against the pinned library versions.

    N/A         The distinction carries no information: the naive baselines (MeanBaseline,
                Persistence, SeasonalNaive, Climatology), which look up or repeat a value per
                future timestamp independently of any other step, and EnsembleForecaster,
                whose own strategy is just whatever mix of members it combines.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator


class Level(StrEnum):
    """Ordinal rating on a scorecard field. Values double as the display labels."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class GrowthShape(StrEnum):
    """How a method's cost moves with record length. Values double as display labels.

    Lives here rather than in `methods.cost` so the declaration vocabulary sits in one
    module and the measurement code can import it without a cycle.
    """

    FLAT = "flat"
    LINEAR = "linear"
    SUPERLINEAR = "superlinear"


class ForecastStrategy(StrEnum):
    """How a method produces a multi-step forecast — see the module rubric."""

    RECURSIVE = "Recursive"
    DIRECT = "Direct"
    NOT_APPLICABLE = "N/A"


class Capabilities(BaseModel):
    """What structure a method can represent at all — see the module rubric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trend: bool
    seasonality: bool
    multiple_seasonality: bool
    exogenous: bool

    @model_validator(mode="after")
    def _multiple_seasonality_implies_seasonality(self) -> Capabilities:
        if self.multiple_seasonality and not self.seasonality:
            raise ValueError("multiple_seasonality=True requires seasonality=True")
        return self


class Scorecard(BaseModel):
    """What a method costs to use — see the module rubric for what each level means."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    history_requirements: Level
    interpretability: Level
    fit_cost: Level
    inference_cost: Level
    tuning_effort: Level
    robustness_to_noise: Level
