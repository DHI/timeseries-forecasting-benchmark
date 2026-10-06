"""What each forecasting method says about itself, beyond its prose description.

Three small pieces of metadata, all assigned as class attributes on a `Forecaster` subclass
and read only by the results viewer's method-info panel (via `tsforecastbench.webdata`).
None has any effect on fit/predict.

`Capabilities` is a set of booleans: what structure the method can represent *at all*.
`Scorecard` is four ratings: what the method costs to use. `ForecastStrategy` is a
one-of-three category: how the method turns one fit into a multi-step forecast. The split
is deliberate — a graded rating answers "how much", a boolean answers "can it", a category
answers "which", and forcing one of these into another's shape reads as precision that
isn't there ("Medium seasonality" is not a thing, and neither is "70% recursive").

That argument is also why two earlier fields are gone. `robustness_to_noise` graded three
different *structures* (a robust aggregate, a parametric fit, a single propagated outlier)
on a Low/Medium/High scale, which is the "Medium seasonality" mistake in this very file; it
read Medium for 25 of 33 methods, and no measurement here can validate it — every proxy
tried ranks accuracy or flatness instead of fragility. What is genuinely known about a
method's fragility now lives in its `limitations` prose, where it can be specific. And the
`fit_cost`/`inference_cost` pair became the single `running_cost` below.

**These ratings are the benchmark authors' judgement, not measurements** — except
`running_cost`, which is read off the results and pinned by
`tests/test_method_scorecards.py`. They are meant to narrow a shortlist before you read the
matrix, not to stand in for it.

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

    `trend`, `seasonality` and `multiple_seasonality` are the exception to that rule: they
    describe the method **as configured in this benchmark**, not in principle. AutoETS and
    AutoARIMA both declare `seasonality=False` because they run at `season_length=1` here,
    not because exponential smoothing has no seasonal component. Their `limitations` say so.

Scorecard:

    history_requirements  How much history before the method is worth using. Lower is better.
                          Low: usable from a handful of points, or needs no training data at
                          all (a zero-shot foundation model, Persistence). Medium: needs a few
                          full seasonal cycles. High: needs thousands of points / many training
                          windows before it beats a cheap baseline.

    interpretability      How readable the forecast's reasoning is. Higher is better.
                          High: the forecast follows from a rule or a handful of named
                          parameters. Medium: inspectable structure — a decomposition, feature
                          importances, attention you can plot. Low: a black box.

    running_cost          What it costs to actually use the method, as a `RunningCost` pair:
                          a Low/Medium/High `level` and a `growth` shape. Lower is better.
                          Both are read off the measured `fit_seconds`/`predict_seconds` — see
                          `methods.cost` for how, and `tests/test_method_scorecards.py` for
                          the guard that stops a declaration drifting from the results.

                          The level bands one fit plus 1,000 forecasts on a 100,000-point
                          record: Low under a minute, Medium one to fifteen, High beyond.
                          One field rather than a fit/inference pair because whichever term
                          is larger dominates, so the fit-once-versus-pay-per-forecast split
                          still reads off one number: TFT's 175s fit amortises to Medium,
                          while MSTL-ETS's trivial fit and 4s-per-forecast cost lands High.

                          Rate the base configuration, not the `+exo` one — a method and its
                          `+exo` variant share one declaration, and covariates are a
                          configuration of the same estimator.

    tuning_effort         How many choices you must get right before it works. Lower is better.
                          Low: nothing that matters, or auto-selected. Medium: a couple of
                          settings with obvious defaults (season_length, lag count). High:
                          several interacting hyperparameters (lookback, learning rate, epochs,
                          batch size, architecture) where a bad pick quietly costs accuracy.

ForecastStrategy (Recursive / Direct / N/A):

    Recursive   The method's own forecast feeds forward into computing a later part of the
                horizon: mlforecast's default lag-feedback strategy (LightGBM, CatBoost,
                LinearLags), where a predicted value is fed back in as the next step's lag
                input, and classical state-space/ARIMA-family recursions (AutoARIMA, AutoETS,
                TBATS, Theta, the MSTL variants) alike, where an h-step forecast for h>1
                substitutes the model's own prior forecast for the unobserved future
                observation and reapplies the fitted recursion. These are the same mechanism,
                not two different ones.

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


class RunningCost(BaseModel):
    """A cost rating and how it moves with record length — see the module rubric.

    Two values on one field rather than two fields: the shape qualifies the level rather
    than standing beside it, and a level without it misreads badly in both directions (TFT
    is expensive at every size, Theta is free at small ones and quadratic beyond).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: Level
    growth: GrowthShape


class Scorecard(BaseModel):
    """What a method costs to use — see the module rubric for what each field means."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    history_requirements: Level
    interpretability: Level
    running_cost: RunningCost
    tuning_effort: Level
