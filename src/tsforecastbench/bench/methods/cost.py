"""Turning the eval loop's timings into the one cost rating a method card carries.

The card's `running_cost` answers a practical question — *what does it cost to actually
use this method?* — so it is not raw `fit_seconds` or `predict_seconds`. Two corrections
turn those into an answer:

1. **Normalise for record length.** The benchmark's cases span 3,052 to 214,476 points, and
   a method's cost is a function of that length. Sixty seconds on a 3k record and sixty on
   a 150k record are not the same cost, so every timing is interpolated to a common
   `REFERENCE_POINTS` record before it is banded. That size sits inside the observed range,
   so this interpolates rather than extrapolates.

2. **Count the forecasts you would actually make.** One fit amortises over however many
   forecasts follow it, so the rating is `fit + REFERENCE_FORECASTS x predict`. That is why
   a single field replaces the old `fit_cost`/`inference_cost` pair: whichever term is
   larger dominates, so the fit-once-versus-pay-per-forecast distinction still reads off
   one number. TFT's 175s fit amortises to Medium; MSTL-ETS's trivial fit and 4s-per-
   forecast cost lands High.

`growth_shape` carries what a single reference size cannot: how the cost moves with record
length. The measured log-log slopes run from ~0 (the neural models train to a step ceiling,
so a 3k record costs what a 150k one does) through ~1 (the tree and linear models grow with
the data) to 2.2 (Theta's roll-forward re-runs its recursion over the whole growing history
at every origin, which is quadratic).

Nothing here reads a `Forecaster`. It takes timings and record lengths, so the same
functions serve the scorecard test, the exported growth-shape label, and any future
re-rating -- rather than three copies of the same regression.
"""

from __future__ import annotations

import math

from tsforecastbench.bench.methods.scorecard import GrowthShape, Level

# The record length every timing is interpolated to before banding, and the number of
# forecasts a fit is amortised over. Both are quoted in the card's tooltip, so a reader can
# reproduce the number on the meter from the results parquet.
REFERENCE_POINTS = 100_000
REFERENCE_FORECASTS = 1_000

# Band edges in seconds, chosen where a cost starts to actually hurt rather than at round
# numbers: under a minute you would not notice, over a quarter of an hour you plan around it.
COST_LOW_SECONDS = 60.0
COST_HIGH_SECONDS = 900.0

# Log-log slope cuts. Below FLAT the cost barely tracks the record; above SUPERLINEAR it
# outruns it.
SLOPE_FLAT = 0.3
SLOPE_SUPERLINEAR = 1.3

# Below this, a timing is measurement noise rather than a cost -- fitting a slope to
# SeasonalNaive's sub-millisecond "fit" describes the clock, not the method.
NEGLIGIBLE_SECONDS = 1e-3


def _is_negligible(seconds: list[float]) -> bool:
    """True when a whole timing series is clock noise rather than a cost.

    Judged on the largest observation, not point by point: dropping only the small points
    of a series would fit the slope to the long cases alone and report growth that the full
    series doesn't show. A zero-shot fit is microseconds on every case, so the series goes
    as one.
    """
    return not seconds or max(seconds) <= NEGLIGIBLE_SECONDS


def _loglog_fit(sizes: list[float], seconds: list[float]) -> tuple[float, float] | None:
    """Least-squares fit of log10(seconds) on log10(size). Returns (slope, intercept).

    None when there is nothing to fit: a negligible series, fewer than two usable points,
    or no spread in record length to regress against. Exact zeros are dropped because
    log10(0) has no value, not because they are small.
    """
    if _is_negligible(seconds):
        return None
    pairs = [
        (math.log10(n), math.log10(s))
        for n, s in zip(sizes, seconds, strict=True)
        if n > 0 and s > 0
    ]
    if len(pairs) < 2:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx
    return slope, my - slope * mx


def _at_reference(sizes: list[float], seconds: list[float]) -> float:
    """One timing series interpolated to a `REFERENCE_POINTS` record.

    Falls back to the mean of the observations when there is no fit to be had -- a method
    whose timings are all at the noise floor (a zero-shot fit, a naive baseline) has no
    slope worth estimating and no cost worth correcting.
    """
    fit = _loglog_fit(sizes, seconds)
    if fit is None:
        return sum(seconds) / len(seconds) if seconds else 0.0
    slope, intercept = fit
    return 10 ** (intercept + slope * math.log10(REFERENCE_POINTS))


def running_cost_seconds(
    sizes: list[float], fit_seconds: list[float], predict_seconds: list[float]
) -> float:
    """What one fit plus `REFERENCE_FORECASTS` forecasts costs on a reference-sized record.

    `sizes` are the record lengths the timings were measured on, one per observation;
    `fit_seconds` is the cost of a single fit and `predict_seconds` the cost of a single
    forecast origin, which is the unit `eval.rolling` already records.
    """
    return _at_reference(sizes, fit_seconds) + REFERENCE_FORECASTS * _at_reference(
        sizes, predict_seconds
    )


def cost_level(seconds: float) -> Level:
    """Band a running cost. Lower is better, so Low is the cheap end."""
    if seconds < COST_LOW_SECONDS:
        return Level.LOW
    if seconds < COST_HIGH_SECONDS:
        return Level.MEDIUM
    return Level.HIGH


def growth_shape(
    sizes: list[float], fit_seconds: list[float], predict_seconds: list[float]
) -> GrowthShape:
    """How the running cost moves with record length.

    The slope is taken from whichever term dominates the running cost, so the label
    describes the number the meter shows rather than a component of it. A method with no
    fittable slope reads `flat`, which is what a constant cost is.
    """
    fit_at_ref = _at_reference(sizes, fit_seconds)
    predict_at_ref = REFERENCE_FORECASTS * _at_reference(sizes, predict_seconds)
    dominant = fit_seconds if fit_at_ref >= predict_at_ref else predict_seconds
    fit = _loglog_fit(sizes, dominant)
    if fit is None:
        return GrowthShape.FLAT
    slope = fit[0]
    if slope < SLOPE_FLAT:
        return GrowthShape.FLAT
    if slope < SLOPE_SUPERLINEAR:
        return GrowthShape.LINEAR
    return GrowthShape.SUPERLINEAR
