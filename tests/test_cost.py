"""Unit tests for the running-cost helper.

These pin the arithmetic and the edge cases; `test_method_scorecards.py` is what checks
the declarations against the real results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.bench.methods.cost import (  # noqa: E402
    COST_HIGH_SECONDS,
    COST_LOW_SECONDS,
    REFERENCE_FORECASTS,
    REFERENCE_POINTS,
    cost_level,
    growth_shape,
    running_cost_seconds,
)
from tsforecastbench.bench.methods.scorecard import GrowthShape, Level  # noqa: E402


def _linear(sizes: list[float], per_point: float) -> list[float]:
    """Timings that grow exactly in proportion to the record."""
    return [n * per_point for n in sizes]


SIZES = [1_000.0, 10_000.0, 100_000.0]


def test_cost_level_bands():
    assert cost_level(0.0) is Level.LOW
    assert cost_level(COST_LOW_SECONDS - 0.1) is Level.LOW
    assert cost_level(COST_LOW_SECONDS) is Level.MEDIUM
    assert cost_level(COST_HIGH_SECONDS - 0.1) is Level.MEDIUM
    assert cost_level(COST_HIGH_SECONDS) is Level.HIGH


def test_a_perfectly_linear_cost_interpolates_to_the_reference():
    """One fit at 1e-4 s per point is 10 s on a 100,000-point record, whatever sizes it
    was measured on."""
    seconds = running_cost_seconds(SIZES, _linear(SIZES, 1e-4), [0.0, 0.0, 0.0])
    assert seconds == pytest.approx(10.0)


def test_forecasts_are_counted_not_just_the_fit():
    """A method with a trivial fit and a small per-forecast cost still lands high, because
    the rating amortises one fit over `REFERENCE_FORECASTS` forecasts."""
    seconds = running_cost_seconds(SIZES, [0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
    assert seconds == REFERENCE_FORECASTS * 0.5


def test_a_constant_cost_is_flat_and_survives_interpolation():
    """The neural models train to a step ceiling, so a 1k record costs what a 100k one
    does — the reference size must not invent growth that isn't there."""
    constant = [80.0, 80.0, 80.0]
    assert running_cost_seconds(SIZES, constant, [0.0, 0.0, 0.0]) == pytest.approx(80.0)
    assert growth_shape(SIZES, constant, [0.0, 0.0, 0.0]) is GrowthShape.FLAT


def test_growth_shape_separates_linear_from_superlinear():
    assert growth_shape(SIZES, _linear(SIZES, 1e-4), [0.0] * 3) is GrowthShape.LINEAR
    quadratic = [n**2 * 1e-9 for n in SIZES]
    assert growth_shape(SIZES, quadratic, [0.0] * 3) is GrowthShape.SUPERLINEAR


def test_growth_shape_follows_whichever_term_dominates():
    """Theta's shape is its quadratic predict, not its mild fit — the label has to describe
    the number the meter shows."""
    mild_fit = _linear(SIZES, 1e-5)
    quadratic_predict = [n**2 * 1e-9 for n in SIZES]
    assert growth_shape(SIZES, mild_fit, quadratic_predict) is GrowthShape.SUPERLINEAR
    # ...and the mirror image: a heavy flat fit outweighs a mild growing predict.
    assert growth_shape(SIZES, [500.0] * 3, [1e-5] * 3) is GrowthShape.FLAT


def test_negligible_timings_do_not_get_a_slope_fitted():
    """A zero-shot fit is microseconds of clock noise; regressing it would describe the
    timer rather than the method."""
    noise = [1e-6, 3e-6, 2e-6]
    assert growth_shape(SIZES, noise, noise) is GrowthShape.FLAT
    assert running_cost_seconds(SIZES, noise, noise) < 1e-2


def test_the_noise_floor_applies_to_a_series_not_to_its_small_points():
    """Keeping only the points above the floor would fit the slope to the long cases alone.
    A flat series that happens to straddle the floor must still read flat."""
    straddling = [8e-4, 9e-4, 2e-3]
    assert growth_shape(SIZES, straddling, [0.0] * 3) is GrowthShape.FLAT


def test_a_single_observation_falls_back_to_its_own_value():
    assert running_cost_seconds([float(REFERENCE_POINTS)], [7.0], [0.0]) == 7.0
