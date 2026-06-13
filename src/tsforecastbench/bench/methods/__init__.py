"""Method wrappers - every method exposes the same fit/predict interface."""

from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.bench.methods.classical import AutoARIMAForecaster, AutoETSForecaster
from tsforecastbench.bench.methods.ml import LightGBMForecaster, LinearLagsForecaster, MLPForecaster
from tsforecastbench.bench.methods.naive import (
    Climatology,
    MeanBaseline,
    Persistence,
    SeasonalNaive,
)

__all__ = [
    "AutoARIMAForecaster",
    "AutoETSForecaster",
    "Climatology",
    "Forecaster",
    "LightGBMForecaster",
    "LinearLagsForecaster",
    "MLPForecaster",
    "MeanBaseline",
    "Persistence",
    "SeasonalNaive",
]
