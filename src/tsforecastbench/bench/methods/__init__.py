"""Method wrappers - every method exposes the same fit/predict interface."""

from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.bench.methods.classical import AutoARIMAForecaster, AutoETSForecaster
from tsforecastbench.bench.methods.ml import (
    LightGBMExoForecaster,
    LightGBMForecaster,
    LinearLagsExoForecaster,
    LinearLagsForecaster,
    MLPForecaster,
)
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
    "LightGBMExoForecaster",
    "LightGBMForecaster",
    "LinearLagsExoForecaster",
    "LinearLagsForecaster",
    "MLPForecaster",
    "MeanBaseline",
    "Persistence",
    "SeasonalNaive",
]
