"""Method wrappers - every method exposes the same fit/predict interface."""

from tsforecastbench.methods.base import Forecaster
from tsforecastbench.methods.classical import AutoARIMAForecaster, AutoETSForecaster
from tsforecastbench.methods.ml import LightGBMForecaster, LinearLagsForecaster, MLPForecaster
from tsforecastbench.methods.naive import (
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
