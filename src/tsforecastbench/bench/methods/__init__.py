"""Method wrappers - every method exposes the same fit/predict interface."""

from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.bench.methods.classical import (
    AutoARIMAForecaster,
    AutoETSForecaster,
    MSTLAutoARIMAForecaster,
    MSTLAutoETSForecaster,
    TBATSForecaster,
    ThetaForecaster,
)
from tsforecastbench.bench.methods.dl import DLinearForecaster
from tsforecastbench.bench.methods.ensemble import (
    EnsembleForecaster,
    ensemble_baselines,
    ensemble_baselines_theta,
)
from tsforecastbench.bench.methods.fm import ChronosForecaster
from tsforecastbench.bench.methods.ml import (
    CatBoostExoForecaster,
    CatBoostForecaster,
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
from tsforecastbench.bench.methods.nf_models import (
    ITransformerForecaster,
    TFTForecaster,
    TimesNetForecaster,
)
from tsforecastbench.bench.methods.torchdl import NHITSForecaster, PatchTSTForecaster

__all__ = [
    "AutoARIMAForecaster",
    "AutoETSForecaster",
    "CatBoostExoForecaster",
    "CatBoostForecaster",
    "ChronosForecaster",
    "Climatology",
    "DLinearForecaster",
    "EnsembleForecaster",
    "Forecaster",
    "ITransformerForecaster",
    "LightGBMExoForecaster",
    "LightGBMForecaster",
    "LinearLagsExoForecaster",
    "LinearLagsForecaster",
    "MLPForecaster",
    "MSTLAutoARIMAForecaster",
    "MSTLAutoETSForecaster",
    "MeanBaseline",
    "NHITSForecaster",
    "PatchTSTForecaster",
    "Persistence",
    "SeasonalNaive",
    "TBATSForecaster",
    "TFTForecaster",
    "ThetaForecaster",
    "TimesNetForecaster",
    "ensemble_baselines",
    "ensemble_baselines_theta",
]
