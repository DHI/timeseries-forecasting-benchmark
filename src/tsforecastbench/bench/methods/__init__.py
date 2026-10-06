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
from tsforecastbench.bench.methods.dl import DLinearExoForecaster, DLinearForecaster
from tsforecastbench.bench.methods.ensemble import (
    EnsembleForecaster,
    ensemble_baselines,
    ensemble_baselines_theta,
)
from tsforecastbench.bench.methods.fm import (
    Chronos2ExoForecaster,
    Chronos2Forecaster,
    ChronosForecaster,
    Moirai2ExoForecaster,
    Moirai2Forecaster,
    TimesFMExoForecaster,
    TimesFMForecaster,
)
from tsforecastbench.bench.methods.ml import (
    CatBoostExoForecaster,
    CatBoostForecaster,
    LightGBMExoForecaster,
    LightGBMForecaster,
    LinearLagsExoForecaster,
    LinearLagsForecaster,
    MLPExoForecaster,
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
    TFTExoForecaster,
    TFTForecaster,
    TimesNetExoForecaster,
    TimesNetForecaster,
)
from tsforecastbench.bench.methods.torchdl import (
    NHITSExoForecaster,
    NHITSForecaster,
    PatchTSTExoForecaster,
    PatchTSTForecaster,
)

__all__ = [
    "AutoARIMAForecaster",
    "AutoETSForecaster",
    "CatBoostExoForecaster",
    "CatBoostForecaster",
    "Chronos2ExoForecaster",
    "Chronos2Forecaster",
    "ChronosForecaster",
    "Climatology",
    "DLinearExoForecaster",
    "DLinearForecaster",
    "EnsembleForecaster",
    "Forecaster",
    "ITransformerForecaster",
    "LightGBMExoForecaster",
    "LightGBMForecaster",
    "LinearLagsExoForecaster",
    "LinearLagsForecaster",
    "MLPExoForecaster",
    "MLPForecaster",
    "MSTLAutoARIMAForecaster",
    "MSTLAutoETSForecaster",
    "MeanBaseline",
    "Moirai2ExoForecaster",
    "Moirai2Forecaster",
    "NHITSExoForecaster",
    "NHITSForecaster",
    "PatchTSTExoForecaster",
    "PatchTSTForecaster",
    "Persistence",
    "SeasonalNaive",
    "TBATSForecaster",
    "TFTExoForecaster",
    "TFTForecaster",
    "ThetaForecaster",
    "TimesFMExoForecaster",
    "TimesFMForecaster",
    "TimesNetExoForecaster",
    "TimesNetForecaster",
    "ensemble_baselines",
    "ensemble_baselines_theta",
]
