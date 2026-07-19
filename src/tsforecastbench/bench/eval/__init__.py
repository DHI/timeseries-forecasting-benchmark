"""Evaluation: splits, metrics, and the run loop."""

from tsforecastbench.bench.eval.config import DEFAULT_ORIGINS
from tsforecastbench.bench.eval.loop import run_case
from tsforecastbench.bench.eval.metrics import kge, mae, mape, mase, nse, rmse, skill_score, smape
from tsforecastbench.bench.eval.rolling import run_case_rolling, run_leadtime_curve

__all__ = [
    "DEFAULT_ORIGINS",
    "kge",
    "mae",
    "mape",
    "mase",
    "nse",
    "rmse",
    "run_case",
    "run_case_rolling",
    "run_leadtime_curve",
    "skill_score",
    "smape",
]
