"""Benchmark concern: forecasting methods, evaluation, and result visualisation.

Everything needed to *compare forecasting models* across the cases lives here. It depends
on the data concern (`tsforecastbench.data`) only through the public `Case` interface — the
two halves are deliberately separable (see research_plan.md on the eventual repo split).
"""

from tsforecastbench.bench.eval import run_case
from tsforecastbench.bench.eval.rolling import run_leadtime_curve

__all__ = ["run_case", "run_leadtime_curve"]
