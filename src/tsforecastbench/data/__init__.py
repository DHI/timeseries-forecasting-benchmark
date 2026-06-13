"""Data concern: case definitions, schema, downloads, and the case registry.

Everything needed to *build and load* the benchmark cases lives here — independent of
any forecasting method. This is the half of the repo that could eventually be released
as a standalone open dataset.
"""

from tsforecastbench.data.registry import discover_cases, get_case
from tsforecastbench.data.schema import Case, load_case_from_toml

__all__ = ["Case", "discover_cases", "get_case", "load_case_from_toml"]
