"""Data loading for the results-matrix app.

Reads `results/pilot_metrics.parquet` directly — no new artifacts, no pipeline changes.
Method family is not a column in that file (adding it would require rerunning the whole
pilot); instead it's read straight off the `Forecaster` classes in
`tsforecastbench.bench.methods`, which is the actual source of truth, and joined in
memory at load time.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import streamlit as st

from tsforecastbench.bench import methods as _methods

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
METRICS_PATH = RESULTS_DIR / "pilot_metrics.parquet"
CASE_REVIEW_DIR = RESULTS_DIR / "case_review"

# Display order for method families (left-to-right column grouping) and case domains
# (top-to-bottom row grouping). Anything not listed sorts after, alphabetically.
FAMILY_ORDER = ["baseline", "ensemble", "classical", "ml", "deep_learning", "foundation_model"]

# EnsembleForecaster's `name` is set per-instance by these factories, not as a class
# attribute, so it can't be read off the class the way every other method's can.
_ENSEMBLE_FACTORIES = ("ensemble_baselines", "ensemble_baselines_theta")


def _method_family_map() -> dict[str, str]:
    """method name -> family, derived from the Forecaster classes."""
    mapping: dict[str, str] = {}
    for attr_name in _methods.__all__:
        obj = getattr(_methods, attr_name)
        if attr_name in _ENSEMBLE_FACTORIES:
            instance = obj()
            mapping[instance.name] = instance.family
        elif (
            inspect.isclass(obj)
            and issubclass(obj, _methods.Forecaster)
            and obj not in (_methods.Forecaster, _methods.EnsembleForecaster)
        ):
            mapping[obj.name] = obj.family
    return mapping


def _rank(order: list[str]):
    return lambda value: order.index(value) if value in order else len(order)


@st.cache_data
def load_metrics(version: int = 0) -> pd.DataFrame:
    """Load pilot_metrics.parquet and join each method's family.

    `version` has no effect on the result; bump it (e.g. from a sidebar reload button)
    to bust the Streamlit cache and pick up a pilot run that finished after the app
    started.
    """
    del version
    df = pd.read_parquet(METRICS_PATH)
    family_map = _method_family_map()
    df["family"] = df["method"].map(family_map).fillna("unclassified")
    return df


def ordered_cases(df: pd.DataFrame) -> list[str]:
    """Case ids sorted by domain (FAMILY_ORDER-style grouping isn't meaningful for
    domains, so alphabetical), then case id, for stable row grouping with separators."""
    pairs = df[["domain", "case_id"]].drop_duplicates()
    pairs = pairs.sort_values(["domain", "case_id"])
    return list(pairs["case_id"])


def ordered_methods(df: pd.DataFrame) -> list[str]:
    """Method names sorted by family (FAMILY_ORDER), then name, for stable column
    grouping with separators."""
    pairs = df[["family", "method"]].drop_duplicates()
    pairs["_family_rank"] = pairs["family"].map(_rank(FAMILY_ORDER))
    pairs = pairs.sort_values(["_family_rank", "method"])
    return list(pairs["method"])


def load_case_notes(case_id: str) -> str | None:
    """Markdown case-review notes for a case, or None if it hasn't been reviewed."""
    path = CASE_REVIEW_DIR / f"{case_id}.md"
    if not path.exists():
        return None
    return path.read_text()
