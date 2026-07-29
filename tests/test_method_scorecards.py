"""Guards on the per-method Capabilities/Scorecard declarations.

These are hand-authored, so nothing in the code can check whether a rating is *right*.
What can be checked is that every method has one, that the field names the web app's
display config mirrors haven't moved, and that the two cross-field rules hold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.bench import methods as _methods  # noqa: E402
from tsforecastbench.bench.methods.scorecard import (  # noqa: E402
    Capabilities,
    Level,
    Scorecard,
)
from tsforecastbench.webdata import _method_registry  # noqa: E402

# The web app duplicates these names in web/src/data/scorecardConfig.ts (labels, order and
# polarity) and web/src/types/data.ts. Renaming a field on either side without the other
# would silently drop a row from the method card, so pin them here.
EXPECTED_CAPABILITY_FIELDS = ("trend", "seasonality", "multiple_seasonality", "exogenous")
EXPECTED_SCORECARD_FIELDS = (
    "history_requirements",
    "interpretability",
    "fit_cost",
    "inference_cost",
    "tuning_effort",
    "robustness_to_noise",
)


def _exo_pairs() -> list[tuple[type, type]]:
    """(+exo class, its non-exo base) for every registered +exo variant."""
    pairs = []
    for attr_name in _methods.__all__:
        cls = getattr(_methods, attr_name)
        if isinstance(cls, type) and cls.__name__.endswith("ExoForecaster"):
            pairs.append((cls, cls.__mro__[1]))
    return pairs


def test_capability_field_names_are_stable():
    assert tuple(Capabilities.model_fields) == EXPECTED_CAPABILITY_FIELDS


def test_scorecard_field_names_are_stable():
    assert tuple(Scorecard.model_fields) == EXPECTED_SCORECARD_FIELDS


def test_every_registered_method_declares_both():
    """A method that never got a declaration raises AttributeError inside the registry
    build (the attributes are annotated but unassigned on `Forecaster`), so reaching this
    assertion at all is most of the check."""
    registry = _method_registry()
    assert registry
    for name, meta in registry.items():
        assert set(meta["capabilities"]) == set(EXPECTED_CAPABILITY_FIELDS), name
        assert set(meta["scorecard"]) == set(EXPECTED_SCORECARD_FIELDS), name
        assert all(isinstance(v, bool) for v in meta["capabilities"].values()), name
        assert all(v in set(Level) for v in meta["scorecard"].values()), name


def test_multiple_seasonality_requires_seasonality():
    with pytest.raises(ValidationError, match="requires seasonality"):
        Capabilities(trend=False, seasonality=False, multiple_seasonality=True, exogenous=False)


def test_exo_variants_inherit_their_base_declarations():
    """A method and its +exo variant are the same method — +exo records which configuration
    this benchmark ran, not a different estimator — so neither block may differ."""
    pairs = _exo_pairs()
    assert pairs, "no +exo variants found; the naming convention this test relies on moved"
    for exo_cls, base_cls in pairs:
        assert exo_cls.capabilities == base_cls.capabilities, exo_cls.__name__
        assert exo_cls.scorecard == base_cls.scorecard, exo_cls.__name__


def test_methods_that_consume_covariates_declare_the_capability():
    """The reverse does not hold: a method can support covariates and still be scored in a
    column that doesn't feed them (plain LightGBM), which is the whole point of the pair."""
    for name, meta in _method_registry().items():
        if meta["uses_exogenous"]:
            assert meta["capabilities"]["exogenous"] is True, name
