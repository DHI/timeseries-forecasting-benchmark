"""Guards on the per-method Capabilities/Scorecard/ForecastStrategy declarations.

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
    ForecastStrategy,
    Level,
    Scorecard,
)
from tsforecastbench.webdata import _method_registry  # noqa: E402

# The web app duplicates these names in web/src/data/scorecardConfig.ts (labels, order and
# polarity) and web/src/types/data.ts. Renaming a field on either side without the other
# would silently drop a row from the method card, so pin them here.
EXPECTED_CAPABILITY_FIELDS = ("trend", "seasonality", "multiple_seasonality", "exogenous")
# What the exported JSON's "capabilities" dict actually carries: the hand-authored model
# fields above, plus "gaps" — computed in _method_registry() from `pools_segments`, not a
# Capabilities model field (see that function's docstring), so it's pinned separately here.
EXPECTED_EXPORTED_CAPABILITY_FIELDS = (*EXPECTED_CAPABILITY_FIELDS, "gaps")
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


def _registered_instances() -> dict[str, object]:
    """name -> instantiated Forecaster, mirroring _method_registry()'s own traversal of
    `_methods.__all__` so a test can read attributes (like `pools_segments`) that never
    make it into the exported JSON."""
    from tsforecastbench.webdata import _ENSEMBLE_FACTORIES

    instances: dict[str, object] = {}
    for attr_name in _methods.__all__:
        obj = getattr(_methods, attr_name)
        if attr_name in _ENSEMBLE_FACTORIES or (
            isinstance(obj, type)
            and issubclass(obj, _methods.Forecaster)
            and obj not in (_methods.Forecaster, _methods.EnsembleForecaster)
        ):
            instance = obj()
        else:
            continue
        instances[instance.name] = instance
    return instances


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
        assert set(meta["capabilities"]) == set(EXPECTED_EXPORTED_CAPABILITY_FIELDS), name
        assert set(meta["scorecard"]) == set(EXPECTED_SCORECARD_FIELDS), name
        assert all(isinstance(v, bool) for v in meta["capabilities"].values()), name
        assert all(v in set(Level) for v in meta["scorecard"].values()), name
        assert meta["forecast_strategy"] in set(ForecastStrategy), name


def test_forecast_strategy_classifications():
    """Locks in the specific classification decisions made for ForecastStrategy: Recursive
    covers both mlforecast's lag-feedback strategy and classical state-space recursions (the
    same own-forecast-feeds-forward mechanism), Direct covers mlforecast's direct
    multi-horizon strategy and every architecture producing the whole horizon in one call,
    and N/A covers the baselines and the ensemble, where the distinction is meaningless."""
    registry = _method_registry()
    expected = {
        "LightGBM": "Recursive",
        "CatBoost": "Recursive",
        "LinearLags": "Recursive",
        "AutoARIMA": "Recursive",
        "AutoETS": "Recursive",
        "TBATS": "Recursive",
        "Theta": "Recursive",
        "MSTL-ETS": "Recursive",
        "MSTL-ARIMA": "Recursive",
        "MLP": "Direct",
        "DLinear": "Direct",
        "N-HiTS": "Direct",
        "PatchTST": "Direct",
        "Chronos-Bolt": "Direct",
        "Mean": "N/A",
        "Persistence": "N/A",
        "SeasonalNaive": "N/A",
        "Climatology": "N/A",
        "EnsembleBaselines": "N/A",
    }
    for name, strategy in expected.items():
        assert registry[name]["forecast_strategy"] == strategy, name


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
        assert exo_cls.forecast_strategy == base_cls.forecast_strategy, exo_cls.__name__


def test_gaps_capability_matches_pools_segments():
    """`gaps` is computed from `pools_segments` in `_method_registry()` rather than
    hand-copied into each `Capabilities(...)` declaration, precisely so a subclass that
    overrides `pools_segments` alone (e.g. `ITransformerForecaster`, which turns it back off
    without touching the base's `capabilities`) can't leave a stale `gaps` value behind."""
    registry = _method_registry()
    for name, instance in _registered_instances().items():
        assert registry[name]["capabilities"]["gaps"] is instance.pools_segments, name


def test_methods_that_consume_covariates_declare_the_capability():
    """The reverse does not hold: a method can support covariates and still be scored in a
    column that doesn't feed them (plain LightGBM), which is the whole point of the pair."""
    for name, meta in _method_registry().items():
        if meta["uses_exogenous"]:
            assert meta["capabilities"]["exogenous"] is True, name
