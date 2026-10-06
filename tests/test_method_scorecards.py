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
from tsforecastbench.bench.methods.cost import (  # noqa: E402
    cost_level,
    growth_shape,
    running_cost_seconds,
)
from tsforecastbench.bench.methods.scorecard import (  # noqa: E402
    Capabilities,
    ForecastStrategy,
    GrowthShape,
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
    "running_cost",
    "tuning_effort",
)
# `running_cost` is the one field that isn't a bare Level: it carries a banded level and the
# measured growth shape together, because the shape qualifies the level rather than standing
# beside it (see `RunningCost` in scorecard.py).
EXPECTED_RUNNING_COST_FIELDS = ("level", "growth")


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
        graded = {k: v for k, v in meta["scorecard"].items() if k != "running_cost"}
        assert all(v in set(Level) for v in graded.values()), name
        cost = meta["scorecard"]["running_cost"]
        assert set(cost) == set(EXPECTED_RUNNING_COST_FIELDS), name
        assert cost["level"] in set(Level), name
        assert cost["growth"] in set(GrowthShape), name
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


def _measured_running_costs() -> dict[str, tuple[Level, GrowthShape]]:
    """Every method's running cost read off the pilot results, keyed by method name.

    Empty when either input is missing, which is how this file stays runnable in a checkout
    without results.
    """
    import json

    metrics = REPO_ROOT / "results" / "pilot_metrics.parquet"
    manifest = REPO_ROOT / "data" / "curated" / "manifest.json"
    if not metrics.exists() or not manifest.exists():
        return {}
    import pandas as pd

    rows = {case: meta["rows"] for case, meta in json.loads(manifest.read_text()).items()}
    frame = pd.read_parquet(metrics).drop_duplicates(subset=["case_id", "method", "horizon"])
    measured = {}
    for method, group in frame.groupby("method"):
        if not set(group["case_id"]).issubset(rows):
            continue
        sizes = [float(rows[c]) for c in group["case_id"]]
        fits = [float(v) for v in group["fit_seconds"]]
        predicts = [float(v) for v in group["predict_seconds"]]
        measured[str(method)] = (
            cost_level(running_cost_seconds(sizes, fits, predicts)),
            growth_shape(sizes, fits, predicts),
        )
    return measured


def test_declared_running_cost_matches_the_measured_results():
    """The one rating on the card that is a measurement rather than a judgement, so it is
    the one that can go stale silently — a method whose implementation gets faster or slower
    keeps whatever level it was given until something checks.

    Rated on the base configuration only: a method and its `+exo` variant share one
    declaration (see `test_exo_variants_inherit_their_base_declarations`), so measuring the
    `+exo` column would ask two different numbers to satisfy one field.
    """
    measured = _measured_running_costs()
    if not measured:
        pytest.skip("no pilot results in this checkout")
    registry = _method_registry()
    mismatches = []
    for name, meta in registry.items():
        if name.endswith("+exo") or name not in measured:
            continue
        level, growth = measured[name]
        declared = meta["scorecard"]["running_cost"]
        if declared["level"] != level or declared["growth"] != growth:
            mismatches.append(
                f"{name}: declared {declared['level']}/{declared['growth']}, "
                f"measured {level}/{growth}"
            )
    assert not mismatches, "running_cost has drifted from the results:\n" + "\n".join(mismatches)


def test_every_scored_method_is_covered_by_the_measurement():
    """Guards the guard: if the results and the registry stop agreeing on method names, the
    check above would pass by skipping everything it cannot match."""
    measured = _measured_running_costs()
    if not measured:
        pytest.skip("no pilot results in this checkout")
    uncovered = {n for n in _method_registry() if not n.endswith("+exo")} - set(measured)
    assert not uncovered, f"declared but never measured: {sorted(uncovered)}"
