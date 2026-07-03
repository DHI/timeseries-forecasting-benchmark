"""Tests for scripts/review_case.py — the per-case quality review.

Exercises the measurable checks on synthetic data so they run headless with no
network and no committed Parquet dependence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import review_case as rc  # noqa: E402

from tsforecastbench.data.registry import get_case  # noqa: E402

PASS, WARN, FAIL = rc.PASS, rc.WARN, rc.FAIL


def test_run_lengths():
    assert rc._run_lengths(np.array([], dtype=bool)) == []
    assert rc._run_lengths(np.array([False, False])) == []
    assert rc._run_lengths(np.array([True, True, False, True])) == [2, 1]
    assert rc._run_lengths(np.array([True, True, True])) == [3]


def test_worst_aggregation():
    assert rc._worst([rc.Check("a", PASS, ""), rc.Check("b", WARN, "")]) == WARN
    assert rc._worst([rc.Check("a", PASS, ""), rc.Check("b", FAIL, "")]) == FAIL
    assert rc._worst([]) == PASS


def test_empty_exogenous_is_flagged_fail():
    """The core motivating bug: a declared exogenous the download never populated."""
    case = get_case("waves_oregon")
    df = case.synth()
    assert case.exogenous, "waves_oregon should declare exogenous"
    blanked = case.exogenous[0]
    df[blanked] = np.nan
    checks = rc.check_exogenous(case, df)
    by_name = {c.name: c for c in checks}
    assert by_name[f"exo[{blanked}]"].level == FAIL


def test_missing_exogenous_column_is_fail():
    case = get_case("waves_oregon")
    df = case.synth().drop(columns=[case.exogenous[0]])
    checks = rc.check_exogenous(case, df)
    assert any(c.level == FAIL for c in checks)


def test_univariate_case_passes_exogenous():
    case = get_case("demand_ltown")  # declares no exogenous
    checks = rc.check_exogenous(case, case.synth())
    assert len(checks) == 1 and checks[0].level == PASS


def test_fullness_flags_large_gap():
    case = get_case("waves_oregon")
    df = case.synth().sort_values("ds").reset_index(drop=True)
    # punch a gap far longer than the horizon into the middle of the target
    df.loc[1000:1500, "y"] = np.nan
    checks = rc.check_fullness(case, df, max_gap=pd.Timedelta("24h"))
    gap = next(c for c in checks if c.name == "target_gaps")
    assert gap.level == WARN
    assert gap.detail["largest_gap_steps"] >= 500


def test_fullness_full_series_has_no_gaps():
    case = get_case("demand_ltown")
    checks = rc.check_fullness(case, case.synth(), max_gap=pd.Timedelta("24h"))
    gap = next(c for c in checks if c.name == "target_gaps")
    assert gap.level == PASS


def test_forecastable_runs_on_synth():
    case = get_case("waves_oregon")
    checks, res = rc.check_forecastable(case, prefer_synth=True)
    assert any(c.name == "forecastable" for c in checks)
    assert len(res) > 0


@pytest.mark.parametrize("cid", ["demand_ltown", "waves_oregon"])
def test_review_returns_valid_structure(cid):
    case = get_case(cid)
    result = rc.review(case, max_gap=pd.Timedelta(case.horizon_primary))
    assert result["verdict"] in (PASS, WARN, FAIL)
    assert set(result["sections"]) >= {
        "config",
        "fullness",
        "exogenous",
        "seasonality",
        "forecastable",
    }
    # reports are serialisable
    jpath, mpath = rc.write_reports(result)
    assert jpath.exists() and mpath.exists()


# --- seasonality drift alarm ------------------------------------------------


def _seasonality_case(seasonalities):
    """A Case with no data files, so check_seasonality falls back to the df we pass."""
    from tsforecastbench.data.schema import Case

    return Case(
        id="dummy_seasonality",
        domain="test",
        target="y",
        freq="1h",
        seasonalities=seasonalities,
        horizon_primary="24h",
        input_window="7D",
        splits={"train_end": "2019-04-01", "val_end": "2019-04-15", "test_end": "2019-05-01"},
        source={"name": "test"},
    )


def _hourly_df(y):
    ds = pd.date_range("2019-01-01", periods=len(y), freq="1h")
    return pd.DataFrame({"unique_id": "dummy", "ds": ds, "y": y})


def _diurnal(n, noise=0.3, seed=3):
    rng = np.random.default_rng(seed)
    return np.sin(2 * np.pi * np.arange(n) / 24) + noise * rng.standard_normal(n)


def _noise(n, seed=3):
    return np.random.default_rng(seed).standard_normal(n)


def test_seasonality_consistent_declaration_passes():
    checks = rc.check_seasonality(_seasonality_case([24]), _hourly_df(_diurnal(24 * 120)))
    assert [c.level for c in checks] == [PASS]


def test_seasonality_unsupported_declaration_warns():
    checks = rc.check_seasonality(_seasonality_case([24]), _hourly_df(_noise(24 * 120)))
    assert any(c.name == "seasonality_declared_unsupported" and c.level == WARN for c in checks)


def test_seasonality_undeclared_supported_period_warns():
    checks = rc.check_seasonality(_seasonality_case([]), _hourly_df(_diurnal(24 * 120)))
    assert any(c.name == "seasonality_undeclared" and c.level == WARN for c in checks)


def test_seasonality_confirmed_nonseasonal_passes():
    checks = rc.check_seasonality(_seasonality_case([]), _hourly_df(_noise(24 * 120)))
    assert [c.level for c in checks] == [PASS]
    assert "non-seasonal" in checks[0].message
