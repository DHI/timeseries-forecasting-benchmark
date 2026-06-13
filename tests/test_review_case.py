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

from tsforecastbench.registry import get_case  # noqa: E402

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
    case = get_case("ndbc_waves")
    df = case.synth()
    assert case.exogenous, "ndbc_waves should declare exogenous"
    blanked = case.exogenous[0]
    df[blanked] = np.nan
    checks = rc.check_exogenous(case, df)
    by_name = {c.name: c for c in checks}
    assert by_name[f"exo[{blanked}]"].level == FAIL


def test_missing_exogenous_column_is_fail():
    case = get_case("ndbc_waves")
    df = case.synth().drop(columns=[case.exogenous[0]])
    checks = rc.check_exogenous(case, df)
    assert any(c.level == FAIL for c in checks)


def test_univariate_case_passes_exogenous():
    case = get_case("wd_battledim")  # declares no exogenous
    checks = rc.check_exogenous(case, case.synth())
    assert len(checks) == 1 and checks[0].level == PASS


def test_fullness_flags_large_gap():
    case = get_case("ndbc_waves")
    df = case.synth().sort_values("ds").reset_index(drop=True)
    # punch a gap far longer than the horizon into the middle of the target
    df.loc[1000:1500, "y"] = np.nan
    checks = rc.check_fullness(case, df, max_gap=pd.Timedelta("24h"))
    gap = next(c for c in checks if c.name == "target_gaps")
    assert gap.level == WARN
    assert gap.detail["largest_gap_steps"] >= 500


def test_fullness_full_series_has_no_gaps():
    case = get_case("wd_battledim")
    checks = rc.check_fullness(case, case.synth(), max_gap=pd.Timedelta("24h"))
    gap = next(c for c in checks if c.name == "target_gaps")
    assert gap.level == PASS


def test_forecastable_runs_on_synth():
    case = get_case("ndbc_waves")
    checks, res = rc.check_forecastable(case, prefer_synth=True)
    assert any(c.name == "forecastable" for c in checks)
    assert len(res) > 0


@pytest.mark.parametrize("cid", ["wd_battledim", "ndbc_waves"])
def test_review_returns_valid_structure(cid):
    case = get_case(cid)
    result = rc.review(case, max_gap=pd.Timedelta(case.horizon_primary))
    assert result["verdict"] in (PASS, WARN, FAIL)
    assert set(result["sections"]) >= {"config", "fullness", "exogenous", "forecastable"}
    # reports are serialisable
    jpath, mpath = rc.write_reports(result)
    assert jpath.exists() and mpath.exists()
