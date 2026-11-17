"""Checks for scripts/merge_pilot_results.py's `--drop` and `--drop-case` options.

`--drop` is how a partial run's known-bad column is kept out of the published matrix while
its file stays on disk (research log entry 101). Both guards below matter: a mistyped tag or
method would otherwise silently republish the column it was meant to hold back.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from merge_pilot_results import _apply_drops, _parse_drops, _tag  # noqa: E402


def test_tag_strips_the_prefix_and_suffix():
    assert _tag(Path("docs/results/partial/pilot_metrics_lookback_v2.parquet")) == "lookback_v2"


def test_parse_drops_groups_methods_by_tag():
    assert _parse_drops(["lookback_v2:TFT", "wm_full:TFT,TFT+exo"], ["lookback_v2", "wm_full"]) == {
        "lookback_v2": {"TFT"},
        "wm_full": {"TFT", "TFT+exo"},
    }


@pytest.mark.parametrize("spec", ["typo_tag:TFT", "lookback_v2", "lookback_v2:"])
def test_parse_drops_rejects_a_spec_it_cannot_honour(spec):
    with pytest.raises(SystemExit):
        _parse_drops([spec], ["lookback_v2"])


def test_apply_drops_removes_only_the_named_method():
    df = pd.DataFrame({"case_id": ["a"] * 3, "method": ["TFT", "TFT", "N-HiTS"]})
    kept = _apply_drops(df, "lookback_v2", "metrics", methods={"TFT"})
    assert kept["method"].tolist() == ["N-HiTS"]


def test_apply_drops_rejects_a_method_the_file_does_not_have():
    df = pd.DataFrame({"case_id": ["a"], "method": ["N-HiTS"]})
    with pytest.raises(SystemExit):
        _apply_drops(df, "lookback_v2", "metrics", methods={"TFT"})


def test_apply_drops_removes_every_row_of_a_dropped_case():
    df = pd.DataFrame(
        {"case_id": ["surge_cuxhaven", "surge_cuxhaven", "waves_oregon"], "method": ["TFT"] * 3}
    )
    kept = _apply_drops(df, "surge_full", "metrics", cases={"surge_cuxhaven"})
    assert kept["case_id"].tolist() == ["waves_oregon"]


def test_apply_drops_rejects_a_case_the_file_does_not_have():
    df = pd.DataFrame({"case_id": ["waves_oregon"], "method": ["TFT"]})
    with pytest.raises(SystemExit):
        _apply_drops(df, "first_six", "metrics", cases={"surge_cuxhaven"})


def test_apply_drops_checks_methods_and_cases_against_the_file_as_read():
    """`surge_full` holds only surge rows: dropping the case and a method together works."""
    df = pd.DataFrame({"case_id": ["surge_cuxhaven"] * 2, "method": ["N-HiTS", "Naive"]})
    kept = _apply_drops(df, "surge_full", "metrics", {"N-HiTS"}, {"surge_cuxhaven"})
    assert kept.empty


def test_apply_drops_skips_a_missing_method_when_not_required():
    """Snapshots: `first_six` holds neuralforecast scores but no neuralforecast forecasts."""
    df = pd.DataFrame({"case_id": ["a"], "method": ["N-HiTS"]})
    kept = _apply_drops(df, "first_six", "snapshot", methods={"TFT"}, required=False)
    assert kept["method"].tolist() == ["N-HiTS"]
