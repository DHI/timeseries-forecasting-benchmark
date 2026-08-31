"""Checks for scripts/merge_pilot_results.py's `--drop` option.

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
    df = pd.DataFrame({"method": ["TFT", "TFT", "N-HiTS"], "value": [1.0, 2.0, 3.0]})
    kept = _apply_drops(df, "lookback_v2", {"TFT"}, "metrics")
    assert kept["method"].tolist() == ["N-HiTS"]


def test_apply_drops_rejects_a_method_the_file_does_not_have():
    df = pd.DataFrame({"method": ["N-HiTS"], "value": [3.0]})
    with pytest.raises(SystemExit):
        _apply_drops(df, "lookback_v2", {"TFT"}, "metrics")
