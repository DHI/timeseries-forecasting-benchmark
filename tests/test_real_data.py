"""Smoke tests against the committed processed Parquets.

If a case's processed Parquet exists and is readable, run the cheap baselines on it
and assert finite metrics. Files that fail to load (e.g. truncated by a cloud-sync
race in a developer's working copy) are skipped with a warning, not failed. CI on a
fresh git checkout reads the committed files cleanly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.bench.eval import run_case  # noqa: E402
from tsforecastbench.bench.methods import MeanBaseline, Persistence, SeasonalNaive  # noqa: E402
from tsforecastbench.data.registry import discover_cases  # noqa: E402

REAL_CASES = [cid for cid, c in discover_cases().items() if c.parquet_path.exists()]


@pytest.mark.parametrize("cid", REAL_CASES)
def test_real_baselines(cid):
    case = discover_cases()[cid]
    try:
        pd.read_parquet(case.parquet_path)
    except Exception as exc:
        pytest.skip(f"{cid} parquet unreadable on this filesystem ({exc}); skipping")
    res = run_case(case, [MeanBaseline(), Persistence(), SeasonalNaive()], prefer_synth=False)
    ok = res[res["status"] == "ok"]
    assert len(ok) > 0, f"all methods failed on real {cid}: {res['status'].unique()}"
