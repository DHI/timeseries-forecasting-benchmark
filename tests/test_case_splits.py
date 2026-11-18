"""Each case's declared `split_fraction` follows the test-length rule on its curated data."""

from __future__ import annotations

import pandas as pd
import pytest

from tsforecastbench.data.curate import split_fraction_for
from tsforecastbench.data.registry import discover_cases

CURATED = [cid for cid, c in discover_cases().items() if c.curated_path.exists()]


@pytest.mark.parametrize("cid", CURATED)
def test_split_fraction_follows_the_test_length_rule(cid):
    case = discover_cases()[cid]
    curated = pd.read_parquet(case.curated_path)
    longest = max(case.seasonalities) if case.seasonalities else 1

    expected = split_fraction_for(curated["ds"][curated["y_observed"]], longest, case.freq)

    assert case.split_fraction == expected
