"""The DL lookback matches the windowed-ML lag reach, floored at two horizons.

`lookback_for` and `_lags_for` are the two halves of one decision -- how far back a method
may look -- so this pins the resulting window for every real case. The table is the point:
the rule moves 11 of the 12 cases, in both directions, and a later edit that silently
widened or narrowed one of them would otherwise pass.
"""

from __future__ import annotations

import pytest

from tsforecastbench.bench.eval.loop import _horizon_to_steps
from tsforecastbench.bench.methods.base import _lags_for, lookback_for
from tsforecastbench.data.registry import discover_cases

# case id -> (expected lookback, which term binds)
EXPECTED = {
    "demand_ltown": (168, "lag"),
    "discharge_flashy_catskills": (365, "lag"),
    "discharge_lowland_denmark": (365, "lag"),
    "discharge_saugeen": (365, "lag"),
    "discharge_snowmelt_yosemite": (365, "lag"),
    "groundwater_yorktown": (365, "lag"),
    "watertemp_james": (365, "lag"),
    "sewerlevel_bellinge": (96, "lag"),
    "surge_cuxhaven": (96, "2h"),
    "waves_massachusetts": (48, "2h"),
    "waves_oregon": (48, "2h"),
    "wwtp_influent_n2o_350k": (48, "2h"),
}


@pytest.mark.parametrize("cid", sorted(EXPECTED))
def test_lookback_is_pinned_per_case(cid):
    case = discover_cases()[cid]
    h = _horizon_to_steps(case.horizon_primary, case.freq)
    expected, binds = EXPECTED[cid]
    assert lookback_for(case, h) == expected
    # Which term binds is part of the claim: where "lag" binds the DL reach equals the ML
    # reach exactly, and where "2h" binds the horizon floor forces it wider.
    if binds == "lag":
        assert max(_lags_for(case)) >= 2 * h
        assert lookback_for(case, h) == max(_lags_for(case))
    else:
        assert 2 * h > max(_lags_for(case))
        assert lookback_for(case, h) == 2 * h


def test_lookback_never_falls_short_of_the_horizon():
    """A window shorter than two horizons is what the floor exists to prevent."""
    for case in discover_cases().values():
        h = _horizon_to_steps(case.horizon_primary, case.freq)
        assert lookback_for(case, h) >= 2 * h


def test_lookback_covers_the_ml_lag_reach():
    """The DL window reaches at least as far back as the sparse ML ladder does."""
    for case in discover_cases().values():
        h = _horizon_to_steps(case.horizon_primary, case.freq)
        assert lookback_for(case, h) >= max(_lags_for(case))
