"""Unit tests for seasonality resolution helpers and detection evidence."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.data.seasonality import (  # noqa: E402
    detect_candidates,
    dominant_period,
    mase_season_for,
    proposed_seasonalities,
    suggest_peak_candidates,
)

RNG = np.random.default_rng(7)


def _case_kwargs(**overrides) -> dict:
    base = dict(
        id="dummy",
        domain="test",
        target="y",
        freq="1h",
        seasonalities=[24, 168],
        horizon_primary="24h",
        input_window="7D",
        splits={"train_end": "2020-01-01", "val_end": "2021-01-01", "test_end": "2022-01-01"},
        source={"name": "test"},
    )
    base.update(overrides)
    return base


# --- schema -----------------------------------------------------------------


def test_case_accepts_declared_seasonalities():
    from tsforecastbench.data.schema import Case

    assert Case(**_case_kwargs()).seasonalities == [24, 168]


def test_case_accepts_explicitly_nonseasonal():
    from tsforecastbench.data.schema import Case

    assert Case(**_case_kwargs(seasonalities=[])).seasonalities == []


def test_case_requires_seasonalities():
    import pytest

    from tsforecastbench.data.schema import Case

    kwargs = _case_kwargs()
    del kwargs["seasonalities"]
    with pytest.raises(ValueError):
        Case(**kwargs)


def test_case_rejects_period_below_two_steps():
    import pytest

    from tsforecastbench.data.schema import Case

    with pytest.raises(ValueError, match=">= 2 steps"):
        Case(**_case_kwargs(seasonalities=[1, 24]))


def test_case_rejects_duplicate_periods():
    import pytest

    from tsforecastbench.data.schema import Case

    with pytest.raises(ValueError, match="unique"):
        Case(**_case_kwargs(seasonalities=[24, 24]))


def test_declared_order_is_preserved_dominant_first():
    from tsforecastbench.data.schema import Case

    assert Case(**_case_kwargs(seasonalities=[365, 7])).seasonalities == [365, 7]


def _sine(n: int, period: float, amp: float = 1.0) -> np.ndarray:
    return amp * np.sin(2 * np.pi * np.arange(n) / period)


# --- resolution -------------------------------------------------------------


def test_dominant_period_is_first_element():
    assert dominant_period([24, 168]) == 24


def test_dominant_period_of_nonseasonal_case_is_one():
    assert dominant_period([]) == 1


def test_mase_season_uses_dominant_when_train_holds_two_cycles():
    assert mase_season_for([365, 7], n_train=800) == 365


def test_mase_season_falls_through_to_next_period_on_short_train():
    assert mase_season_for([365, 7], n_train=400) == 7


def test_mase_season_is_one_when_no_period_fits():
    assert mase_season_for([365], n_train=400) == 1


def test_mase_season_is_one_for_nonseasonal_case():
    assert mase_season_for([], n_train=10_000) == 1


# --- detection --------------------------------------------------------------


def test_detects_daily_cycle_in_hourly_series():
    y = _sine(24 * 90, period=24) + 0.3 * RNG.standard_normal(24 * 90)

    evidence = detect_candidates(y, "1h")

    assert proposed_seasonalities(evidence) == [24]


def test_detects_both_daily_and_weekly_cycles_dominant_first():
    n = 24 * 7 * 30
    y = _sine(n, period=24, amp=2.0) + _sine(n, period=168, amp=1.0)
    y += 0.3 * RNG.standard_normal(n)

    evidence = detect_candidates(y, "1h")

    assert proposed_seasonalities(evidence) == [24, 168]


def test_detects_annual_cycle_in_daily_series():
    y = _sine(365 * 6, period=365) + 0.3 * RNG.standard_normal(365 * 6)

    evidence = detect_candidates(y, "1D")

    assert proposed_seasonalities(evidence) == [365]


def test_genuine_cycle_in_heavy_noise_is_accepted_despite_diluted_acf():
    # A real cycle buried in strong high-frequency noise concentrates in one periodogram
    # bin (huge spectral ratio) while its ACF is diluted to ~0.1 (regression: demand_ltown).
    y = _sine(24 * 90, period=24) + 2.0 * RNG.standard_normal(24 * 90)

    evidence = detect_candidates(y, "1h")

    assert 24 in proposed_seasonalities(evidence)


def test_white_noise_accepts_nothing():
    evidence = detect_candidates(RNG.standard_normal(24 * 90), "1h")

    assert proposed_seasonalities(evidence) == []


def test_red_noise_persistence_is_not_mistaken_for_a_cycle():
    # AR(1) with high persistence has large ACF at any modest lag but no spectral peak.
    n = 24 * 90
    y = np.zeros(n)
    eps = RNG.standard_normal(n)
    for i in range(1, n):
        y[i] = 0.98 * y[i - 1] + eps[i]

    evidence = detect_candidates(y, "1h")

    assert proposed_seasonalities(evidence) == []


def test_rejected_candidates_are_still_reported_with_evidence():
    y = _sine(24 * 90, period=24) + 0.3 * RNG.standard_normal(24 * 90)

    evidence = detect_candidates(y, "1h")

    assert {e["period"] for e in evidence} == {12, 24, 168}
    rejected = [e for e in evidence if not e["accepted"]]
    assert all(np.isfinite(e["acf"]) or np.isnan(e["acf"]) for e in rejected)


def test_extra_candidate_joins_the_menu():
    y = _sine(25 * 200, period=25) + 0.1 * RNG.standard_normal(25 * 200)

    evidence = detect_candidates(y, "1h", extra_candidates=(25,))

    assert proposed_seasonalities(evidence) == [25]


def test_candidate_without_two_full_cycles_is_rejected():
    y = _sine(365, period=365) + 0.1 * RNG.standard_normal(365)

    evidence = detect_candidates(y, "1D")

    annual = next(e for e in evidence if e["period"] == 365)
    assert not annual["accepted"]


def test_detection_tolerates_interior_nans():
    y = _sine(24 * 90, period=24) + 0.3 * RNG.standard_normal(24 * 90)
    y[100:130] = np.nan

    evidence = detect_candidates(y, "1h")

    assert proposed_seasonalities(evidence) == [24]


# --- peak suggestions --------------------------------------------------------


def test_suggests_off_menu_peak():
    # 17 is not on the 1h menu (12, 24, 168), so this cycle can only surface as a
    # periodogram-peak suggestion, not through the menu-based detect_candidates.
    y = _sine(17 * 100, period=17) + 0.1 * RNG.standard_normal(17 * 100)

    assert 17 in suggest_peak_candidates(y, "1h")


def test_suggestions_exclude_menu_periods():
    y = _sine(24 * 90, period=24) + 0.3 * RNG.standard_normal(24 * 90)

    assert 24 not in suggest_peak_candidates(y, "1h")


def test_suggestions_exclude_caller_supplied_periods():
    y = _sine(17 * 100, period=17) + 0.1 * RNG.standard_normal(17 * 100)

    assert 17 not in suggest_peak_candidates(y, "1h", exclude=(17,))


def test_white_noise_suggests_nothing():
    y = RNG.standard_normal(24 * 90)

    assert suggest_peak_candidates(y, "1h") == []


def test_suggestions_drop_leakage_near_a_menu_period():
    # A 13-step candidate is spectral leakage of the on-menu 12-step cycle (within 10%),
    # not an independent seasonality, so it should not be suggested.
    y = _sine(12 * 200, period=12) + 0.1 * RNG.standard_normal(12 * 200)

    assert 13 not in suggest_peak_candidates(y, "1h")
    assert 11 not in suggest_peak_candidates(y, "1h")
