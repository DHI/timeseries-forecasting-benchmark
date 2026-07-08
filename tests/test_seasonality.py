"""Unit tests for seasonality resolution helpers and detection evidence."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.data.seasonality import (  # noqa: E402
    detect_candidates,
    dominant_period,
    mase_season_for,
    position_in_cycle,
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


# --- position_in_cycle ------------------------------------------------------


def test_position_in_cycle_canonical_hourly_is_hour_of_day():
    idx = pd.date_range("2020-03-01", periods=50, freq="1h")
    assert list(position_in_cycle("1h", 24, idx)) == [t % 24 for t in range(50)]


def test_position_in_cycle_canonical_daily_is_weekday():
    # 2020-01-06 is a Monday (dayofweek 0).
    idx = pd.date_range("2020-01-06", periods=9, freq="1d")
    assert list(position_in_cycle("1d", 7, idx)) == [0, 1, 2, 3, 4, 5, 6, 0, 1]


def test_position_in_cycle_normalizes_freq_alias():
    idx = pd.date_range("2020-03-01", periods=30, freq="1h")
    assert np.array_equal(position_in_cycle("H", 24, idx), position_in_cycle("1h", 24, idx))


def test_position_in_cycle_noncanonical_period_falls_back_to_step_count():
    # A period not in the canonical menu keys on an epoch-anchored step count: contiguous
    # 0..p-1 that wraps at p, so every position is populated and stays within range.
    idx = pd.date_range("2020-01-01", periods=40, freq="1h")
    pos = position_in_cycle("1h", 13, idx)
    assert set(pos.tolist()) == set(range(13))
    assert pos.min() == 0 and pos.max() == 12


def test_position_in_cycle_bounded_below_period():
    idx = pd.date_range("2020-01-01", periods=1000, freq="15min")
    for period in (96, 672, 100):  # two canonical, one epoch-fallback
        pos = position_in_cycle("15min", period, idx)
        assert pos.min() >= 0 and pos.max() < period


def test_position_in_cycle_is_deterministic_across_calls():
    # Fit and predict must derive the same position for the same timestamp.
    idx = pd.date_range("2021-07-01 03:00", periods=200, freq="1h")
    assert np.array_equal(position_in_cycle("1h", 168, idx), position_in_cycle("1h", 168, idx))


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


def test_dominant_period_picks_period_that_explains_most_variance():
    n = 24 * 60
    y = _sine(n, period=168, amp=3.0) + _sine(n, period=24, amp=0.5)
    # 168 dominates even though 24 is declared first.
    assert dominant_period([24, 168], y) == 168


def test_dominant_period_picks_first_when_it_explains_most_variance():
    n = 24 * 60
    y = _sine(n, period=24, amp=3.0) + _sine(n, period=168, amp=0.5)
    assert dominant_period([24, 168], y) == 24


def test_dominant_period_of_single_declared_period_is_that_period():
    # A lone declared period is returned without scoring, whatever the target looks like.
    assert dominant_period([24], RNG.standard_normal(500)) == 24


def test_dominant_period_of_nonseasonal_case_is_one():
    assert dominant_period([], RNG.standard_normal(500)) == 1


def test_dominant_period_falls_back_to_first_when_target_is_degenerate():
    # Too few finite points to detrend -> keep the curated most-prominent-first order.
    assert dominant_period([24, 168], np.array([1.0, 2.0])) == 24


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


def test_mstl_corroborates_when_acf_is_too_diluted_to_pass_alone():
    # Heavier noise than the ACF-dilution case above: raw-lag ACF drops below ACF_MIN,
    # but MSTL's LOESS-smoothed seasonal component still separates the cycle from iid
    # noise, so it corroborates the spectral peak and the candidate is still accepted.
    # Uses its own RNG (not the shared module-level RNG) so it doesn't perturb the draw
    # sequence other tests rely on.
    rng = np.random.default_rng(11)
    y = _sine(24 * 90, period=24) + 4.0 * rng.standard_normal(24 * 90)

    evidence = detect_candidates(y, "1h")

    e24 = next(e for e in evidence if e["period"] == 24)
    assert e24["acf"] < 0.1
    assert e24["mstl_strength"] >= 0.05
    assert e24["accepted"]


def test_mstl_strength_is_nan_when_period_too_long_for_the_sample_window():
    # 600 doesn't fit MIN_CYCLES within MSTL_SAMPLE_ROWS=1000, so MSTL is skipped for it
    # even though the full training split easily supports it; evidence falls back to ACF.
    rng = np.random.default_rng(12)
    y = _sine(600 * 5, period=600, amp=5.0) + 0.1 * rng.standard_normal(600 * 5)

    evidence = detect_candidates(y, "1D", extra_candidates=(600,))

    e600 = next(e for e in evidence if e["period"] == 600)
    assert np.isnan(e600["mstl_strength"])
    assert e600["accepted"]


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


# --- RobustPeriod: MODWT -----------------------------------------------------

from tsforecastbench.data.seasonality import (  # noqa: E402
    _band_period,
    _biweight_midvariance,
    _huber_clean,
    _huber_periodogram_at,
    _modwt,
)


def test_modwt_preserves_energy():
    # The MODWT is an energy-preserving transform: the sum of squared detail coefficients
    # across all levels plus the squared final scaling coefficients equals the input energy.
    # This is the tightest single check that the hardcoded sym4 filter bank is correct.
    for n in (256, 2556, 5000):
        x = RNG.standard_normal(n)
        level = min(int(np.floor(np.log2(n / 4))), 11)
        details, scaling = _modwt(x, level)
        energy_out = float(np.sum(details**2) + np.sum(scaling**2))
        assert energy_out == pytest.approx(float(np.sum(x**2)), rel=1e-9)


def test_modwt_localises_period_to_its_dyadic_band():
    # A cycle of period P concentrates its energy in the level whose band [2**j, 2**(j+1)]
    # contains P: period 24 -> level 4 ([16, 32]), period 168 -> level 7 ([128, 256]).
    n = 24 * 7 * 40
    y = _sine(n, period=24) + 0.7 * _sine(n, period=168)
    details, _ = _modwt(y, 9)
    variances = [_biweight_midvariance(w) for w in details]
    assert int(np.argmax(variances)) + 1 == 4
    # The runner-up energy level holds the weekly cycle.
    assert int(np.argsort(variances)[-2]) + 1 == 7


# --- RobustPeriod: robust estimators -----------------------------------------


def test_huber_clean_pulls_in_spikes_but_keeps_the_cycle():
    n = 24 * 40
    y = _sine(n, period=24)
    y[::97] += 50.0  # sparse large spikes
    cleaned = _huber_clean(y)
    # The spikes are winsorised to near the bulk range, but the cycle is preserved.
    assert cleaned.max() < 5.0
    assert np.corrcoef(cleaned, _sine(n, period=24))[0, 1] > 0.9


def test_huber_clean_still_winsorises_when_the_mad_is_zero():
    # A zero-inflated series (most values identical) has MAD 0; the cleaning must fall back
    # to another scale and still pull the spikes in, not return the data untouched.
    y = np.zeros(200)
    y[::20] = 1000.0
    cleaned = _huber_clean(y)
    assert cleaned.max() < 300.0


def test_huber_periodogram_recovers_amplitude_despite_outliers():
    n = 24 * 60
    amp = 3.0
    clean = amp * np.sin(2 * np.pi * np.arange(n) / 24)
    spiked = clean.copy()
    spiked[::50] += 40.0  # outliers that would wreck an ordinary (least-squares) periodogram
    power = _huber_periodogram_at(spiked, 1.0 / 24)
    recovered_amp = np.sqrt(4.0 * power / n)
    assert recovered_amp == pytest.approx(amp, rel=0.15)


def test_band_period_reads_period_within_the_band():
    n = 24 * 80
    coeffs = _sine(n, period=24)
    assert _band_period(coeffs, 16, 32) == 24
    # A period outside the queried band is not returned; the strongest in-band lag is.
    assert _band_period(coeffs, 100, 200) != 24


# --- RobustPeriod: discovery -------------------------------------------------

from tsforecastbench.data.seasonality import (  # noqa: E402
    robust_period_candidates,
)


def _prominent_periods(result) -> list[int]:
    return [c["period"] for c in result["candidates"] if c["prominent"]]


def test_discovers_multiple_coexisting_cycles():
    n = 24 * 7 * 40
    y = _sine(n, period=24, amp=2.0) + _sine(n, period=168, amp=1.0)
    y += 0.3 * RNG.standard_normal(n)

    prominent = _prominent_periods(robust_period_candidates(y, "1h"))

    assert 24 in prominent
    assert any(abs(p - 168) / 168 <= 0.05 for p in prominent)


def test_discovery_ranks_by_spectral_prominence():
    # The stronger cycle (24) must outrank the weaker (168) in the returned order.
    n = 24 * 7 * 40
    y = _sine(n, period=24, amp=3.0) + _sine(n, period=168, amp=1.0)
    y += 0.3 * RNG.standard_normal(n)

    periods = [c["period"] for c in robust_period_candidates(y, "1h")["candidates"]]

    assert periods[0] == 24


def test_discovery_survives_trend_and_spikes():
    # RobustPeriod's whole point: a linear trend and sparse spikes do not stop it recovering
    # the cycles, where a plain periodogram would be swamped.
    n = 24 * 7 * 40
    y = _sine(n, period=24, amp=2.0) + _sine(n, period=168, amp=1.0)
    y += 0.002 * np.arange(n)  # trend
    spikes = RNG.choice(n, size=n // 200, replace=False)
    y[spikes] += 30.0 * RNG.standard_normal(len(spikes))

    prominent = _prominent_periods(robust_period_candidates(y, "1h"))

    assert 24 in prominent


def test_discovers_off_menu_period_and_leaves_it_unsnapped():
    # 17 is not on the 1h menu (12, 24, 168); it should surface as a prominent candidate with
    # no menu_period attached.
    y = _sine(17 * 200, period=17) + 0.3 * RNG.standard_normal(17 * 200)

    result = robust_period_candidates(y, "1h")
    seventeen = next(c for c in result["candidates"] if c["period"] == 17)

    assert seventeen["prominent"]
    assert seventeen["menu_period"] is None


def test_near_menu_discovery_carries_the_menu_period():
    # A cycle a little off a menu period still points at it for reference (within 10%).
    y = _sine(366 * 8, period=366) + 0.2 * RNG.standard_normal(366 * 8)

    result = robust_period_candidates(y, "1D")
    annual = next(c for c in result["candidates"] if c["menu_period"] == 365)

    assert annual["prominent"]
    assert abs(annual["period"] - 365) / 365 <= 0.1


def test_flags_a_harmonic_of_a_stronger_cycle():
    # A non-sinusoidal (here, two-harmonic) cycle of period 24 puts real power at 12 as well;
    # 12 must be flagged as a harmonic of the stronger 24, not read as an independent cycle.
    n = 24 * 120
    t = np.arange(n)
    y = np.sin(2 * np.pi * t / 24) + 0.5 * np.sin(2 * np.pi * t / 12)
    y += 0.2 * RNG.standard_normal(n)

    candidates = robust_period_candidates(y, "1h")["candidates"]
    twelve = next(c for c in candidates if c["period"] == 12)

    assert twelve["harmonic_of"] == 24


def test_nyquist_alternating_series_yields_no_garbage_amplitude():
    # A period-2 (Nyquist) cycle sits where the Huber periodogram's cos/sin design is
    # degenerate: sin(pi*t) is numerically zero at every sample, so the fit returns an
    # astronomically wrong amplitude (~1e11 for a unit signal). Such candidates are skipped.
    rng = np.random.default_rng(3)  # a seed that surfaced a period-2 candidate before the guard
    y = (-1.0) ** np.arange(400) + 0.3 * rng.standard_normal(400)

    candidates = robust_period_candidates(y, "1h")["candidates"]

    assert all(c["period"] != 2 for c in candidates)
    assert all(c["robust_amplitude"] < 10.0 for c in candidates)


def test_white_noise_yields_no_prominent_cycles():
    result = robust_period_candidates(RNG.standard_normal(24 * 90), "1h")

    assert _prominent_periods(result) == []
    assert set(result["menu_periods_missing"]) == {12, 24, 168}


def test_long_series_white_noise_yields_no_prominent_cycles():
    # The spectral-peak window spans ~0.04*n/period Fourier bins, and the ratio takes the max
    # over them: at n=100k a fixed bar let that max clear POWER_RATIO_MIN on a quarter of pure-
    # noise draws. The size-adjusted prominence bar must keep long noise non-prominent.
    y = np.random.default_rng(0).standard_normal(100_000)

    assert _prominent_periods(robust_period_candidates(y, "1h")) == []


def test_red_noise_yields_no_prominent_cycles():
    # Strongly persistent AR(1) noise is the classic seasonality decoy on environmental data;
    # discovery may list its bumps but must not flag them prominent.
    from scipy.signal import lfilter

    for seed in (0, 3):
        e = np.random.default_rng(seed).standard_normal(20_000)
        ar = lfilter([1.0], [1.0, -0.99], e)

        assert _prominent_periods(robust_period_candidates(ar, "1h")) == []


def test_long_period_is_flagged_approximate():
    # An annual cycle seen over only six years (< 10 cycles) is resolution-limited, so it is
    # flagged approximate even though it is found; with many cycles the flag clears.
    n = 365 * 6
    y = _sine(n, period=365) + 0.2 * RNG.standard_normal(n)

    result = robust_period_candidates(y, "1D")
    annual = next(c for c in result["candidates"] if c["menu_period"] == 365)

    assert annual["approximate"]
    assert annual["prominent"]


def test_discovery_is_fast_on_a_long_series():
    # Guards against an accidental O(n^2) path: a 150k-point series must finish quickly.
    import time

    n = 150_000
    y = _sine(n, period=24) + _sine(n, period=168) + 0.5 * RNG.standard_normal(n)
    start = time.perf_counter()
    robust_period_candidates(y, "1h")
    assert time.perf_counter() - start < 15.0
