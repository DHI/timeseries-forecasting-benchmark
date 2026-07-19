"""Per-case seasonality: resolution helpers and offline detection evidence.

The benchmark has a single source of truth for seasonal periods: the `seasonalities`
field each case declares in its case.toml (periods in steps of the case frequency,
curated most-prominent first, `[]` = non-seasonal). Which declared period dominates a
given training window is re-selected at fit time by variance explained (see
`dominant_period`); the committed order is what curation records and what
`mase_season_for` walks. The resolution helpers here are the only seasonality logic that
runs during the method train/predict cycle.

The detection half is curation tooling, run once per case when the field is set (see
notebooks/seasonality.py): candidates come from a fixed, physically motivated menu per
frequency and are accepted or rejected by evidence measured on the training split — a
periodogram peak prominent against its local spectral neighborhood (robust to the red
spectra typical of environmental series), corroborated by either the autocorrelation at
that lag or an MSTL seasonal-strength score. The human reviews the evidence table and
commits the accepted periods; nothing here is consulted at fit/predict time.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd

# Physically motivated candidate periods (in steps) per sampling frequency: tidal/semi-
# diurnal, diurnal, weekly for sub-daily data; weekly, annual for daily data. Detection
# only ever accepts periods from this menu (plus explicitly justified per-case extras),
# so spectral noise can never mint a meaningless period into benchmark metadata.
CANDIDATE_MENU: dict[str, tuple[int, ...]] = {
    "1d": (7, 365),
    "1h": (12, 24, 168),
    "15min": (96, 672),
    "1w": (52,),
}

# Evidence thresholds (documented once, applied everywhere). A candidate is accepted
# when the training split holds >= MIN_CYCLES full cycles, its periodogram peak exceeds
# POWER_RATIO_MIN x the median power of its local frequency neighborhood, and *either*
# the ACF at the candidate lag clears ACF_MIN *or* its MSTL seasonal strength clears
# MSTL_STRENGTH_MIN. The spectral ratio is the discriminator (red-noise persistence
# inflates ACF at any modest lag but produces no local spectral peak); ACF/MSTL are two
# independent ways to corroborate that the peak is a real cycle and not an artifact. The
# ACF floor is deliberately low: a genuine cycle buried in heavy high-frequency noise
# concentrates in a single periodogram bin but its ACF is diluted toward zero
# (demand_ltown's unmistakable diurnal cycle — spectral ratio ~700 — sits at ACF(24)
# ~= 0.10). MSTL's LOESS-smoothed seasonal component is far less diluted by iid noise,
# so it can corroborate exactly the cases where ACF is weakest.
MIN_CYCLES = 2
POWER_RATIO_MIN = 10.0
ACF_MIN = 0.1
MSTL_STRENGTH_MIN = 0.05

# MSTL is only run on candidates that already clear the periodogram+cycle gates above
# (no point decomposing spectral noise), and only over the most recent MSTL_SAMPLE_ROWS
# observations rather than the full training split: STL cost scales with n x period
# (benchmarked: ~40s for n=100_000/period=672, ~0.1s for n=1_000/period=365), and 1000
# rows still holds >= MIN_CYCLES for every period this project has ever declared (the
# longest, 365, gets ~2.7 cycles). Periods too long to clear MIN_CYCLES within that
# window are silently skipped (evidence falls back to ACF alone, as before MSTL existed)
# rather than paying for a slow fit on a barely-resolved cycle.
MSTL_SAMPLE_ROWS = 1000


def normalize_freq(freq: str) -> str:
    """Canonical key for a pandas-style frequency string ("1D", "h", "15T" ...)."""
    f = freq.lower()
    aliases = {
        "d": "1d",
        "1d": "1d",
        "h": "1h",
        "1h": "1h",
        "15t": "15min",
        "15min": "15min",
        "w": "1w",
        "1w": "1w",
    }
    return aliases.get(f, f)


def _variance_explained(yd: np.ndarray, period: int) -> float:
    """R^2 of `yd` regressed on one sinusoid (sin + cos) at `period` steps: the portion of
    variance a signal of that period explains. Uses two parameters regardless of period, so
    it is comparable across periods (unlike a per-position seasonal mean, whose parameter
    count grows with the period and mechanically favours the longest one), and is phase-
    invariant because both sin and cos are fit, so it needs no calendar anchoring."""
    n = len(yd)
    t = np.arange(n)
    ang = 2 * np.pi * t / period
    design = np.column_stack([np.sin(ang), np.cos(ang), np.ones(n)])
    coef, *_ = np.linalg.lstsq(design, yd, rcond=None)
    resid = yd - design @ coef
    ss_tot = float(np.sum((yd - yd.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - float(resid @ resid) / ss_tot


def dominant_period(seasonalities: list[int], y: np.ndarray) -> int:
    """Dominant declared period, chosen at fit time by variance explained on `y`; 1 if
    non-seasonal. Among the periods a case declares, the dominant one is whichever sinusoid
    of that period explains the most variance of the detrended training target — measured
    per fold, so each training window keys on the cycle it actually shows most strongly
    rather than on a fixed order baked in at curation time. Ties break to the earlier
    declared period (the curated most-prominent order). Callers guarantee every declared
    period is supported by the window; an unsupportable period must not be declared in the
    case.toml. Falls back to the first declared period when the target is too degenerate to
    detrend."""
    if not seasonalities:
        return 1
    if len(seasonalities) == 1:
        return int(seasonalities[0])
    try:
        yd = _fill_and_detrend(y)
    except ValueError:
        return int(seasonalities[0])
    scores = [_variance_explained(yd, p) for p in seasonalities]
    return int(seasonalities[int(np.argmax(scores))])


# Position-in-cycle from calendar fields for the canonical (freq, period) pairs: exact
# across DST-free naive timestamps. For the 365-day "annual" cycle, drop Feb 29 in leap
# years so positions stay in 0..364.
def _dayofyear_365(idx: pd.DatetimeIndex) -> np.ndarray:
    doy0 = idx.dayofyear.to_numpy(dtype=int) - 1
    leap = idx.is_leap_year.to_numpy(dtype=bool)
    after_feb28 = doy0 >= 59
    return doy0 - (leap & after_feb28).astype(int)

_CANONICAL_POSITIONS = {
    ("1d", 7): lambda idx: idx.dayofweek,
    ("1d", 365): _dayofyear_365,
    ("1h", 12): lambda idx: idx.hour % 12,
    ("1h", 24): lambda idx: idx.hour,
    ("1h", 168): lambda idx: idx.dayofweek * 24 + idx.hour,
    ("15min", 96): lambda idx: idx.hour * 4 + idx.minute // 15,
    ("15min", 672): lambda idx: idx.dayofweek * 96 + idx.hour * 4 + idx.minute // 15,
}


def position_in_cycle(freq: str, period: int, idx: pd.DatetimeIndex) -> np.ndarray:
    """Integer position `0..period-1` of each timestamp within a cycle of `period` steps.

    Canonical (freq, period) pairs read the position off calendar fields — exact across
    leap years and DST-free naive timestamps; any other period falls back to an epoch-
    anchored step count modulo `period`, which matches at fit and predict time because
    both use the same naive timestamps. This is the single position-in-cycle rule shared
    by the sin/cos ML features and the Climatology baseline, so both key on the case's
    declared seasonality the same way at every frequency.
    """
    calendar_pos = _CANONICAL_POSITIONS.get((normalize_freq(freq), period))
    if calendar_pos is not None:
        return np.asarray(calendar_pos(idx))
    step_ns = pd.Timedelta(pd.tseries.frequencies.to_offset(freq)).value
    return (idx.asi8 // step_ns) % period


def mase_season_for(seasonalities: list[int], n_train: int) -> int:
    """Season for the MASE denominator: the first declared period the training series
    can support (>= 2 full cycles), walking the list dominant-first; 1 (persistence
    scaling) when none fits or the case is non-seasonal. The denominator is therefore
    always a real season or the unit lag, never an arbitrary truncated one."""
    for p in seasonalities:
        if n_train >= 2 * p:
            return int(p)
    return 1


class CandidateEvidence(TypedDict):
    period: int
    n_cycles: float
    power_ratio: float
    acf: float
    mstl_strength: float
    accepted: bool


def _fill_and_detrend(y: np.ndarray) -> np.ndarray:
    """Interpolate interior NaNs and remove mean + linear trend (trend leakage would
    otherwise pile power into the lowest frequencies and skew neighborhood medians)."""
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 3:
        raise ValueError("need at least 3 finite values for seasonality detection")
    idx = np.arange(len(y))
    y = np.interp(idx, idx[finite], y[finite])
    slope, intercept = np.polyfit(idx, y, 1)
    return y - (slope * idx + intercept)


def _acf_at(y: np.ndarray, lag: int) -> float:
    if lag >= len(y) - 1:
        return float("nan")
    a, b = y[:-lag], y[lag:]
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _pacf(y: np.ndarray, max_lag: int) -> np.ndarray:
    """Partial autocorrelation for lags 1..max_lag (Yule-Walker, unadjusted). Unlike
    `_acf_at`, each lag's value nets out the shorter lags already regressed out — so a
    candidate that's an exact multiple of a shorter, more prominent one already on the
    menu (e.g. 24 = 2x12 on hourly data) only shows a partial-autocorrelation spike if it
    carries information beyond what the shorter cycle already explains, rather than
    inheriting a spike purely from the shorter cycle's own persistence (as ACF does).
    Reviewer-facing diagnostic only — not part of the `detect_candidates` acceptance
    gate, since Yule-Walker requires max_lag < half the sample size."""
    from statsmodels.tsa.stattools import pacf

    return pacf(y, nlags=max_lag, method="ywm")[1:]


def _spectral_ratio(freqs: np.ndarray, power: np.ndarray, period: int) -> float:
    """Peak periodogram power within 2% of the candidate frequency, relative to the
    median power of the surrounding octave band (excluding the peak bins). The window
    is deliberately tight: widening it inflates the max over more noise bins, which is
    how red-noise variance fakes a peak."""
    f0 = 1.0 / period
    in_peak = np.abs(freqs - f0) <= 0.02 * f0
    if not in_peak.any():
        in_peak[np.argmin(np.abs(freqs - f0))] = True
    peak = float(power[in_peak].max())
    neighborhood = (freqs >= f0 / 2) & (freqs <= 2 * f0) & ~in_peak & (freqs > 0)
    base = float(np.median(power[neighborhood])) if neighborhood.any() else float("nan")
    if not np.isfinite(base) or base <= 0:
        return float("nan")
    return peak / base


def _mstl_decompose(y: np.ndarray, periods: list[int]) -> tuple[object, list[int]] | None:
    """Interpolate the last MSTL_SAMPLE_ROWS observations and run one joint MSTL
    decomposition (statsforecast's, a thin wrapper over `statsmodels.tsa.STL`) over the
    subset of `periods` that clear MIN_CYCLES within that window — shortest period first,
    so a longer period is decomposed against the residual left after shorter cycles are
    already stripped out, discounting a period that's really just spectral leakage of a
    shorter one. Returns `(decomposition_df, fitted_periods)`, or None if nothing in
    `periods` is fittable or the series is too degenerate to interpolate."""
    from statsforecast.mstl import mstl

    sample = np.asarray(y, dtype=float)[-MSTL_SAMPLE_ROWS:]
    finite = np.isfinite(sample)
    if finite.sum() < 3:
        return None
    idx = np.arange(len(sample))
    sample = np.interp(idx, idx[finite], sample[finite])

    fittable = sorted(p for p in periods if len(sample) // p >= MIN_CYCLES)
    if not fittable:
        return None

    decomposition = mstl(sample, period=fittable if len(fittable) > 1 else fittable[0])
    return decomposition, fittable


def _mstl_strengths(y: np.ndarray, periods: list[int]) -> dict[int, float]:
    """Seasonal strength (Hyndman & Athanasopoulos: 1 - Var(remainder) / Var(remainder +
    seasonal)) for each of `periods`, from `_mstl_decompose`. Periods that don't clear
    MIN_CYCLES within the MSTL_SAMPLE_ROWS window are dropped from the result; returns {}
    if none do or MSTL can't fit (degenerate series)."""
    fit = _mstl_decompose(y, periods)
    if fit is None:
        return {}
    decomposition, fittable = fit
    remainder = decomposition["remainder"].to_numpy()
    strengths: dict[int, float] = {}
    for p in fittable:
        seasonal = decomposition["seasonal" if len(fittable) == 1 else f"seasonal{p}"].to_numpy()
        total_var = np.var(remainder + seasonal)
        strengths[p] = max(0.0, 1.0 - np.var(remainder) / total_var) if total_var > 0 else 0.0
    return strengths


def detect_candidates(
    y: np.ndarray,
    freq: str,
    extra_candidates: tuple[int, ...] = (),
) -> list[CandidateEvidence]:
    """Evidence for every candidate period of `freq` (menu + justified extras), measured
    on the training target `y`. Returns all candidates — rejected ones included, so the
    reviewer sees why — ordered by descending spectral prominence."""
    from scipy.signal import periodogram

    candidates = sorted(set(CANDIDATE_MENU.get(normalize_freq(freq), ())) | set(extra_candidates))
    yd = _fill_and_detrend(y)
    freqs, power = periodogram(yd, fs=1.0)

    n_cycles_by_period = {p: len(yd) / p for p in candidates}
    ratio_by_period = {p: _spectral_ratio(freqs, power, p) for p in candidates}
    plausible = [
        p
        for p in candidates
        if n_cycles_by_period[p] >= MIN_CYCLES
        and np.isfinite(ratio_by_period[p])
        and ratio_by_period[p] >= POWER_RATIO_MIN
    ]
    mstl_strength_by_period = _mstl_strengths(y, plausible)

    evidence: list[CandidateEvidence] = []
    for p in candidates:
        n_cycles = n_cycles_by_period[p]
        ratio = ratio_by_period[p]
        acf = _acf_at(yd, p)
        strength = mstl_strength_by_period.get(p, float("nan"))
        accepted = (
            n_cycles >= MIN_CYCLES
            and np.isfinite(ratio)
            and ratio >= POWER_RATIO_MIN
            and (
                (np.isfinite(acf) and acf >= ACF_MIN)
                or (np.isfinite(strength) and strength >= MSTL_STRENGTH_MIN)
            )
        )
        evidence.append(
            CandidateEvidence(
                period=int(p),
                n_cycles=round(n_cycles, 1),
                power_ratio=float(ratio),
                acf=float(acf),
                mstl_strength=float(strength),
                accepted=bool(accepted),
            )
        )
    evidence.sort(key=lambda e: -e["power_ratio"] if np.isfinite(e["power_ratio"]) else np.inf)
    return evidence


def suggest_peak_candidates(
    y: np.ndarray,
    freq: str,
    exclude: tuple[int, ...] = (),
    max_candidates: int = 5,
) -> list[int]:
    """Periods (in steps) at local periodogram maxima that aren't already on the fixed
    menu for `freq` — a discovery aid for cycles the menu doesn't anticipate (e.g. a
    site-specific swell period). Ranked by spectral ratio, filtered to >= MIN_CYCLES full
    cycles and a spectral ratio clearing POWER_RATIO_MIN (the same prominence bar
    `detect_candidates` applies) so ordinary periodogram noise doesn't get surfaced.
    Peaks within 10% of a menu/excluded/more-prominent period are dropped as spectral
    leakage of that already-covered cycle, not an independent one.

    This is advisory only: it does not run the ACF acceptance check from
    `detect_candidates`. The reviewer adds a promising suggestion to the notebook's
    extra-candidates box to get its full evidence before deciding whether to accept it.
    """
    from scipy.signal import periodogram

    yd = _fill_and_detrend(y)
    freqs, power = periodogram(yd, fs=1.0)
    skip = set(CANDIDATE_MENU.get(normalize_freq(freq), ())) | set(exclude)

    is_peak = np.zeros(len(power), dtype=bool)
    is_peak[1:-1] = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
    peak_idx = np.where(is_peak & (freqs > 0))[0]

    scored: list[tuple[int, float]] = []
    for i in peak_idx:
        period = round(1.0 / freqs[i])
        if (
            period < 2
            or len(yd) / period < MIN_CYCLES
            or any(abs(period - s) / s <= 0.1 for s in skip)
        ):
            continue
        ratio = _spectral_ratio(freqs, power, period)
        if np.isfinite(ratio) and ratio >= POWER_RATIO_MIN:
            scored.append((period, ratio))
    scored.sort(key=lambda t: -t[1])

    suggestions: list[int] = []
    for period, _ in scored:
        if all(abs(period - p) / p > 0.1 for p in suggestions):
            suggestions.append(period)
        if len(suggestions) >= max_candidates:
            break
    return suggestions


def proposed_seasonalities(evidence: list[CandidateEvidence]) -> list[int]:
    """Accepted periods ordered dominant-first (descending spectral prominence) — the
    value to review and commit as `seasonalities` in the case.toml.

    Accepted periods within 10% of a more prominent accepted period are dropped: they
    are the same cycle seen through spectral leakage (e.g. a genuine 25-step cycle also
    lights up a 24-step candidate), not a second seasonality.
    """
    accepted = sorted((e for e in evidence if e["accepted"]), key=lambda e: -e["power_ratio"])
    proposed: list[int] = []
    for e in accepted:
        if all(abs(e["period"] - p) / p > 0.1 for p in proposed):
            proposed.append(e["period"])
    return proposed
