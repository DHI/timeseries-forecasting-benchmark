"""Per-case seasonality: resolution helpers and offline detection evidence.

The benchmark has a single source of truth for seasonal periods: the `seasonalities`
field each case declares in its case.toml (periods in steps of the case frequency,
dominant period first, `[]` = non-seasonal). The resolution helpers here are the only
seasonality logic that runs during the method train/predict cycle.

The detection half is curation tooling, run once per case when the field is set (see
notebooks/seasonality.py): candidates come from a fixed, physically motivated menu per
frequency and are accepted or rejected by evidence measured on the training split — a
periodogram peak prominent against its local spectral neighborhood (robust to the red
spectra typical of environmental series) cross-checked by the autocorrelation at that
lag. The human reviews the evidence table and commits the accepted periods; nothing
here is consulted at fit/predict time.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

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
# POWER_RATIO_MIN x the median power of its local frequency neighborhood, and the ACF at
# the candidate lag clears ACF_MIN. The spectral ratio is the discriminator (red-noise
# persistence inflates ACF at any modest lag but produces no local spectral peak); the
# ACF floor guards against isolated periodogram artifacts.
MIN_CYCLES = 2
POWER_RATIO_MIN = 10.0
ACF_MIN = 0.15


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


def dominant_period(seasonalities: list[int]) -> int:
    """Dominant declared period (first element by convention); 1 if non-seasonal."""
    return seasonalities[0] if seasonalities else 1


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

    evidence: list[CandidateEvidence] = []
    for p in candidates:
        n_cycles = len(yd) / p
        ratio = _spectral_ratio(freqs, power, p)
        acf = _acf_at(yd, p)
        accepted = (
            n_cycles >= MIN_CYCLES
            and np.isfinite(ratio)
            and ratio >= POWER_RATIO_MIN
            and np.isfinite(acf)
            and acf >= ACF_MIN
        )
        evidence.append(
            CandidateEvidence(
                period=int(p),
                n_cycles=round(n_cycles, 1),
                power_ratio=float(ratio),
                acf=float(acf),
                accepted=bool(accepted),
            )
        )
    evidence.sort(key=lambda e: -e["power_ratio"] if np.isfinite(e["power_ratio"]) else np.inf)
    return evidence


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
