"""Per-case seasonality: position-in-cycle resolution and offline detection evidence.

The benchmark has a single source of truth for seasonal periods, and it is entirely
declarative: the `seasonalities` field each case declares in its case.toml (the set of
periods it carries, in steps of the case frequency, `[]` = non-seasonal) plus
`dominant_seasonality`, the one period every consumer that needs a single season reads.
Nothing here re-derives either at fit time. That used to happen — the dominant period was
re-selected per fit window by variance explained while the MASE denominator took the first
declared period instead, so the metric and the methods could resolve different periods for
one case (research log id 103). `position_in_cycle` is now the only seasonality logic left
in the train/predict cycle.

The detection half is curation tooling, run once per case when the fields are set (see
notebooks/seasonality.py): candidates come from a fixed, physically motivated menu per
frequency and are accepted or rejected by evidence measured on the training split — a
periodogram peak prominent against its local spectral neighborhood (robust to the red
spectra typical of environmental series), corroborated by either the autocorrelation at
that lag or an MSTL seasonal-strength score. `_variance_explained` scores a single sinusoid
at a candidate period, which is one input to the dominance choice the reviewer makes there.
The human reviews the evidence and commits both fields; nothing here is consulted at
fit/predict time.
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

# --- RobustPeriod discovery constants (curation tooling) --------------------
# The detection above scores a fixed menu of periods. RobustPeriod (Wen et al., SIGMOD 2021)
# is the complementary discovery half: it finds periods from the data itself — robust to
# outliers, trend, and red noise, and able to separate coexisting cycles — so an off-menu
# site-specific period (a swell period, a control-cycle) surfaces without a human eyeballing
# a periodogram. See `robust_period_candidates` for the pipeline and the reconciliation with
# CANDIDATE_MENU.

# sym4 (least-asymmetric Daubechies, 8-tap) decomposition filter bank, hardcoded so the MODWT
# needs no wavelet dependency. Values are PyWavelets' `Wavelet("sym4").dec_lo/.dec_hi`; the
# energy-preservation test guards against a transcription error.
_SYM4_DEC_LO = np.array(
    [
        -0.07576571478927333,
        -0.02963552764599851,
        0.49761866763201545,
        0.8037387518059161,
        0.29785779560527736,
        -0.09921954357684722,
        -0.012603967262037833,
        0.0322231006040427,
    ]
)
_SYM4_DEC_HI = np.array(
    [
        -0.0322231006040427,
        -0.012603967262037833,
        0.09921954357684722,
        0.29785779560527736,
        -0.8037387518059161,
        0.49761866763201545,
        0.02963552764599851,
        -0.07576571478927333,
    ]
)

# Huber knee for the robust periodogram/cleaning: 1.345 gives 95% efficiency under Gaussian
# noise (the standard M-estimator tuning, and what the RobustPeriod reference uses).
HUBER_ZETA = 1.345
# Cap on the number of MODWT levels (each level ~doubles the period band); 11 reaches periods
# of ~2000-4000 steps, past the longest cycle this project resolves (672).
MODWT_MAX_LEVEL = 11
# Proximity for snapping a discovered period to a menu period and for de-duplicating the same
# cycle found at adjacent wavelet levels — the 10% rule `proposed_seasonalities` already uses.
ROBUST_PROXIMITY = 0.1


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


# Position-in-cycle from calendar fields for the canonical (freq, period) pairs: exact
# across DST-free naive timestamps. For the 365-day "annual" cycle, drop Feb 29 in leap
# years so positions stay in 0..364.
def _dayofyear_365(idx: pd.DatetimeIndex) -> np.ndarray:
    doy0 = idx.dayofyear.to_numpy(dtype=int) - 1
    leap = np.asarray(idx.is_leap_year, dtype=bool)
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


class CandidateEvidence(TypedDict):
    period: int
    n_cycles: float
    power_ratio: float
    acf: float
    mstl_strength: float
    accepted: bool


def fill_and_detrend(y: np.ndarray) -> np.ndarray:
    """Interpolate interior NaNs and remove mean + linear trend (trend leakage would
    otherwise pile power into the lowest frequencies and skew neighborhood medians). The
    shared detrending convention for this module's own detection logic, and for anything
    outside it that needs the seasonal shape isolated from a case's long-run drift (e.g.
    the case panel's seasonal-pattern plot)."""
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 3:
        raise ValueError("need at least 3 finite values to detrend")
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


def _peak_window(freqs: np.ndarray, period: int) -> np.ndarray:
    """Boolean mask of the periodogram bins within 2% of the candidate frequency (at
    minimum the single nearest bin) — the window `_spectral_ratio` takes its peak over."""
    f0 = 1.0 / period
    in_peak = np.abs(freqs - f0) <= 0.02 * f0
    if not in_peak.any():
        in_peak[np.argmin(np.abs(freqs - f0))] = True
    return in_peak


def _spectral_ratio(freqs: np.ndarray, power: np.ndarray, period: int) -> float:
    """Peak periodogram power within 2% of the candidate frequency, relative to the
    median power of the surrounding octave band (excluding the peak bins). The window
    is deliberately tight: widening it inflates the max over more noise bins, which is
    how red-noise variance fakes a peak."""
    f0 = 1.0 / period
    in_peak = _peak_window(freqs, period)
    peak = float(power[in_peak].max())
    neighborhood = (freqs >= f0 / 2) & (freqs <= 2 * f0) & ~in_peak & (freqs > 0)
    base = float(np.median(power[neighborhood])) if neighborhood.any() else float("nan")
    if not np.isfinite(base) or base <= 0:
        return float("nan")
    return peak / base


def _mstl_decompose(y: np.ndarray, periods: list[int]) -> tuple[object, list[int]] | None:
    """Interpolate the last MSTL_SAMPLE_ROWS observations and run one joint MSTL
    decomposition (`statsmodels.tsa.seasonal.MSTL`) over the subset of `periods` that
    clear MIN_CYCLES within that window — shortest period first, so a longer period is
    decomposed against the residual left after shorter cycles are already stripped out,
    discounting a period that's really just spectral leakage of a shorter one. Returns
    `(decomposition_result, fitted_periods)`, or None if nothing in `periods` is fittable
    or the series is too degenerate to interpolate."""
    from statsmodels.tsa.seasonal import MSTL

    sample = np.asarray(y, dtype=float)[-MSTL_SAMPLE_ROWS:]
    finite = np.isfinite(sample)
    if finite.sum() < 3:
        return None
    idx = np.arange(len(sample))
    sample = np.interp(idx, idx[finite], sample[finite])

    fittable = sorted(p for p in periods if len(sample) // p >= MIN_CYCLES)
    if not fittable:
        return None

    decomposition = MSTL(
        sample, periods=fittable, iterate=10, stl_kwargs=dict(seasonal_deg=0)
    ).fit()
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
    remainder = np.asarray(decomposition.resid)
    # A single fitted period comes back as a 1D array (one seasonal component); reshape
    # to a (n, 1) column so the indexing below is uniform with the multi-period case.
    seasonal = np.asarray(decomposition.seasonal)
    if seasonal.ndim == 1:
        seasonal = seasonal[:, None]
    strengths: dict[int, float] = {}
    for i, p in enumerate(fittable):
        total_var = np.var(remainder + seasonal[:, i])
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
    yd = fill_and_detrend(y)
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

    yd = fill_and_detrend(y)
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


# --- RobustPeriod discovery -------------------------------------------------


def _modwt(x: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray]:
    """Maximal-overlap discrete wavelet transform of `x` to `level` levels (sym4 filters).

    Returns `(W, V)` where `W[j-1]` is the level-`j` detail coefficients and `V` the final
    scaling coefficients, all the same length as `x`. Unlike the ordinary DWT, the MODWT is
    shift-invariant and defined for any length, so a level's coefficients line up with the
    original samples — which is what lets a periodicity be localised to the level whose dyadic
    band (periods `2**j .. 2**(j+1)`) contains it. Each level is a circular convolution with
    the filter dilated by `2**(j-1)` (the a-trous scheme), computed via FFT so the whole
    transform is O(n log n)."""
    n = len(x)
    h = _SYM4_DEC_HI / np.sqrt(2.0)
    g = _SYM4_DEC_LO / np.sqrt(2.0)
    taps = np.arange(len(h))
    v = np.asarray(x, dtype=float)
    details = []
    for j in range(1, level + 1):
        idx = (2 ** (j - 1) * taps) % n  # dilated-filter support, wrapped circularly
        filt_h = np.zeros(n)
        filt_g = np.zeros(n)
        np.add.at(filt_h, idx, h)
        np.add.at(filt_g, idx, g)
        vf = np.fft.fft(v)
        details.append(np.real(np.fft.ifft(vf * np.fft.fft(filt_h))))
        v = np.real(np.fft.ifft(vf * np.fft.fft(filt_g)))
    return np.array(details), v


def _biweight_midvariance(x: np.ndarray, c: float = 9.0) -> float:
    """Tukey biweight midvariance: a robust variance that all but ignores outliers (points
    past `c` MADs get zero weight). Used to rank wavelet levels by how much genuine energy
    each carries without a few spikes inflating the ranking (Beers et al. 1990)."""
    m = np.median(x)
    mad = np.median(np.abs(x - m))
    if mad <= 0:
        return 0.0
    u = (x - m) / (c * mad)
    mask = np.abs(u) < 1.0
    if not mask.any():
        return 0.0
    um = u[mask]
    num = np.sum((x[mask] - m) ** 2 * (1 - um**2) ** 4)
    den = np.sum((1 - um**2) * (1 - 5 * um**2))
    if den == 0:
        return 0.0
    return float(len(x) * num / den**2)


def _huber_clean(y: np.ndarray) -> np.ndarray:
    """Winsorise `y` at the Huber knee: standardise by median and (consistent) MAD, clip the
    standardised series to +-HUBER_ZETA, and rescale. This is RobustPeriod's preprocessing
    step — it pulls spikes and heavy-tailed excursions back to the bulk of the distribution so
    a few outliers can't dominate the spectrum, without touching the ordinary variation."""
    mu = np.median(y)
    scale = 1.4826 * np.median(np.abs(y - mu))
    if scale <= 0:
        # More than half the values are identical (flat, censored or zero-inflated series):
        # the MAD is zero but spikes may remain, and returning the data uncleaned would let
        # them dominate the spectrum. Fall back to the mean absolute deviation (Gaussian-
        # consistent via sqrt(pi/2)), which is nonzero unless the series is constant.
        scale = float(np.mean(np.abs(y - mu))) * np.sqrt(np.pi / 2)
    if scale <= 0:
        return y - mu
    return np.clip((y - mu) / scale, -HUBER_ZETA, HUBER_ZETA) * scale


def _huber_periodogram_at(y: np.ndarray, freq: float, max_iter: int = 25) -> float:
    """RobustPeriod's Huber periodogram at a single frequency: `P = (n/4)|β̂|²` where β̂ fits a
    cos/sin pair at `freq` to `y` under Huber loss (Wen et al. Eq. 6/8). Solved by IRLS, which
    reaches the same optimum as the paper's ADMM for this convex two-parameter fit. Evaluating
    the exact robust estimate only at a candidate peak frequency — rather than at every Fourier
    frequency, which is O(n²) — is what keeps it affordable on long series. `√(4P/n)` recovers
    the cycle's robust amplitude in the units of `y`."""
    n = len(y)
    t = np.arange(n)
    w = 2 * np.pi * freq
    design = np.column_stack([np.cos(w * t), np.sin(w * t)])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    for _ in range(max_iter):
        resid = y - design @ beta
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if scale <= 0:
            break
        u = resid / scale
        weight = np.ones(n)
        big = np.abs(u) > HUBER_ZETA
        weight[big] = HUBER_ZETA / np.abs(u[big])
        beta_new = np.linalg.solve(design.T @ (weight[:, None] * design), design.T @ (weight * y))
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return float(n / 4.0 * (beta[0] ** 2 + beta[1] ** 2))


def _band_period(coeffs: np.ndarray, lo: int, hi: int) -> int | None:
    """Dominant period of one MODWT level, read off the autocorrelation of its coefficients:
    the lag of the tallest ACF peak within the level's dyadic band `[lo, hi]`. The ACF (via the
    Wiener-Khinchin theorem — the inverse transform of the power spectrum) gives an exact
    integer lag, so short periods land precisely; a level whose band holds no ACF peak falls
    back to the strongest in-band lag. Returns None if the band is empty or degenerate."""
    from scipy.signal import find_peaks

    n = len(coeffs)
    power = np.abs(np.fft.rfft(coeffs)) ** 2
    power[0] = 0.0
    acf = np.fft.irfft(power, n=n)[: n // 2]
    if acf[0] <= 0:
        return None
    acf = acf / acf[0]
    lags = np.arange(len(acf))
    in_band = (lags >= lo) & (lags <= hi)
    if not in_band.any():
        return None
    band_lags = lags[in_band]
    band_acf = acf[in_band]
    peaks, props = find_peaks(band_acf, height=0.05)
    if len(peaks) == 0:
        return int(band_lags[int(np.argmax(band_acf))])
    return int(band_lags[peaks[int(np.argmax(props["peak_heights"]))]])


class RobustCandidate(TypedDict):
    period: int
    spectral_ratio: float
    prominence_bar: float
    robust_amplitude: float
    n_cycles: float
    level: int
    variance_fraction: float
    prominent: bool
    approximate: bool
    harmonic_of: int | None
    menu_period: int | None


class RobustPeriodResult(TypedDict):
    candidates: list[RobustCandidate]
    menu_periods_missing: list[int]


def robust_period_candidates(
    y: np.ndarray,
    freq: str,
    var_floor: float = 0.02,
) -> RobustPeriodResult:
    """Discover the periodic structure of `y` from the data alone (RobustPeriod, Wen et al.
    2021), knowing nothing about the case's declared seasonalities.

    The series is linearly detrended and Huber-cleaned (robust to trend and spikes), then a
    MODWT splits it into dyadic frequency bands so coexisting cycles land in different levels.
    Each level with real energy contributes its dominant in-band period (from the ACF of the Huber-cleaned coefficients),
    scored by two independent measures: `spectral_ratio` — the peak's prominence over its local
    spectral neighborhood, the same red-noise-robust discriminator `detect_candidates` uses —
    and `robust_amplitude`, the cycle's amplitude in the units of `y` from the exact Huber
    periodogram at that frequency.

    Returns every candidate found, ranked by spectral prominence — nothing is accepted or
    rejected. Each is flagged `prominent` (clears `prominence_bar`: POWER_RATIO_MIN plus
    log2 of the Fourier-bin count of the peak window, so the max over a wide window on a
    long series can't fake prominence out of noise),
    `harmonic_of` (a shorter period that is an integer fraction of a more prominent one, e.g.
    12 under 24), `approximate` (fewer than 10 cycles in the window, so the period is
    resolution-limited), and `menu_period` (the nearest fixed-menu period within 10%, for
    reference — not a target). `menu_periods_missing` lists menu periods no prominent candidate
    corroborates. The reviewer reads the evidence and decides what to log."""
    menu = sorted(CANDIDATE_MENU.get(normalize_freq(freq), ()))
    empty: RobustPeriodResult = {"candidates": [], "menu_periods_missing": list(menu)}
    try:
        yd = fill_and_detrend(y)
    except ValueError:
        return empty
    yc = _huber_clean(yd)
    n = len(yc)
    # Go as deep as the data supports: the deepest band's lower edge (2**level) must still hold
    # >= MIN_CYCLES cycles, so level <= log2(n / MIN_CYCLES). Capped at MODWT_MAX_LEVEL.
    level = min(int(np.floor(np.log2(n / MIN_CYCLES))), MODWT_MAX_LEVEL) if n >= 16 else 0
    if level < 1:
        return empty

    details, _ = _modwt(yc, level)
    variances = np.array([_biweight_midvariance(w) for w in details])
    vmax = float(variances.max()) if variances.size else 0.0
    freqs = np.fft.rfftfreq(n)
    power = np.abs(np.fft.rfft(yc)) ** 2
    power[0] = 0.0

    raw: list[RobustCandidate] = []
    for j in range(level):
        if vmax > 0 and variances[j] < var_floor * vmax:
            continue
        lo, hi = 2 ** (j + 1), 2 ** (j + 2)  # dyadic band of level j+1 (1-based)
        period = _band_period(details[j], lo, hi)
        # Period 2 is the Nyquist limit: sin(pi*t) is numerically zero at every integer
        # sample, so the Huber periodogram's cos/sin design is degenerate there and the
        # amplitude estimate is meaningless.
        if period is None or period < 3:
            continue
        ratio = _spectral_ratio(freqs, power, period)
        if not np.isfinite(ratio):
            continue
        # The peak window spans ~0.04*n/period Fourier bins, and the spectral ratio takes
        # the max over them; on noise alone that max grows like ln(bins), so a fixed bar
        # would let long series mint spurious prominence. Raising the bar by log2(bins)
        # holds the noise clearance probability at its single-bin level.
        bar = POWER_RATIO_MIN + np.log2(int(_peak_window(freqs, period).sum()))
        huber_power = _huber_periodogram_at(yc, 1.0 / period)
        n_cycles = n / period
        raw.append(
            RobustCandidate(
                period=period,
                spectral_ratio=float(ratio),
                prominence_bar=round(float(bar), 1),
                robust_amplitude=float(np.sqrt(4.0 * huber_power / n)),
                n_cycles=round(n_cycles, 1),
                level=j + 1,
                variance_fraction=round(variances[j] / vmax, 3) if vmax > 0 else 0.0,
                prominent=bool(ratio >= bar),
                approximate=bool(n_cycles < 10),
                harmonic_of=None,
                menu_period=None,
            )
        )

    # De-duplicate the same cycle surfacing at adjacent levels: keep the most prominent within
    # the 10% window (the rule `proposed_seasonalities` uses).
    raw.sort(key=lambda c: -c["spectral_ratio"])
    candidates: list[RobustCandidate] = []
    for c in raw:
        if all(abs(c["period"] - k["period"]) / k["period"] > ROBUST_PROXIMITY for k in candidates):
            candidates.append(c)

    for c in candidates:
        # Harmonic: a shorter period that is ~1/k of a strictly more prominent one.
        for other in candidates:
            if other["spectral_ratio"] <= c["spectral_ratio"]:
                continue
            if any(
                abs(c["period"] * k - other["period"]) / other["period"] <= 0.06 for k in (2, 3, 4)
            ):
                c["harmonic_of"] = other["period"]
                break
        near = [m for m in menu if abs(c["period"] - m) / m <= ROBUST_PROXIMITY]
        if near:
            c["menu_period"] = min(near, key=lambda m: abs(c["period"] - m))

    corroborated = {c["menu_period"] for c in candidates if c["prominent"] and c["menu_period"]}
    missing = [m for m in menu if m not in corroborated]
    return {"candidates": candidates, "menu_periods_missing": missing}
