"""Morphological feature vector for time series characterisation.

The working 4-feature vector (seasonal_strength, hurst, amplitude_dispersion,
l_skewness) is assembled by ``compute_morphology.py`` from the full 13-candidate
set returned by ``compute_features_all``.
All functions are stateless and independently importable.

Trend and seasonal features come from a single, explainable least-squares
decomposition ``y = T + S + e`` (see ``_decompose``): a low-order polynomial
trend plus ``_N_HARMONICS`` sine/cosine pairs per automatically detected
period.  This replaces the earlier MSTL decomposition — one closed-form solve
handles multiple seasonalities, runs in well under a second even on hundreds
of thousands of samples, and needs no downsampling.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.signal import find_peaks, periodogram
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.stattools import acf as sm_acf

# Maps freq string → primary period in steps.  Used only for gap-handling
# thresholds (gap_interp_max, min_segment_len) and as the AR-order hint for
# exog_r2_gain.  Feature computation is fully automated via detect_periods.
PERIOD_FOR_FREQ: dict[str, int] = {
    "1D": 365,
    "1h": 24,
    "15min": 96,
}

# ---------------------------------------------------------------------------
# Detection hyper-parameters
# ---------------------------------------------------------------------------

# A periodogram peak must exceed this multiple of the median PSD to be accepted.
_PEAK_THRESHOLD: float = 10.0

# A peak must also *rise* this multiple of the median PSD above its local
# background (scipy prominence).  Height alone is a poor test on a red (1/f)
# spectrum — a high-Hurst series piles power at low frequencies and a global
# median threshold admits the slope itself as a spurious long "period".
# Prominence requires the peak to stand out from its surroundings, so the same
# significance bar applies whether the spectrum is red or white.
_PEAK_PROMINENCE: float = 5.0

# Detected period P is rejected unless the series contains at least this many
# complete cycles: len(y) / P >= _MIN_CYCLES.
_MIN_CYCLES: int = 3

# Candidate period P is suppressed when too close to an already-accepted period
# A: |P - A| / A <= _CLUSTER_TOL.  Prevents spectral-leakage sidebands from
# being accepted as independent seasons.  Note we deliberately do NOT suppress
# sub-harmonics here: a genuine 12 h semidiurnal and 24 h diurnal cycle must
# both survive.  Redundant frequencies (e.g. the 24 h cycle's 2nd harmonic vs
# the 12 h fundamental) are de-duplicated later, in the Fourier design.
_CLUSTER_TOL: float = 0.25

# Minimum meaningful period in samples (shorter periods are noise).
_MIN_PERIOD: int = 4

# Keep at most the strongest this-many periods (ranked by periodogram power).
# Without sub-harmonic suppression a richly structured series produces a long
# cascade of harmonic peaks; capping to the dominant handful keeps the feature
# space (and the design matrix) small and interpretable while still admitting
# genuinely multi-seasonal cases (e.g. semidiurnal + diurnal + weekly).
_MAX_PERIODS: int = 4

# Relative-prominence gate (two tiers, both measured against the dominant peak).
# This is what lets the detector return *fewer* than _MAX_PERIODS: when one
# cycle dominates, the cap is no longer the binding constraint and weak spurious
# peaks are rejected instead of mechanically filling the top-4 slots.
#
#  - A peak at or above _STRONG_FRACTION of the dominant prominence is always a
#    season, regardless of how many cycles it spans (so a genuinely dominant
#    long-period cycle that only fits a handful of cycles still survives).
#  - A fainter peak (between _FAINT_FRACTION and _STRONG_FRACTION) is admitted
#    only if it completes at least _FAINT_MIN_CYCLES cycles.  This is the test
#    that separates a real-but-secondary season from a spurious low-frequency
#    bump: a weekly cycle dwarfed by a sharp diurnal peak sits at ~6% prominence
#    but spans ~100 cycles (a sharp, well-resolved line, kept), whereas a
#    ~950-sample "period" also at ~6% spans only ~5 cycles (a broad bump from
#    imperfect detrending, rejected).
#  - Anything below _FAINT_FRACTION is treated as noise.
_STRONG_FRACTION: float = 0.10
_FAINT_FRACTION: float = 0.05
_FAINT_MIN_CYCLES: int = 12

# Sine/cosine pairs fitted per detected period in the structural decomposition.
# Small enough to stay collision-free and avoid fitting noise, large enough to
# register non-sinusoidal (peaky) seasonal shape.
_N_HARMONICS: int = 3


# ---------------------------------------------------------------------------
# Public helper: period lookup for gap-handling
# ---------------------------------------------------------------------------

def period_for_freq(freq: str) -> int:
    """Return the primary period in steps for a given freq string.

    Used only for gap-handling thresholds and exog_r2_gain AR order.
    Feature computation uses detect_periods instead.
    """
    try:
        return PERIOD_FOR_FREQ[freq]
    except KeyError:
        raise KeyError(
            f"No period defined for freq {freq!r}. Known: {list(PERIOD_FOR_FREQ)}"
        ) from None


# ---------------------------------------------------------------------------
# Polynomial detrending
# ---------------------------------------------------------------------------

def _poly_detrend(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AIC-adaptive polynomial detrend (degree 1–3).

    Returns (y_detrended, poly_coeffs). poly_coeffs can be passed to
    np.polyval to reconstruct the fitted trend.
    """
    t = np.arange(len(y), dtype=float)
    n = len(y)
    best_aic = np.inf
    best_coeffs: np.ndarray | None = None
    for deg in range(1, 4):
        if n <= deg + 1:
            break
        coeffs = np.polyfit(t, y, deg)
        resid = y - np.polyval(coeffs, t)
        rss = float(np.dot(resid, resid))
        aic = n * np.log(rss / n + 1e-14) + 2.0 * (deg + 1)
        if aic < best_aic:
            best_aic = aic
            best_coeffs = coeffs
    if best_coeffs is None:
        best_coeffs = np.polyfit(t, y, 1)
    return y - np.polyval(best_coeffs, t), best_coeffs


# ---------------------------------------------------------------------------
# Period detection
# ---------------------------------------------------------------------------

def _detect_periods(y_det: np.ndarray) -> list[int]:
    """Periodogram-based period detection on a pre-detrended series."""
    n = len(y_det)
    _, psd = periodogram(y_det)
    psd = psd[1:]  # drop DC; psd[i] corresponds to period n/(i+1)
    if len(psd) == 0:
        return []

    med = float(np.median(psd))
    if med <= 0.0:
        return []

    # A peak must be both tall (height) and locally distinct (prominence).
    peak_idxs, props = find_peaks(
        psd,
        height=_PEAK_THRESHOLD * med,
        prominence=_PEAK_PROMINENCE * med,
    )
    prominences = props["prominences"]

    # Rank by *prominence*, not raw power: on a red (1/f) spectrum low
    # frequencies carry the most absolute power, so ranking by height would
    # crowd genuine short cycles (diurnal, tidal) out of the top _MAX_PERIODS.
    # Prominence measures how far a peak rises above its local background, which
    # lets a modest-power but sharp seasonal peak compete fairly.
    candidates: list[tuple[float, float]] = []
    for i, prom in zip(peak_idxs, prominences):
        period_f = n / (i + 1)
        if period_f < _MIN_PERIOD:
            continue
        if n / period_f < _MIN_CYCLES:
            continue
        candidates.append((period_f, float(prom)))

    if not candidates:
        return []

    # Greedy suppression of spectral-leakage sidebands only: process the most
    # prominent peaks first and reject a candidate within _CLUSTER_TOL of an
    # accepted period.  Sub-harmonics are intentionally NOT suppressed (see
    # _CLUSTER_TOL note); harmonically related but physically distinct periods
    # survive and any redundant frequencies are removed later in _structural_fit.
    candidates.sort(key=lambda x: -x[1])
    top_prom = candidates[0][1]
    strong = _STRONG_FRACTION * top_prom
    faint = _FAINT_FRACTION * top_prom
    accepted: list[float] = []
    for period_f, prom in candidates:
        # Candidates are sorted by descending prominence, so once one falls
        # below the faint floor every remaining one does too.
        if prom < faint:
            break
        # Strong peaks pass outright; faint peaks must be well-resolved (span
        # many cycles) to be distinguished from broad low-frequency bumps.
        if prom < strong and n / period_f < _FAINT_MIN_CYCLES:
            continue
        if any(abs(period_f - acc) / acc <= _CLUSTER_TOL for acc in accepted):
            continue
        accepted.append(period_f)
        if len(accepted) >= _MAX_PERIODS:
            break

    return sorted({max(2, round(p)) for p in accepted})


def detect_periods(y: np.ndarray) -> list[int]:
    """Detect seasonal periods from a raw series.

    Polynomial-detrends ``y`` before running the periodogram so trend power
    does not suppress seasonal peaks. Returns a sorted list of integer periods;
    empty when no significant periodicity is found.
    """
    y_det, _ = _poly_detrend(y)
    return _detect_periods(y_det)


# ---------------------------------------------------------------------------
# Structural least-squares decomposition
# ---------------------------------------------------------------------------

class _Decomp:
    """Result of the structural decomposition ``y = T + S + e``.

    Attributes
    ----------
    periods         : sorted detected periods (empty when none found)
    trend           : fitted polynomial trend T
    seasonal_arrays : per-period seasonal components S_p, aligned with ``periods``
    resid           : residual e = y - T - sum(S_p)
    dom_idx         : index into ``periods`` of the dominant period (None if no
                      periods), chosen as the period explaining the most variance
    """

    __slots__ = ("periods", "trend", "seasonal_arrays", "resid", "dom_idx")

    def __init__(self, periods, trend, seasonal_arrays, resid, dom_idx):
        self.periods = periods
        self.trend = trend
        self.seasonal_arrays = seasonal_arrays
        self.resid = resid
        self.dom_idx = dom_idx

    @property
    def seasonal_total(self):
        """Sum of all per-period seasonal components (0.0 when none)."""
        return sum(self.seasonal_arrays) if self.seasonal_arrays else 0.0


def _structural_fit(
    y: np.ndarray, periods: list[int], degree: int
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """One OLS solve of ``y`` on a polynomial trend + Fourier seasonal basis.

    The trend basis uses powers of a [-1, 1]-scaled time axis (well conditioned
    even for very long series).  Each detected period contributes ``_N_HARMONICS``
    sine/cosine pairs.  Frequencies are de-duplicated across periods via reduced
    fractions h/p: a shared frequency (e.g. the 24 h cycle's 2nd harmonic and the
    12 h fundamental) is fitted once and attributed to the *shortest* period that
    generates it (its most fundamental owner).  This keeps the design full-rank
    while honouring physically distinct seasonalities.

    Returns (trend, seasonal_arrays, resid), with seasonal_arrays aligned to
    ``sorted(periods)``.
    """
    n = len(y)
    t = np.arange(n, dtype=float)
    ts = (2.0 * t / (n - 1) - 1.0) if n > 1 else np.zeros(n)

    trend_cols = [ts ** k for k in range(degree + 1)]
    n_trend = len(trend_cols)

    seasonal_cols: list[np.ndarray] = []
    period_col_idx: dict[int, list[int]] = {p: [] for p in periods}
    claimed: set[tuple[int, int]] = set()
    for p in sorted(periods):  # shortest first → fundamentals own shared freqs
        for h in range(1, _N_HARMONICS + 1):
            g = math.gcd(h, p)
            key = (h // g, p // g)  # reduced fraction = unique frequency h/p
            if key in claimed:
                continue
            claimed.add(key)
            w = 2.0 * np.pi * h / p
            ci = n_trend + len(seasonal_cols)
            seasonal_cols.append(np.cos(w * t))
            si = n_trend + len(seasonal_cols)
            seasonal_cols.append(np.sin(w * t))
            period_col_idx[p].extend([ci, si])

    X = np.column_stack(trend_cols + seasonal_cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    trend = X[:, :n_trend] @ beta[:n_trend]
    resid = y - X @ beta
    seasonal_arrays = [
        (X[:, idx] @ beta[idx] if (idx := period_col_idx[p]) else np.zeros(n))
        for p in sorted(periods)
    ]
    return trend, seasonal_arrays, resid


def _decompose(y: np.ndarray) -> _Decomp:
    """Poly-detrend → detect periods → single structural OLS fit.

    Period detection runs on the full-resolution series (a cheap FFT).  The
    polynomial trend degree is the AIC-selected degree from ``_poly_detrend``.
    When no periods are detected the trend-only polynomial fit is returned.
    """
    y_det, coeffs = _poly_detrend(y)
    degree = len(coeffs) - 1
    periods = _detect_periods(y_det)

    if not periods:
        # Trend-only: _poly_detrend already gives trend = y - y_det.
        return _Decomp([], y - y_det, [], y_det, None)

    trend, seasonal_arrays, resid = _structural_fit(y, periods, degree)
    dom_idx = max(
        range(len(periods)), key=lambda i: float(np.var(seasonal_arrays[i], ddof=1))
    )
    return _Decomp(periods, trend, seasonal_arrays, resid, dom_idx)


# ---------------------------------------------------------------------------
# Feature helpers (amplitude_dispersion and phase_dispersion inner logic)
# ---------------------------------------------------------------------------

def _cycle_amplitude_phase(
    g: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-cycle amplitude and phase of the dominant fundamental.

    Folds ``g`` (the dominant seasonal plus residual, ``e + S_dom``) into
    complete cycles of ``period`` and projects each cycle onto the fundamental
    sine/cosine: amplitude = sqrt(a^2 + b^2), phase = atan2(b, a).  Returns None
    when fewer than two complete cycles are available.
    """
    n_complete = len(g) // period
    if n_complete < 2:
        return None
    chunks = g[: n_complete * period].reshape(n_complete, period)
    j = np.arange(period, dtype=float)
    cos_j = np.cos(2.0 * np.pi * j / period)
    sin_j = np.sin(2.0 * np.pi * j / period)
    a = chunks @ cos_j * (2.0 / period)
    b = chunks @ sin_j * (2.0 / period)
    return np.sqrt(a ** 2 + b ** 2), np.arctan2(b, a)


def _amplitude_dispersion_from_component(
    g: np.ndarray, period: int, y: np.ndarray
) -> float:
    res = _cycle_amplitude_phase(g, period)
    # No measurable seasonal amplitude (too few cycles, or amplitude below the
    # noise floor) → maximal-instability sentinel of 1.0, the limit of the
    # bounded transform below. A genuinely steady cycle returns a small value;
    # 1.0 reads as "no measurable stable cycle", never as "perfectly stable".
    if res is None:
        return 1.0
    amplitudes, _ = res
    mean_amp = float(amplitudes.mean())
    if mean_amp < 1e-2 * (y.max() - y.min() + 1e-10) or mean_amp < 1e-10:
        return 1.0
    # Bound the unbounded CV onto [0, 1): cv/(1+cv). Monotonic, so 0 still means
    # perfectly steady amplitude and the ordering across series is preserved.
    cv = float(amplitudes.std(ddof=1) / mean_amp)
    return cv / (1.0 + cv)


def _phase_dispersion_from_component(
    g: np.ndarray, period: int, y: np.ndarray
) -> float:
    res = _cycle_amplitude_phase(g, period)
    # No measurable seasonal phase (too few cycles, or amplitude below the
    # noise floor) → maximal dispersion. A sentinel of 1.0 reads correctly:
    # "no coherent phase", distinct from a genuine near-0 phase-locked cycle.
    if res is None:
        return 1.0
    amplitudes, phases = res
    if amplitudes.mean() < 1e-2 * (y.max() - y.min() + 1e-10):
        return 1.0
    R = float(np.abs(np.exp(1j * phases).mean()))
    return float(np.clip(1.0 - R, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public feature functions
# ---------------------------------------------------------------------------

def acf1(y: np.ndarray) -> float:
    """Lag-1 autocorrelation coefficient."""
    return float(sm_acf(y, nlags=1, fft=True)[1])


def trend_strength(y: np.ndarray) -> float:
    """Trend strength: max(0, 1 - Var(resid) / Var(trend + resid)).

    ``trend + resid`` is ``y`` with the seasonal component removed. With no
    detected periods this reduces to the polynomial trend's R² against ``y``.
    """
    d = _decompose(y)
    var_resid = np.var(d.resid, ddof=1)
    var_tr = np.var(y - d.seasonal_total, ddof=1)
    return float(max(0.0, 1.0 - var_resid / var_tr)) if var_tr > 0 else 0.0


def seasonal_strength(y: np.ndarray) -> float:
    """Seasonal strength against the combined seasonal signal.

    ``S + resid`` is ``y`` with the trend removed; the ratio measures the share
    of de-trended variance explained by the fitted Fourier seasonality across
    all detected periods. Returns 0 when no seasonal periods are detected.
    """
    d = _decompose(y)
    if not d.periods:
        return 0.0
    var_resid = np.var(d.resid, ddof=1)
    var_sr = np.var(y - d.trend, ddof=1)
    return float(max(0.0, 1.0 - var_resid / var_sr)) if var_sr > 0 else 0.0


def amplitude_dispersion(y: np.ndarray) -> float:
    """Bounded dispersion of the dominant cycle's amplitude across cycles, [0, 1).

    ``cv / (1 + cv)`` where ``cv`` is the coefficient of variation of the
    per-cycle amplitude of the dominant period. 0 means the seasonal swing has
    identical magnitude every cycle (perfectly steady); values approach 1 as the
    amplitude becomes increasingly erratic. Returns 1 (the limit) when no
    seasonal period is detected or there are fewer than 2 complete cycles —
    there is no measurable cycle whose amplitude dispersion could be assessed.
    """
    d = _decompose(y)
    if d.dom_idx is None:
        return 1.0
    g = d.resid + d.seasonal_arrays[d.dom_idx]
    return _amplitude_dispersion_from_component(g, d.periods[d.dom_idx], y)


def phase_dispersion(y: np.ndarray) -> float:
    """Circular spread of the dominant cycle's phase across cycles.

    Computed as ``1 - R`` where ``R`` is the resultant length of the per-cycle
    phase. Near 0 means the cycle is phase-locked across cycles (a clean,
    regular seasonal pattern); larger values mean the timing drifts.
    Returns 1 (maximal dispersion) when no seasonal period is detected or there
    are fewer than 2 complete cycles — there is no coherent phase to measure.
    """
    d = _decompose(y)
    if d.dom_idx is None:
        return 1.0
    g = d.resid + d.seasonal_arrays[d.dom_idx]
    return _phase_dispersion_from_component(g, d.periods[d.dom_idx], y)


def spectral_entropy(y: np.ndarray) -> float:
    """Normalised spectral entropy in [0, 1].

    Computed on the linearly detrended series so the DC component does not
    dominate the PSD. Normalised by log(n_bins) so values are in [0, 1].
    High entropy → broad, irregular spectrum (unpredictable).
    Low entropy  → energy concentrated at a few frequencies (periodic/smooth).
    """
    t = np.arange(len(y), dtype=float)
    y_det = y - np.polyval(np.polyfit(t, y, 1), t)
    _, psd = periodogram(y_det)
    psd = psd[1:]  # drop DC bin
    total = psd.sum()
    if total == 0.0:
        return 0.0
    pmf = psd / total
    pmf = pmf[pmf > 0]
    ent = -np.sum(pmf * np.log(pmf))
    return float(ent / np.log(len(pmf))) if len(pmf) > 1 else 0.0


def boxcox_lambda(y: np.ndarray) -> float:
    """Optimal Box-Cox lambda (MLE). Shifted to be strictly positive."""
    y_shifted = y - y.min() + 1e-6
    if np.std(y_shifted) < 1e-10:
        return 1.0  # constant series; lambda undefined → identity (no transform)
    _, lam = stats.boxcox(y_shifted)
    return float(np.clip(lam, -3.0, 3.0))


def excess_kurtosis(y: np.ndarray) -> float:
    """Fisher excess kurtosis (0 for a normal distribution)."""
    return float(stats.kurtosis(y, fisher=True))


def hurst_exponent(y: np.ndarray) -> float:
    """Hurst exponent via R/S analysis (slope of log-log regression over window sizes).

    H > 0.5 → long-range persistence (e.g. groundwater).
    H ≈ 0.5 → random walk.
    H < 0.5 → anti-persistent.
    Clipped to [0, 1]. Returns 0.5 on failure.
    """
    n = len(y)
    if n < 20:
        return 0.5
    sizes = np.unique(np.logspace(1, np.log10(n // 2), num=20).astype(int))
    rs_vals: list[tuple[int, float]] = []
    for size in sizes:
        sub = y[:size]
        mean_adj = sub - sub.mean()
        cumsum = np.cumsum(mean_adj)
        R = float(cumsum.max() - cumsum.min())
        S = float(sub.std(ddof=1))
        if S > 0:
            rs_vals.append((size, R / S))
    if len(rs_vals) < 2:
        return 0.5
    arr = np.array(rs_vals)
    h = float(np.polyfit(np.log(arr[:, 0]), np.log(arr[:, 1]), 1)[0])
    return float(np.clip(h, 0.0, 1.0))


def permutation_entropy(y: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Normalised permutation entropy in [0, 1].

    Embeds y into ordinal patterns of length ``order`` with step ``delay``.
    Low → highly structured/predictable; High → complex/random.
    Normalised by log₂(order!) so the range is [0, 1].
    """
    n = len(y)
    m = order
    n_patterns = n - (m - 1) * delay
    if n_patterns < 1:
        return 1.0
    embedded = np.stack([y[i: i + (m - 1) * delay + 1: delay] for i in range(n_patterns)])
    ordinals = np.argsort(np.argsort(embedded, axis=1), axis=1)
    _, counts = np.unique(ordinals, axis=0, return_counts=True)
    pmf = counts / counts.sum()
    entropy = float(-np.sum(pmf * np.log2(pmf + 1e-14)))
    return float(np.clip(entropy / math.log2(math.factorial(m)), 0.0, 1.0))


def asymmetry_index(y: np.ndarray) -> float:
    """Temporal rise/fall asymmetry index in (−1, 1).

    Positive → mean rise rate exceeds mean fall rate (e.g. flashy discharge).
    Negative → mean fall rate exceeds mean rise rate.
    Near-zero → symmetric dynamics (smooth/tidal series).
    A purely monotonic series is maximally asymmetric: +1 (only rises) or
    -1 (only falls). Returns 0 only for a flat series (no rises and no falls).
    """
    diff = np.diff(y)
    rises = diff[diff > 0]
    falls = -diff[diff < 0]
    if len(rises) == 0 and len(falls) == 0:
        return 0.0
    if len(falls) == 0:
        return 1.0
    if len(rises) == 0:
        return -1.0
    mean_rise = rises.mean()
    mean_fall = falls.mean()
    denom = mean_rise + mean_fall
    if denom < 1e-10:
        return 0.0
    return float((mean_rise - mean_fall) / denom)


def l_skewness(y: np.ndarray) -> float:
    """L-skewness (τ₃ = L₃/L₂) via probability-weighted moments.

    More robust to outliers than conventional skewness.
    0 → symmetric; positive → right-skewed (heavy upper tail).
    Returns 0 when n < 4 or L₂ ≈ 0.
    """
    n = len(y)
    if n < 4:
        return 0.0
    x = np.sort(y)
    i = np.arange(1, n + 1, dtype=float)
    b0 = x.mean()
    b1 = float(np.sum((i - 1) / (n - 1) * x)) / n
    b2 = float(np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x)) / n
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    if abs(l2) < 1e-10:
        return 0.0
    return float(l3 / l2)


def zero_proportion(y: np.ndarray, threshold: float | None = None) -> float:
    """Fraction of time steps where the signal is near-zero.

    Threshold defaults to 1 % of the 95th percentile (auto-scaled per series).
    Useful for distinguishing perennial from intermittent flow regimes.
    """
    if threshold is None:
        p95 = float(np.percentile(y, 95))
        threshold = max(1e-6, 0.01 * p95)
    return float(np.mean(y <= threshold))


def compute_features_all(y: np.ndarray) -> dict[str, float]:
    """Compute all 13 candidate features with fully automatic period detection.

    Runs the structural decomposition once (``_decompose``); all four trend and
    seasonal features share that single fit. ``trend_strength`` reduces to the
    polynomial trend's R² and the seasonal features to 0 when no periods are
    detected.

    Parameters
    ----------
    y:
        Clean (NaN-free) 1-D array of the target variable at native frequency.

    Returns
    -------
    dict with keys: acf1, trend_strength, seasonal_strength, spectral_entropy,
    boxcox_lambda, kurtosis, hurst, perm_entropy, amplitude_dispersion,
    phase_dispersion, asymmetry, l_skewness, zero_proportion.
    """
    d = _decompose(y)
    var_resid = np.var(d.resid, ddof=1)

    var_tr = np.var(y - d.seasonal_total, ddof=1)
    ts = float(max(0.0, 1.0 - var_resid / var_tr)) if var_tr > 0 else 0.0

    if d.dom_idx is not None:
        var_sr = np.var(y - d.trend, ddof=1)
        ss = float(max(0.0, 1.0 - var_resid / var_sr)) if var_sr > 0 else 0.0
        g = d.resid + d.seasonal_arrays[d.dom_idx]
        dom_period = d.periods[d.dom_idx]
        ad = _amplitude_dispersion_from_component(g, dom_period, y)
        pc = _phase_dispersion_from_component(g, dom_period, y)
    else:
        # No seasonal period: strength collapses to 0, but stability and phase
        # dispersion are maximal (no coherent cycle to measure).
        ss = 0.0
        ad = pc = 1.0

    return {
        # Core 6
        "acf1": acf1(y),
        "trend_strength": ts,
        "seasonal_strength": ss,
        "spectral_entropy": spectral_entropy(y),
        "boxcox_lambda": boxcox_lambda(y),
        "kurtosis": excess_kurtosis(y),
        # Extended 7
        "hurst": hurst_exponent(y),
        "perm_entropy": permutation_entropy(y),
        "amplitude_dispersion": ad,
        "phase_dispersion": pc,
        "asymmetry": asymmetry_index(y),
        "l_skewness": l_skewness(y),
        "zero_proportion": zero_proportion(y),
    }


def exog_r2_gain(
    y: np.ndarray,
    X_exog: np.ndarray | None,
    period: int,
    n_exog_lags: int,
    poly_degree: int = 1,
) -> float:
    """R² gain of a ridge ARX model over a pure ridge AR model.

    Both models are fit on the same data (training split). Returns 0.0 for
    univariate cases. Clipped to [0, 1].

    When ``poly_degree`` > 1, polynomial features are computed on the raw
    series *before* lagging (poly → lags). Concretely: y is expanded to
    [y, y², …] and X_exog to its degree-``poly_degree`` expansion, then each
    derived series is lagged independently. This avoids the cross-lag
    interaction explosion that occurs with the reverse order (lags → poly)
    and keeps the feature count tractable even for long lag depths.

    Limitation: a weak linear AR baseline on impulsive or nonlinear series
    can inflate the gain beyond genuine external forcing dependence.

    Parameters
    ----------
    y:            Clean 1-D target array (training split).
    X_exog:       2-D exogenous array (n_samples × n_vars) or None.
    period:       Dominant seasonality period; AR order p = min(period, 30).
    n_exog_lags:  Exogenous lags included: 0..n_exog_lags.
    poly_degree:  Polynomial expansion degree applied before lagging (1 = linear).
    """
    if X_exog is None or X_exog.ndim < 2 or X_exog.shape[1] == 0:
        return 0.0

    p = min(period, 30)
    q = n_exog_lags
    max_lag = max(p, q)
    n = len(y)

    if n - max_lag < max(20, p + q * X_exog.shape[1] + 5):
        return 0.0

    Y = y[max_lag:]

    poly_y = PolynomialFeatures(degree=poly_degree, include_bias=False)
    poly_x = PolynomialFeatures(degree=poly_degree, include_bias=False)

    y_exp = poly_y.fit_transform(y.reshape(-1, 1))
    x_exp = poly_x.fit_transform(X_exog)

    AR  = np.hstack([y_exp[max_lag - lag: n - lag] for lag in range(1, p + 1)])
    ARX = np.hstack(
        [AR] + [x_exp[max_lag - lag: n - lag] for lag in range(0, q + 1)]
    )

    y_mean, y_std = Y.mean(), Y.std() + 1e-10
    Y_s = (Y - y_mean) / y_std

    def _r2(X: np.ndarray) -> float:
        X_s = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        pred = Ridge(alpha=1.0).fit(X_s, Y_s).predict(X_s)
        ss_tot = ((Y_s - Y_s.mean()) ** 2).sum()
        return float(1.0 - ((Y_s - pred) ** 2).sum() / ss_tot) if ss_tot > 0 else 0.0

    return float(np.clip(_r2(ARX) - _r2(AR), 0.0, 1.0))
