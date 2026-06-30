"""Morphological feature vector for time series characterisation.

The working 5-feature vector (seasonal_strength, hurst, trend_strength,
phase_consistency, l_skewness) is assembled by ``compute_morphology.py``
from the full 13-candidate set returned by ``compute_features_all``.
All functions are stateless and independently importable.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.signal import find_peaks, periodogram
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.seasonal import MSTL
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

# Detected period P is rejected unless the series contains at least this many
# complete cycles: len(y) / P >= _MIN_CYCLES.
_MIN_CYCLES: int = 3

# Candidate period P is suppressed as a harmonic of accepted period A when
# |P - A/k| / (A/k) <= _HARMONIC_TOL for any integer k in {2, 3, 4}.
_HARMONIC_TOL: float = 0.05

# Candidate period P is suppressed when too close to an already-accepted period
# A: |P - A| / A <= _CLUSTER_TOL.  Prevents spectral-leakage sidebands from
# being accepted as independent seasons.
_CLUSTER_TOL: float = 0.25

# Minimum meaningful period in samples (shorter periods are noise).
_MIN_PERIOD: int = 4

# MSTL is O(N) per iteration; cap series length to keep runtimes sane.
_MAX_N: int = 50_000


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

    threshold = _PEAK_THRESHOLD * float(np.median(psd))
    peak_idxs, _ = find_peaks(psd, height=threshold)

    candidates: list[tuple[float, float]] = []
    for i in peak_idxs:
        period_f = n / (i + 1)
        if period_f < _MIN_PERIOD:
            continue
        if n / period_f < _MIN_CYCLES:
            continue
        candidates.append((period_f, float(psd[i])))

    if not candidates:
        return []

    # Greedy suppression: process strongest peaks first.
    # Reject a candidate if it is (a) a sub-harmonic of an accepted period or
    # (b) within _CLUSTER_TOL of an accepted period (spectral-leakage sideband).
    candidates.sort(key=lambda x: -x[1])
    accepted: list[float] = []
    for period_f, _ in candidates:
        is_suppressed = False
        for acc in accepted:
            # Sub-harmonic: period_f ≈ acc/k for k in {2,3,4}
            for k in range(2, 5):
                ref = acc / k
                if ref > 0 and abs(period_f - ref) / ref <= _HARMONIC_TOL:
                    is_suppressed = True
                    break
            if not is_suppressed and abs(period_f - acc) / acc <= _CLUSTER_TOL:
                is_suppressed = True
            if is_suppressed:
                break
        if not is_suppressed:
            accepted.append(period_f)

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
# MSTL decomposition
# ---------------------------------------------------------------------------

def _fit_mstl(y: np.ndarray, periods: list[int]):
    """Fit a robust MSTL model, truncating very long series."""
    if len(y) > _MAX_N:
        max_p = max(periods)
        n_keep = max((_MAX_N // max_p) * max_p, 3 * max_p)
        y = y[-n_keep:]
    return MSTL(y, periods=periods, stl_kwargs={"robust": True}).fit()


def _mstl_seasonal_arrays(mstl_result, n_periods: int) -> list[np.ndarray]:
    """Return per-period seasonal components as a list of 1-D arrays."""
    s = np.asarray(mstl_result.seasonal)
    if s.ndim == 1:
        return [s]
    return [s[:, i] for i in range(n_periods)]


def _dominant_idx(seasonal_arrays: list[np.ndarray], resid: np.ndarray) -> int:
    """Index of the period with the highest per-period seasonal strength."""
    var_resid = np.var(resid, ddof=1)

    def _strength(s: np.ndarray) -> float:
        var_sr = np.var(s + resid, ddof=1)
        return 0.0 if var_sr == 0.0 else max(0.0, 1.0 - var_resid / var_sr)

    return max(range(len(seasonal_arrays)), key=lambda i: _strength(seasonal_arrays[i]))


# ---------------------------------------------------------------------------
# Shared decomposition
# ---------------------------------------------------------------------------

def _decompose(y: np.ndarray):
    """Poly-detrend → detect periods → fit MSTL.

    Returns
    -------
    periods         : list[int]            detected periods (may be empty)
    mstl_result     : MSTL result | None   None when no periods detected
    y_det           : np.ndarray           polynomial-detrended series
    seasonal_arrays : list[np.ndarray]     per-period MSTL seasonal components
    dom_idx         : int | None           index of dominant period
    """
    y_det, _ = _poly_detrend(y)
    periods = _detect_periods(y_det)
    if not periods:
        return [], None, y_det, [], None
    mstl = _fit_mstl(y, periods)
    seasonal_arrays = _mstl_seasonal_arrays(mstl, len(periods))
    resid = np.asarray(mstl.resid)
    dom_idx = _dominant_idx(seasonal_arrays, resid)
    return periods, mstl, y_det, seasonal_arrays, dom_idx


# ---------------------------------------------------------------------------
# Feature helpers (seasonal_stability and phase_consistency inner logic)
# ---------------------------------------------------------------------------

def _seasonal_stability_from_component(
    seasonal_dom: np.ndarray, period: int, y: np.ndarray
) -> float:
    n_complete = len(seasonal_dom) // period
    if n_complete < 2:
        return 0.0
    chunks = seasonal_dom[: n_complete * period].reshape(n_complete, period)
    amplitudes = chunks.max(axis=1) - chunks.min(axis=1)
    if amplitudes.mean() < 1e-2 * (y.max() - y.min() + 1e-10):
        return 0.0
    mean_amp = amplitudes.mean()
    if mean_amp < 1e-10:
        return 0.0
    return float(amplitudes.std(ddof=1) / mean_amp)


def _phase_consistency_from_component(
    seasonal_dom: np.ndarray, period: int, y: np.ndarray
) -> float:
    n_complete = len(seasonal_dom) // period
    if n_complete < 2:
        return 0.0
    chunks = seasonal_dom[: n_complete * period].reshape(n_complete, period)
    amplitudes = chunks.max(axis=1) - chunks.min(axis=1)
    if amplitudes.mean() < 1e-2 * (y.max() - y.min() + 1e-10):
        return 0.0
    peak_phases = chunks.argmax(axis=1) / period
    angles = 2 * np.pi * peak_phases
    R = float(np.abs(np.exp(1j * angles).mean()))
    return float(np.clip(1.0 - R, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public feature functions
# ---------------------------------------------------------------------------

def acf1(y: np.ndarray) -> float:
    """Lag-1 autocorrelation coefficient."""
    return float(sm_acf(y, nlags=1, fft=True)[1])


def trend_strength(y: np.ndarray) -> float:
    """Trend strength from MSTL: max(0, 1 - Var(resid) / Var(trend + resid)).

    Falls back to polynomial R² when no seasonal periods are detected.
    """
    _, mstl, y_det, _, _ = _decompose(y)
    if mstl is None:
        var_y = np.var(y, ddof=1)
        var_det = np.var(y_det, ddof=1)
        return float(max(0.0, 1.0 - var_det / var_y)) if var_y > 0 else 0.0
    resid = np.asarray(mstl.resid)
    trend = np.asarray(mstl.trend)
    var_resid = np.var(resid, ddof=1)
    var_tr = np.var(trend + resid, ddof=1)
    return float(max(0.0, 1.0 - var_resid / var_tr)) if var_tr > 0 else 0.0


def seasonal_strength(y: np.ndarray) -> float:
    """Seasonal strength from MSTL against the combined seasonal signal.

    ``total_seasonal`` is the sum of all per-period MSTL components.
    Returns 0 when no seasonal periods are detected.
    """
    _, mstl, _, seasonal_arrays, _ = _decompose(y)
    if mstl is None:
        return 0.0
    resid = np.asarray(mstl.resid)
    seasonal_total = sum(seasonal_arrays)
    var_resid = np.var(resid, ddof=1)
    var_sr = np.var(seasonal_total + resid, ddof=1)
    return float(max(0.0, 1.0 - var_resid / var_sr)) if var_sr > 0 else 0.0


def seasonal_stability(y: np.ndarray) -> float:
    """CV of seasonal amplitude across complete cycles of the dominant period.

    Uses the dominant period's MSTL seasonal component. Returns 0 when fewer
    than 2 complete cycles or no seasonal periods are detected.
    """
    periods, mstl, _, seasonal_arrays, dom_idx = _decompose(y)
    if mstl is None:
        return 0.0
    return _seasonal_stability_from_component(
        seasonal_arrays[dom_idx], periods[dom_idx], y
    )


def phase_consistency(y: np.ndarray) -> float:
    """Circular spread of peak-of-season phase across cycles of the dominant period.

    Uses the dominant period's MSTL seasonal component. Returns 0 when fewer
    than 2 complete cycles or no seasonal periods are detected.
    """
    periods, mstl, _, seasonal_arrays, dom_idx = _decompose(y)
    if mstl is None:
        return 0.0
    return _phase_consistency_from_component(
        seasonal_arrays[dom_idx], periods[dom_idx], y
    )


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
        return 0.0  # constant series; lambda undefined
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
    Returns 0 when all differences are zero.
    """
    diff = np.diff(y)
    rises = diff[diff > 0]
    falls = -diff[diff < 0]
    if len(rises) == 0 or len(falls) == 0:
        return 0.0
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

    Fits MSTL once on the detected periods; all seasonal features share the
    same decomposition. Falls back to polynomial R² for trend_strength and
    zeros for seasonal features when no periods are detected.

    Parameters
    ----------
    y:
        Clean (NaN-free) 1-D array of the target variable at native frequency.

    Returns
    -------
    dict with keys: acf1, trend_strength, seasonal_strength, spectral_entropy,
    boxcox_lambda, kurtosis, hurst, perm_entropy, seasonal_stability,
    phase_consistency, asymmetry, l_skewness, zero_proportion.
    """
    periods, mstl, y_det, seasonal_arrays, dom_idx = _decompose(y)

    if mstl is not None:
        resid = np.asarray(mstl.resid)
        trend = np.asarray(mstl.trend)
        seasonal_total = sum(seasonal_arrays)
        dom_period = periods[dom_idx]
        seasonal_dom = seasonal_arrays[dom_idx]

        var_resid = np.var(resid, ddof=1)

        var_tr = np.var(trend + resid, ddof=1)
        ts = float(max(0.0, 1.0 - var_resid / var_tr)) if var_tr > 0 else 0.0

        var_sr = np.var(seasonal_total + resid, ddof=1)
        ss = float(max(0.0, 1.0 - var_resid / var_sr)) if var_sr > 0 else 0.0

        sstab = _seasonal_stability_from_component(seasonal_dom, dom_period, y)
        pc = _phase_consistency_from_component(seasonal_dom, dom_period, y)
    else:
        var_y = np.var(y, ddof=1)
        var_det = np.var(y_det, ddof=1)
        ts = float(max(0.0, 1.0 - var_det / var_y)) if var_y > 0 else 0.0
        ss = sstab = pc = 0.0

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
        "seasonal_stability": sstab,
        "phase_consistency": pc,
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
