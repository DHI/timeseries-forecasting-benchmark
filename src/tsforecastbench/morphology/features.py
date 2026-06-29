"""Morphological feature vector for time series characterisation.

Six core features are computed by ``compute_features``; thirteen candidate
features (core + extended) are computed by ``compute_features_all`` and used
for the feature-selection analysis in ``notebooks/feature_selection.py``.
All functions are stateless and independently importable.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.signal import periodogram
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf as sm_acf

# Maps the freq string used in case.toml to a seasonal period in steps.
# Used as the primary period for trend_strength, phase_consistency, seasonal_stability,
# exog_r2_gain, and as the fallback when no seasonal_periods list is provided.
PERIOD_FOR_FREQ: dict[str, int] = {
    "1D": 365,
    "1h": 24,
    "15min": 96,
}

# Candidate periods for seasonal_strength — all plausible dominant cycles per freq.
# seasonal_strength runs STL at each and returns the max so that series with a
# dominant annual cycle (e.g. waves, surge) are not incorrectly scored as zero
# simply because the primary period targets the diurnal cycle.
SEASONAL_PERIODS_FOR_FREQ: dict[str, list[int]] = {
    "1D": [7, 365],
    "1h": [24, 168, 8760],       # diurnal, weekly, annual
    "15min": [96, 672, 35040],   # diurnal, weekly, annual
}

# STL is O(N) per iteration; cap series length to keep runtimes reasonable.
_STL_MAX_N = 50_000


def period_for_freq(freq: str) -> int:
    """Return the dominant seasonality period in steps for a given freq string."""
    try:
        return PERIOD_FOR_FREQ[freq]
    except KeyError:
        raise KeyError(
            f"No period defined for freq {freq!r}. Known: {list(PERIOD_FOR_FREQ)}"
        ) from None


def seasonal_periods_for_freq(freq: str) -> list[int]:
    """Return all candidate seasonality periods in steps for a given freq string."""
    try:
        return SEASONAL_PERIODS_FOR_FREQ[freq]
    except KeyError:
        raise KeyError(
            f"No seasonal periods defined for freq {freq!r}. Known: {list(SEASONAL_PERIODS_FOR_FREQ)}"
        ) from None


def acf1(y: np.ndarray) -> float:
    """Lag-1 autocorrelation coefficient."""
    return float(sm_acf(y, nlags=1, fft=True)[1])


def trend_strength(y: np.ndarray, period: int) -> float:
    """Trend strength from STL: max(0, 1 - Var(resid) / Var(trend + resid))."""
    stl = _fit_stl(y, period)
    var_resid = np.var(stl.resid, ddof=1)
    var_tr = np.var(stl.trend + stl.resid, ddof=1)
    if var_tr == 0.0:
        return 0.0
    return float(max(0.0, 1.0 - var_resid / var_tr))


def seasonal_strength(y: np.ndarray, periods: int | list[int]) -> float:
    """Seasonal strength: max across candidate periods.

    Runs STL independently at each period and returns the highest score so
    that series with a dominant annual cycle are not penalised for having no
    diurnal cycle (and vice-versa). Formula per period:
        max(0, 1 - Var(resid) / Var(seasonal + resid))
    """
    if isinstance(periods, int):
        periods = [periods]
    return max(_seasonal_strength_at(y, p) for p in periods)


def _seasonal_strength_at(y: np.ndarray, period: int) -> float:
    stl = _fit_stl(y, period)
    var_resid = np.var(stl.resid, ddof=1)
    var_sr = np.var(stl.seasonal + stl.resid, ddof=1)
    if var_sr == 0.0:
        return 0.0
    return float(max(0.0, 1.0 - var_resid / var_sr))


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
    # Build embedding matrix and rank each row to get ordinal pattern.
    idx = np.arange(m) * delay
    embedded = np.stack([y[i: i + (m - 1) * delay + 1: delay] for i in range(n_patterns)])
    ordinals = np.argsort(np.argsort(embedded, axis=1), axis=1)
    _, counts = np.unique(ordinals, axis=0, return_counts=True)
    pmf = counts / counts.sum()
    entropy = float(-np.sum(pmf * np.log2(pmf + 1e-14)))
    return float(np.clip(entropy / math.log2(math.factorial(m)), 0.0, 1.0))


def seasonal_stability(y: np.ndarray, period: int) -> float:
    """Coefficient of variation of seasonal amplitude across complete periods.

    Uses the STL seasonal component. Each complete chunk of length ``period``
    has its amplitude (max − min) computed; returns std/mean of those amplitudes.
    Low → consistent seasonality year-on-year; High → shifting/unstable seasonality.
    Returns 0 when fewer than 2 complete periods or when the seasonal signal is flat.
    """
    stl = _fit_stl(y, period)
    seasonal = stl.seasonal
    n_complete = len(seasonal) // period
    if n_complete < 2:
        return 0.0
    chunks = seasonal[: n_complete * period].reshape(n_complete, period)
    amplitudes = chunks.max(axis=1) - chunks.min(axis=1)
    # Guard against a near-flat seasonal component (non-seasonal series).
    if amplitudes.mean() < 1e-2 * (y.max() - y.min() + 1e-10):
        return 0.0
    mean_amp = amplitudes.mean()
    if mean_amp < 1e-10:
        return 0.0
    return float(amplitudes.std(ddof=1) / mean_amp)


def phase_consistency(y: np.ndarray, period: int) -> float:
    """Circular spread of peak-of-season phase across complete periods.

    Uses the STL seasonal component. Peak phase per chunk is normalised to [0, 1].
    Returns 1 − mean-resultant-length of the circular distribution.
    0 → perfectly consistent peak timing; 1 → uniformly scattered phases.
    Returns 0 when fewer than 2 complete periods.
    """
    stl = _fit_stl(y, period)
    seasonal = stl.seasonal
    n_complete = len(seasonal) // period
    if n_complete < 2:
        return 0.0
    chunks = seasonal[: n_complete * period].reshape(n_complete, period)
    amplitudes = chunks.max(axis=1) - chunks.min(axis=1)
    if amplitudes.mean() < 1e-2 * (y.max() - y.min() + 1e-10):
        return 0.0
    peak_phases = chunks.argmax(axis=1) / period  # [0, 1)
    angles = 2 * np.pi * peak_phases
    R = float(np.abs(np.exp(1j * angles).mean()))
    return float(np.clip(1.0 - R, 0.0, 1.0))


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


def compute_features(
    y: np.ndarray,
    period: int,
    seasonal_periods: list[int] | None = None,
) -> dict[str, float]:
    """Compute the full 6-feature morphological vector.

    Parameters
    ----------
    y:
        Clean (NaN-free) 1-D array of the target variable at native frequency.
    period:
        Primary seasonality period in steps (see ``period_for_freq``). Used by
        trend_strength and all STL-based features except seasonal_strength.
    seasonal_periods:
        Candidate periods for seasonal_strength (see ``seasonal_periods_for_freq``).
        Defaults to ``[period]`` when omitted (backward-compatible).

    Returns
    -------
    dict with keys: acf1, trend_strength, seasonal_strength,
    spectral_entropy, boxcox_lambda, kurtosis.
    """
    if seasonal_periods is None:
        seasonal_periods = [period]
    return {
        "acf1": acf1(y),
        "trend_strength": trend_strength(y, period),
        "seasonal_strength": seasonal_strength(y, seasonal_periods),
        "spectral_entropy": spectral_entropy(y),
        "boxcox_lambda": boxcox_lambda(y),
        "kurtosis": excess_kurtosis(y),
    }


def compute_features_all(
    y: np.ndarray,
    period: int,
    seasonal_periods: list[int] | None = None,
) -> dict[str, float]:
    """Compute all 13 candidate features (core 6 + extended 7) for feature selection.

    Parameters
    ----------
    y:
        Clean (NaN-free) 1-D array of the target variable at native frequency.
    period:
        Primary seasonality period in steps (see ``period_for_freq``). Used by
        trend_strength, phase_consistency, seasonal_stability, and exog_r2_gain.
    seasonal_periods:
        Candidate periods for seasonal_strength (see ``seasonal_periods_for_freq``).
        Defaults to ``[period]`` when omitted (backward-compatible).

    Returns
    -------
    dict with keys: acf1, trend_strength, seasonal_strength, spectral_entropy,
    boxcox_lambda, kurtosis, hurst, perm_entropy, seasonal_stability,
    phase_consistency, asymmetry, l_skewness, zero_proportion.
    """
    if seasonal_periods is None:
        seasonal_periods = [period]
    return {
        # Core 6
        "acf1": acf1(y),
        "trend_strength": trend_strength(y, period),
        "seasonal_strength": seasonal_strength(y, seasonal_periods),
        "spectral_entropy": spectral_entropy(y),
        "boxcox_lambda": boxcox_lambda(y),
        "kurtosis": excess_kurtosis(y),
        # Extended 7
        "hurst": hurst_exponent(y),
        "perm_entropy": permutation_entropy(y),
        "seasonal_stability": seasonal_stability(y, period),
        "phase_consistency": phase_consistency(y, period),
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

    # Expand raw series before lagging, then build lag matrices.
    poly_y = PolynomialFeatures(degree=poly_degree, include_bias=False)
    poly_x = PolynomialFeatures(degree=poly_degree, include_bias=False)

    y_exp = poly_y.fit_transform(y.reshape(-1, 1))          # (n, d_y)
    x_exp = poly_x.fit_transform(X_exog)                    # (n, d_x)

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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _fit_stl(y: np.ndarray, period: int):
    """Fit a robust STL model, truncating very long series to _STL_MAX_N samples."""
    if len(y) > _STL_MAX_N:
        n_keep = (_STL_MAX_N // period) * period
        y = y[-n_keep:]
    period = max(2, min(period, len(y) // 2))
    return STL(y, period=period, robust=True).fit()
