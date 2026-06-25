"""Morphological feature vector for time series characterisation.

Six features computed on a raw (NaN-free) numpy array. All functions are
stateless and independently importable; ``compute_features`` is the normal
entry point.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.signal import periodogram
from sklearn.linear_model import Ridge
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf as sm_acf

# Maps the freq string used in case.toml to a seasonal period in steps.
PERIOD_FOR_FREQ: dict[str, int] = {
    "1D": 365,
    "1h": 24,
    "15min": 96,
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


def seasonal_strength(y: np.ndarray, period: int) -> float:
    """Seasonal strength from STL: max(0, 1 - Var(resid) / Var(seasonal + resid))."""
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


def compute_features(y: np.ndarray, period: int) -> dict[str, float]:
    """Compute the full 6-feature morphological vector.

    Parameters
    ----------
    y:
        Clean (NaN-free) 1-D array of the target variable at native frequency.
    period:
        Dominant seasonality period in steps (see ``period_for_freq``).

    Returns
    -------
    dict with keys: acf1, trend_strength, seasonal_strength,
    spectral_entropy, boxcox_lambda, kurtosis.
    """
    return {
        "acf1": acf1(y),
        "trend_strength": trend_strength(y, period),
        "seasonal_strength": seasonal_strength(y, period),
        "spectral_entropy": spectral_entropy(y),
        "boxcox_lambda": boxcox_lambda(y),
        "kurtosis": excess_kurtosis(y),
    }


def exog_r2_gain(
    y: np.ndarray,
    X_exog: np.ndarray | None,
    period: int,
    n_exog_lags: int,
) -> float:
    """R² gain of a ridge ARX model over a pure ridge AR model.

    Both models are fit on the same data (training split). Returns 0.0 for
    univariate cases. Clipped to [0, 1].

    Limitation: a weak linear AR baseline on impulsive or nonlinear series
    can inflate the gain beyond genuine external forcing dependence.

    Parameters
    ----------
    y:            Clean 1-D target array (training split).
    X_exog:       2-D exogenous array (n_samples × n_vars) or None.
    period:       Dominant seasonality period; AR order p = min(period, 30).
    n_exog_lags:  Exogenous lags included: 0..n_exog_lags.
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
    AR = np.column_stack([y[max_lag - lag: n - lag] for lag in range(1, p + 1)])
    exog_blocks = [X_exog[max_lag - lag: n - lag, :] for lag in range(0, q + 1)]
    ARX = np.hstack([AR] + exog_blocks)

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
