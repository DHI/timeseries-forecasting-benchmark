"""Forecast metrics. Plain numpy — no fitting state."""

from __future__ import annotations

import numpy as np


def mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - y_hat)))


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def mase(y: np.ndarray, y_hat: np.ndarray, y_train: np.ndarray, season: int = 1) -> float:
    """Mean Absolute Scaled Error.

    Denominator is the in-sample MAE of a (seasonal) naive forecast on the train series.
    MASE < 1 means the forecast is better than (seasonal) naive on training data.
    """
    if len(y_train) <= season:
        raise ValueError(f"Need len(y_train) > season ({len(y_train)} <= {season})")
    naive_errors = np.abs(y_train[season:] - y_train[:-season])
    denom = float(np.mean(naive_errors))
    if denom == 0.0:
        return float("nan")
    return float(np.mean(np.abs(y - y_hat)) / denom)


def skill_score(y: np.ndarray, y_hat: np.ndarray, y_hat_reference: np.ndarray) -> float:
    """Skill score vs a reference forecast: 1 - MSE(model) / MSE(reference).

    Positive => model better than reference. 1 = perfect. 0 = no better than reference.
    """
    mse_model = float(np.mean((y - y_hat) ** 2))
    mse_ref = float(np.mean((y - y_hat_reference) ** 2))
    if mse_ref == 0.0:
        return float("nan")
    return 1.0 - mse_model / mse_ref


def mape(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Points where y == 0 are excluded (undefined);
    nan if none survive."""
    mask = y != 0.0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y[mask] - y_hat[mask]) / y[mask])))


def smape(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Symmetric MAPE: denominator is (|y| + |y_hat|) / 2, so a point is only excluded
    when both y and y_hat are 0; nan if none survive."""
    denom = (np.abs(y) + np.abs(y_hat)) / 2.0
    mask = denom != 0.0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y[mask] - y_hat[mask]) / denom[mask]))


def nse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency: 1 - sum((y - y_hat)^2) / sum((y - mean(y))^2).

    1 = perfect, 0 = no better than the mean of `y`, negative = worse than the mean.
    Unlike `skill_score`, the reference here is the mean of the values being scored, not
    a forecast carried over from training. nan if `y` is constant (zero variance).
    """
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom == 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y - y_hat) ** 2)) / denom


def kge(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Kling-Gupta Efficiency (Gupta et al. 2009): 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2),
    with r = Pearson correlation, alpha = std(y_hat)/std(y), beta = mean(y_hat)/mean(y).

    1 = perfect. nan if `y` is constant or has zero mean (alpha/beta undefined).
    """
    std_y, std_hat = float(np.std(y)), float(np.std(y_hat))
    mean_y = float(np.mean(y))
    if std_y == 0.0 or mean_y == 0.0:
        return float("nan")
    r = float(np.corrcoef(y, y_hat)[0, 1]) if std_hat > 0.0 else 0.0
    alpha = std_hat / std_y
    beta = float(np.mean(y_hat)) / mean_y
    return 1.0 - float(np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
