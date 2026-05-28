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
