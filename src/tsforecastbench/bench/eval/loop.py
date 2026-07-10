"""Evaluation loop - single hold-out (rolling-origin in Phase B).

Methods are fit and scored on the *curated* series (`Case.load(curated=True)`): the gap
policy in `tsforecastbench.data.curate` has already filled small gaps and trimmed each
case to its tail contiguous span, so the target is complete and every method sees
identical data. Scoring still counts only genuine observations — the curated frame's
`y_observed` mask marks which points were measured vs. filled, and metrics use only the
measured ones. The fraction of finite test points is reported so a gappy test window is
visible, not silently scored.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from tsforecastbench.bench.eval.metrics import mae, mase, rmse, skill_score
from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.data.schema import Case, _observed_mask
from tsforecastbench.data.seasonality import mase_season_for


def _horizon_to_steps(horizon: str, freq: str) -> int:
    delta = pd.Timedelta(horizon)
    step = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    return int(delta / step)


def _raw_y(df: pd.DataFrame) -> np.ndarray:
    """Target with filled points masked back to NaN, for honest scoring/origin choice."""
    return np.where(_observed_mask(df).to_numpy(), df["y"].to_numpy(dtype=float), np.nan)


def _pick_origin(
    raw_y: np.ndarray, test_start: int, last_origin: int, horizon: int, min_frac: float = 0.5
) -> int:
    """First origin in [test_start, last_origin] whose horizon window is >= min_frac
    observed; else the origin with the densest window. Keeps the single-holdout metric
    meaningful when the test split begins in a gap."""
    if last_origin <= test_start:
        return test_start
    step = max(1, horizon // 4)
    best_o, best_frac = test_start, -1.0
    for o in range(test_start, last_origin + 1, step):
        frac = float(np.isfinite(raw_y[o : o + horizon]).mean())
        if frac >= min_frac:
            return o
        if frac > best_frac:
            best_frac, best_o = frac, o
    return best_o


def run_case(case: Case, methods: list[Forecaster], prefer_synth: bool = False) -> pd.DataFrame:
    df = case.load(prefer_synth=prefer_synth, curated=True)
    splits = case.split_frame(df)
    train_val = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    test = splits["test"]
    if len(test) == 0:
        raise ValueError(f"Case {case.id}: test split is empty")

    horizon_steps = _horizon_to_steps(case.horizon_primary, case.freq)
    horizon_steps = min(horizon_steps, len(test))

    exo_cols = list(dict.fromkeys([*case.exogenous, *case.exogenous_known_future]))

    # Choose the forecast origin within the test split. The naive choice (the very first
    # test timestamp) can land on a filled (non-observed) point, making the single-holdout
    # window mostly unscored and the metric meaningless. Instead slide to the first origin
    # whose horizon window is at least half genuinely observed (falling back to the densest
    # window). History is then extended to that origin so the forecast is still made from
    # genuine observations.
    full = pd.concat([train_val, test], ignore_index=True)
    full_fit = full  # curated: the target is already complete within the span
    raw_y = _raw_y(full)
    test_start = len(train_val)
    last_origin = len(full) - horizon_steps
    origin = _pick_origin(raw_y, test_start, last_origin, horizon_steps)

    train_fit = full_fit.iloc[:origin].reset_index(drop=True)
    future_exog = None
    if exo_cols:
        fe = full_fit.iloc[origin : origin + horizon_steps]
        future_exog = fe[["ds", *[c for c in exo_cols if c in fe.columns]]].reset_index(drop=True)

    # Test target kept raw; we score only on finite (observed) points.
    y_true_raw = raw_y[origin : origin + horizon_steps]
    finite = np.isfinite(y_true_raw)
    frac_finite = float(finite.mean()) if len(finite) else 0.0

    y_train = train_fit["y"].to_numpy(dtype=float)
    # Declared dominant period the train series can support (>= 2 cycles), else the next
    # declared period, else 1 — a real season or honest persistence scaling, never an
    # arbitrary truncated lag.
    season = mase_season_for(case.seasonalities, len(y_train))
    ref_persistence = np.full(horizon_steps, float(train_fit["y"].iloc[-1]))
    ref_mean = np.full(horizon_steps, float(np.nanmean(y_train)))

    metric_names = ("MAE", "RMSE", "MASE", "Skill_vs_Persistence", "Skill_vs_Mean")
    records: list[dict] = []
    for method in methods:
        t0 = time.perf_counter()
        try:
            method.fit(case, train_fit)
            preds = method.predict(case, train_fit, horizon_steps, future_exog=future_exog)
            y_hat = preds["y_hat"].to_numpy(dtype=float)
            assert len(y_hat) == horizon_steps, (
                f"{method.name} returned {len(y_hat)} predictions, expected {horizon_steps}"
            )
            if finite.sum() == 0:
                raise ValueError("no finite test points to score")
            yt, yh = y_true_raw[finite], y_hat[finite]
            metrics = {
                "MAE": mae(yt, yh),
                "RMSE": rmse(yt, yh),
                "MASE": mase(yt, yh, y_train, season=season),
                "Skill_vs_Persistence": skill_score(yt, yh, ref_persistence[finite]),
                "Skill_vs_Mean": skill_score(yt, yh, ref_mean[finite]),
            }
            status = "ok"
        except Exception as exc:
            metrics = {k: float("nan") for k in metric_names}
            status = f"error: {type(exc).__name__}: {str(exc)[:70]}"
        seconds = time.perf_counter() - t0
        for metric_name, value in metrics.items():
            records.append(
                {
                    "case_id": case.id,
                    "domain": case.domain,
                    "case_exo_tier": case.exo_tier,
                    "method": method.name,
                    "uses_exogenous": method.uses_exogenous,
                    "horizon": case.horizon_primary,
                    "metric": metric_name,
                    "value": value,
                    "seconds": seconds,
                    "frac_finite_test": frac_finite,
                    "status": status,
                }
            )
        print(f"  {method.name:>18s}: {status[:60]:<60s}  ({seconds:.2f}s)")

    if frac_finite < 1.0:
        print(
            f"  [note] test window {frac_finite:.0%} observed; "
            f"{int((1 - frac_finite) * horizon_steps)} of {horizon_steps} points were gaps (excluded)."
        )
    return pd.DataFrame(records)
