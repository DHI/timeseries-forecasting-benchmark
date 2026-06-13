"""Evaluation loop - single hold-out (rolling-origin in Phase B).

Real observed series contain gaps. We impute gaps in the *training* frame so methods
that require a complete target (e.g. mlforecast/LightGBM) can fit, while scoring only
on genuinely observed test points (NaN-masked metrics). The fraction of finite test
points is reported so a case with a gappy test window is visible, not silently scored.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from tsforecastbench.bench.eval.metrics import mae, mase, rmse, skill_score
from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.bench.methods.naive import _default_season_length
from tsforecastbench.data.schema import Case


def _horizon_to_steps(horizon: str, freq: str) -> int:
    delta = pd.Timedelta(horizon)
    step = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    return int(delta / step)


def _complete_for_fit(df: pd.DataFrame, cols: list[str], interp_limit: int | None = None) -> pd.DataFrame:
    """Return a copy with `cols` gap-filled so models needing complete data can fit.

    Strategy: linear time-interpolation (optionally limited), then fill any remaining
    edge/long-gap NaNs with the column mean. Training-only; never used for scoring.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        s = s.interpolate(method="linear", limit=interp_limit, limit_direction="both")
        if s.isna().any():
            s = s.fillna(s.mean())
        out[c] = s
    return out


def run_case(case: Case, methods: list[Forecaster], prefer_synth: bool = False) -> pd.DataFrame:
    df = case.load(prefer_synth=prefer_synth)
    splits = case.split_frame(df)
    train_val = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    test = splits["test"]
    if len(test) == 0:
        raise ValueError(f"Case {case.id}: test split is empty")

    horizon_steps = _horizon_to_steps(case.horizon_primary, case.freq)
    horizon_steps = min(horizon_steps, len(test))

    # Impute training gaps for methods that need a complete target/covariates.
    exo_cols = list(dict.fromkeys([*case.exogenous, *case.exogenous_known_future]))
    fit_cols = ["y", *exo_cols]
    train_fit = _complete_for_fit(train_val, fit_cols)

    # Future covariates over the horizon (a perfect-forecast proxy), imputed and passed to
    # exogenous-aware methods. Univariate methods receive but ignore it.
    future_exog = None
    if exo_cols:
        fe = _complete_for_fit(test.head(horizon_steps), exo_cols)
        future_exog = fe[["ds", *[c for c in exo_cols if c in fe.columns]]]

    # Test target kept raw; we score only on finite (observed) points.
    y_true_raw = test["y"].head(horizon_steps).to_numpy(dtype=float)
    finite = np.isfinite(y_true_raw)
    frac_finite = float(finite.mean()) if len(finite) else 0.0

    y_train = train_fit["y"].to_numpy(dtype=float)
    season = _default_season_length(case.freq)
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
                "MASE": mase(yt, yh, y_train, season=min(season, len(y_train) // 2)),
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
                    "method": method.name,
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
        print(f"  [note] test window {frac_finite:.0%} observed; "
              f"{int((1 - frac_finite) * horizon_steps)} of {horizon_steps} points were gaps (excluded).")
    return pd.DataFrame(records)
