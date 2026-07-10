"""Evaluation loop - single hold-out (rolling-origin in Phase B).

Every method is fit and scored on the *curated* series (`Case.load(curated=True)`): the
gap policy in `tsforecastbench.data.curate` has already filled small gaps and trimmed
each case to its tail contiguous span, so the target is complete. Scoring still counts
only genuine observations — the curated frame's `y_observed` mask marks which points
were measured vs. filled, and metrics use only the measured ones. The fraction of
finite test points is reported so a gappy test window is visible, not silently scored.

`Forecaster.is_global` methods additionally get a second, longer training frame
(`Case.load_extended`) reaching back before the tail segment — *when that extended
history is itself gap-free*. If it contains a gap bigger than the per-case cap, using
it as-is would require either fabricating a value across the gap (dishonest) or
silently treating pre-gap and post-gap rows as adjacent time steps in a windowed
model (wrong) — no current method's `fit` is gap-aware. Rather than do either, an
`is_global` method falls back to the same curated data every other method uses for
that case; `used_extended_history` on each result row records which happened. Building
genuine gap-aware (or cross-series pooled) training is future work — see
`Forecaster.fit_pooled`.
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
    # Rows the origin-picking slide pulled from the test split into training -- carried
    # over verbatim onto the extended frame too, so global methods forecast from the
    # same origin as everyone else.
    slide_in = full_fit.iloc[test_start:origin].reset_index(drop=True)
    future_exog = None
    if exo_cols:
        fe = full_fit.iloc[origin : origin + horizon_steps]
        future_exog = fe[["ds", *[c for c in exo_cols if c in fe.columns]]].reset_index(drop=True)

    # Test target kept raw; we score only on finite (observed) points.
    y_true_raw = raw_y[origin : origin + horizon_steps]
    finite = np.isfinite(y_true_raw)
    frac_finite = float(finite.mean()) if len(finite) else 0.0

    def _train_stats(train_df: pd.DataFrame) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
        y_train = train_df["y"].to_numpy(dtype=float)
        # Declared dominant period the train series can support (>= 2 cycles), else the
        # next declared period, else 1 — a real season or honest persistence scaling,
        # never an arbitrary truncated lag.
        season = mase_season_for(case.seasonalities, len(y_train))
        ref_persistence = np.full(horizon_steps, float(train_df["y"].iloc[-1]))
        ref_mean = np.full(horizon_steps, float(np.nanmean(y_train)))
        return y_train, season, ref_persistence, ref_mean

    y_train, season, ref_persistence, ref_mean = _train_stats(train_fit)

    # Extended history for `is_global` methods -- only if the case actually has one, and
    # only if it's itself gap-free. A gap bigger than the per-case cap would otherwise
    # need fabricating across (dishonest) or silently treating pre/post-gap rows as
    # adjacent (wrong); no method's fit is gap-aware, so we fall back to the shared
    # curated data instead of risking either. See the module docstring.
    train_fit_ext = None
    if any(m.is_global for m in methods):
        try:
            extended = case.load_extended(prefer_synth=prefer_synth)
        except Exception as exc:
            print(
                f"  [note] load_extended failed ({type(exc).__name__}: {exc}); "
                "is_global methods fall back to the shared curated data."
            )
        else:
            if extended["y"].isna().any():
                print(
                    "  [note] extended history has a gap bigger than the per-case cap; "
                    "is_global methods fall back to the shared curated data."
                )
            else:
                train_fit_ext = pd.concat([extended, slide_in], ignore_index=True)
    ext_stats = _train_stats(train_fit_ext) if train_fit_ext is not None else None

    metric_names = ("MAE", "RMSE", "MASE", "Skill_vs_Persistence", "Skill_vs_Mean")
    records: list[dict] = []
    for method in methods:
        t0 = time.perf_counter()
        used_extended = method.is_global and train_fit_ext is not None
        m_train_fit, m_y_train, m_season, m_ref_persistence, m_ref_mean = (
            (train_fit_ext, *ext_stats)
            if used_extended
            else (train_fit, y_train, season, ref_persistence, ref_mean)
        )
        try:
            method.fit(case, m_train_fit)
            preds = method.predict(case, m_train_fit, horizon_steps, future_exog=future_exog)
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
                "MASE": mase(yt, yh, m_y_train, season=m_season),
                "Skill_vs_Persistence": skill_score(yt, yh, m_ref_persistence[finite]),
                "Skill_vs_Mean": skill_score(yt, yh, m_ref_mean[finite]),
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
                    "is_global": method.is_global,
                    "used_extended_history": used_extended,
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
