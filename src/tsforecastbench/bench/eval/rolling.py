"""Rolling-origin evaluation producing error-vs-lead-time curves.

For each of `n_origins` cutoffs spaced across the test period, fit each method on the
history up to the cutoff (gaps imputed) and forecast `horizon` steps. Errors are
collected per lead time and averaged over all origins, giving the canonical
"skill degrades with lead time" curve used to compare methods.

Refit-per-origin is honest but costs more for ARIMA/LightGBM; keep n_origins modest
for those, or run the cheap baselines (Mean/Persistence/SeasonalNaive/Climatology)
which refit trivially.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from tsforecastbench.bench.eval.config import DEFAULT_ORIGINS
from tsforecastbench.bench.eval.loop import _complete_for_fit, _horizon_to_steps
from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.bench.methods.naive import _default_season_length
from tsforecastbench.data.schema import Case


def _select_rolling_origins(
    test_start: int, last_origin: int, raw_y: np.ndarray, H: int, n_origins: int
) -> list[int]:
    """Pick up to `n_origins` origins spread across [test_start, last_origin].

    Oversamples candidates, drops any whose horizon window is <50% observed, then
    evenly downsamples the survivors back to `n_origins` — evenly, not by truncating
    the sorted list — so a few dropped candidates near the end don't collapse the
    selection onto the front of the range.
    """
    cand = np.unique(
        np.linspace(test_start, last_origin, max(n_origins * 2, n_origins)).astype(int)
    )
    valid = [int(o) for o in cand if np.isfinite(raw_y[o : o + H]).mean() >= 0.5]
    if not valid:
        return [int(cand[0])]
    if len(valid) <= n_origins:
        return valid
    idx = np.unique(np.linspace(0, len(valid) - 1, n_origins).astype(int))
    return [valid[i] for i in idx]


def run_case_rolling(
    case: Case,
    methods: list[Forecaster],
    n_origins: int = DEFAULT_ORIGINS,
    prefer_synth: bool = False,
) -> pd.DataFrame:
    """Rolling-origin scores for the cases x methods matrix — robust to single-window luck.

    Each method is refit at `n_origins` cutoffs spread across the test period and forecasts
    the primary horizon; errors are POOLED across all origins x leads (observed points only)
    and the matrix metrics computed once on the pool. Returns the same tidy record shape as
    `loop.run_case` (case_id, domain, method, horizon, metric, value, seconds,
    frac_finite_test, status) so the matrix renderer is unchanged.
    """
    df = case.load(prefer_synth=prefer_synth)
    splits = case.split_frame(df)
    train_val = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    test = splits["test"]
    if len(test) == 0:
        raise ValueError(f"Case {case.id}: empty test split")
    full = pd.concat([train_val, test], ignore_index=True)

    H = _horizon_to_steps(case.horizon_primary, case.freq)
    exo_cols = list(dict.fromkeys([*case.exogenous, *case.exogenous_known_future]))
    full_fit = _complete_for_fit(full, ["y", *exo_cols])
    exo_present = [c for c in exo_cols if c in full_fit.columns]
    raw_y = full["y"].to_numpy(dtype=float)

    test_start = len(train_val)
    last_origin = len(full) - H
    if last_origin <= test_start:
        raise ValueError(f"Case {case.id}: test span too short for horizon {H}")
    origins = _select_rolling_origins(test_start, last_origin, raw_y, H, n_origins)

    season = _default_season_length(case.freq)
    y_pre = full_fit["y"].iloc[:test_start].to_numpy(dtype=float)
    s = min(season, max(1, len(y_pre) // 2))
    naive_scale = float(np.mean(np.abs(y_pre[s:] - y_pre[:-s]))) if len(y_pre) > s else float("nan")

    metric_names = ("MAE", "RMSE", "MASE", "Skill_vs_Persistence", "Skill_vs_Mean")
    records: list[dict] = []
    for method in methods:
        t0 = time.perf_counter()
        a, p, ref_p, ref_m = [], [], [], []  # pooled actual, pred, persistence, mean
        n_ok = 0
        try:
            for o in origins:
                history = full_fit.iloc[:o]
                fexog = full_fit.iloc[o : o + H][["ds", *exo_present]] if exo_present else None
                method.fit(case, history)
                y_hat = method.predict(case, history, H, future_exog=fexog)["y_hat"].to_numpy(float)
                actual = raw_y[o : o + H]
                m = np.isfinite(actual)
                if not m.any():
                    continue
                n_ok += 1
                a.extend(actual[m])
                p.extend(y_hat[m])
                ref_p.extend(np.full(int(m.sum()), float(history["y"].iloc[-1])))
                ref_m.extend(np.full(int(m.sum()), float(np.nanmean(history["y"].to_numpy(float)))))
            if not a:
                raise ValueError("no finite test points across origins")
            a = np.array(a)
            p = np.array(p)
            ref_p = np.array(ref_p)
            ref_m = np.array(ref_m)
            mse = float(np.mean((a - p) ** 2))
            metrics = {
                "MAE": float(np.mean(np.abs(a - p))),
                "RMSE": float(np.sqrt(mse)),
                "MASE": float(np.mean(np.abs(a - p)) / naive_scale)
                if naive_scale
                else float("nan"),
                "Skill_vs_Persistence": 1.0 - mse / float(np.mean((a - ref_p) ** 2)),
                "Skill_vs_Mean": 1.0 - mse / float(np.mean((a - ref_m) ** 2)),
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
                    "frac_finite_test": float(len(a) / max(1, len(origins) * H))
                    if isinstance(a, np.ndarray)
                    else 0.0,
                    "status": status,
                    "n_origins": n_ok,
                }
            )
        print(f"  {method.name:>18s}: {status[:50]:<50s} ({n_ok} origins, {seconds:.1f}s)")
    return pd.DataFrame(records)


def run_leadtime_curve(
    case: Case,
    methods: list[Forecaster],
    n_origins: int = DEFAULT_ORIGINS,
    max_horizon: str | None = None,
    prefer_synth: bool = False,
) -> pd.DataFrame:
    """Return tidy per-lead-time errors averaged over rolling origins.

    Columns: case_id, method, lead, MAE, RMSE, bias, n_origins.
    `lead` is in steps (1 = first step ahead).
    """
    df = case.load(prefer_synth=prefer_synth)
    splits = case.split_frame(df)
    train_val = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    test = splits["test"]
    full = pd.concat([train_val, test], ignore_index=True)
    if len(test) == 0:
        raise ValueError(f"Case {case.id}: empty test split")

    H = _horizon_to_steps(max_horizon or case.horizon_primary, case.freq)
    exo_cols = list(dict.fromkeys([*case.exogenous, *case.exogenous_known_future]))
    fit_cols = ["y", *exo_cols]
    full_fit = _complete_for_fit(full, fit_cols)
    exo_present = [c for c in exo_cols if c in full_fit.columns]

    test_start = len(train_val)
    last_origin = len(full) - H - 1
    if last_origin <= test_start:
        raise ValueError(f"Case {case.id}: test span too short for horizon {H}")
    origins = np.unique(np.linspace(test_start, last_origin, n_origins).astype(int))

    raw_y = full["y"].to_numpy(dtype=float)  # for honest scoring (keeps NaN)
    rows: list[dict] = []
    for method in methods:
        # accumulate errors per lead
        abs_err = [[] for _ in range(H)]
        sq_err = [[] for _ in range(H)]
        signed = [[] for _ in range(H)]
        n_ok = 0
        for o in origins:
            history = full_fit.iloc[: o + 1]
            future_exog = (
                full_fit.iloc[o + 1 : o + 1 + H][["ds", *exo_present]] if exo_present else None
            )
            try:
                method.fit(case, history)
                y_hat = method.predict(case, history, H, future_exog=future_exog)["y_hat"].to_numpy(
                    dtype=float
                )
            except Exception:
                continue
            actual = raw_y[o + 1 : o + 1 + H]
            n_ok += 1
            for h in range(H):
                yt = actual[h]
                if np.isfinite(yt):
                    e = yt - y_hat[h]
                    abs_err[h].append(abs(e))
                    sq_err[h].append(e * e)
                    signed[h].append(e)
        for h in range(H):
            if abs_err[h]:
                rows.append(
                    {
                        "case_id": case.id,
                        "method": method.name,
                        "lead": h + 1,
                        "MAE": float(np.mean(abs_err[h])),
                        "RMSE": float(np.sqrt(np.mean(sq_err[h]))),
                        "bias": float(np.mean(signed[h])),
                        "n_origins": n_ok,
                    }
                )
        print(f"  {method.name:>14s}: {n_ok} origins x {H} leads")

    return pd.DataFrame(rows)
