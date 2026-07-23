"""Rolling-origin evaluation producing error-vs-lead-time curves.

For each of `n_origins` cutoffs spaced across the test period, forecast `horizon` steps;
errors are pooled per lead time and averaged over all origins, giving the canonical
"skill degrades with lead time" curve used to compare methods. How the model behind each
forecast is obtained depends on the method's `is_global` flag (see `methods.base`):

- Local methods (AutoARIMA, AutoETS, Theta, TBATS, MSTL, and the naive baselines) have no
  separable trained model, so they are refit on the curated history up to each cutoff --
  that refit is simply how they fold in the newer data.
- Global methods (LightGBM, CatBoost, MLP, LinearLags, DLinear, Chronos, and the optional
  torch/neuralforecast models) estimate a reusable model once, on data strictly before
  the test period, then evaluate every cutoff by inference alone -- each origin's predict
  conditions on the real observed history up to that origin without re-estimating any
  parameters. This stays leak-free and makes raising `n_origins` cheap for the methods
  that otherwise dominate runtime.

Scoring uses only genuinely observed points (the curated frame's `y_observed` mask);
filled points never enter a metric.

`run_case_rolling` is `scripts/run_pilot.py`'s default path (single hold-out `run_case`
only runs under `--fast`), so it gets the same segment-pooled extended-history scoring
as `eval.loop.run_case` -- see that module's docstring. Methods that are `is_global` and
`pools_segments` get a second (`used_extended_history=True`) score from a single fit on
the full pre-curated history (its gap-bounded segments), alongside the local one.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from tsforecastbench.bench.eval.config import DEFAULT_ORIGINS
from tsforecastbench.bench.eval.loop import _horizon_to_steps, _raw_y, _segment_extended
from tsforecastbench.bench.eval.metrics import kge, mape, nse, smape
from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.data.schema import Case
from tsforecastbench.data.seasonality import mase_season_for


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


SNAPSHOT_COLUMNS = (
    "case_id",
    "method",
    "used_extended_history",
    "origin_id",
    "origin_ds",
    "ds",
    "y_hat",
)


def run_case_rolling(
    case: Case,
    methods: list[Forecaster],
    n_origins: int = DEFAULT_ORIGINS,
    prefer_synth: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling-origin scores for the cases x methods matrix — robust to single-window luck.

    Each method forecasts the primary horizon at `n_origins` cutoffs spread across the test
    period; errors are POOLED across all origins x leads (observed points only) and the
    matrix metrics computed once on the pool. Local methods refit at each cutoff; global
    methods (`is_global`) are fit once before the test period and evaluated across all
    cutoffs by inference (see the module docstring).

    Returns `(metrics, snapshots)`:
    - `metrics` keeps the tidy record shape of `loop.run_case` (case_id, domain, method,
      horizon, metric, value, seconds, frac_finite_test, status) so the matrix renderer is
      unchanged.
    - `snapshots` holds every origin's full predicted series (columns `SNAPSHOT_COLUMNS`:
      case_id, method, used_extended_history, origin_id, origin_ds, ds, y_hat) so the
      results viewer can draw real forecasts. Origins are case-level, so `origin_id`/
      `origin_ds` align across methods and fits, which is what lets the viewer overlay
      several methods (and swap in a method's extended-history fit) on one origin's
      panel. A method that errors contributes no snapshot rows for that fit (its metrics
      row still records the failure).
    """
    df = case.load(prefer_synth=prefer_synth, curated=True)
    splits = case.split_frame(df)
    train_val = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    test = splits["test"]
    if len(test) == 0:
        raise ValueError(f"Case {case.id}: empty test split")
    full = pd.concat([train_val, test], ignore_index=True)

    H = _horizon_to_steps(case.horizon_primary, case.freq)
    exo_cols = list(dict.fromkeys([*case.exogenous, *case.exogenous_known_future]))
    full_fit = full  # curated: the target is already complete within the span
    exo_present = [c for c in exo_cols if c in full_fit.columns]
    raw_y = _raw_y(full)

    test_start = len(train_val)
    last_origin = len(full) - H
    if last_origin <= test_start:
        raise ValueError(f"Case {case.id}: test span too short for horizon {H}")
    origins = _select_rolling_origins(test_start, last_origin, raw_y, H, n_origins)

    def _naive_scale(y_pre: np.ndarray) -> float:
        s = mase_season_for(case.seasonalities, len(y_pre))
        return float(np.mean(np.abs(y_pre[s:] - y_pre[:-s]))) if len(y_pre) > s else float("nan")

    naive_scale = _naive_scale(full_fit["y"].iloc[:test_start].to_numpy(dtype=float))

    # Extended-history training for is_global methods that pool segments -- the full
    # single-series record split at its big gaps into contiguous per-`unique_id`
    # segments, so no window spans a gap (see `eval.loop._segment_extended`). A gap-free
    # record has no history beyond the curated span and yields a single segment, so no
    # extended fit runs. The extended fit reuses the local MASE/persistence/mean
    # references, so its scores differ from the local fit only through the model.
    ext_segments = None
    if any(m.is_global and m.pools_segments for m in methods):
        try:
            extended = case.load_extended(prefer_synth=prefer_synth)
        except Exception as exc:
            print(
                f"  [note] load_extended failed ({type(exc).__name__}: {exc}); "
                "is_global methods use the curated span only."
            )
        else:
            segs = _segment_extended(extended, case.id, case.freq, H)
            if len(segs) >= 2:
                ext_segments = segs
            else:
                print(
                    "  [note] no history beyond the curated span; "
                    "is_global methods use the curated span only."
                )

    metric_names = (
        "MAE",
        "RMSE",
        "MASE",
        "Skill_vs_Persistence",
        "Skill_vs_Mean",
        "MAPE",
        "sMAPE",
        "NSE",
        "KGE",
    )
    records: list[dict] = []
    snap_frames: list[pd.DataFrame] = []
    for method in methods:
        # Local fit for everyone; a second extended fit for segment-pooling methods when
        # the case has history beyond the curated span. Both fits' forecasts are saved as
        # snapshots, tagged by `used_extended_history`, so the viewer can show either.
        variants = [(None, False)]
        if method.is_global and method.pools_segments and ext_segments is not None:
            variants.append((ext_segments, True))
        for segs, used_extended in variants:
            t0 = time.perf_counter()
            a, p, ref_p, ref_m = [], [], [], []  # pooled actual, pred, persistence, mean
            method_snaps: list[pd.DataFrame] = []
            n_ok = 0
            try:
                # A global method keeps a trained model apart from its input, so it is fit
                # once, on data strictly before the test period — train+val locally, or the
                # pre-curated segments for the extended fit — then evaluated across every
                # origin by inference alone. No origin's forecast sees test data in its
                # weights, so this stays leak-free while cutting the cost of `--origins` to
                # one fit per case. A local (non-global) method, whose parameters are a
                # direct function of the fitted series, refits at each origin below instead.
                if method.is_global:
                    fit_frame = (
                        pd.concat([*segs], ignore_index=True) if used_extended else train_val
                    )
                    method.fit(case, fit_frame)
                for i, o in enumerate(origins):
                    local_hist = full_fit.iloc[:o]
                    if used_extended:
                        slide = full_fit.iloc[test_start:o].copy()
                        slide["unique_id"] = segs[-1]["unique_id"].iloc[0]
                        history = pd.concat([*segs, slide], ignore_index=True)
                    else:
                        history = local_hist
                    fexog = full_fit.iloc[o : o + H][["ds", *exo_present]] if exo_present else None
                    if not method.is_global:
                        method.fit(case, history)
                    preds = method.predict(case, history, H, future_exog=fexog)
                    y_hat = preds["y_hat"].to_numpy(float)
                    # Full predicted series (all H steps, before masking) so the viewer
                    # draws a continuous forecast line even where the actual has gaps.
                    method_snaps.append(
                        pd.DataFrame(
                            {
                                "case_id": case.id,
                                "method": method.name,
                                "used_extended_history": used_extended,
                                "origin_id": i,
                                "origin_ds": full_fit["ds"].iloc[o],
                                "ds": preds["ds"].to_numpy(),
                                "y_hat": y_hat,
                            }
                        )
                    )
                    actual = raw_y[o : o + H]
                    m = np.isfinite(actual)
                    if not m.any():
                        continue
                    n_ok += 1
                    a.extend(actual[m])
                    p.extend(y_hat[m])
                    # Local-history references, shared by both fits so the extended fit's
                    # skill scores isolate the model, not a shifted baseline.
                    ref_p.extend(np.full(int(m.sum()), float(local_hist["y"].iloc[-1])))
                    ref_m.extend(
                        np.full(int(m.sum()), float(np.nanmean(local_hist["y"].to_numpy(float))))
                    )
                if not a:
                    raise ValueError("no finite test points across origins")
                a = np.array(a)
                p = np.array(p)
                ref_p = np.array(ref_p)
                ref_m = np.array(ref_m)
                mse = float(np.mean((a - p) ** 2))
                mse_ref_p = float(np.mean((a - ref_p) ** 2))
                mse_ref_m = float(np.mean((a - ref_m) ** 2))
                metrics = {
                    "MAE": float(np.mean(np.abs(a - p))),
                    "RMSE": float(np.sqrt(mse)),
                    "MASE": float(np.mean(np.abs(a - p)) / naive_scale)
                    if naive_scale
                    else float("nan"),
                    "Skill_vs_Persistence": 1.0 - mse / mse_ref_p
                    if mse_ref_p != 0.0
                    else float("nan"),
                    "Skill_vs_Mean": 1.0 - mse / mse_ref_m if mse_ref_m != 0.0 else float("nan"),
                    "MAPE": mape(a, p),
                    "sMAPE": smape(a, p),
                    "NSE": nse(a, p),
                    "KGE": kge(a, p),
                }
                status = "ok"
                snap_frames.extend(method_snaps)
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
                        "frac_finite_test": float(len(a) / max(1, len(origins) * H))
                        if isinstance(a, np.ndarray)
                        else 0.0,
                        "status": status,
                        "n_origins": n_ok,
                    }
                )
            tag = " [ext]" if used_extended else ""
            print(
                f"  {method.name:>18s}{tag:>6s}: {status[:48]:<48s} ({n_ok} origins, {seconds:.1f}s)"
            )
    snapshots = (
        pd.concat(snap_frames, ignore_index=True)
        if snap_frames
        else pd.DataFrame(columns=list(SNAPSHOT_COLUMNS))
    )
    return pd.DataFrame(records), snapshots


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
    df = case.load(prefer_synth=prefer_synth, curated=True)
    splits = case.split_frame(df)
    train_val = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    test = splits["test"]
    full = pd.concat([train_val, test], ignore_index=True)
    if len(test) == 0:
        raise ValueError(f"Case {case.id}: empty test split")

    H = _horizon_to_steps(max_horizon or case.horizon_primary, case.freq)
    exo_cols = list(dict.fromkeys([*case.exogenous, *case.exogenous_known_future]))
    full_fit = full  # curated: the target is already complete within the span
    exo_present = [c for c in exo_cols if c in full_fit.columns]

    test_start = len(train_val)
    last_origin = len(full) - H - 1
    if last_origin <= test_start:
        raise ValueError(f"Case {case.id}: test span too short for horizon {H}")
    origins = np.unique(np.linspace(test_start, last_origin, n_origins).astype(int))

    raw_y = _raw_y(full)  # for honest scoring (filled points masked back to NaN)
    rows: list[dict] = []
    for method in methods:
        # accumulate errors per lead
        abs_err = [[] for _ in range(H)]
        sq_err = [[] for _ in range(H)]
        signed = [[] for _ in range(H)]
        n_ok = 0
        # Same protocol as the matrix (see `run_case_rolling`): a global method is fit once,
        # on train+val only, then evaluated across every origin by inference; a local method
        # refits per origin below. A failed single fit makes the method unusable on this
        # case, so it contributes no rows.
        if method.is_global:
            try:
                method.fit(case, full_fit.iloc[:test_start])
            except Exception:
                continue
        for o in origins:
            history = full_fit.iloc[: o + 1]
            future_exog = (
                full_fit.iloc[o + 1 : o + 1 + H][["ds", *exo_present]] if exo_present else None
            )
            try:
                if not method.is_global:
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
