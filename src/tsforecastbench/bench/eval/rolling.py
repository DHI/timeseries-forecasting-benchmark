"""Rolling-origin evaluation producing error-vs-lead-time curves.

For each of `n_origins` cutoffs spaced across the test period, forecast `horizon` steps;
errors are pooled per lead time and averaged over all origins, giving the canonical
"skill degrades with lead time" curve used to compare methods.

Every method follows one rule (gh #120): its parameters are estimated once, on data strictly
before the test period, and frozen; every forecast reads all observations up to its origin.
Each origin's predict conditions on the real observed history up to that origin without
re-estimating any parameters. The naive baselines have no parameters, so for them reading the
history is the whole forecast. This stays leak-free, and it means only the fit cost is paid
once; predict cost still scales with
  `n_origins`, and for some methods predict is the expensive half (Theta and MSTL, below).
  Most of the classical ones roll forward through statsforecast's own `forward(y, h)`,
  which applies the fitted model to a new series without calling an optimizer; TBATS has
  no such public method, so it rolls forward through two of statsforecast's own internals
  (`calcTBATSFaster`/`tbats_forecast`) instead, pinned bit-exact against the library's own
  output. Two deterministic pieces (Theta's and MSTL's seasonal decomposition) are
  re-derived per origin, no differently from the ML methods recomputing their calendar
  features -- see `methods.classical`. The ensembles are global for the same reason but
  mixed inside: whichever members keep a trained model are fit once, while the naive
  members they combine are re-derived from each origin's history inside `predict` -- so
  that per-origin cost lands in `predict_seconds`, and `fit_seconds` describes only the
  trained members (see `methods.ensemble`).

`fit_seconds` is a method's one fit and `predict_seconds` the mean predict call, so
neither depends on how many origins a given run used, and fit and predict stay comparable
across method families that pay for very different things (a zero-shot foundation model's `fit_seconds` is ~0;
its real cost is inference, i.e. `predict_seconds`).

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

from tsforecastbench.bench.eval.config import DEFAULT_ORIGINS, DEFAULT_SEEDS
from tsforecastbench.bench.eval.loop import (
    _first_origin_after_observed,
    _horizon_to_steps,
    _raw_y,
    _segment_extended,
    _split_segments_at_train_end,
    fit_before_test,
)
from tsforecastbench.bench.eval.metrics import POOLED_METRICS, pooled_metrics
from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.data.schema import Case


def _select_rolling_origins(
    test_start: int, last_origin: int, raw_y: np.ndarray, H: int, n_origins: int
) -> list[int]:
    """Pick up to `n_origins` origins spread across [test_start, last_origin].

    Oversamples candidates, drops any whose horizon window is <50% observed or whose
    history ends on a filled value (interpolated from the observation after its gap, so
    it carries the future; gh #119), then evenly downsamples the survivors back to
    `n_origins` — evenly, not by truncating the sorted list — so a few dropped
    candidates near the end don't collapse the selection onto the front of the range.
    """
    cand = np.unique(
        np.linspace(test_start, last_origin, max(n_origins * 2, n_origins)).astype(int)
    )
    valid = [
        int(o)
        for o in cand
        if np.isfinite(raw_y[o - 1]) and np.isfinite(raw_y[o : o + H]).mean() >= 0.5
    ]
    if not valid:
        return [_first_origin_after_observed(raw_y, test_start, last_origin)]
    if len(valid) <= n_origins:
        return valid
    idx = np.unique(np.linspace(0, len(valid) - 1, n_origins).astype(int))
    return [valid[i] for i in idx]


def _leadtime_origins(
    test_start: int, last_origin: int, raw_y: np.ndarray, n_origins: int
) -> list[int]:
    """`n_origins` evenly spaced origins in [test_start, last_origin], each moved forward
    to the first one whose history ends on an observed value (gh #119). An origin with
    none before `last_origin` is dropped; raises `ValueError` if all are."""
    evenly = np.linspace(test_start, last_origin, n_origins).astype(int)
    shifted = [o + int(np.argmax(np.isfinite(raw_y[o - 1 : last_origin]))) for o in evenly]
    origins = sorted({int(o) for o in shifted if np.isfinite(raw_y[o - 1])})
    if not origins:
        raise ValueError(
            f"no forecast origin in [{test_start}, {last_origin}] has an observed value before it"
        )
    return origins


SNAPSHOT_COLUMNS = (
    "case_id",
    "method",
    "used_extended_history",
    "origin_id",
    "origin_ds",
    "ds",
    "y_hat",
    "y",
    "y_persistence",
    "y_mean",
)


def run_case_rolling(
    case: Case,
    methods: list[Forecaster],
    n_origins: int = DEFAULT_ORIGINS,
    prefer_synth: bool = False,
    n_seeds: int = DEFAULT_SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling-origin scores for the cases x methods matrix — robust to single-window luck.

    Each method forecasts the primary horizon at `n_origins` cutoffs spread across the test
    period; errors are POOLED across all origins x leads (observed points only) and the
    matrix metrics computed once on the pool. Every method is fit once before the test
    period and evaluated across all cutoffs by inference (see the module docstring).

    Returns `(metrics, snapshots)`:
    - `metrics` keeps the tidy record shape of `loop.run_case` (case_id, domain, method,
      horizon, metric, value, fit_seconds, predict_seconds, frac_finite_test, status) so
      the matrix renderer is unchanged.
    - `snapshots` holds every origin's full predicted series (columns `SNAPSHOT_COLUMNS`:
      case_id, method, used_extended_history, origin_id, origin_ds, ds, y_hat, y,
      y_persistence, y_mean) so the results viewer can draw real forecasts and the
      bootstrap intervals can recompute the score. Origins are case-level, so `origin_id`/
      `origin_ds` align across methods and fits, which is what lets the viewer overlay
      several methods (and swap in a method's extended-history fit) on one origin's
      panel. A method that errors contributes no snapshot rows for that fit (its metrics
      row still records the failure).

    An `is_stochastic` method is run `n_seeds` times per cell and the median-MSE run is
    kept WHOLE — its nine metrics, its snapshots, its timings, its `train_steps` — so the
    cell stays one real fit rather than a composite of several (issue #106). Every other
    method runs once. Averaging the forecasts instead would benchmark a deep ensemble
    against non-ensembled methods, and medianing each metric separately would leave a row
    whose numbers came from different runs and whose drawn forecast matched none of them.
    """
    df = case.load(prefer_synth=prefer_synth, curated=True)
    splits = case.split_frame(df)
    train_end, _val_end, _test_end = case.split_bounds(df)
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

    val_start = len(splits["train"])
    test_start = len(train_val)
    last_origin = len(full) - H
    if last_origin <= test_start:
        raise ValueError(f"Case {case.id}: test span too short for horizon {H}")
    origins = _select_rolling_origins(test_start, last_origin, raw_y, H, n_origins)

    # Parameters are estimated on everything before the test period; the train and val
    # splits also go over apart, for the iteratively trained methods to find their stopping
    # step on (`fit_before_test`, gh #120). None of these is the history a forecast
    # conditions on -- that grows with the origin, further down.
    local_fit_frame = full_fit.iloc[:val_start]
    local_val_frame = full_fit.iloc[val_start:test_start]
    local_pre_test = full_fit.iloc[:test_start]

    def _naive_scale(y_pre: np.ndarray) -> float:
        """In-sample MAE of a seasonal-naive forecast at the case's pinned dominant period
        -- the MASE denominator. Reading the pinned field is what keeps this anchored on the
        same period `SeasonalNaive` forecasts with (research log id 103)."""
        s = case.dominant_seasonality
        if len(y_pre) <= s:
            raise ValueError(
                f"Case {case.id}: {len(y_pre)} pre-test points cannot support a "
                f"seasonal-naive scale at the pinned dominant_seasonality {s}"
            )
        return float(np.mean(np.abs(y_pre[s:] - y_pre[:-s])))

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
            segs = _segment_extended(extended, case, H)
            if len(segs) >= 2:
                ext_segments = segs
            else:
                print(
                    "  [note] no history beyond the curated span; "
                    "is_global methods use the curated span only."
                )

    records: list[dict] = []
    snap_frames: list[pd.DataFrame] = []

    def _run_once(method: Forecaster, segs, used_extended: bool, seed: int) -> dict:
        """Fit and score one (method, variant) at one seed, returning the whole run.

        A closure rather than a module-level helper: it reads a dozen-odd frames and
        indices off the enclosing scope, all of which are seed-independent and computed
        once per case. Everything the caller needs comes back in the returned bundle
        instead of being written into `records`/`snap_frames` here, so several seeds can
        be run and only the one that is kept gets committed.
        """
        # Timed separately: one fit, then a predict per origin.
        fit_seconds = float("nan")
        predict_times: list[float] = []
        a, p, ref_p, ref_m = [], [], [], []  # pooled actual, pred, persistence, mean
        method_snaps: list[pd.DataFrame] = []
        n_ok = 0
        # Rows handed to the fit.
        train_points: int | None = None
        val_points: int | None = None
        # Ranks the seeds of a stochastic method. Pooled over the same origins against the
        # same observed points every time, so only the predictions differ between runs.
        # Infinite for a run that failed, which keeps it out of the median.
        mse = float("inf")
        method.last_fit_steps = None
        method.last_best_step = None
        method.seed = seed
        try:
            # Fit once, on everything before the test period -- plus the pre-curated
            # segments for the extended fit -- then evaluate every origin by inference
            # alone. No origin's forecast sees test data in its weights.
            if used_extended:
                fit_frame, val_frame = _split_segments_at_train_end(segs, train_end)
                pre_test = pd.concat(segs, ignore_index=True)
            else:
                fit_frame, val_frame = local_fit_frame, local_val_frame
                pre_test = local_pre_test
            t_fit0 = time.perf_counter()
            train_points, val_points = fit_before_test(method, case, fit_frame, val_frame, pre_test)
            fit_seconds = time.perf_counter() - t_fit0
            for i, o in enumerate(origins):
                local_hist = full_fit.iloc[:o]
                if used_extended:
                    slide = full_fit.iloc[test_start:o].copy()
                    slide["unique_id"] = segs[-1]["unique_id"].iloc[0]
                    history = pd.concat([*segs, slide], ignore_index=True)
                else:
                    history = local_hist
                fexog = full_fit.iloc[o : o + H][["ds", *exo_present]] if exo_present else None
                t_pred0 = time.perf_counter()
                preds = method.predict(case, history, H, future_exog=fexog)
                predict_times.append(time.perf_counter() - t_pred0)
                y_hat = preds["y_hat"].to_numpy(float)
                actual = raw_y[o : o + H]
                # Local-history references, shared by both fits so the extended fit's
                # skill scores isolate the model, not a shifted baseline.
                persistence = float(local_hist["y"].iloc[-1])
                mean = float(np.nanmean(local_hist["y"].to_numpy(float)))
                # Full predicted series (all H steps, before masking) so the viewer
                # draws a continuous forecast line even where the actual has gaps. The
                # observed value (NaN where filled) and both references go alongside, so
                # the score can be recomputed from the snapshots for the intervals.
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
                            "y": actual,
                            "y_persistence": persistence,
                            "y_mean": mean,
                        }
                    )
                )
                m = np.isfinite(actual)
                if not m.any():
                    continue
                n_ok += 1
                a.extend(actual[m])
                p.extend(y_hat[m])
                ref_p.extend(np.full(int(m.sum()), persistence))
                ref_m.extend(np.full(int(m.sum()), mean))
            if not a:
                raise ValueError("no finite test points across origins")
            a = np.array(a)
            p = np.array(p)
            ref_p = np.array(ref_p)
            ref_m = np.array(ref_m)
            mse = float(np.mean((a - p) ** 2))
            metrics = pooled_metrics(a, p, ref_p, ref_m, naive_scale)
            status = "ok"
        except Exception as exc:
            metrics = {k: float("nan") for k in POOLED_METRICS}
            status = f"error: {type(exc).__name__}: {str(exc)[:70]}"
        return {
            "seed": seed,
            "status": status,
            "mse": mse,
            "metrics": metrics,
            # A failed run contributes no snapshots, so the viewer never draws a line for
            # a cell whose metrics row records an error.
            "snaps": method_snaps if status == "ok" else [],
            "fit_seconds": fit_seconds,
            "predict_seconds": float(np.mean(predict_times)) if predict_times else float("nan"),
            "n_ok": n_ok,
            "frac_finite_test": float(len(a) / max(1, len(origins) * H))
            if isinstance(a, np.ndarray)
            else 0.0,
            "train_points": train_points,
            "val_points": val_points,
            # Read here rather than by the caller: the next seed's fit overwrites it.
            "train_steps": method.last_fit_steps,
            "best_step": method.last_best_step,
        }

    for method in methods:
        # Local fit for everyone; a second extended fit for segment-pooling methods when
        # the case has history beyond the curated span. Both fits' forecasts are saved as
        # snapshots, tagged by `used_extended_history`, so the viewer can show either.
        variants = [(None, False)]
        if method.is_global and method.pools_segments and ext_segments is not None:
            variants.append((ext_segments, True))
        for segs, used_extended in variants:
            # One draw is not a score for a method whose forecast moves with the seed, so
            # take several and keep the middle one whole (issue #106). Every run is pooled
            # over the same origins against the same observed points, so the MSEs rank
            # runs of the same experiment and the chosen run's nine metrics, snapshots and
            # timings all describe one real fit.
            cell_seeds = list(range(n_seeds)) if method.is_stochastic else [0]
            runs = []
            for seed in cell_seeds:
                run = _run_once(method, segs, used_extended, seed)
                if len(cell_seeds) > 1:
                    print(
                        f"    seed {seed}: {run['status'][:40]:<40s} "
                        f"(fit {run['fit_seconds']:.2f}s, MASE {run['metrics']['MASE']:.6g})"
                    )
                runs.append(run)
            ok = [r for r in runs if r["status"] == "ok"]
            # Lower-middle when a seed failed and an even number survived; the true median
            # when all five did. If none did, report the first run's failure, which is what
            # a single-seed run would have reported.
            chosen = sorted(ok, key=lambda r: r["mse"])[(len(ok) - 1) // 2] if ok else runs[0]
            snap_frames.extend(chosen["snaps"])
            for metric_name, value in chosen["metrics"].items():
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
                        "fit_seconds": chosen["fit_seconds"],
                        "predict_seconds": chosen["predict_seconds"],
                        "frac_finite_test": chosen["frac_finite_test"],
                        "status": chosen["status"],
                        "n_origins": chosen["n_ok"],
                        # Which run this row describes and how many it was picked from, so
                        # a cell's provenance is in the data rather than inferred from the
                        # method's name. `n_seeds` is 1 for every deterministic method.
                        "seed": chosen["seed"],
                        "n_seeds": len(runs),
                        # What this fit was actually given and what it spent. Training
                        # budgets used to be per-method constants that differed by two
                        # orders of magnitude (issue #95); recording them per cell keeps
                        # any future divergence visible in the data rather than only in
                        # the code. `train_steps` is null for methods that do not train
                        # by gradient descent, and `best_step` -- the step whose weights
                        # were kept -- with it. `train_points` is what the parameters were
                        # estimated on and `val_points` the holdout that found the stopping
                        # step, 0 for a method that does not stop on val.
                        "train_points": chosen["train_points"],
                        "val_points": chosen["val_points"],
                        "train_steps": chosen["train_steps"],
                        "best_step": chosen["best_step"],
                    }
                )
            tag = " [ext]" if used_extended else ""
            picked = f", seed {chosen['seed']} of {len(runs)}" if len(runs) > 1 else ""
            print(
                f"  {method.name:>18s}{tag:>6s}: {chosen['status'][:48]:<48s} "
                f"({chosen['n_ok']} origins, fit {chosen['fit_seconds']:.2f}s, "
                f"predict {chosen['predict_seconds']:.2f}s{picked})"
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

    val_start = len(splits["train"])
    test_start = len(train_val)
    last_origin = len(full) - H - 1
    if last_origin <= test_start:
        raise ValueError(f"Case {case.id}: test span too short for horizon {H}")
    raw_y = _raw_y(full)  # for honest scoring (filled points masked back to NaN)
    origins = _leadtime_origins(test_start, last_origin, raw_y, n_origins)
    rows: list[dict] = []
    for method in methods:
        # accumulate errors per lead
        abs_err = [[] for _ in range(H)]
        sq_err = [[] for _ in range(H)]
        signed = [[] for _ in range(H)]
        n_ok = 0
        # Same protocol as the matrix (see `run_case_rolling`): fit once before the test
        # period, then evaluate every origin by inference. A failed fit makes the method
        # unusable on this case, so it contributes no rows.
        try:
            fit_before_test(
                method,
                case,
                full_fit.iloc[:val_start],
                full_fit.iloc[val_start:test_start],
                full_fit.iloc[:test_start],
            )
        except Exception:
            continue
        for o in origins:
            history = full_fit.iloc[: o + 1]
            future_exog = (
                full_fit.iloc[o + 1 : o + 1 + H][["ds", *exo_present]] if exo_present else None
            )
            try:
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
