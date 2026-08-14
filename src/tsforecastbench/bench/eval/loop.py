"""Evaluation loop - single hold-out (rolling-origin in Phase B).

Every method is fit and scored on the *curated* series (`Case.load(curated=True)`): the
gap policy in `tsforecastbench.data.curate` has already filled small gaps and trimmed
each case to its tail contiguous span, so the target is complete. Scoring still counts
only genuine observations — the curated frame's `y_observed` mask marks which points
were measured vs. filled, and metrics use only the measured ones. The fraction of
finite test points is reported so a gappy test window is visible, not silently scored.

Methods that are `is_global` and `pools_segments` are additionally scored on the full
single-series history (`Case.load_extended`) reaching back before the tail segment. That
history is split at its big gaps into contiguous segments, each carrying its own
`unique_id`, so a windowed method trains across all of them without ever building a
sample that straddles a gap (see `_segment_extended` and `methods.base.segment_arrays`).
The tail segment equals the shared curated span, so the local and extended fits are
scored against identical reference scales and the same held-out window — only the
training data the model sees differs. Each method thus emits a local result row
(`used_extended_history=False`) and, when the case has history beyond the curated span,
a second extended row (`=True`); a gap-free record has no extra history and produces the
local row only.

Each row also carries `fit_seconds`/`predict_seconds`, timed separately (this path has
no rolling origins, so each is a single measurement, not an average — see
`eval.rolling` for the rolling-origin case).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from tsforecastbench.bench.eval.config import DEFAULT_SEEDS
from tsforecastbench.bench.eval.metrics import kge, mae, mape, mase, nse, rmse, skill_score, smape
from tsforecastbench.bench.methods.base import Forecaster, lookback_for
from tsforecastbench.data.schema import Case, _observed_mask
from tsforecastbench.data.seasonality import mase_season_for


def _horizon_to_steps(horizon: str, freq: str) -> int:
    delta = pd.Timedelta(horizon)
    step = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    return int(delta / step)


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """`[start, stop)` index ranges of each maximal run of True in `mask`."""
    runs: list[tuple[int, int]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i + 1
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _segment_extended(
    extended: pd.DataFrame, case_id: str, freq: str, horizon: int
) -> list[pd.DataFrame]:
    """Split the extended-history frame into contiguous segments at its big gaps.

    The extended frame is a regular-frequency grid whose gaps larger than the per-case
    fill cap are explicit NaN-`y` rows (small gaps are already filled). Each maximal run
    of non-NaN rows becomes one segment, relabeled with its own `unique_id`
    (`{case_id}__s{k}`) so a windowed method never builds a training sample across a gap.
    Segments too short to yield even one lookback+horizon window are dropped; the tail
    segment (ending at `val_end`) always survives, being the curated training span.

    Returns segments in time order (tail last). A gap-free record yields a single
    segment, i.e. no history beyond the curated span.
    """
    observed = pd.to_numeric(extended["y"], errors="coerce").notna().to_numpy()
    floor = lookback_for(freq, horizon) + horizon
    segments: list[pd.DataFrame] = []
    for start, stop in _contiguous_runs(observed):
        if stop - start < floor:
            continue
        seg = extended.iloc[start:stop].copy()
        seg["unique_id"] = f"{case_id}__s{len(segments)}"
        segments.append(seg)
    return segments


def _naive_ds(df: pd.DataFrame) -> pd.Series:
    """`ds` as tz-naive UTC, matching what `Case.split_bounds` returns."""
    ds = pd.to_datetime(df["ds"])
    if ds.dt.tz is not None:
        ds = ds.dt.tz_convert("UTC").dt.tz_localize(None)
    return ds


def _split_segments_at_train_end(
    segments: list[pd.DataFrame], train_end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a segmented extended-history panel into its fit frame and its val frame.

    The declared val period lies at the end of the tail segment, since `load_extended`
    cuts the record at `val_end`. Splitting every segment by `train_end` rather than only
    the tail one costs nothing and stays correct if a gap ever falls inside the val period,
    which would put val rows at the end of an earlier segment too.
    """
    fit_parts, val_parts = [], []
    for seg in segments:
        ds = _naive_ds(seg)
        fit_parts.append(seg[ds <= train_end])
        val_parts.append(seg[ds > train_end])
    return (
        pd.concat(fit_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
    )


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


def run_case(
    case: Case,
    methods: list[Forecaster],
    prefer_synth: bool = False,
    n_seeds: int = DEFAULT_SEEDS,
) -> pd.DataFrame:
    """Single-holdout scores for the cases x methods matrix.

    An `is_stochastic` method is run `n_seeds` times and the median-MSE run is kept whole
    — all nine of its metrics and its timings — so the cell stays one real fit rather than
    a composite (issue #106). See `eval.rolling.run_case_rolling`, which does the same for
    the rolling-origin path and carries the full reasoning.
    """
    df = case.load(prefer_synth=prefer_synth, curated=True)
    splits = case.split_frame(df)
    train_end, _val_end, _test_end = case.split_bounds(df)
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
    val_start = len(splits["train"])
    test_start = len(train_val)
    last_origin = len(full) - horizon_steps
    origin = _pick_origin(raw_y, test_start, last_origin, horizon_steps)

    # Three frames, not one. Parameters are estimated on the train split alone; the val
    # split is handed over separately, for the iteratively trained methods to stop on and
    # for no other purpose; and `history` -- everything before the origin, val and the
    # slid-in test prefix included -- is what predict conditions on (gh #99).
    fit_frame = full_fit.iloc[:val_start].reset_index(drop=True)
    val_frame = full_fit.iloc[val_start:test_start].reset_index(drop=True)
    history = full_fit.iloc[:origin].reset_index(drop=True)
    # Rows the origin-picking slide pulled from the test split into the history -- carried
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

    def _train_stats(history_df: pd.DataFrame) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
        """Reference scales for MASE and the skill scores.

        Read off the pre-origin history, not the (shorter) fit frame: these are the
        yardstick every method is measured against, so reserving the val split from
        training must not move them.
        """
        y_train = history_df["y"].to_numpy(dtype=float)
        # Declared dominant period the train series can support (>= 2 cycles), else the
        # next declared period, else 1 — a real season or honest persistence scaling,
        # never an arbitrary truncated lag.
        season = mase_season_for(case.seasonalities, len(y_train))
        ref_persistence = np.full(horizon_steps, float(history_df["y"].iloc[-1]))
        ref_mean = np.full(horizon_steps, float(np.nanmean(y_train)))
        return y_train, season, ref_persistence, ref_mean

    y_train, season, ref_persistence, ref_mean = _train_stats(history)

    # Extended-history training for `is_global` methods that pool segments. The extended
    # frame is split at its big gaps into contiguous segments (see `_segment_extended`);
    # the tail segment equals the local train span, so the metric reference scales stay
    # identical and only the training data the model sees grows. A gap-free record yields
    # a single segment (no history beyond the curated span), so only the local fit runs.
    fit_frame_ext = val_frame_ext = history_ext = None
    if any(m.is_global and m.pools_segments for m in methods):
        try:
            extended = case.load_extended(prefer_synth=prefer_synth)
        except Exception as exc:
            print(
                f"  [note] load_extended failed ({type(exc).__name__}: {exc}); "
                "is_global methods use the curated span only."
            )
        else:
            segments = _segment_extended(extended, case.id, case.freq, horizon_steps)
            if len(segments) >= 2:
                slide = slide_in.copy()
                slide["unique_id"] = segments[-1]["unique_id"].iloc[0]
                fit_frame_ext, val_frame_ext = _split_segments_at_train_end(segments, train_end)
                history_ext = pd.concat([*segments, slide], ignore_index=True)
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

    def _run_once(method: Forecaster, m_fit, m_val, m_history, seed) -> dict:
        """Fit and score one (method, variant) at one seed, returning the whole run.

        Returns a bundle rather than appending to `records`, so several seeds can be run
        and only the one that is kept gets written out.
        """
        fit_seconds = float("nan")
        predict_seconds = float("nan")
        # Ranks the seeds of a stochastic method. Every run is scored on the same holdout
        # points, so only the predictions differ. Infinite for a run that failed, which
        # keeps it out of the median.
        mse = float("inf")
        method.last_fit_steps = None
        method.seed = seed
        try:
            t_fit0 = time.perf_counter()
            method.fit(case, m_fit, m_val)
            fit_seconds = time.perf_counter() - t_fit0
            t_pred0 = time.perf_counter()
            preds = method.predict(case, m_history, horizon_steps, future_exog=future_exog)
            predict_seconds = time.perf_counter() - t_pred0
            y_hat = preds["y_hat"].to_numpy(dtype=float)
            assert len(y_hat) == horizon_steps, (
                f"{method.name} returned {len(y_hat)} predictions, expected {horizon_steps}"
            )
            if finite.sum() == 0:
                raise ValueError("no finite test points to score")
            yt, yh = y_true_raw[finite], y_hat[finite]
            mse = float(np.mean((yt - yh) ** 2))
            metrics = {
                "MAE": mae(yt, yh),
                "RMSE": rmse(yt, yh),
                "MASE": mase(yt, yh, y_train, season=season),
                "Skill_vs_Persistence": skill_score(yt, yh, ref_persistence[finite]),
                "Skill_vs_Mean": skill_score(yt, yh, ref_mean[finite]),
                "MAPE": mape(yt, yh),
                "sMAPE": smape(yt, yh),
                "NSE": nse(yt, yh),
                "KGE": kge(yt, yh),
            }
            status = "ok"
        except Exception as exc:
            metrics = {k: float("nan") for k in metric_names}
            status = f"error: {type(exc).__name__}: {str(exc)[:70]}"
        return {
            "seed": seed,
            "status": status,
            "mse": mse,
            "metrics": metrics,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            # Read here rather than by the caller: the next seed's fit overwrites it.
            "train_steps": method.last_fit_steps,
        }

    for method in methods:
        # Every method is scored on the local span. Methods that pool segments are scored
        # a second time on the extended history, when the case has any. Both fits share
        # the local reference scales (persistence/mean/seasonal-naive), so the extended
        # row differs from the local one only through the model, not the yardstick.
        variants = [(fit_frame, val_frame, history, False)]
        if method.is_global and method.pools_segments and fit_frame_ext is not None:
            variants.append((fit_frame_ext, val_frame_ext, history_ext, True))
        for m_fit, m_val, m_history, used_extended in variants:
            # Several draws for a method whose forecast moves with the seed, keeping the
            # middle one whole; one run for everyone else. See `eval.rolling` on why the
            # run is kept whole rather than averaged or medianed metric by metric.
            cell_seeds = list(range(n_seeds)) if method.is_stochastic else [0]
            runs = [
                _run_once(method, m_fit, m_val, m_history, seed) for seed in cell_seeds
            ]
            ok = [r for r in runs if r["status"] == "ok"]
            # Lower-middle when a seed failed and an even number survived; if none did,
            # report the first run's failure, as a single-seed run would have.
            chosen = sorted(ok, key=lambda r: r["mse"])[(len(ok) - 1) // 2] if ok else runs[0]
            fit_seconds = chosen["fit_seconds"]
            predict_seconds = chosen["predict_seconds"]
            status = chosen["status"]
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
                        "fit_seconds": fit_seconds,
                        "predict_seconds": predict_seconds,
                        "frac_finite_test": frac_finite,
                        "status": status,
                        # Which run this row describes and how many it was picked from,
                        # so a cell's provenance is in the data. `n_seeds` is 1 for every
                        # deterministic method.
                        "seed": chosen["seed"],
                        "n_seeds": len(runs),
                        # What this fit was given and what it spent -- see the same set
                        # in `eval.rolling` and `eval.budget` on why they are recorded.
                        # `val_points` is the holdout handed over for early stopping; a
                        # method that does not train iteratively ignores it.
                        "train_points": len(m_fit),
                        "val_points": len(m_val),
                        "train_steps": chosen["train_steps"],
                    }
                )
            tag = " [ext]" if used_extended else ""
            picked = f", seed {chosen['seed']} of {len(runs)}" if len(runs) > 1 else ""
            print(
                f"  {method.name:>18s}{tag:>6s}: {status[:54]:<54s}  "
                f"(fit {fit_seconds:.2f}s, predict {predict_seconds:.2f}s{picked})"
            )

    if frac_finite < 1.0:
        print(
            f"  [note] test window {frac_finite:.0%} observed; "
            f"{int((1 - frac_finite) * horizon_steps)} of {horizon_steps} points were gaps (excluded)."
        )
    return pd.DataFrame(records)
