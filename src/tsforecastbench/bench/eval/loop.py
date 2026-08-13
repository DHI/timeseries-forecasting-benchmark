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

    # Extended-history training for `is_global` methods that pool segments. The extended
    # frame is split at its big gaps into contiguous segments (see `_segment_extended`);
    # the tail segment equals the local train span, so the metric reference scales stay
    # identical and only the training data the model sees grows. A gap-free record yields
    # a single segment (no history beyond the curated span), so only the local fit runs.
    train_fit_ext = None
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
                train_fit_ext = pd.concat([*segments, slide], ignore_index=True)
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
    for method in methods:
        # Every method is scored on the local span. Methods that pool segments are scored
        # a second time on the extended history, when the case has any. Both fits share
        # the local reference scales (persistence/mean/seasonal-naive), so the extended
        # row differs from the local one only through the model, not the yardstick.
        variants = [(train_fit, False)]
        if method.is_global and method.pools_segments and train_fit_ext is not None:
            variants.append((train_fit_ext, True))
        for m_train_fit, used_extended in variants:
            fit_seconds = float("nan")
            predict_seconds = float("nan")
            method.last_fit_steps = None
            try:
                t_fit0 = time.perf_counter()
                method.fit(case, m_train_fit)
                fit_seconds = time.perf_counter() - t_fit0
                t_pred0 = time.perf_counter()
                preds = method.predict(case, m_train_fit, horizon_steps, future_exog=future_exog)
                predict_seconds = time.perf_counter() - t_pred0
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
                    "MAPE": mape(yt, yh),
                    "sMAPE": smape(yt, yh),
                    "NSE": nse(yt, yh),
                    "KGE": kge(yt, yh),
                }
                status = "ok"
            except Exception as exc:
                metrics = {k: float("nan") for k in metric_names}
                status = f"error: {type(exc).__name__}: {str(exc)[:70]}"
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
                        "fit_seconds": fit_seconds,
                        "predict_seconds": predict_seconds,
                        "frac_finite_test": frac_finite,
                        "status": status,
                        # What this fit was given and what it spent -- see the same pair
                        # in `eval.rolling` and `eval.budget` on why they are recorded.
                        "train_points": len(m_train_fit),
                        "train_steps": method.last_fit_steps,
                    }
                )
            tag = " [ext]" if used_extended else ""
            print(
                f"  {method.name:>18s}{tag:>6s}: {status[:54]:<54s}  "
                f"(fit {fit_seconds:.2f}s, predict {predict_seconds:.2f}s)"
            )

    if frac_finite < 1.0:
        print(
            f"  [note] test window {frac_finite:.0%} observed; "
            f"{int((1 - frac_finite) * horizon_steps)} of {horizon_steps} points were gaps (excluded)."
        )
    return pd.DataFrame(records)
