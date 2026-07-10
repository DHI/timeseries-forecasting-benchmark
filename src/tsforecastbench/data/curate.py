"""Gap policy: turn a raw, gappy series into a benchmark-ready one.

Two decisions, applied uniformly to every case (see RESEARCH_LOG.md):

1. **Per-case gap cap.** A gap is small enough to interpolate only if its observed
   endpoints still carry information about the missing interior. That is exactly what
   autocorrelation measures, so the cap is the lag at which the target's ACF first
   falls below `ACF_THRESHOLD`, bounded above by one dominant seasonal cycle. Flashy
   series decorrelate in a step or two (tiny cap, almost nothing filled); smooth series
   stay correlated for longer (a modest fill is genuinely informed by the data).

2. **Tail-segment trim.** Gaps larger than the cap are hard boundaries — never bridged.
   They split the record into contiguous segments; we keep the last one (the span
   ending at the record's most recent observation) and drop the rest. Anchoring to the
   tail rather than the longest segment keeps the held-out test window at the true end
   of the record, and leaves everything before the segment's start available as extra
   history for methods that can use it (`Forecaster.is_global`). "If it has no target,
   it is nonexistent": we use every observation in that span and invent nothing across
   a real void. A tail segment too short to give the train split 2 full cycles of the
   longest declared seasonal period, or the test split a full forecast horizon, fails
   loudly rather than silently falling back to a different span.

The materialized frame keeps a `y_observed` boolean column: `y` is complete (small gaps
filled) so any method can fit, but scoring counts only genuinely observed points.

`Forecaster.is_global` methods can use more than the shared tail segment: `extend_frame`
applies the same small-gap fill to the full record up to the tail segment's `val_end`,
without the segment trim, for `Case.load_extended` to serve.

This module is the single source of truth for the policy. `scripts/curate_data.py`
applies it once to write `data/curated/`; `Case.load(curated=True)` reads that, falling
back to applying `curate_frame` on the fly when the parquet is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACF_THRESHOLD = 0.8


def _season_ceiling(freq: str) -> int:
    """Typical dominant period in steps for the frequency — the upper bound on the gap
    cap so a fill can never span a whole cycle. Deliberately freq-based, not declared:
    curation produces the very data the per-case `seasonalities` evidence is measured
    on, so it cannot depend on that field (and it never feeds train/predict)."""
    f = freq.lower()
    if f in {"15min", "15t"}:
        return 96
    if f in {"1h", "h"}:
        return 24
    if f in {"1d", "d"}:
        return 365
    if f in {"1w", "w"}:
        return 52
    return 24


def gap_cap(y: pd.Series, freq: str, threshold: float = ACF_THRESHOLD) -> int:
    """Largest gap (in steps) worth interpolating for this series.

    The lag at which the target autocorrelation first drops below `threshold`, clipped
    to [1, one seasonal cycle]. Computed on the observed values (pandas `.autocorr`
    drops NaN pairs, so gaps do not bias it)."""
    ceil = _season_ceiling(freq)
    y = pd.to_numeric(y, errors="coerce")
    for k in range(1, ceil + 1):
        r = y.autocorr(lag=k)
        if not np.isfinite(r) or r < threshold:
            return int(np.clip(k, 1, ceil))
    return ceil


def _nan_runs(mask: np.ndarray):
    """Yield (start, stop) half-open ranges of consecutive True values in `mask`."""
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            yield i, j
            i = j
        else:
            i += 1


def tail_segment_bounds(observed: np.ndarray, cap: int) -> tuple[int, int]:
    """`[start, stop)` of the last span separated by gaps longer than `cap`.

    Gaps of `cap` steps or fewer do not break a segment (they will be filled); longer
    gaps are boundaries. The tail segment is the one ending at the record's most recent
    *observation* — trailing rows with no observed value at all (e.g. padding to a
    common end date) are not a segment, just an absence of record, and are excluded
    before the gap search so they can't be mistaken for the tail span. The returned
    bounds are trimmed to observed points on both ends, so the span begins and ends on
    real data."""
    if not observed.any():
        return 0, 0
    last_obs = int(np.flatnonzero(observed)[-1]) + 1
    big = [(a, b) for a, b in _nan_runs(~observed[:last_obs]) if (b - a) > cap]
    start = big[-1][1] if big else 0
    obs_idx = np.flatnonzero(observed[start:last_obs])
    return start + int(obs_idx[0]), start + int(obs_idx[-1]) + 1


def _horizon_steps(horizon: str, freq: str) -> int:
    delta = pd.Timedelta(horizon)
    step = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    return int(delta / step)


def _fill_small_gaps(s: pd.Series, cap: int) -> pd.Series:
    """Linearly interpolate only NaN runs of length `<= cap`; longer runs are left
    as NaN rather than bridged."""
    na = s.isna().to_numpy()
    fillable = np.zeros(len(s), dtype=bool)
    for a, b in _nan_runs(na):
        if (b - a) <= cap:
            fillable[a:b] = True
    if not fillable.any():
        return s
    filled = s.interpolate(method="linear", limit_direction="both")
    return s.where(~fillable, filled)


def min_tail_length(
    seasonalities: list[int], horizon_primary: str, freq: str, split_fraction: list[float]
) -> int:
    """Shortest tail segment usable for benchmarking.

    The train split alone (fraction `split_fraction[0]` of the segment) must hold 2
    full cycles of the *longest* declared seasonal period — not the most prominent one
    (`seasonalities` is ordered by detection prominence, not length; a long period is
    the more data-hungry requirement regardless of where it ranks). The test split (the
    remaining `1 - split_fraction[1]`) must hold at least one full forecast horizon.
    """
    period = max(seasonalities) if seasonalities else 1
    horizon_steps = _horizon_steps(horizon_primary, freq)
    f0, f1 = split_fraction
    return int(np.ceil(max(2 * period / f0, horizon_steps / (1 - f1))))


def curate_frame(
    df: pd.DataFrame,
    freq: str,
    *,
    seasonalities: list[int],
    horizon_primary: str,
    split_fraction: list[float],
    exog: list[str] | None = None,
    threshold: float = ACF_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """Apply the gap policy to a raw case frame.

    Returns `(curated, info)` where `curated` is the tail contiguous span with small
    gaps filled and a `y_observed` mask added, and `info` records the policy outcome
    (cap, gap counts, span, kept fraction) for the manifest. Raises `ValueError` if the
    tail segment is shorter than `min_tail_length` allows — a case that fails this
    needs a human look, not a silent fallback to a different span.
    """
    df = df.sort_values("ds").reset_index(drop=True)
    y = pd.to_numeric(df["y"], errors="coerce")
    observed = y.notna().to_numpy()
    n_total = len(df)

    cap = gap_cap(y, freq, threshold)
    start, stop = tail_segment_bounds(observed, cap)
    floor = min_tail_length(seasonalities, horizon_primary, freq, split_fraction)
    if stop - start < floor:
        raise ValueError(
            f"tail segment is {stop - start} steps, below the {floor}-step floor "
            f"(2 cycles of the longest declared period in train, >=1 horizon of "
            f"'{horizon_primary}' in test) — needs a human look, not a smaller span."
        )
    seg = df.iloc[start:stop].copy().reset_index(drop=True)
    seg_observed = observed[start:stop].copy()

    # Fill target and exogenous within the span. By construction the span holds no gap
    # longer than `cap` in the *target*; exogenous columns are filled unconditionally
    # (their own gaps aren't cap-bounded the way the target's are).
    fill_cols = ["y", *[c for c in (exog or []) if c in seg.columns]]
    n_filled = int((~seg_observed).sum())
    for c in fill_cols:
        s = pd.to_numeric(seg[c], errors="coerce").interpolate(
            method="linear", limit_direction="both"
        )
        if s.isna().any():  # exogenous column that is all-NaN or gappy at the edges
            s = s.ffill().bfill()
        seg[c] = s

    seg["y_observed"] = seg_observed

    big_runs = [(a, b) for a, b in _nan_runs(~observed) if (b - a) > cap]
    info = {
        "cap": int(cap),
        "acf_threshold": float(threshold),
        "orig_rows": int(n_total),
        "rows": len(seg),
        "kept_frac": float(len(seg) / n_total) if n_total else 0.0,
        "n_observed": int(seg_observed.sum()),
        "n_filled": n_filled,
        "n_big_gaps": len(big_runs),
        "min_tail_length": floor,
        "span_start": str(seg["ds"].iloc[0]) if len(seg) else None,
        "span_end": str(seg["ds"].iloc[-1]) if len(seg) else None,
    }
    return seg, info


def extend_frame(
    df: pd.DataFrame,
    freq: str,
    cutoff: pd.Timestamp,
    exog: list[str] | None = None,
    threshold: float = ACF_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """Gap-fill the full raw record up to `cutoff`, for `Forecaster.is_global` methods
    that can use more history than the shared tail segment.

    Unlike `curate_frame`, this does not trim to one contiguous segment: every gap
    `<= cap` is filled throughout the kept range, but gaps larger than `cap` are left
    unfilled (and marked unobserved) rather than used as a trim boundary — a
    global-capable method sees the record's true shape, gaps and all, not a
    stitched-together approximation of one. `cutoff` should be the tail segment's
    `val_end`, so this frame never reaches past what the corresponding local matrix
    holds out for testing.
    """
    df = df.sort_values("ds").reset_index(drop=True)
    ds_cmp = pd.to_datetime(df["ds"])
    if ds_cmp.dt.tz is not None:
        ds_cmp = ds_cmp.dt.tz_convert("UTC").dt.tz_localize(None)
    df = df[ds_cmp <= cutoff].reset_index(drop=True)

    y = pd.to_numeric(df["y"], errors="coerce")
    observed = y.notna().to_numpy()
    if not observed.any():
        raise ValueError(f"no observed y values at or before cutoff {cutoff}")
    first_obs = int(np.flatnonzero(observed)[0])
    df = df.iloc[first_obs:].copy().reset_index(drop=True)
    observed = observed[first_obs:]
    n_total = len(df)

    cap = gap_cap(pd.to_numeric(df["y"], errors="coerce"), freq, threshold)

    fill_cols = ["y", *[c for c in (exog or []) if c in df.columns]]
    n_filled = 0
    for c in fill_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        filled = _fill_small_gaps(s, cap)
        if c == "y":
            n_filled = int((s.isna() & filled.notna()).sum())
        if filled.isna().any():  # exogenous column that is all-NaN or gappy at the edges
            filled = filled.ffill().bfill()
        df[c] = filled

    df["y_observed"] = observed

    big_runs = [(a, b) for a, b in _nan_runs(~observed) if (b - a) > cap]
    info = {
        "cap": int(cap),
        "acf_threshold": float(threshold),
        "orig_rows": int(n_total),
        "rows": len(df),
        "n_observed": int(observed.sum()),
        "n_filled": n_filled,
        "n_big_gaps": len(big_runs),
        "cutoff": str(cutoff),
        "span_start": str(df["ds"].iloc[0]) if len(df) else None,
        "span_end": str(df["ds"].iloc[-1]) if len(df) else None,
    }
    return df, info
