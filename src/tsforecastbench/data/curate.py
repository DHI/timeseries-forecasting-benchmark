"""Gap policy: turn a raw, gappy series into a benchmark-ready one.

Two decisions, applied uniformly to every case (see research_log/details.md, id 15):

1. **Per-case gap cap.** A gap is small enough to interpolate only if its observed
   endpoints still carry information about the missing interior. That is exactly what
   autocorrelation measures, so the cap is the lag at which the target's ACF first
   falls below `ACF_THRESHOLD`, bounded above by one dominant seasonal cycle. Flashy
   series decorrelate in a step or two (tiny cap, almost nothing filled); smooth series
   stay correlated for longer (a modest fill is genuinely informed by the data). The
   ACF is read on the pre-test record only (up to `val_end`), so the test period does
   not shape the data it is scored on (gh #119).

2. **Segment trim.** Gaps larger than the cap are hard boundaries — never bridged.
   They split the record into contiguous segments, and one is kept. The last segment
   is kept when it is long enough, so the test window is the end of the record and
   everything before the segment's `val_end` stays available as extra history for
   methods that can use it (`Forecaster.pools_segments`). Otherwise the longest segment is
   kept (the latest on a tie), and data after it goes unused. A segment is long enough
   when its train split spans `MIN_TRAIN_CYCLES` cycles of the longest declared
   seasonal period and its test split spans 2 cycles and one forecast horizon (gh
   #121). When neither segment is, curation fails and the case needs a human look.
   "If it has no target, it is nonexistent": we use every observation in that span and
   invent nothing across a real void.

The materialized frame keeps a `y_observed` boolean column: `y` is complete (small gaps
filled) so any method can fit, but scoring counts only genuinely observed points.

`Forecaster.pools_segments` methods can use more than the shared segment: `extend_frame`
applies the same small-gap fill to the full record up to the segment's `val_end`,
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
    gaps are boundaries. Trailing rows with no observed value (e.g. padding to a common
    end date) are not part of the segment. The span begins and ends on observed
    points."""
    if not observed.any():
        return 0, 0
    last_obs = int(np.flatnonzero(observed)[-1]) + 1
    big = [(a, b) for a, b in _nan_runs(~observed[:last_obs]) if (b - a) > cap]
    start = big[-1][1] if big else 0
    obs_idx = np.flatnonzero(observed[start:last_obs])
    return start + int(obs_idx[0]), start + int(obs_idx[-1]) + 1


def longest_segment_bounds(observed: np.ndarray, cap: int) -> tuple[int, int]:
    """`[start, stop)` of the longest span separated by gaps longer than `cap`, the
    latest one on a tie.

    Gaps of `cap` steps or fewer do not break a segment (they will be filled); longer
    gaps are boundaries. Each segment is trimmed to observed points on both ends, so the
    span begins and ends on real data, and leading or trailing rows with no observed
    value are not part of any segment."""
    if not observed.any():
        return 0, 0
    big = [(a, b) for a, b in _nan_runs(~observed) if (b - a) > cap]
    edges = [0, *[x for ab in big for x in ab], len(observed)]
    best = (0, 0)
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        obs_idx = np.flatnonzero(observed[a:b])
        if len(obs_idx) == 0:
            continue
        seg = (a + int(obs_idx[0]), a + int(obs_idx[-1]) + 1)
        if seg[1] - seg[0] >= best[1] - best[0]:
            best = seg
    return best


def fraction_split_dates(
    observed_ds: pd.Series, split_fraction: list[float]
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """`(train_end, val_end)` as quantiles `split_fraction` of the observed timestamps,
    tz-naive UTC."""
    ds = pd.to_datetime(observed_ds)
    if ds.dt.tz is not None:
        ds = ds.dt.tz_convert("UTC").dt.tz_localize(None)
    if ds.empty:
        raise ValueError("no observed y values to split on")
    ns = ds.astype("int64").to_numpy()
    f0, f1 = split_fraction
    return pd.Timestamp(int(np.quantile(ns, f0))), pd.Timestamp(int(np.quantile(ns, f1)))


VAL_FRACTION = 0.15
MIN_TEST_FRACTION = 0.15


def split_fraction_for(observed_ds: pd.Series, longest_period: int, freq: str) -> list[float]:
    """`split_fraction` for a curated segment with observations at `observed_ds` (gh #121).

    The test split is the larger of `MIN_TEST_FRACTION` of the observations and 2 cycles
    of the longest declared seasonal period, measured in time from the last observation;
    the val split is `VAL_FRACTION`; train gets the rest. Splits are quantiles of the
    observed timestamps (`fraction_split_dates`), so the 2-cycle test start is converted
    to the quantile of the last observation before it, `(k - 1) / (n - 1)` for `k` of
    `n` observations before it. Fractions are rounded down to 3 decimals, which can only
    lengthen the test split. A case's `case.toml` holds the result, computed once on its
    curated data.
    """
    ds = pd.to_datetime(observed_ds)
    if ds.dt.tz is not None:
        ds = ds.dt.tz_convert("UTC").dt.tz_localize(None)
    step = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    test_start = ds.max() - 2 * longest_period * step
    n_before = int((ds < test_start).sum())
    val_end = min(1 - MIN_TEST_FRACTION, (n_before - 1) / (len(ds) - 1))
    val_end = np.floor(val_end * 1000) / 1000
    train_end = np.floor((val_end - VAL_FRACTION) * 1000) / 1000
    return [float(train_end), float(val_end)]


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


MIN_TRAIN_CYCLES = 4


def _segment_shortfall(
    observed_ds: pd.Series, longest_period: int, horizon_primary: str, freq: str
) -> str | None:
    """Why a segment with observations at `observed_ds` is too short, or None.

    Its split comes from `split_fraction_for`. The train split must span
    `MIN_TRAIN_CYCLES` cycles of `longest_period`; the test split must span 2 cycles
    and one `horizon_primary`."""
    step = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    cycle = longest_period * step
    need_test = max(2 * cycle, pd.Timedelta(horizon_primary))
    need_train = MIN_TRAIN_CYCLES * cycle
    if observed_ds.empty or observed_ds.max() - observed_ds.min() < need_train + need_test:
        span = observed_ds.max() - observed_ds.min() if len(observed_ds) else pd.Timedelta(0)
        return f"spans {span}, below {need_train} of train plus {need_test} of test"
    train_end, val_end = fraction_split_dates(
        observed_ds, split_fraction_for(observed_ds, longest_period, freq)
    )
    if train_end - observed_ds.min() < need_train:
        return f"train spans {train_end - observed_ds.min()}, below {need_train}"
    if observed_ds.max() - val_end < need_test:
        return f"test spans {observed_ds.max() - val_end}, below {need_test}"
    return None


def _pick_segment(
    observed: np.ndarray,
    ds: pd.Series,
    cap: int,
    longest_period: int,
    horizon_primary: str,
    freq: str,
) -> tuple[int, int, str, list[str]]:
    """`(start, stop, kind, shortfalls)` of the segment to keep: the last one if it is
    long enough, else the longest. When neither is, `kind` is "longest" and
    `shortfalls` says why each failed."""
    shortfalls = []
    for kind, bounds in (("last", tail_segment_bounds), ("longest", longest_segment_bounds)):
        start, stop = bounds(observed, cap)
        why = _segment_shortfall(
            ds.iloc[start:stop][observed[start:stop]], longest_period, horizon_primary, freq
        )
        if why is None:
            return start, stop, kind, []
        shortfalls.append(f"{kind} segment {why}")
    return start, stop, "longest", shortfalls


def mask_sensor_faults(
    df: pd.DataFrame,
    *,
    stuck_steps: int,
    valid_min: float | None,
    valid_max: float | None,
) -> tuple[pd.DataFrame, dict]:
    """Set target readings a working sensor cannot produce to NaN (gh #151).

    Two kinds are masked, so the gap policy treats them as missing like any other gap:

    - **Stuck:** a run of identical consecutive readings longer than `stuck_steps`. The
      caller passes the case's dominant seasonal period: within a full cycle a working
      sensor has to move, so an exactly flat run longer than that is a held or frozen
      value.
    - **Out of range:** a reading at or below `valid_min` or at or above `valid_max`, the
      physical bounds a case declares (None = no bound on that side).

    Returns the masked copy and the count of each kind.
    """
    df = df.sort_values("ds").reset_index(drop=True)
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    observed = ~np.isnan(y)

    stuck = np.zeros(len(y), dtype=bool)
    idx = np.flatnonzero(observed)
    if len(idx):
        v = y[idx]
        new_run = np.r_[True, (v[1:] != v[:-1]) | (np.diff(idx) != 1)]
        starts = np.flatnonzero(new_run)
        lengths = np.diff(np.r_[starts, len(v)])
        for s, n in zip(starts, lengths, strict=True):
            if n > stuck_steps:
                stuck[idx[s : s + n]] = True

    out_of_range = np.zeros(len(y), dtype=bool)
    if valid_min is not None:
        out_of_range |= observed & (y <= valid_min)
    if valid_max is not None:
        out_of_range |= observed & (y >= valid_max)

    df["y"] = np.where(stuck | out_of_range, np.nan, y)
    return df, {
        "n_stuck": int(stuck.sum()),
        "n_out_of_range": int((out_of_range & ~stuck).sum()),
    }


def curate_frame(
    df: pd.DataFrame,
    freq: str,
    *,
    seasonalities: list[int],
    horizon_primary: str,
    exog: list[str] | None = None,
    threshold: float = ACF_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """Apply the gap policy to a raw case frame.

    Returns `(curated, info)` where `curated` is the kept segment (the last one if long
    enough, else the longest) with small gaps filled and a `y_observed` mask added, and
    `info` records the policy outcome (cap, segment kind, split, gap counts, span, kept
    fraction) for the manifest. `info["split_fraction"]` is what the case's `case.toml`
    must declare. Raises `ValueError` when neither segment is long enough: the case
    needs a human look.
    """
    df = df.sort_values("ds").reset_index(drop=True)
    y = pd.to_numeric(df["y"], errors="coerce")
    observed = y.notna().to_numpy()
    n_total = len(df)

    ds_naive = pd.to_datetime(df["ds"])
    if ds_naive.dt.tz is not None:
        ds_naive = ds_naive.dt.tz_convert("UTC").dt.tz_localize(None)
    longest_period = max(seasonalities) if seasonalities else 1

    # The cap sets the segment, the segment sets `val_end`, and the cap is read on
    # the record up to `val_end`. Start from the largest cap the frequency allows, which
    # reads no data, and repeat until the cap stops changing. Starting from the full
    # record's cap would let test values pick where the loop settles. The segment length is
    # checked on the settled segment only.
    cap = _season_ceiling(freq)
    for _ in range(10):
        start, stop, kind, shortfalls = _pick_segment(
            observed, ds_naive, cap, longest_period, horizon_primary, freq
        )
        seg_obs_ds = ds_naive.iloc[start:stop][observed[start:stop]]
        split_fraction = split_fraction_for(seg_obs_ds, longest_period, freq)
        if shortfalls:
            break  # flagged below; the split of a too-short segment can be meaningless
        _, val_end = fraction_split_dates(seg_obs_ds, split_fraction)
        pre_test_cap = gap_cap(y[ds_naive <= val_end], freq, threshold)
        if pre_test_cap == cap:
            break
        cap = pre_test_cap
    else:
        raise ValueError("gap cap did not settle on the pre-test record after 10 passes")
    if shortfalls:
        raise ValueError(
            f"no segment is long enough at cap {cap} ({'; '.join(shortfalls)}): needs a human look."
        )
    seg = df.iloc[start:stop].copy().reset_index(drop=True)
    seg_observed = observed[start:stop].copy()

    # Fill target and exogenous gaps of up to `cap` steps within the span. By
    # construction the span holds no longer gap in the target; a longer gap in an
    # exogenous column is a data problem for the case's download step to fix (gh #119).
    fill_cols = ["y", *[c for c in (exog or []) if c in seg.columns]]
    n_filled = int((~seg_observed).sum())
    for c in fill_cols:
        filled = _fill_small_gaps(pd.to_numeric(seg[c], errors="coerce"), cap)
        if filled.isna().any():
            runs = [b - a for a, b in _nan_runs(filled.isna().to_numpy())]
            raise ValueError(
                f"exogenous column {c!r} has {len(runs)} gap(s) longer than the "
                f"{cap}-step cap inside the curated segment (longest {max(runs)} steps)"
            )
        seg[c] = filled

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
        "segment": kind,
        "split_fraction": split_fraction,
        "val_end": str(val_end),
        "span_start": str(seg["ds"].iloc[0]) if len(seg) else None,
        "span_end": str(seg["ds"].iloc[-1]) if len(seg) else None,
    }
    return seg, info


def extend_frame(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    cap: int,
    exog: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Gap-fill the full raw record up to `cutoff`, for `Forecaster.pools_segments`
    methods that can use more history than the shared curated segment.

    Unlike `curate_frame`, this does not trim to one contiguous segment: every gap
    `<= cap` is filled throughout the kept range, but gaps larger than `cap` are left
    unfilled (and marked unobserved) rather than used as a trim boundary — a
    segment-pooling method sees the record's true shape, gaps and all, not a
    stitched-together approximation of one. `cutoff` should be the curated segment's
    `val_end`, so this frame never reaches past what the corresponding local matrix
    holds out for testing, and `cap` the gap cap `curate_frame` reported, so both views
    fill the same gaps.
    """
    df = df.sort_values("ds").reset_index(drop=True)

    ds_cmp = pd.to_datetime(df["ds"])
    if ds_cmp.dt.tz is not None:
        ds_cmp = ds_cmp.dt.tz_convert("UTC").dt.tz_localize(None)
    df = df[ds_cmp <= cutoff].reset_index(drop=True)

    n_total = len(df)
    y = pd.to_numeric(df["y"], errors="coerce")
    observed = y.notna().to_numpy()
    if not observed.any():
        raise ValueError(f"no observed y values at or before cutoff {cutoff}")
    first_obs = int(np.flatnonzero(observed)[0])
    df = df.iloc[first_obs:].copy().reset_index(drop=True)
    observed = observed[first_obs:]

    fill_cols = ["y", *[c for c in (exog or []) if c in df.columns]]
    n_filled = 0
    for c in fill_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        filled = _fill_small_gaps(s, cap)
        if c == "y":
            n_filled = int((s.isna() & filled.notna()).sum())
        df[c] = filled

    df["y_observed"] = observed

    big_runs = [(a, b) for a, b in _nan_runs(~observed) if (b - a) > cap]
    info = {
        "cap": int(cap),
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
