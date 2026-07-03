"""Gap policy: turn a raw, gappy series into a benchmark-ready one.

Two decisions, applied uniformly to every case (see RESEARCH_LOG.md):

1. **Per-case gap cap.** A gap is small enough to interpolate only if its observed
   endpoints still carry information about the missing interior. That is exactly what
   autocorrelation measures, so the cap is the lag at which the target's ACF first
   falls below `ACF_THRESHOLD`, bounded above by one dominant seasonal cycle. Flashy
   series decorrelate in a step or two (tiny cap, almost nothing filled); smooth series
   stay correlated for longer (a modest fill is genuinely informed by the data).

2. **Longest-segment trim.** Gaps larger than the cap are hard boundaries — never
   bridged. They split the record into contiguous segments; we keep the single longest
   one and drop the rest. "If it has no target, it is nonexistent": we use every
   observation in the longest usable span and invent nothing across a real void.

The materialized frame keeps a `y_observed` boolean column: `y` is complete (small gaps
filled) so any method can fit, but scoring counts only genuinely observed points.

This module is the single source of truth for the policy. `scripts/curate_data.py`
applies it once to write `data/curated/`; `Case.load(curated=True)` reads that, falling
back to applying `curate_frame` on the fly when the parquet is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACF_THRESHOLD = 0.8


def _season_ceiling(freq: str) -> int:
    """Dominant seasonal period in steps — the upper bound on the gap cap so a fill can
    never span a whole cycle. Mirrors bench.methods.naive._default_season_length."""
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


def longest_segment_bounds(observed: np.ndarray, cap: int) -> tuple[int, int]:
    """`[start, stop)` of the longest span separated by gaps longer than `cap`.

    Gaps of `cap` steps or fewer do not break a segment (they will be filled); longer
    gaps are boundaries. The returned bounds are trimmed to observed points on both
    ends, so the span begins and ends on real data."""
    n = len(observed)
    if n == 0:
        return 0, 0
    na = ~observed
    big = [(a, b) for a, b in _nan_runs(na) if (b - a) > cap]
    segments: list[tuple[int, int]] = []
    prev = 0
    for a, b in big:
        segments.append((prev, a))
        prev = b
    segments.append((prev, n))

    best = max(segments, key=lambda s: int(observed[s[0] : s[1]].sum()))
    start, stop = best
    obs_idx = np.flatnonzero(observed[start:stop])
    if len(obs_idx) == 0:
        return start, stop
    return start + int(obs_idx[0]), start + int(obs_idx[-1]) + 1


def curate_frame(
    df: pd.DataFrame, freq: str, exog: list[str] | None = None, threshold: float = ACF_THRESHOLD
) -> tuple[pd.DataFrame, dict]:
    """Apply the gap policy to a raw case frame.

    Returns `(curated, info)` where `curated` is the longest contiguous span with small
    gaps filled and a `y_observed` mask added, and `info` records the policy outcome
    (cap, gap counts, span, kept fraction) for the manifest.
    """
    df = df.sort_values("ds").reset_index(drop=True)
    y = pd.to_numeric(df["y"], errors="coerce")
    observed = y.notna().to_numpy()
    n_total = len(df)

    cap = gap_cap(y, freq, threshold)
    start, stop = longest_segment_bounds(observed, cap)
    seg = df.iloc[start:stop].copy().reset_index(drop=True)
    seg_observed = observed[start:stop].copy()

    # Fill target and exogenous within the span. By construction the span holds no gap
    # longer than `cap`, so linear interpolation fully completes the target.
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
        "span_start": str(seg["ds"].iloc[0]) if len(seg) else None,
        "span_end": str(seg["ds"].iloc[-1]) if len(seg) else None,
    }
    return seg, info
