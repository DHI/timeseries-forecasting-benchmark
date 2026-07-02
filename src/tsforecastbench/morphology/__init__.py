"""Morphological characterisation of time series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.morphology.features import compute_features_all, period_for_freq

# ---------------------------------------------------------------------------
# Gap-handling thresholds (shared by compute_morphology.py and
# compute_features_all.py so both scripts use identical behaviour).
# ---------------------------------------------------------------------------
# Gaps of at most max(GAP_INTERP_MIN, period * GAP_INTERP_FRAC) consecutive
# NaN steps are filled by linear interpolation. Longer gaps become segment
# boundaries.
GAP_INTERP_FRAC: float = 1 / 12
GAP_INTERP_MIN: int = 3

# Segments shorter than max(MIN_SEGMENT_ABS, period * MIN_SEGMENT_FRAC) are
# dropped from feature aggregation.
MIN_SEGMENT_FRAC: float = 1 / 4
MIN_SEGMENT_ABS: int = 20


def gap_aware_segments(
    case,
    period: int,
    gap_interp_max: int,
    min_segment_len: int,
) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
    """Split a case's training series into gap-handled segments.

    Short gaps (≤ ``gap_interp_max`` consecutive NaN steps) are filled by
    linear interpolation in both ``y`` and any exogenous columns. Longer gaps
    become segment boundaries. Segments shorter than ``min_segment_len`` are
    discarded.

    Returns
    -------
    y_segs : list of 1-D float arrays
    X_segs : parallel list of 2-D float arrays (or None when no exogenous)
    """
    df = case.load()
    train = case.split_frame(df)["train"].copy()

    # Identify short-gap positions in y and interpolate them.
    y_s = train["y"].copy()
    is_nan = y_s.isna().to_numpy()
    padded = np.concatenate(([False], is_nan, [False])).view(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    nan_starts = edges[0::2]
    nan_ends = edges[1::2]  # exclusive

    short_gap_mask = np.zeros(len(y_s), dtype=bool)
    for s, e in zip(nan_starts, nan_ends):
        if (e - s) <= gap_interp_max:
            short_gap_mask[s:e] = True

    if short_gap_mask.any():
        y_s = y_s.copy()
        y_s[short_gap_mask] = np.nan
        y_s = y_s.interpolate(method="linear", limit_direction="both")
        train["y"] = y_s

        if case.exogenous:
            for col in case.exogenous:
                s_col = train[col].copy()
                s_col[short_gap_mask] = np.nan
                s_col = s_col.interpolate(method="linear", limit_direction="both")
                if s_col.isna().any():
                    s_col = s_col.fillna(s_col.mean())
                train[col] = s_col

    # Split at remaining NaN positions into contiguous segments.
    notna = train["y"].notna().to_numpy()
    padded2 = np.concatenate(([False], notna, [False])).view(np.int8)
    edges2 = np.flatnonzero(np.diff(padded2))
    seg_starts = edges2[0::2]
    seg_ends = edges2[1::2]  # exclusive

    y_segs: list[np.ndarray] = []
    X_segs: list[np.ndarray | None] = []

    for s, e in zip(seg_starts, seg_ends):
        if (e - s) < min_segment_len:
            continue
        y_segs.append(train["y"].iloc[s:e].to_numpy(dtype=float))
        if not case.exogenous:
            X_segs.append(None)
        else:
            X_seg = train[case.exogenous].iloc[s:e].to_numpy(dtype=float)
            col_means = np.nanmean(X_seg, axis=0)
            for j in range(X_seg.shape[1]):
                bad = np.isnan(X_seg[:, j])
                if bad.any():
                    X_seg[bad, j] = col_means[j]
            X_segs.append(X_seg)

    return y_segs, X_segs


def aggregate_features(
    feat_dicts: list[dict[str, float]],
    lengths: list[int],
) -> dict[str, float]:
    """Length-weighted average of per-segment feature dicts."""
    total = sum(lengths)
    return {
        key: sum(d[key] * w for d, w in zip(feat_dicts, lengths)) / total
        for key in feat_dicts[0]
    }


__all__ = [
    "compute_features_all",
    "period_for_freq",
    "gap_aware_segments",
    "aggregate_features",
    "GAP_INTERP_FRAC",
    "GAP_INTERP_MIN",
    "MIN_SEGMENT_FRAC",
    "MIN_SEGMENT_ABS",
]
