"""Compute the 4-feature morphological vector for all benchmark cases and
produce a PCA biplot.

Outputs
-------
results/morphology_features.csv        raw feature table (case × 4 features)
docs/figures/morphology/pca_biplot.png PCA biplot — 2-component PCA (PC1/PC2)

Usage
-----
    uv run python scripts/compute_morphology.py
    uv run python scripts/compute_morphology.py --no-plot
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.data.registry import discover_cases
from tsforecastbench.morphology.features import (
    compute_features_all,
    exog_r2_gain,
    period_for_freq,
)

RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = REPO_ROOT / "docs" / "figures" / "morphology"

FEATURE_COLS = [
    "seasonal_strength",
    "hurst",
    "amplitude_dispersion",
    "l_skewness",
]

# Friendly display names for the loading arrows in the biplot. Keys must match
# FEATURE_COLS.
FEATURE_LABELS = {
    "seasonal_strength": "seasonal",
    "hurst": "Hurst",
    "amplitude_dispersion": "amplitude\ndispersion",
    "l_skewness": "L-skewness",
}

# ---------------------------------------------------------------------------
# Gap-handling thresholds
# ---------------------------------------------------------------------------
# Gaps of at most (max(GAP_INTERP_MIN, period * GAP_INTERP_FRAC)) consecutive
# NaN steps are filled by linear interpolation before feature computation.
# Longer gaps become segment boundaries.
GAP_INTERP_FRAC: float = 1 / 12
GAP_INTERP_MIN: int = 3

# Segments shorter than max(MIN_SEGMENT_ABS, period * MIN_SEGMENT_FRAC) are
# excluded from feature aggregation.
MIN_SEGMENT_FRAC: float = 1 / 4
MIN_SEGMENT_ABS: int = 20


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _horizon_steps(case) -> int:
    """Convert horizon_primary + freq strings to an integer step count."""
    h = pd.Timedelta(case.horizon_primary)
    f = pd.Timedelta(case.freq)
    return max(1, int(round(h / f)))


def _gap_aware_segments(
    case,
    period: int,
    gap_interp_max: int,
    min_segment_len: int,
) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
    """Return gap-handled training segments as parallel lists (y_segs, X_segs).

    Short gaps (≤ gap_interp_max consecutive NaN steps) are filled by linear
    interpolation in both y and exogenous columns.  Longer gaps become segment
    boundaries.  Segments shorter than min_segment_len are discarded.
    """
    df = case.load()
    train = case.split_frame(df)["train"].copy()

    # --- interpolate short gaps in y ---
    y_s = train["y"].copy()
    is_nan = y_s.isna().to_numpy()
    # identify contiguous NaN run boundaries
    padded = np.concatenate(([False], is_nan, [False])).view(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    nan_starts = edges[0::2]
    nan_ends = edges[1::2]  # exclusive

    short_gap_mask = np.zeros(len(y_s), dtype=bool)
    for s, e in zip(nan_starts, nan_ends):
        if (e - s) <= gap_interp_max:
            short_gap_mask[s:e] = True

    if short_gap_mask.any():
        # temporarily mark short-gap positions; interpolate fills them
        y_s = y_s.copy()
        y_s[short_gap_mask] = np.nan  # already NaN, but explicit
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

    # --- split at remaining NaN positions ---
    notna = train["y"].notna().to_numpy()
    # find starts/ends of contiguous True runs
    padded2 = np.concatenate(([False], notna, [False])).view(np.int8)
    edges2 = np.flatnonzero(np.diff(padded2))
    seg_starts = edges2[0::2]
    seg_ends = edges2[1::2]  # exclusive

    y_segs: list[np.ndarray] = []
    X_segs: list[np.ndarray | None] = []

    for s, e in zip(seg_starts, seg_ends):
        if (e - s) < min_segment_len:
            continue
        y_seg = train["y"].iloc[s:e].to_numpy(dtype=float)
        y_segs.append(y_seg)
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


# ---------------------------------------------------------------------------
# Feature aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate_features(
    feat_dicts: list[dict[str, float]],
    lengths: list[int],
) -> dict[str, float]:
    """Length-weighted average of per-segment feature dicts."""
    total = sum(lengths)
    agg: dict[str, float] = {}
    for key in feat_dicts[0]:
        agg[key] = sum(d[key] * w for d, w in zip(feat_dicts, lengths)) / total
    return agg


def _compute_features_segmented(
    y_segs: list[np.ndarray],
    X_segs: list[np.ndarray | None],
    period: int,
    n_exog_lags: int,
) -> tuple[dict[str, float], float]:
    """Compute morphological features and exog_r2_gain across all segments.

    Returns (feature_dict, exog_gain) aggregated by length-weighted average.
    """
    feat_dicts: list[dict[str, float]] = []
    gains: list[float] = []
    lengths: list[int] = []

    for y_seg, X_seg in zip(y_segs, X_segs):
        feat_dicts.append(compute_features_all(y_seg))
        gains.append(exog_r2_gain(y_seg, X_seg, period, n_exog_lags, poly_degree=2))
        lengths.append(len(y_seg))

    feats = _aggregate_features(feat_dicts, lengths)
    total = sum(lengths)
    gain = sum(g * w for g, w in zip(gains, lengths)) / total
    return feats, gain


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def _run_pca(
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Standardise and run PCA.

    Returns
    -------
    scores_df    DataFrame(index=case_id, columns=[PC1, PC2])
    loadings_df  DataFrame(index=feature, columns=[PC1, PC2])
    ev_ratio     explained variance ratios for PC1, PC2
    """
    X = StandardScaler().fit_transform(features_df[FEATURE_COLS].to_numpy())
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)
    scores_df = pd.DataFrame(scores, index=features_df.index, columns=["PC1", "PC2"])
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=FEATURE_COLS,
        columns=["PC1", "PC2"],
    )
    return scores_df, loadings_df, pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def _plot_biplot(
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
    ev_ratio: np.ndarray,
    domains: dict[str, str],
    out: Path,
) -> None:
    """Single-panel biplot: PC1/PC2 axes, uniform marker size."""
    fig, ax = plt.subplots(figsize=(10, 7.5))

    unique_domains = sorted(set(domains.values()))
    palette = plt.cm.tab10.colors
    domain_colour = {d: palette[i % len(palette)] for i, d in enumerate(unique_domains)}

    for case_id, row in scores.iterrows():
        colour = domain_colour.get(domains.get(case_id, ""), "#888")
        ax.scatter(
            row["PC1"],
            row["PC2"],
            color=colour,
            s=90,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
        )
        ax.text(row["PC1"] + 0.05, row["PC2"] + 0.05, case_id, fontsize=7.5, color=colour, zorder=4)

    # Scale loading arrows to ~80 % of PC1/PC2 score spread
    load_sub = loadings[["PC1", "PC2"]]
    score_range = max(scores[["PC1", "PC2"]].abs().max().max(), 1e-9)
    loading_max = max(load_sub.abs().max().max(), 1e-9)
    arrow_scale = 0.8 * score_range / loading_max

    for feat, row in load_sub.iterrows():
        l1, l2 = row["PC1"], row["PC2"]
        ax.annotate(
            "",
            xy=(l1 * arrow_scale, l2 * arrow_scale),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": "#333", "lw": 1.3},
            zorder=2,
        )
        label = FEATURE_LABELS.get(feat, feat)
        ax.text(
            l1 * arrow_scale * 1.10,
            l2 * arrow_scale * 1.10,
            label,
            fontsize=8.5,
            color="#222",
            ha="center",
            va="center",
            zorder=5,
        )

    ax.axhline(0, color="#ccc", linewidth=0.7, zorder=1)
    ax.axvline(0, color="#ccc", linewidth=0.7, zorder=1)

    ax.set_xlabel(f"PC1  ({ev_ratio[0]:.1%} variance explained)", fontsize=10)
    ax.set_ylabel(f"PC2  ({ev_ratio[1]:.1%} variance explained)", fontsize=10)
    ax.set_title(
        f"Morphological feature space — PCA biplot  ·  {ev_ratio[:2].sum():.1%} variance explained (2 PCs)\n"
        f"Water benchmark cases  ·  {len(FEATURE_COLS)}-feature vector",
        fontsize=10,
    )

    # Domain colour legend — outside, bottom, 2 rows × 4 columns
    colour_handles = [plt.scatter([], [], color=c, s=90, label=d) for d, c in domain_colour.items()]
    fig.legend(
        handles=colour_handles,
        title="Domain",
        fontsize=8,
        title_fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=4,
        frameon=False,
    )

    fig.subplots_adjust(bottom=0.15, right=0.95)

    ax.grid(True, alpha=0.18)
    ax.set_aspect("equal", "box")
    xlo, xhi = ax.get_xlim()
    ax.set_xlim(-max(abs(xlo), abs(xhi)), max(abs(xlo), abs(xhi)))
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(-max(abs(ylo), abs(yhi)), max(abs(ylo), abs(yhi)))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Biplot saved → {out.relative_to(REPO_ROOT)}")


def _plot_exog_bar(
    exog_series: pd.Series,
    domains: dict[str, str],
    out: Path,
) -> None:
    """Horizontal bar chart of exog_r2_gain, cases sorted descending."""
    ordered = exog_series.sort_values(ascending=True)  # ascending so top bar is highest

    unique_domains = sorted(set(domains.values()))
    palette = plt.cm.tab10.colors
    domain_colour = {d: palette[i % len(palette)] for i, d in enumerate(unique_domains)}
    colours = [domain_colour.get(domains.get(case_id, ""), "#888") for case_id in ordered.index]

    fig, ax = plt.subplots(figsize=(7, 0.45 * len(ordered) + 1.2))
    bars = ax.barh(
        range(len(ordered)), ordered.values, color=colours, edgecolor="white", linewidth=0.5
    )

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered.index, fontsize=8.5)
    ax.set_xlabel("Exogenous R² gain  (ARX − AR, ridge, training split)", fontsize=9)
    ax.set_title("External forcing dependence by case", fontsize=10)
    ax.set_xlim(0, max(ordered.values.max() * 1.15, 0.02))
    ax.axvline(0, color="#ccc", linewidth=0.7)
    ax.grid(axis="x", alpha=0.2)

    for bar, val in zip(bars, ordered.values):
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=7.5,
            color="#444",
        )

    colour_handles = [plt.scatter([], [], color=c, s=60, label=d) for d, c in domain_colour.items()]
    ax.legend(
        handles=colour_handles,
        title="Domain",
        fontsize=7.5,
        title_fontsize=8,
        loc="lower right",
        frameon=False,
    )

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Exog bar chart saved → {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--no-plot", action="store_true", help="Skip PCA and biplot; only write the CSV."
    )
    args = p.parse_args()

    cases = discover_cases()
    records: dict[str, dict[str, float]] = {}
    exog_gains: dict[str, float] = {}
    domains: dict[str, str] = {}

    for case_id, case in cases.items():
        if not case.parquet_path.exists():
            print(f"[{case_id}] no parquet — skipping.")
            continue
        period = period_for_freq(case.freq)
        gap_interp_max = max(GAP_INTERP_MIN, int(period * GAP_INTERP_FRAC))
        min_segment_len = max(MIN_SEGMENT_ABS, int(period * MIN_SEGMENT_FRAC))
        y_segs, X_segs = _gap_aware_segments(case, period, gap_interp_max, min_segment_len)
        if not y_segs:
            print(f"[{case_id}] no segments meet min_segment_len={min_segment_len} — skipping.")
            continue
        n_segs = len(y_segs)
        print(
            f"[{case_id}] {n_segs} seg(s), gap_interp_max={gap_interp_max} ...",
            end="  ",
            flush=True,
        )
        n_exog_lags = min(_horizon_steps(case), 30)
        feats, gain = _compute_features_segmented(y_segs, X_segs, period, n_exog_lags)
        exog_gains[case_id] = gain
        records[case_id] = feats
        domains[case_id] = case.domain
        print("  ".join(f"{k}={v:+.3f}" for k, v in feats.items()) + f"  exog_r2_gain={gain:+.3f}")

    if not records:
        print("No cases with processed parquet found — nothing to do.")
        return 1

    features_df = pd.DataFrame.from_dict(records, orient="index")[FEATURE_COLS]
    features_df.index.name = "case_id"
    features_df["exog_r2_gain"] = pd.Series(exog_gains)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "morphology_features.csv"
    features_df.to_csv(csv_path)
    print(f"\nFeatures saved → {csv_path.relative_to(REPO_ROOT)}")
    print(features_df.to_string(float_format="{:+.3f}".format))

    if not args.no_plot:
        scores_df, loadings_df, ev_ratio = _run_pca(features_df[FEATURE_COLS])
        _plot_biplot(
            scores_df,
            loadings_df,
            ev_ratio,
            domains,
            out=FIG_DIR / "pca_biplot.png",
        )
        _plot_exog_bar(
            features_df["exog_r2_gain"],
            domains,
            out=FIG_DIR / "exog_r2_gain.png",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
