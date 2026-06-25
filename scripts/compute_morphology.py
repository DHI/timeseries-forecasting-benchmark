"""Compute the 6-feature morphological vector for all benchmark cases and
produce a PCA biplot.

Outputs
-------
results/morphology_features.csv        raw feature table (case × 6 features)
docs/figures/morphology/pca_biplot.png PCA biplot — PC1/PC2 scores + loadings

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
from tsforecastbench.morphology.features import compute_features, period_for_freq

RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = REPO_ROOT / "docs" / "figures" / "morphology"

FEATURE_COLS = [
    "acf1",
    "trend_strength",
    "seasonal_strength",
    "spectral_entropy",
    "boxcox_lambda",
    "kurtosis",
]

# Friendly display names for the loading arrows in the biplot.
FEATURE_LABELS = {
    "acf1": "ACF₁",
    "trend_strength": "trend",
    "seasonal_strength": "seasonal",
    "spectral_entropy": "entropy",
    "boxcox_lambda": "λ (Box-Cox)",
    "kurtosis": "kurtosis",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _extract_y(case) -> np.ndarray:
    """Load processed parquet for a case and return the clean target array."""
    df = case.load()
    return df["y"].dropna().to_numpy(dtype=float)


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
    ev_ratio     explained variance ratios for PC1 and PC2
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
    """Matplotlib biplot: labelled case scores + feature loading arrows."""
    fig, ax = plt.subplots(figsize=(10, 7.5))

    unique_domains = sorted(set(domains.values()))
    palette = plt.cm.tab10.colors
    domain_colour = {d: palette[i % len(palette)] for i, d in enumerate(unique_domains)}

    for case_id, (pc1, pc2) in scores.iterrows():
        colour = domain_colour.get(domains.get(case_id, ""), "#888")
        ax.scatter(pc1, pc2, color=colour, s=90, zorder=3, edgecolors="white", linewidths=0.5)
        ax.text(pc1 + 0.05, pc2 + 0.05, case_id, fontsize=7.5, color=colour, zorder=4)

    # Scale loading arrows to ~80 % of the score spread
    score_range = max(scores.abs().max().max(), 1e-9)
    loading_max = max(loadings.abs().max().max(), 1e-9)
    arrow_scale = 0.8 * score_range / loading_max

    for feat, (l1, l2) in loadings.iterrows():
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

    # Conceptual axis interpretation: double-headed arrows + labels at each pole.
    # Placed in axes-fraction coordinates so they sit at fixed positions regardless
    # of the data range.  Low zorder keeps them behind everything else.
    _ann_kw = {
        "xycoords": "axes fraction",
        "textcoords": "axes fraction",
        "arrowprops": {"arrowstyle": "<->", "color": "#bbb", "lw": 1.3},
        "zorder": 0,
    }
    _txt_kw = {
        "transform": ax.transAxes,
        "fontsize": 8,
        "color": "#aaa",
        "style": "italic",
        "zorder": 0,
    }

    # Horizontal: smooth (left) ↔ impulsive (right) — PC1
    ax.annotate("", xy=(0.42, 0.05), xytext=(0.58, 0.05), **_ann_kw)
    ax.text(0.40, 0.05, "smooth", ha="right", va="center", **_txt_kw)
    ax.text(0.60, 0.05, "impulsive", ha="left", va="center", **_txt_kw)

    # Vertical: seasonal (bottom) ↔ trending (top) — PC2
    ax.annotate("", xy=(0.05, 0.42), xytext=(0.05, 0.58), **_ann_kw)
    ax.text(0.05, 0.40, "seasonal", ha="center", va="top", rotation=90, **_txt_kw)
    ax.text(0.05, 0.60, "trending", ha="center", va="bottom", rotation=90, **_txt_kw)
    ax.set_xlabel(f"PC1  ({ev_ratio[0]:.1%} variance explained)", fontsize=10)
    ax.set_ylabel(f"PC2  ({ev_ratio[1]:.1%} variance explained)", fontsize=10)
    ax.set_title(
        "Morphological feature space — PCA biplot\n"
        "Water benchmark cases  ·  6-feature vector on raw series at native frequency",
        fontsize=10,
    )

    for domain, colour in domain_colour.items():
        ax.scatter([], [], color=colour, label=domain, s=60)
    ax.legend(
        title="Domain",
        fontsize=8,
        title_fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncols=4,
        frameon=False,
    )

    ax.grid(True, alpha=0.18)
    ax.set_aspect("equal", "box")
    xmin, xmax = ax.get_xlim()
    m = max(abs(xmin), abs(xmax))
    ax.set_xlim(-m, m)
    ymin, ymax = ax.get_ylim()
    m = max(abs(ymin), abs(ymax))
    ax.set_ylim(-m, m)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Biplot saved → {out.relative_to(REPO_ROOT)}")


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
    domains: dict[str, str] = {}

    for case_id, case in cases.items():
        if not case.parquet_path.exists():
            print(f"[{case_id}] no parquet — skipping.")
            continue
        print(f"[{case_id}] computing ...", end="  ", flush=True)
        y = _extract_y(case)
        period = period_for_freq(case.freq)
        feats = compute_features(y, period)
        records[case_id] = feats
        domains[case_id] = case.domain
        print("  ".join(f"{k}={v:+.3f}" for k, v in feats.items()))

    if not records:
        print("No cases with processed parquet found — nothing to do.")
        return 1

    features_df = pd.DataFrame.from_dict(records, orient="index")[FEATURE_COLS]
    features_df.index.name = "case_id"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "morphology_features.csv"
    features_df.to_csv(csv_path)
    print(f"\nFeatures saved → {csv_path.relative_to(REPO_ROOT)}")
    print(features_df.to_string(float_format="{:+.3f}".format))

    if not args.no_plot:
        scores_df, loadings_df, ev_ratio = _run_pca(features_df)
        _plot_biplot(
            scores_df,
            loadings_df,
            ev_ratio,
            domains,
            out=FIG_DIR / "pca_biplot.png",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
