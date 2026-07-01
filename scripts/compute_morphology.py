"""Compute morphological features for all benchmark cases and produce a PCA biplot.

All 13 candidate features are computed and saved (plus exog_r2_gain and domain).

Outputs
-------
results/morphology_features.csv        full feature table (case × 13 features
                                       + exog_r2_gain + domain)
docs/figures/morphology/exog_r2_gain.png external-forcing dependence bar chart

Usage
-----
    uv run python scripts/compute_morphology.py
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.data.registry import discover_cases
from tsforecastbench.morphology import (
    GAP_INTERP_FRAC,
    GAP_INTERP_MIN,
    MIN_SEGMENT_ABS,
    MIN_SEGMENT_FRAC,
    aggregate_features,
    gap_aware_segments,
)
from tsforecastbench.morphology.features import (
    compute_features_all,
    exog_r2_gain,
    period_for_freq,
)

RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = REPO_ROOT / "docs" / "figures" / "morphology"


# The 4 features used for the PCA taxonomy biplot.
# Written by notebooks/feature_selection.py (Save selection button);
# falls back to the defaults below if the file is absent.
def _load_feature_cols() -> list[str]:
    import json as _json

    _cfg = RESULTS_DIR / "selected_features.json"
    if _cfg.exists():
        return _json.loads(_cfg.read_text())["feature_cols"]
    return [
        "seasonal_strength",
        "hurst",
        "amplitude_dispersion",
        "l_skewness",
    ]


# Friendly display names for the loading arrows in the biplot. All 13 candidates
# are listed so the biplot is always readable regardless of which features are active.
FEATURE_LABELS = {
    "acf1": "ACF₁",
    "trend_strength": "Trend\nstrength",
    "seasonal_strength": "Seasonal\nstrength",
    "spectral_entropy": "Spectral\nentropy",
    "boxcox_lambda": "Box-Cox λ",
    "kurtosis": "Kurtosis",
    "hurst": "Hurst",
    "perm_entropy": "Perm.\nentropy",
    "amplitude_dispersion": "Amplitude\ndispersion",
    "phase_dispersion": "Phase\ndispersion",
    "asymmetry": "Asymmetry",
    "l_skewness": "L-skewness",
    "zero_proportion": "Zero\nproportion",
}

# Column order for the output CSV (all 13 candidates + exog_r2_gain + domain).
_ALL_COLS = [
    "acf1",
    "trend_strength",
    "seasonal_strength",
    "spectral_entropy",
    "boxcox_lambda",
    "kurtosis",
    "hurst",
    "perm_entropy",
    "amplitude_dispersion",
    "phase_dispersion",
    "asymmetry",
    "l_skewness",
    "zero_proportion",
    "exog_r2_gain",
    "domain",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _horizon_steps(case) -> int:
    """Convert horizon_primary + freq strings to an integer step count."""
    h = pd.Timedelta(case.horizon_primary)
    f = pd.Timedelta(case.freq)
    return max(1, int(round(h / f)))


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

    feats = aggregate_features(feat_dicts, lengths)
    total = sum(lengths)
    gain = sum(g * w for g, w in zip(gains, lengths)) / total
    return feats, gain


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


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
        y_segs, X_segs = gap_aware_segments(case, period, gap_interp_max, min_segment_len)
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

    if not records:
        print("No cases with processed parquet found — nothing to do.")
        return 1

    features_df = pd.DataFrame.from_dict(records, orient="index")
    features_df.index.name = "case_id"
    features_df["exog_r2_gain"] = pd.Series(exog_gains)
    features_df["domain"] = pd.Series(domains)
    features_df = features_df[[c for c in _ALL_COLS if c in features_df.columns]]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "morphology_features.csv"
    features_df.to_csv(csv_path)
    print(f"\nFeatures saved → {csv_path.relative_to(REPO_ROOT)}")

    _plot_exog_bar(
        features_df["exog_r2_gain"],
        domains,
        out=FIG_DIR / "exog_r2_gain.png",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
