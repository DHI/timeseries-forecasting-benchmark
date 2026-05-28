"""Plot forecast error as a function of lead time, one line per method."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_leadtime_curve(
    curve: pd.DataFrame,
    metric: str = "MAE",
    out_path: Path | None = None,
    title: str | None = None,
    freq_label: str = "steps",
) -> plt.Figure:
    """curve: tidy frame from run_leadtime_curve (method, lead, MAE, RMSE, bias)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, g in curve.groupby("method"):
        g = g.sort_values("lead")
        ax.plot(g["lead"], g[metric], marker="o", ms=3, label=method)
    ax.set_xlabel(f"Lead time ({freq_label})")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Forecast {metric} vs lead time (avg over origins)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
    return fig
