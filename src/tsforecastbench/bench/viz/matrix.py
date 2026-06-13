"""Colour-coded matrix: rows = cases, cols = methods, cell value = chosen metric."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

_METRIC_CONFIG = {
    "MASE": {
        "cmap": "RdYlGn_r",
        "center": 1.0,
        "label": "MASE (lower is better; 1 = seasonal naive)",
        "fmt": "{:.2f}",
    },
    "Skill_vs_Persistence": {
        "cmap": "RdYlGn",
        "center": 0.0,
        "label": "Skill vs persistence (higher is better; 0 = no better)",
        "fmt": "{:+.2f}",
    },
    "MAE": {"cmap": "Reds", "center": None, "label": "MAE (lower is better)", "fmt": "{:.3g}"},
    "RMSE": {"cmap": "Reds", "center": None, "label": "RMSE (lower is better)", "fmt": "{:.3g}"},
}


def render_matrix(
    results: pd.DataFrame,
    metric: str = "MASE",
    out_path: Path | None = None,
    title: str | None = None,
    method_order: list[str] | None = None,
) -> plt.Figure:
    cfg = _METRIC_CONFIG.get(
        metric, {"cmap": "viridis", "center": None, "label": metric, "fmt": "{:.3g}"}
    )

    pivot = (
        results[results["metric"] == metric]
        .pivot(index="case_id", columns="method", values="value")
    )
    if method_order is not None:
        pivot = pivot.reindex(columns=[m for m in method_order if m in pivot.columns])

    data = pivot.to_numpy(dtype=float)
    n_rows, n_cols = data.shape

    fig, ax = plt.subplots(figsize=(1.2 * n_cols + 3, 0.6 * n_rows + 2.5))

    if cfg["center"] is not None and np.isfinite(data).any():
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        vmin = min(vmin, cfg["center"] - 1e-9)
        vmax = max(vmax, cfg["center"] + 1e-9)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=cfg["center"], vmax=vmax)
        im = ax.imshow(data, cmap=cfg["cmap"], norm=norm, aspect="auto")
    else:
        im = ax.imshow(data, cmap=cfg["cmap"], aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Method")
    ax.set_ylabel("Case")
    ax.set_title(title or f"RK2026 benchmark — {metric}")

    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            txt = "—" if not np.isfinite(v) else cfg["fmt"].format(v)
            rgba = im.cmap(im.norm(v)) if np.isfinite(v) else (0.5, 0.5, 0.5, 1.0)
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="black" if lum > 0.6 else "white",
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(cfg["label"])

    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
    return fig
