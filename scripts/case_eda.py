"""Per-case EDA panel for the documentation.

Produces a single PNG per case with three panels: data-availability strip,
long-range target series (with a key exogenous overlaid), and a zoom window
showing the dynamics. A small stats box reports rows, span, mean, std, and
observed fraction.

Run:
    uv run python scripts/case_eda.py             # all cases with processed Parquet
    uv run python scripts/case_eda.py --case waves_oregon
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Force non-interactive backend BEFORE pyplot import; we only save PNGs.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.data.registry import discover_cases  # noqa: E402

FIG_DIR = REPO_ROOT / "docs" / "figures" / "cases"

# Per-case visualisation hints: which exogenous to overlay, zoom window length.
CASE_HINTS = {
    "discharge_lowland_denmark":  {"overlay": "precip",            "overlay_kind": "rain",  "zoom": "365D", "target_label": "Discharge (mm/d)"},
    "discharge_flashy_catskills": {"overlay": "precip",            "overlay_kind": "rain",  "zoom": "365D", "target_label": "Discharge (m3/s)"},
    "groundwater_yorktown":     {"overlay": "precip",            "overlay_kind": "rain",  "zoom": "365D", "target_label": "Depth to water (m)"},
    "discharge_snowmelt_yosemite": {"overlay": "temp_air",       "overlay_kind": "line",  "zoom": "365D", "target_label": "Discharge (m3/s)"},
    "watertemp_james":     {"overlay": "temp_air",          "overlay_kind": "line",  "zoom": "365D", "target_label": "Water temp (C)"},
    "sewerlevel_bellinge": {"overlay": "rain_gauge",        "overlay_kind": "rain",  "zoom": "14D",  "target_label": "Sewer level (m)"},
    "demand_ltown":         {"overlay": None,                "overlay_kind": None,    "zoom": "14D",  "target_label": "Inlet flow (L/s)"},
    "wwtp_influent_n2o_350k":        {"overlay": None,                "overlay_kind": None,    "zoom": "14D",  "target_label": "Influent flow"},
    "surge_cuxhaven":    {"overlay": "mslp",              "overlay_kind": "line",  "zoom": "30D",  "target_label": "Residual (m)"},
    "waves_oregon":           {"overlay": "wspd",              "overlay_kind": "line",  "zoom": "30D",  "target_label": "H_s (m)"},
    "waves_massachusetts":  {"overlay": "wspd",              "overlay_kind": "line",  "zoom": "30D",  "target_label": "H_s (m)"},
}


def _overlay(ax, ds, values, label, kind):
    """Plot an exogenous overlay on a twin-y axis. Rain uses a filled hyetograph for
    visibility (mostly-zero series with occasional spikes); other variables use a line."""
    ax2 = ax.twinx()
    v = pd.to_numeric(values, errors="coerce")
    if kind == "rain":
        # filled bars from 0; the strong colour + alpha makes spikes pop
        ax2.fill_between(ds, 0, v, step="mid", color="#cb181d", alpha=0.55, linewidth=0)
        ax2.set_ylim(bottom=0)
    else:
        ax2.plot(ds, v, color="#cb181d", linewidth=0.4, alpha=0.6)
    ax2.set_ylabel(label, color="#cb181d")
    ax2.tick_params(axis="y", labelcolor="#cb181d")
    return ax2


def _pick_zoom_window(ds: pd.Series, y: pd.Series, mask: np.ndarray, length: pd.Timedelta):
    """Find the window of `length` with the HIGHEST variance of y (= most dynamic).

    A dense window of constant or near-zero values is uninformative; we want the part
    of the record where the signal is varying. Slides a window over the record at ~32
    candidate positions and picks the one with the largest std(y).
    """
    if mask.sum() == 0:
        return ds.iloc[0], ds.iloc[-1]
    obs_y = pd.to_numeric(y, errors="coerce")
    span = ds.iloc[-1] - ds.iloc[0]
    if length >= span:
        return ds.iloc[0], ds.iloc[-1]
    # ~32 candidate start positions across the record
    n_candidates = 48
    starts = pd.date_range(ds.iloc[0], ds.iloc[-1] - length, periods=n_candidates)
    best_t0, best_std = starts[0], -1.0
    for t0 in starts:
        t1 = t0 + length
        sub = obs_y[(ds >= t0) & (ds <= t1)]
        if sub.notna().sum() < 10:
            continue
        s = float(sub.std())
        if s > best_std:
            best_std, best_t0 = s, t0
    return best_t0, best_t0 + length


def make_eda(case_id: str) -> Path:
    cases = discover_cases()
    if case_id not in cases:
        raise KeyError(case_id)
    case = cases[case_id]
    pq = case.parquet_path
    if not pq.exists():
        print(f"[{case_id}] no processed Parquet at {pq}; skipping.")
        return None
    df = pd.read_parquet(pq)
    df["ds"] = pd.to_datetime(df["ds"], utc=True)
    df = df.sort_values("ds").reset_index(drop=True)
    hint = CASE_HINTS.get(case_id, {})

    y = pd.to_numeric(df["y"], errors="coerce")
    mask = y.notna().to_numpy()
    n = len(df)
    n_obs = int(mask.sum())
    span = df["ds"].iloc[-1] - df["ds"].iloc[0]

    fig = plt.figure(figsize=(11, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.6, 2.4, 2.4])
    ax_av = fig.add_subplot(gs[0])
    ax_long = fig.add_subplot(gs[1])
    ax_zoom = fig.add_subplot(gs[2])

    # --- availability strip ---
    ax_av.fill_between(df["ds"], 0, mask.astype(float),
                       step="mid", color="#2c7fb8", alpha=0.85, linewidth=0)
    ax_av.set_ylim(0, 1.1)
    ax_av.set_yticks([])
    ax_av.set_xlim(df["ds"].iloc[0], df["ds"].iloc[-1])
    ax_av.set_title(f"{case_id} · {case.domain} · {case.freq} · horizon {case.horizon_primary}",
                    loc="left", fontsize=11)
    ax_av.text(0.99, 0.5, f"observed: {n_obs/n:.0%}  ·  {n_obs:,} of {n:,} rows",
               transform=ax_av.transAxes, ha="right", va="center", fontsize=9, color="#555")

    # --- long view: target + optional exogenous overlay ---
    ax_long.plot(df["ds"], y, color="#08519c", linewidth=0.5)
    ax_long.set_ylabel(hint.get("target_label", "y"))
    ax_long.grid(True, alpha=0.25)
    ovl = hint.get("overlay")
    overlay_drawn = False
    if ovl and ovl in df.columns:
        # diagnostic: how much overlay data actually exists?
        vo = pd.to_numeric(df[ovl], errors="coerce")
        ovl_finite = int(vo.notna().sum())
        ovl_nonzero = int((vo.abs() > 1e-9).sum())
        ovl_max = float(vo.max()) if ovl_finite else float("nan")
        print(f"[{case_id}]   overlay {ovl}: finite={ovl_finite:,} "
              f"nonzero={ovl_nonzero:,} max={ovl_max:.4g}")
        if ovl_nonzero > 0:
            _overlay(ax_long, df["ds"], df[ovl], ovl, hint.get("overlay_kind", "line"))
            overlay_drawn = True
    elif ovl:
        print(f"[{case_id}]   overlay {ovl}: NOT in columns {list(df.columns)}")

    # If no useful overlay, add a weekly-scale rolling mean of the target as a
    # secondary trend line so the long view always shows some auxiliary structure.
    if not overlay_drawn:
        # window = ~7 days expressed in samples; safe lower bound for short series
        seven_days_steps = int(pd.Timedelta("7D") / pd.Timedelta(pd.tseries.frequencies.to_offset(case.freq)))
        window_n = max(seven_days_steps, 7)
        rolling = y.rolling(window_n, min_periods=max(2, window_n // 4), center=True).mean()
        ax_long.plot(df["ds"], rolling, color="#cb181d", linewidth=1.2, alpha=0.75,
                     label="7-day rolling mean")
        ax_long.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # mark splits
    splits = case.split_frame(df)
    if len(splits["train"]) and len(splits["val"]) and len(splits["test"]):
        tv = splits["val"]["ds"].iloc[0]
        vt = splits["test"]["ds"].iloc[0]
        for x, label in [(tv, "val"), (vt, "test")]:
            ax_long.axvline(x, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
            ax_long.text(x, ax_long.get_ylim()[1], f" {label}", color="grey",
                         fontsize=8, va="top")

    # --- zoom window ---
    zoom_len = pd.Timedelta(hint.get("zoom", "30D"))
    t0, t1 = _pick_zoom_window(df["ds"], y, mask, zoom_len)
    sub = df[(df["ds"] >= t0) & (df["ds"] <= t1)]
    ax_zoom.plot(sub["ds"], pd.to_numeric(sub["y"], errors="coerce"),
                 color="#08519c", linewidth=0.9)
    ax_zoom.set_ylabel(hint.get("target_label", "y"))
    ax_zoom.set_title(f"Zoom: {t0.strftime('%Y-%m-%d')} → {t1.strftime('%Y-%m-%d')}",
                      loc="left", fontsize=9, color="#555")
    if ovl and ovl in sub.columns:
        sub_v = pd.to_numeric(sub[ovl], errors="coerce")
        if (sub_v.abs() > 1e-9).sum() > 0:
            _overlay(ax_zoom, sub["ds"], sub[ovl], ovl, hint.get("overlay_kind", "line"))
    ax_zoom.grid(True, alpha=0.25)

    # nicer date ticks
    for ax in (ax_av, ax_long):
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax_zoom.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_zoom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_zoom.xaxis.get_major_locator()))

    # stats box
    mean = float(np.nanmean(y))
    std = float(np.nanstd(y))
    stats = (f"span  {span.days} days\n"
             f"mean  {mean:.3g}\n"
             f"std   {std:.3g}\n"
             f"freq  {case.freq}")
    ax_long.text(0.005, 0.97, stats, transform=ax_long.transAxes, va="top", ha="left",
                 fontsize=8, family="monospace",
                 bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#ccc", "pad": 4})

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{case_id}_eda.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[{case_id}] wrote {out.relative_to(REPO_ROOT)}  ({n_obs:,}/{n:,} obs, span {span.days}d)")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case", help="single case id; default: all with processed Parquet")
    args = p.parse_args()

    if args.case:
        make_eda(args.case)
    else:
        for cid in discover_cases():
            make_eda(cid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
