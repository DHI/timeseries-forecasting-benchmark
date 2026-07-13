"""Forecast-snapshot panel for the results-matrix app.

Draws, for one case, a grid of small charts — one per curated rolling-origin snapshot the
eval pipeline persisted to `results/pilot_snapshots.parquet`. Rather than showing every
stored origin in storage order, the panel picks up to three representative ones (best,
median, and worst by whichever metric is selected on the matrix — see `curate_origins`)
so a viewer sees a meaningful spread of outcomes instead of an arbitrary sequence. Each
panel shows the case's real actual series around a forecast origin with one predicted
line per selected method overlaid, so several methods can be compared on the same
curated origins by clicking cells along a matrix row.

The grid is capped at `MAX_COLS` columns and stacks rows until every curated origin is
placed. Actuals are not stored in the snapshot artifact; they're reloaded from the case's
curated series so the target lives in exactly one place.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from data import (
    load_case_split,
    load_case_split_counts,
    load_case_split_values,
    load_case_training_tail,
)
from tsforecastbench.bench.eval.metrics import mae as _mae_score
from tsforecastbench.bench.eval.metrics import mase as _mase_score
from tsforecastbench.bench.eval.metrics import rmse as _rmse_score
from tsforecastbench.bench.eval.metrics import skill_score as _skill_score
from tsforecastbench.data.registry import get_case
from tsforecastbench.data.seasonality import mase_season_for

# Sort direction per matrix metric (see tsforecastbench.bench.viz.METRIC_CONFIG for the
# same set of names) -- lower is better for error metrics, higher is better for skill
# scores. Falls back to "min" for any metric not listed here.
METRIC_LOWER_IS_BETTER = {
    "MAE": True,
    "RMSE": True,
    "MASE": True,
    "Skill_vs_Persistence": False,
    "Skill_vs_Mean": False,
}

MAX_COLS = 3
TRAINING_TAIL_N = 1000

# Shared height for the case-panel's train/test snapshots and distribution plot — kept
# short so all four case-panel plots fit on screen together without much scrolling.
CASE_PLOT_HEIGHT = 150

ACTUAL_COLOR = "black"
# Method identity is the thing being compared across overlaid lines, so colour encodes it
# (Actual stays black). Palette is colourblind-safe (Wong 2011, dropping its low-contrast
# yellow) and cycles if more methods are overlaid than it has entries.
METHOD_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#8B008B"]

# Greyscale, chronological-order encoding for the train/val/test split-proportions bar —
# darkest first, since these panels aren't comparing method identity like the palette above.
SPLIT_COLORS = {"train": "#000000", "val": "#888888", "test": "#cccccc"}


def _horizon_steps(case) -> int:
    return int(pd.Timedelta(case.horizon_primary) // pd.Timedelta(case.freq))


@cache
def _actual_series(case_id: str) -> pd.Series:
    """The case's real curated target, indexed by timestamp -- gap-filled, like the
    `y` column the eval pipeline fits models on and builds persistence/mean reference
    forecasts from (see `bench.eval.rolling.run_case_rolling`). Cached: curated data is
    static, and the same series is reused across every origin panel and re-click."""
    case = get_case(case_id)
    df = case.load(curated=True)
    return pd.Series(df["y"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["ds"]))


@cache
def _observed_actual_series(case_id: str) -> pd.Series:
    """Like `_actual_series`, but with filled (non-observed) points masked back to NaN
    -- mirrors `bench.eval.loop._raw_y`, so only genuinely-observed points count towards
    a per-origin metric, matching how the matrix itself scores."""
    case = get_case(case_id)
    df = case.load(curated=True)
    mask = df["y_observed"] if "y_observed" in df.columns else df["y"].notna()
    y = np.where(mask.to_numpy(), df["y"].to_numpy(dtype=float), np.nan)
    return pd.Series(y, index=pd.DatetimeIndex(df["ds"]))


@cache
def _mase_context(case_id: str) -> tuple[np.ndarray, int]:
    """(train+val `y`, seasonal period) for this case's MASE denominator -- mirrors the
    once-per-case `naive_scale` in `bench.eval.rolling.run_case_rolling`, so a per-origin
    MASE uses the same fixed scale as the matrix's pooled MASE rather than a per-origin
    one."""
    case = get_case(case_id)
    df = case.load(curated=True)
    splits = case.split_frame(df)
    y_train = pd.concat([splits["train"], splits["val"]])["y"].to_numpy(dtype=float)
    return y_train, mase_season_for(case.seasonalities, len(y_train))


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"size": 14})
    fig.update_layout(
        height=300,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def _line_snapshot(df: pd.DataFrame, empty_message: str) -> go.Figure:
    """Shared quick-look line plot for a single split — used for both the training and
    test snapshots, which differ only in what data they're handed."""
    if df.empty:
        return _empty_fig(empty_message)
    fig = go.Figure(
        go.Scatter(
            x=df["ds"],
            y=df["y"],
            mode="lines",
            line={"color": ACTUAL_COLOR, "width": 1.5},
        )
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False, showline=True, linecolor="black", ticks="outside"
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="black",
        ticks="outside",
        automargin=True,
    )
    fig.update_layout(
        autosize=True,
        height=CASE_PLOT_HEIGHT,
        margin={"l": 0, "t": 10, "r": 10, "b": 30},
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "system-ui, sans-serif"},
    )
    return fig


def render_training_snapshot(
    case_id: str, n: int = TRAINING_TAIL_N, extended: bool = False
) -> go.Figure:
    """Quick-look line plot of the case's last `n` training-split points — shown in the
    side panel independent of any method selection. `extended` swaps in the
    extended-history train split where the pilot run found one (see
    `data.load_case_split`)."""
    df = load_case_training_tail(case_id, n, extended=extended)
    return _line_snapshot(df, f"No training data available for {case_id}.")


def render_test_snapshot(case_id: str) -> go.Figure:
    """Quick-look line plot of the case's full test split — always the curated span,
    since extended history never reaches into the test window."""
    df = load_case_split(case_id, "test")[["ds", "y"]]
    return _line_snapshot(df, f"No test data available for {case_id}.")


def _kde_curve(values: pd.Series, x_grid: np.ndarray) -> np.ndarray | None:
    """Gaussian-KDE density of `values` over `x_grid`, or None if there isn't enough
    variance to fit one (a near-empty split, or one with a single repeated value)."""
    values = values.to_numpy(dtype=float)
    if len(values) < 2 or np.ptp(values) == 0:
        return None
    return gaussian_kde(values)(x_grid)


def render_split_distribution(case_id: str, extended: bool = False) -> go.Figure:
    """Train vs test value distributions for one case, as KDE lines rather than
    overlaid histogram bars — two curves on the same axes are easier to compare than
    two sets of overlapping bars."""
    train = load_case_split_values(case_id, "train", extended=extended)
    test = load_case_split_values(case_id, "test")
    if train.empty and test.empty:
        return _empty_fig(f"No data available for {case_id}.")

    combined = pd.concat([train, test])
    x_grid = np.linspace(combined.min(), combined.max(), 200)
    train_density = _kde_curve(train, x_grid)
    test_density = _kde_curve(test, x_grid)
    if train_density is None and test_density is None:
        return _empty_fig(f"Not enough variance to estimate a distribution for {case_id}.")

    fig = go.Figure()
    if train_density is not None:
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=train_density,
                name="Train",
                mode="lines",
                line={"color": "#000000", "width": 2},
            )
        )
    if test_density is not None:
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=test_density,
                name="Test",
                mode="lines",
                line={"color": "#888888", "width": 2, "dash": "dash"},
            )
        )
    fig.update_xaxes(
        showgrid=False, zeroline=False, showline=True, linecolor="black", ticks="outside"
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="black",
        ticks="outside",
        showticklabels=False,
    )
    fig.update_layout(
        height=CASE_PLOT_HEIGHT,
        margin={"l": 0, "t": 10, "r": 10, "b": 30},
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "system-ui, sans-serif"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
        },
    )
    return fig


def render_split_proportions(case_id: str, extended: bool = False) -> go.Figure:
    """Horizontal stacked bar of the train/val/test proportion of observed points.
    `extended` swaps in the extended-history train count (see `data.load_case_split_counts`),
    so the train segment grows when the toggle reveals history beyond the curated span."""
    counts = load_case_split_counts(case_id, extended=extended)
    total = sum(counts.values())
    if total == 0:
        return _empty_fig(f"No data available for {case_id}.")

    fig = go.Figure()
    for split, color in SPLIT_COLORS.items():
        pct = counts[split] / total
        fig.add_trace(
            go.Bar(
                y=[""],
                x=[pct],
                name=split.capitalize(),
                orientation="h",
                marker_color=color,
                text=f"{split.capitalize()} {pct:.0%}",
                textposition="inside",
                insidetextanchor="middle",
                textfont={"color": "black" if split == "test" else "white"},
                hovertemplate=f"{split}: %{{x:.0%}}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        height=90,
        margin={"l": 0, "t": 10, "r": 10, "b": 10},
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        font={"family": "system-ui, sans-serif"},
        xaxis={"visible": False, "range": [0, 1]},
        yaxis={"visible": False},
    )
    return fig


def _origin_metric(case_id: str, origin_preds: pd.DataFrame, metric: str) -> float:
    """One origin's score on `metric`, computed the same way the matrix's pooled score
    is (see `bench.eval.rolling.run_case_rolling`), just over a single origin's window
    instead of pooled across all of them. NaN if the window has no observed actual, or
    (MASE only) the case's train+val span can't support a seasonal-naive scale.

    MAE/RMSE/MASE compare against the observed-only actual (`_observed_actual_series`);
    MASE's denominator is the case-level `_mase_context` scale, not a per-origin one.
    The skill scores compare against a reference forecast built from the *filled*
    history strictly before this origin (`_actual_series`) -- persistence repeats its
    last value, mean repeats its average -- mirroring how the pipeline isolates the
    model's skill from a shifted baseline.
    """
    ds = pd.DatetimeIndex(origin_preds["ds"])
    aligned = _observed_actual_series(case_id).reindex(ds)
    mask = aligned.notna().to_numpy()
    if not mask.any():
        return float("nan")
    y = aligned.to_numpy()[mask]
    y_hat = origin_preds["y_hat"].to_numpy(dtype=float)[mask]

    if metric == "RMSE":
        return _rmse_score(y, y_hat)
    if metric == "MASE":
        y_train, season = _mase_context(case_id)
        try:
            return _mase_score(y, y_hat, y_train, season=season)
        except ValueError:
            return float("nan")
    if metric in ("Skill_vs_Persistence", "Skill_vs_Mean"):
        origin_ds = pd.Timestamp(origin_preds["origin_ds"].iloc[0])
        history = _actual_series(case_id)
        local_hist = history[history.index < origin_ds].to_numpy()
        if len(local_hist) == 0:
            return float("nan")
        reference = local_hist[-1] if metric == "Skill_vs_Persistence" else np.nanmean(local_hist)
        return _skill_score(y, y_hat, np.full(mask.sum(), float(reference)))
    return _mae_score(y, y_hat)  # default / "MAE"


def curate_origins(
    case_preds: pd.DataFrame, anchor_method: str, case_id: str, metric: str = "MAE"
) -> pd.DataFrame:
    """Pick up to 3 representative origins for one case — the best, median, and worst
    performing by `metric` (whichever the matrix currently shows) for `anchor_method`
    (the first-selected/"anchor" method of the overlay) — so the snapshot grid shows a
    meaningful spread of outcomes rather than whatever origins happen to be first in
    storage order. All overlaid methods share this same set of origins so their
    forecasts remain directly comparable.

    Returns a frame with columns (origin_id, origin_ds, value, label), ordered
    Best -> Median -> Worst ("Best" always means most-favorable by `metric`, so lowest
    for an error metric but highest for a skill score -- see `METRIC_LOWER_IS_BETTER`).
    Origins with an undefined score are dropped before ranking; if fewer than 3 remain,
    only the labels that make sense are emitted (e.g. just Best/Worst for 2 origins).
    """
    anchor_preds = case_preds[case_preds["method"] == anchor_method]
    origins = anchor_preds[["origin_id", "origin_ds"]].drop_duplicates().reset_index(drop=True)
    origins["value"] = [
        _origin_metric(case_id, anchor_preds[anchor_preds["origin_id"] == oid], metric)
        for oid in origins["origin_id"]
    ]
    ascending = METRIC_LOWER_IS_BETTER.get(metric, True)
    origins = origins.dropna(subset=["value"]).sort_values("value", ascending=ascending)
    origins = origins.reset_index(drop=True)

    n = len(origins)
    if n == 0:
        return origins.assign(label=pd.Series(dtype=object))
    if n == 1:
        picks, labels = [0], ["Best"]
    elif n == 2:
        picks, labels = [0, 1], ["Best", "Worst"]
    else:
        picks = sorted({0, n // 2, n - 1})
        label_of = {0: "Best", n // 2: "Median", n - 1: "Worst"}
        labels = [label_of[i] for i in picks]

    curated = origins.loc[picks].copy()
    curated["label"] = labels
    return curated


def render_snapshots(
    case_id: str,
    methods: list[str],
    preds: pd.DataFrame,
    show_global: bool = False,
    max_cols: int = MAX_COLS,
    metric: str = "MAE",
) -> go.Figure:
    """Grid of curated forecast-origin snapshots for one case, overlaying each of
    `methods`.

    `preds` is the snapshot artifact (see `data.load_snapshots`). Rather than showing
    every stored origin, origins are curated by `curate_origins` — best/median/worst by
    `metric` (the matrix's currently-selected metric) for `methods[0]` (the anchor) — so
    every method shares the same curated panels and the overlays line up. When
    `show_global` is True, a method with an extended-history fit for this case (see
    `bench.eval.rolling.run_case_rolling`) is drawn from that fit instead of the local one
    -- matching the matrix's toggle -- with its legend entry flagged accordingly.
    `max_cols` caps how many origin panels share a row; pass 1 to stack them in a single
    column, e.g. when rendering into a narrow vertical panel.
    """
    case_preds = preds[preds["case_id"] == case_id]
    if case_preds.empty:
        return _empty_fig(f"No stored forecasts for {case_id} (needs a rolling pilot run).")

    has_extended = case_preds.groupby("method")["used_extended_history"].any()
    use_extended = {m: bool(show_global and has_extended.get(m, False)) for m in methods}
    case_preds = pd.concat(
        [
            case_preds[
                (case_preds["method"] == m)
                & (case_preds["used_extended_history"] == use_extended[m])
            ]
            for m in methods
        ],
        ignore_index=True,
    )
    if case_preds.empty:
        return _empty_fig(f"No stored forecasts for {case_id} (needs a rolling pilot run).")

    curated = curate_origins(case_preds, methods[0], case_id, metric)
    if curated.empty:
        return _empty_fig(f"No observed actuals to score any stored origin for {case_id}.")
    origins = list(curated.itertuples(index=False))

    case = get_case(case_id)
    horizon_steps = _horizon_steps(case)
    n_history = max(horizon_steps // 4, 1)
    step = pd.Timedelta(case.freq)
    actual = _actual_series(case_id)

    n = len(origins)
    n_cols = min(max_cols, n)
    n_rows = math.ceil(n / n_cols)

    titles = [f"{o.label} · ({metric} {o.value:.3g})" for o in origins]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_yaxes=False,
        subplot_titles=titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.04,
    )

    color_of = {m: METHOD_COLORS[i % len(METHOD_COLORS)] for i, m in enumerate(methods)}

    for i, origin in enumerate(origins):
        row, col = i // n_cols + 1, i % n_cols + 1
        first_panel = i == 0
        origin_ds = pd.Timestamp(origin.origin_ds)
        origin_preds = case_preds[case_preds["origin_id"] == origin.origin_id]

        last_pred_ds = pd.Timestamp(origin_preds["ds"].max())
        window = actual.loc[origin_ds - n_history * step : last_pred_ds]
        fig.add_trace(
            go.Scatter(
                x=window.index,
                y=window.to_numpy(),
                name="Actual",
                mode="lines",
                line={"color": ACTUAL_COLOR, "width": 2},
                legendgroup="Actual",
                showlegend=first_panel,
            ),
            row=row,
            col=col,
        )

        for method in methods:
            mpred = origin_preds[origin_preds["method"] == method].sort_values("ds")
            if mpred.empty:
                continue
            label = f"{method} (global fit)" if use_extended[method] else method
            fig.add_trace(
                go.Scatter(
                    x=pd.DatetimeIndex(mpred["ds"]),
                    y=mpred["y_hat"].to_numpy(dtype=float),
                    name=label,
                    mode="lines",
                    line={"color": color_of[method], "width": 2},
                    legendgroup=method,
                    showlegend=first_panel,
                ),
                row=row,
                col=col,
            )

        fig.add_vline(
            x=origin_ds,
            line={"color": "#888888", "dash": "dash", "width": 1},
            row=row,
            col=col,
        )

    fig.update_xaxes(
        showgrid=False, zeroline=False, showline=True, linecolor="black", ticks="outside"
    )
    fig.update_yaxes(
        showgrid=False, zeroline=False, showline=True, linecolor="black", ticks="outside"
    )
    fig.update_layout(
        height=240 * n_rows + 60,
        margin={"l": 50, "t": 60, "r": 20, "b": 40},
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "system-ui, sans-serif"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
        },
    )
    return fig
