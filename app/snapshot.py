"""Forecast-snapshot panel for the results-matrix app.

Draws, for one case, a grid of small charts — one per rolling-origin snapshot the eval
pipeline persisted to `results/pilot_snapshots.parquet`. Each panel shows the case's real
actual series around a forecast origin with one predicted line per selected method
overlaid, so several methods can be compared on the same origins by clicking cells along a
matrix row.

The grid is capped at `MAX_COLS` columns and stacks rows until every stored origin is
placed. Actuals are not stored in the snapshot artifact; they're reloaded from the case's
curated series so the target lives in exactly one place.
"""

from __future__ import annotations

import math
from functools import cache

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tsforecastbench.data.registry import get_case

MAX_COLS = 3

ACTUAL_COLOR = "black"
# Method identity is the thing being compared across overlaid lines, so colour encodes it
# (Actual stays black). Palette is colourblind-safe (Wong 2011, dropping its low-contrast
# yellow) and cycles if more methods are overlaid than it has entries.
METHOD_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#8B008B"]


def _horizon_steps(case) -> int:
    return int(pd.Timedelta(case.horizon_primary) // pd.Timedelta(case.freq))


@cache
def _actual_series(case_id: str) -> pd.Series:
    """The case's real curated target, indexed by timestamp. Cached: curated data is
    static, and the same series is reused across every origin panel and re-click."""
    case = get_case(case_id)
    df = case.load(curated=True)
    return pd.Series(df["y"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["ds"]))


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


def render_snapshots(case_id: str, methods: list[str], preds: pd.DataFrame) -> go.Figure:
    """Grid of forecast-origin snapshots for one case, overlaying each of `methods`.

    `preds` is the snapshot artifact (see `data.load_snapshots`); origins come from it, so
    every method shares the same panels and the overlays line up.
    """
    case_preds = preds[preds["case_id"] == case_id]
    if case_preds.empty:
        return _empty_fig(f"No stored forecasts for {case_id} (needs a rolling pilot run).")

    origins = case_preds[["origin_id", "origin_ds"]].drop_duplicates().sort_values("origin_id")
    origins = list(origins.itertuples(index=False))

    case = get_case(case_id)
    horizon_steps = _horizon_steps(case)
    n_history = max(horizon_steps // 4, 1)
    step = pd.Timedelta(case.freq)
    actual = _actual_series(case_id)

    n = len(origins)
    n_cols = min(MAX_COLS, n)
    n_rows = math.ceil(n / n_cols)

    titles = [pd.Timestamp(o.origin_ds).strftime("%Y-%m-%d %H:%M") for o in origins]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_yaxes="all",
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
            fig.add_trace(
                go.Scatter(
                    x=pd.DatetimeIndex(mpred["ds"]),
                    y=mpred["y_hat"].to_numpy(dtype=float),
                    name=method,
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
