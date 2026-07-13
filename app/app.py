"""Interactive viewer for the benchmark results matrix.

Reads `results/pilot_metrics.parquet` and renders the same case x method grid as the
static `results/pilot_matrix_*.png` plots, but with hover detail, domain/method-family
filtering, and a metric switcher. Clicking a cell opens a snapshot panel with forecast
examples for that case x method pair.

Built on Dash: the matrix is a `dcc.Graph`, and a click callback reads the cell's
`clickData` to populate the snapshot panel — no selection-event workarounds needed.

Run with: uv run --extra app python app/app.py
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate
from snapshot import (
    METHOD_COLORS,
    TRAINING_TAIL_N,
    render_snapshots,
    render_split_distribution,
    render_split_proportions,
    render_test_snapshot,
    render_training_snapshot,
)

from data import (
    FAMILY_ORDER,
    case_has_extended_history,
    case_summary,
    load_metrics,
    load_snapshots,
    method_info,
    method_info_detail,
    ordered_cases,
    ordered_methods,
)
from tsforecastbench.bench.viz import METRIC_CONFIG

# Pixel size of one heatmap cell, and the fixed margin (title/colorbar/labels) around it —
# together they size the heatmap figure so cells render square (see `render_matrix`).
CELL = 42
MATRIX_MARGIN = 120

DOMAIN_KEY_DICT = {
    "groundwater": "GrW",
    "hydrology": "Hyd",
    "spectral_waves": "SpW",
    "storm_surge": "StS",
    "urban_drainage": "UrD",
    "wastewater_treatment": "WaT",
    "water_distribution": "WaD",
    "water_quality": "WaQ",
}


def pd_isna(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def _metric_cfg(metric: str) -> dict:
    return METRIC_CONFIG.get(
        metric, {"cmap": "viridis", "center": None, "vlo": None, "vhi": None, "fmt": "{:.3g}"}
    )


def _d3_format(py_fmt: str) -> str:
    """Convert this module's Python format spec ("{:.2f}") to a d3-format spec
    (".2f") for use inside a Plotly hovertemplate/texttemplate."""
    return py_fmt[2:-1]


def _pivot(df_metric, index_order, columns_order, column):
    return df_metric.pivot(index="case_id", columns="method", values=column).reindex(
        index=index_order, columns=columns_order
    )


def _group_boundaries(ordered_keys: list[str], key_of: dict[str, str]) -> list[float]:
    """Grid positions (between cell i-1 and i) where the group (domain/family) changes."""
    boundaries = []
    for i in range(1, len(ordered_keys)):
        if key_of[ordered_keys[i]] != key_of[ordered_keys[i - 1]]:
            boundaries.append(i - 0.5)
    return boundaries


def _group_labels(ordered_keys: list[str], key_of: dict[str, str]) -> list[tuple[float, str]]:
    """(center position, label) for each contiguous run of the same group."""
    labels = []
    start = 0
    for i in range(1, len(ordered_keys) + 1):
        at_end = i == len(ordered_keys)
        if at_end or key_of[ordered_keys[i]] != key_of[ordered_keys[start]]:
            labels.append(((start + i - 1) / 2, key_of[ordered_keys[start]]))
            start = i
    return labels


def _hatch_path(x0: float, y0: float, x1: float, y1: float, n_lines: int = 6) -> str:
    """SVG path for a thin diagonal hash filling one cell's box.

    Plotly shapes have no `fillpattern` in this version (only trace markers do), so the
    hash is drawn by hand as a set of disconnected diagonal strokes, each clipped to the
    box: a "M x,y L x,y" subpath per line, at intercepts spanning corner to corner.
    """
    s = x1 - x0
    segments = []
    for c in np.linspace(-s, s, n_lines + 2)[1:-1]:
        if c >= 0:
            xa, ya, xb, yb = x0, y0 + c, x1 - c, y1
        else:
            xa, ya, xb, yb = x0 - c, y0, x1, y1 + c
        segments.append(f"M {xa},{ya} L {xb},{yb}")
    return " ".join(segments)


def _global_overlay_shapes(
    cells: list[tuple[str, str]], row_order: list[str], col_order: list[str]
) -> list[dict]:
    """Thin hotpink diagonal-hash pattern over cells overridden with the extended-history
    (full-data) fit, so they read as "not directly comparable to the rest of the row"."""
    shapes = []
    for case_id, method in cells:
        if case_id not in row_order or method not in col_order:
            continue
        y = row_order.index(case_id)
        x = col_order.index(method)
        shapes.append(
            {
                "type": "path",
                "path": _hatch_path(x - 0.5, y - 0.5, x + 0.5, y + 0.5),
                "line": {"color": "#FF1493", "width": 1.2},
                "fillcolor": "rgba(0,0,0,0)",
                "opacity": 0.8,
                "layer": "above",
            }
        )
    return shapes


def _selection_shapes(
    selection: dict | None, row_order: list[str], col_order: list[str]
) -> list[dict]:
    """Outline shapes for the selected case's clicked cells, one per selected method,
    colour-matched to that method's chip/forecast line (see `_method_chips`)."""
    selection = selection or {}
    case_id, methods = selection.get("case_id"), selection.get("methods") or []
    if case_id not in row_order:
        return []
    y = row_order.index(case_id)
    shapes = []
    for i, method in enumerate(methods):
        if method not in col_order:
            continue
        x = col_order.index(method)
        color = METHOD_COLORS[i % len(METHOD_COLORS)]
        shapes.append(
            {
                "type": "rect",
                "x0": x - 0.5,
                "x1": x + 0.5,
                "y0": y - 0.5,
                "y1": y + 0.5,
                "line": {"color": color, "width": 3},
                "fillcolor": "rgba(0,0,0,0)",
                "layer": "above",
            }
        )
    return shapes


def render_matrix(
    df,
    metric: str,
    case_ids: list[str],
    methods: list[str],
    selection: dict | None = None,
    show_global: bool = False,
) -> tuple[go.Figure, int]:
    """Returns the heatmap figure plus its row count (unused by the current caller, kept
    since it falls out of the same pivot work the figure needs anyway).

    Every method has a `used_extended_history=False` ("local") row on the shared
    curated span; that's what's plotted by default. `is_global` methods additionally
    get a `used_extended_history=True` row wherever the case has real history beyond
    the curated span (a gappy record with a segment before the tail). When
    `show_global` is True, those cells are swapped in for the local score and outlined
    with a hash (see `_global_overlay_shapes`), since they're fit on more data than the
    rest of the row and aren't directly comparable.
    """
    df_metric = df[
        (df["metric"] == metric) & df["case_id"].isin(case_ids) & df["method"].isin(methods)
    ]
    df_local = df_metric[~df_metric["used_extended_history"]]
    row_order = ordered_cases(df_local)
    col_order = ordered_methods(df_local)

    values = _pivot(df_local, row_order, col_order, "value")
    status = _pivot(df_local, row_order, col_order, "status").fillna("—")
    n_origins = _pivot(df_local, row_order, col_order, "n_origins")
    frac_finite = _pivot(df_local, row_order, col_order, "frac_finite_test")
    seconds = _pivot(df_local, row_order, col_order, "seconds")
    horizon = _pivot(df_local, row_order, col_order, "horizon").fillna("—")

    overlay_cells: list[tuple[str, str]] = []
    if show_global:
        df_global = df_metric[df_metric["used_extended_history"] & df_metric["is_global"]]
        for row in df_global.itertuples(index=False):
            if row.case_id not in values.index or row.method not in values.columns:
                continue
            values.loc[row.case_id, row.method] = row.value
            status.loc[row.case_id, row.method] = row.status
            n_origins.loc[row.case_id, row.method] = row.n_origins
            frac_finite.loc[row.case_id, row.method] = row.frac_finite_test
            seconds.loc[row.case_id, row.method] = row.seconds
            overlay_cells.append((row.case_id, row.method))

    cfg = _metric_cfg(metric)
    fmt = cfg.get("fmt", "{:.3g}")
    text = values.map(lambda v: "—" if pd_isna(v) else fmt.format(v))

    def fmt_pct(v):
        return "—" if pd_isna(v) else f"{v:.0%}"

    def fmt_secs(v):
        return "—" if pd_isna(v) else f"{v:.2f}s"

    def fmt_int(v):
        return "—" if pd_isna(v) else f"{int(v)}"

    customdata = np.dstack(
        [
            status.to_numpy(dtype=object),
            n_origins.map(fmt_int).to_numpy(dtype=object),
            frac_finite.map(fmt_pct).to_numpy(dtype=object),
            seconds.map(fmt_secs).to_numpy(dtype=object),
            horizon.to_numpy(dtype=object),
        ]
    )

    heatmap_kwargs = dict(
        z=values.to_numpy(dtype=float),
        x=col_order,
        y=row_order,
        colorscale=cfg["cmap"],
        text=text.to_numpy(dtype=object),
        texttemplate="%{text}",
        textfont={"size": 10},
        customdata=customdata,
        # Native per-cell hover label is off; the row/col name tooltips are drawn as HTML
        # overlays by the `hover-tooltips` clientside callback. "none" (not "skip") keeps
        # the plotly_hover event firing so `matrix.hoverData` still updates.
        hoverinfo="none",
        colorbar={
            "title": metric.replace("_", "\n"),
            "len": 0.6,
            "thickness": 16,
            "x": 1.0,
            "xanchor": "left",
        },
    )
    if cfg.get("center") is not None:
        center = cfg["center"]
        finite = values.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        lo = (
            cfg.get("vlo") if cfg.get("vlo") is not None else (finite.min() if len(finite) else 0.0)
        )
        hi = (
            cfg.get("vhi") if cfg.get("vhi") is not None else (finite.max() if len(finite) else 1.0)
        )
        heatmap_kwargs["zmin"] = min(lo, center)
        heatmap_kwargs["zmax"] = max(hi, center)
        heatmap_kwargs["zmid"] = center

    fig = go.Figure(go.Heatmap(**heatmap_kwargs))

    n_rows, n_cols = len(row_order), len(col_order)
    domain_of = df_metric.drop_duplicates("case_id").set_index("case_id")["domain"].to_dict()
    family_of = df_metric.drop_duplicates("method").set_index("method")["family"].to_dict()

    boundary_line = {"color": "white", "width": 4}
    for y in _group_boundaries(row_order, domain_of):
        fig.add_shape(type="line", x0=-0.5, x1=n_cols - 0.5, y0=y, y1=y, line=boundary_line)
    for x in _group_boundaries(col_order, family_of):
        fig.add_shape(type="line", x0=x, x1=x, y0=-0.5, y1=n_rows - 0.5, line=boundary_line)
    for shape in _global_overlay_shapes(overlay_cells, row_order, col_order):
        fig.add_shape(**shape)
    for shape in _selection_shapes(selection, row_order, col_order):
        fig.add_shape(**shape)

    for pos, label in _group_labels(col_order, family_of):
        fig.add_annotation(
            x=pos,
            y=-0.5,
            xref="x",
            yref="y",
            yanchor="bottom",
            yshift=6,
            text=f"<b>{label.replace('_', '<br>').capitalize()}</b>",
            showarrow=False,
            font={"size": 13},
        )
    for pos, label in _group_labels(row_order, domain_of):
        fig.add_annotation(
            x=-0.5,
            y=pos,
            xref="x",
            yref="y",
            xanchor="right",
            xshift=-6,
            text=f"<b>{DOMAIN_KEY_DICT.get(label, label)}</b>",
            showarrow=False,
            font={"size": 13},
            textangle=-90,
        )

    # Size the figure so square cells fill the plot area exactly: with the height fixed
    # at CELL*n_rows (plus title margins), matching the width to CELL*n_cols leaves no
    # centering gap, so the colorbar and the group titles sit flush against the grid.
    fig.update_layout(
        height=CELL * n_rows + MATRIX_MARGIN,
        width=CELL * n_cols + 170,
        margin={"l": 90, "t": 40, "r": 80, "b": 80},
        paper_bgcolor="white",
        plot_bgcolor="white",
        dragmode=False,
        xaxis={
            "side": "bottom",
            "tickangle": -35,
            "constrain": "domain",
            "fixedrange": True,
        },
        yaxis={
            "autorange": "reversed",
            "scaleanchor": "x",
            "scaleratio": 1,
            "fixedrange": True,
        },
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig, n_rows


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"size": 16})
    fig.update_layout(
        height=300,
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def _domain_options(df) -> list[dict]:
    return [
        {"label": d.replace("_", " ").capitalize(), "value": d}
        for d in sorted(df["domain"].dropna().unique())
    ]


def _family_options(df) -> list[str]:
    return [f for f in FAMILY_ORDER if f in df["family"].unique()] + sorted(
        set(df["family"].unique()) - set(FAMILY_ORDER)
    )


app = Dash(__name__, title="Timeseries Forecasting Benchmark")

_df0 = load_metrics(0)
_metric_options = [m for m in METRIC_CONFIG if m in _df0["metric"].unique()]
_domain_opts = _domain_options(_df0)
_family_opts = _family_options(_df0)

_SIDEBAR_STYLE = {
    "width": "260px",
    "flex": "0 0 260px",
    "padding": "16px",
    "borderRight": "1px solid #ddd",
    "boxSizing": "border-box",
    "overflow": "hidden",
}
# Collapsed state keeps the div mounted (so dropdown state survives) but reclaims its
# width for the matrix/snapshot panels.
_SIDEBAR_COLLAPSED_STYLE = {
    **_SIDEBAR_STYLE,
    "width": "0px",
    "flex": "0 0 0px",
    "padding": "0px",
    "borderRight": "none",
}
_SIDEBAR_TOGGLE_STYLE = {
    "flex": "0 0 auto",
    "alignSelf": "flex-start",
    "marginTop": "16px",
    "border": "1px solid #ddd",
    "background": "white",
    "cursor": "pointer",
    "padding": "4px 8px",
}
_LABEL_STYLE = {"fontWeight": "bold", "marginTop": "16px", "display": "block"}

# A single top-level column, sibling to the sidebar rather than nested next to the
# matrix, so it starts flush with the top of the page instead of below the intro text.
# `position: sticky` keeps it in view as the (often much taller) matrix column scrolls
# underneath. Case info and test snapshots used to be separate side-by-side panels;
# they're now stacked in reading order (case context, then forecasts) inside one
# scrolling column, with the case-info block collapsible to leave more room for
# snapshots (see the `html.Details` wrapper around it in `app.layout`).
_DETAIL_COLUMN_STYLE = {
    "flex": "0 0 460px",
    "width": "460px",
    "boxSizing": "border-box",
    "position": "sticky",
    "top": "0",
    "maxHeight": "100vh",
    "padding": "16px",
    "borderLeft": "1px solid #ddd",
    "display": "flex",
    "flexDirection": "column",
    "minHeight": "0",
    "overflowY": "auto",
}
_CASE_PANEL_VISIBLE_STYLE = {"display": "flex", "flexDirection": "column", "flex": "0 0 auto"}
_CASE_PANEL_HIDDEN_STYLE = {"display": "none"}
_CASE_PLOT_CAPTION_STYLE = {
    "color": "#666",
    "fontSize": "0.85em",
    "margin": "12px 0 4px 0",
    "flex": "0 0 auto",
}

# Full-width strip docked to the bottom of the viewport (sticky, not fixed, so it scrolls
# into place with the rest of the page rather than always overlapping content) — one card
# per selected method, colour-matched to that method's chip/forecast line.
_METHOD_INFO_STYLE = {
    "position": "sticky",
    "bottom": "0",
    "display": "flex",
    "flexWrap": "wrap",
    "gap": "16px",
    "padding": "12px 16px",
    "background": "white",
    "borderTop": "1px solid #ddd",
    "boxShadow": "0 -2px 6px rgba(0,0,0,0.08)",
    "boxSizing": "border-box",
}
_METHOD_INFO_HIDDEN_STYLE = {"display": "none"}
_METHOD_INFO_CARD_STYLE = {
    "flex": "1 1 320px",
    "minWidth": "320px",
    "paddingLeft": "10px",
}

# Locks the heatmap down to a read-only view: keeps the click-to-select and
# hover interactions the app depends on, but strips every pan/zoom/select tool
# from the mode bar so the grid itself can't be dragged, scaled, or rearranged.
# The download-as-png button ("toImage") is the only mode-bar tool left.
_MATRIX_CONFIG = {
    "displayModeBar": True,
    "modeBarButtonsToRemove": [
        "zoom2d",
        "pan2d",
        "select2d",
        "lasso2d",
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "resetScale2d",
    ],
    "displaylogo": False,
    "scrollZoom": False,
    "doubleClick": False,
    "showAxisDragHandles": False,
    "showAxisRangeEntryBoxes": False,
}

# Two HTML overlays that replace the native per-cell hover label: a row tooltip pinned to
# the grid's left edge (triangle pointing right at the hovered row) showing the case name,
# and a column tooltip pinned to the top edge (triangle pointing down at the hovered
# column) showing the method name. Positioned by the `hover-tooltips` clientside callback.
_TOOLTIP_BG = "#2a3f5f"
_TOOLTIP_BOX_STYLE = {
    "background": _TOOLTIP_BG,
    "color": "white",
    "padding": "3px 8px",
    "borderRadius": "3px",
    "fontSize": "12px",
    "whiteSpace": "nowrap",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.3)",
}
_ROW_TRIANGLE_STYLE = {
    "width": "0",
    "height": "0",
    "borderTop": "6px solid transparent",
    "borderBottom": "6px solid transparent",
    "borderLeft": f"7px solid {_TOOLTIP_BG}",
}
_COL_TRIANGLE_STYLE = {
    "width": "0",
    "height": "0",
    "borderLeft": "6px solid transparent",
    "borderRight": "6px solid transparent",
    "borderTop": f"7px solid {_TOOLTIP_BG}",
}
# Base (hidden) styles; the clientside callback flips `display` and sets left/top/transform.
_ROW_TOOLTIP_STYLE = {
    "position": "absolute",
    "display": "none",
    "alignItems": "center",
    "pointerEvents": "none",
    "zIndex": 10,
}
_COL_TOOLTIP_STYLE = {
    "position": "absolute",
    "display": "none",
    "flexDirection": "column",
    "alignItems": "center",
    "pointerEvents": "none",
    "zIndex": 10,
}

app.layout = html.Div(
    style={"display": "flex", "flexDirection": "column", "fontFamily": "system-ui, sans-serif"},
    children=[
        html.Div(
            style={"display": "flex"},
            children=[
                html.Div(
                    id="sidebar",
                    style=_SIDEBAR_STYLE,
                    children=[
                        html.H3("Controls"),
                        html.Button("Reload data", id="reload-btn", n_clicks=0),
                        html.Label("Metric", style=_LABEL_STYLE),
                        dcc.Dropdown(
                            id="metric",
                            options=_metric_options,
                            value=_metric_options[0] if _metric_options else None,
                            clearable=False,
                        ),
                        html.Label("Domains", style=_LABEL_STYLE),
                        dcc.Checklist(
                            id="domains",
                            options=_domain_opts,
                            value=[o["value"] for o in _domain_opts],
                            labelStyle={"display": "block"},
                        ),
                        html.Label("Method families", style=_LABEL_STYLE),
                        dcc.Dropdown(
                            id="families",
                            options=_family_opts,
                            value=_family_opts,
                            multi=True,
                        ),
                        html.Label("Global methods", style=_LABEL_STYLE),
                        dcc.Checklist(
                            id="show-global",
                            options=[
                                {"label": " Score on full history where available", "value": "on"}
                            ],
                            value=[],
                        ),
                        html.P(
                            "For is_global methods on cases with data beyond the curated span, "
                            "swaps in the full-history fit (a hash marks affected cells).",
                            style={"color": "#666", "fontSize": "0.8em", "marginTop": "4px"},
                        ),
                    ],
                ),
                html.Button(
                    "◀",
                    id="sidebar-toggle",
                    n_clicks=0,
                    title="Collapse/expand controls",
                    style=_SIDEBAR_TOGGLE_STYLE,
                ),
                html.Div(
                    style={"flex": "1", "padding": "16px", "minWidth": "0"},
                    children=[
                        html.H1("Timeseries Forecasting Benchmark"),
                        html.P(
                            "The following matrix compiles the forecasting results of multiple methods across a variety of timeseries in different water domains. "
                            "Each cell represents one case x method run, colored by the metric chosen in the left sidebar. In addition, you can use the control panel in the sidebar to filter by domain, method family, ",
                            style={"color": "#444", "maxWidth": "1000px"},
                        ),
                        html.P(" "),
                        html.P(
                            "Click a cell to get more details on the training data, and explore forecast snapshots. "
                            "You can click up to 3 cells in the same row to stack their forecasts on the same plot for comparison.",
                            style={"color": "#444", "maxWidth": "1000px"},
                        ),
                        html.Div(
                            id="matrix-wrapper",
                            style={"position": "relative"},
                            children=[
                                dcc.Graph(
                                    id="matrix",
                                    config=_MATRIX_CONFIG,
                                    clear_on_unhover=True,
                                ),
                                html.Div(
                                    id="row-tooltip",
                                    style=_ROW_TOOLTIP_STYLE,
                                    children=[
                                        html.Div(id="row-tooltip-text", style=_TOOLTIP_BOX_STYLE),
                                        html.Div(style=_ROW_TRIANGLE_STYLE),
                                    ],
                                ),
                                html.Div(
                                    id="col-tooltip",
                                    style=_COL_TOOLTIP_STYLE,
                                    children=[
                                        html.Div(id="col-tooltip-text", style=_TOOLTIP_BOX_STYLE),
                                        html.Div(style=_COL_TRIANGLE_STYLE),
                                    ],
                                ),
                            ],
                        ),
                        html.H2("Benchmark Design"),
                        html.Div(
                            style={"color": "#444", "maxWidth": "1000px"},
                            children=[
                                html.P(
                                    [
                                        html.B("Splits. "),
                                        "Each case declares its train/val/test split in "
                                        "case.toml, either as calendar dates (train_end, "
                                        "val_end, test_end) or as fractions (default "
                                        "70/15/15) of the observed timestamps, so series "
                                        "with gaps still split cleanly. Splits are strictly "
                                        "chronological — no shuffling.",
                                    ]
                                ),
                                html.P(
                                    [
                                        html.B("Curation. "),
                                        "Curation enforces minimum lengths: train must "
                                        "cover at least two full seasonal cycles, and "
                                        "val+test together must cover at least one "
                                        "forecast horizon. Evaluation is rolling-origin by "
                                        "default (errors pooled across many refit points "
                                        "in the test window), with a --fast single-holdout "
                                        "mode for quick checks. Missing data is kept as "
                                        "explicit NaN: imputed for training, masked out "
                                        "during scoring so only real observations count.",
                                    ]
                                ),
                                html.P(
                                    [
                                        html.B("Model parameters. "),
                                        "There's no per-case tuning or grid search. Each "
                                        "method uses fixed, hard-coded defaults — e.g., "
                                        "LightGBM and CatBoost both use 200 iterations at "
                                        "a 0.05 learning rate, so the two gradient-boosting "
                                        "methods compete on equal footing. Classical "
                                        "methods (AutoARIMA, AutoETS, etc.) run their own "
                                        "internal search, but training size is capped per "
                                        "method to bound fit time, not to improve accuracy. "
                                        "Any deviation from a method's default is manual "
                                        "and explicit, not automated.",
                                    ]
                                ),
                                html.P(
                                    [
                                        html.B("Case description. "),
                                        "Cases carry structured metadata in case.toml: "
                                        "domain, target, frequency, horizons, exogenous "
                                        "variables, source citation, and declared "
                                        "seasonalities. Seasonalities are the single "
                                        "source of truth for every downstream method — a "
                                        "candidate only qualifies with real evidence "
                                        "(enough training cycles, a clear periodogram "
                                        "peak, sufficient autocorrelation). There's no "
                                        "ground-truth seasonality to validate against; any "
                                        "override needs a written justification. The same "
                                        "evidence-based standard applies to marking "
                                        "exogenous variables as necessary, since that's a "
                                        "domain judgment, not something a model's error "
                                        "alone can tell you.",
                                    ]
                                ),
                            ],
                        ),
                        dcc.Store(id="snapshot-selection", data={"case_id": None, "methods": []}),
                    ],
                ),
                html.Div(
                    id="detail-column",
                    style=_DETAIL_COLUMN_STYLE,
                    children=[
                        html.Div(
                            id="case-panel",
                            style=_CASE_PANEL_HIDDEN_STYLE,
                            children=[
                                html.Details(
                                    open=True,
                                    children=[
                                        html.Summary(
                                            id="case-panel-title",
                                            style={
                                                "cursor": "pointer",
                                                "fontWeight": "bold",
                                                "fontSize": "1.1em",
                                                "margin": "0 0 4px 0",
                                            },
                                        ),
                                        html.P(
                                            id="case-training-caption",
                                            style={
                                                **_CASE_PLOT_CAPTION_STYLE,
                                                "margin": "0 0 4px 0",
                                            },
                                        ),
                                        dcc.Graph(
                                            id="case-training-graph",
                                            config={"displayModeBar": False},
                                            style={"flex": "0 0 auto", "width": "100%"},
                                        ),
                                        html.P(
                                            "Test data (full test split)",
                                            style=_CASE_PLOT_CAPTION_STYLE,
                                        ),
                                        dcc.Graph(
                                            id="case-test-graph",
                                            config={"displayModeBar": False},
                                            style={"flex": "0 0 auto", "width": "100%"},
                                        ),
                                        html.P(
                                            "Train vs test value distribution",
                                            style=_CASE_PLOT_CAPTION_STYLE,
                                        ),
                                        dcc.Graph(
                                            id="case-distribution-graph",
                                            config={"displayModeBar": False},
                                            style={"flex": "0 0 auto", "width": "100%"},
                                        ),
                                        html.P(
                                            "Train / val / test split",
                                            style=_CASE_PLOT_CAPTION_STYLE,
                                        ),
                                        dcc.Graph(
                                            id="case-proportions-graph",
                                            config={"displayModeBar": False},
                                            style={"flex": "0 0 auto", "width": "100%"},
                                        ),
                                        html.Hr(style={"flex": "0 0 auto", "width": "100%"}),
                                        html.P(
                                            "Case summary",
                                            style={
                                                "color": "#666",
                                                "fontSize": "0.9em",
                                                "flex": "0 0 auto",
                                            },
                                        ),
                                        html.Div(
                                            id="case-summary",
                                            style={"flex": "0 0 auto"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Hr(style={"flex": "0 0 auto", "width": "100%"}),
                        html.Div(
                            id="snapshot-column",
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "flex": "0 0 auto",
                            },
                            children=[
                                html.Div(
                                    id="snapshot-hint",
                                    children=(
                                        "Click a matrix cell to see real forecast snapshots: the "
                                        "best, median, and worst performing origins, by the metric "
                                        "selected above, for that method. Click more cells in the "
                                        "same row to overlay other methods on those same origins."
                                    ),
                                    style={"color": "#666"},
                                ),
                                html.Div(
                                    id="snapshot-panel",
                                    style={"display": "none"},
                                    children=[
                                        html.H2(id="snapshot-title"),
                                        html.P(
                                            id="snapshot-caption",
                                            style={
                                                "color": "#666",
                                                "fontSize": "0.85em",
                                                "margin": "0 0 8px 0",
                                            },
                                        ),
                                        html.Div(
                                            style={
                                                "display": "flex",
                                                "gap": "8px",
                                                "alignItems": "center",
                                                "flexWrap": "wrap",
                                                "marginBottom": "8px",
                                            },
                                            children=[
                                                html.Span(
                                                    id="snapshot-chips",
                                                    style={"display": "contents"},
                                                ),
                                                html.Button(
                                                    "Clear",
                                                    id="clear-btn",
                                                    n_clicks=0,
                                                    style={"marginLeft": "auto"},
                                                ),
                                            ],
                                        ),
                                        dcc.Graph(id="snapshot-graph"),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(id="method-info-panel", style=_METHOD_INFO_HIDDEN_STYLE),
    ],
)


# Position the row/col name tooltips from the hovered cell. Runs in the browser so it can
# read the live pixel geometry off the Plotly graph div (`_fullLayout` axis offsets and the
# data->pixel converters) and reacts to every hover without a server round-trip. Returns
# hidden styles when the cursor leaves the grid (dcc.Graph `clear_on_unhover` nulls hoverData).
app.clientside_callback(
    """
    function(hoverData) {
        const rowHidden = {position:'absolute', display:'none', alignItems:'center',
                           pointerEvents:'none', zIndex:10};
        const colHidden = {position:'absolute', display:'none', flexDirection:'column',
                           alignItems:'center', pointerEvents:'none', zIndex:10};
        if (!hoverData || !hoverData.points || !hoverData.points.length) {
            return [rowHidden, '', colHidden, ''];
        }
        const pt = hoverData.points[0];
        const wrapper = document.getElementById('matrix-wrapper');
        const gd = wrapper ? wrapper.querySelector('.js-plotly-plot') : null;
        if (!gd || !gd._fullLayout) {
            return [rowHidden, '', colHidden, ''];
        }
        const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
        const d2pix = (ax, v) => (ax.d2p ? ax.d2p(v) : ax.c2p(ax.d2c(v)));
        // Offset from the wrapper's origin to the plot div's origin, so absolute-positioned
        // tooltips line up with the axis pixel coordinates below.
        const wr = wrapper.getBoundingClientRect(), gr = gd.getBoundingClientRect();
        const dx = gr.left - wr.left, dy = gr.top - wr.top;
        const colCenter = dx + xa._offset + d2pix(xa, pt.x);
        const rowCenter = dy + ya._offset + d2pix(ya, pt.y);
        const gridLeft = dx + xa._offset, gridTop = dy + ya._offset;
        const rowStyle = Object.assign({}, rowHidden, {
            display:'flex', left: gridLeft + 'px', top: rowCenter + 'px',
            transform: 'translate(-100%, -50%)'
        });
        const colStyle = Object.assign({}, colHidden, {
            display:'flex', left: colCenter + 'px', top: gridTop + 'px',
            transform: 'translate(-50%, -100%)'
        });
        return [rowStyle, pt.y, colStyle, pt.x];
    }
    """,
    Output("row-tooltip", "style"),
    Output("row-tooltip-text", "children"),
    Output("col-tooltip", "style"),
    Output("col-tooltip-text", "children"),
    Input("matrix", "hoverData"),
)


@app.callback(
    Output("sidebar", "style"),
    Output("sidebar-toggle", "children"),
    Input("sidebar-toggle", "n_clicks"),
)
def toggle_sidebar(n_clicks):
    collapsed = (n_clicks or 0) % 2 == 1
    style = _SIDEBAR_COLLAPSED_STYLE if collapsed else _SIDEBAR_STYLE
    return style, "▶" if collapsed else "◀"


@app.callback(
    Output("matrix", "figure"),
    Input("metric", "value"),
    Input("domains", "value"),
    Input("families", "value"),
    Input("reload-btn", "n_clicks"),
    Input("snapshot-selection", "data"),
    Input("show-global", "value"),
)
def update_matrix(metric, domains, families, n_reload, selection, show_global):
    df = load_metrics(n_reload or 0)
    show_global = "on" in (show_global or [])
    if not metric or not domains or not families:
        return _empty_fig("Select a metric, at least one domain, and one method family.")

    methods = sorted(df.loc[df["family"].isin(families), "method"].unique())
    case_ids = sorted(df.loc[df["domain"].isin(domains), "case_id"].unique())
    if not case_ids or not methods:
        return _empty_fig("No cases or methods match the current filters.")

    fig, _n_rows = render_matrix(df, metric, case_ids, methods, selection, show_global)
    return fig


def _method_chips(methods: list[str]) -> list:
    """One removable chip per selected method, colour-matched to its forecast line."""
    chips = []
    for i, method in enumerate(methods):
        color = METHOD_COLORS[i % len(METHOD_COLORS)]
        chips.append(
            html.Button(
                [
                    html.Span(
                        style={
                            "display": "inline-block",
                            "width": "10px",
                            "height": "10px",
                            "borderRadius": "50%",
                            "backgroundColor": color,
                            "marginRight": "6px",
                        }
                    ),
                    f"{method}  ✕",
                ],
                id={"type": "method-chip", "method": method},
                n_clicks=0,
                title="Remove from overlay",
                style={
                    "border": f"1px solid {color}",
                    "borderRadius": "14px",
                    "padding": "2px 10px",
                    "background": "white",
                    "cursor": "pointer",
                    "fontSize": "0.85em",
                },
            )
        )
    return chips


def _method_info_cards(case_id: str, methods: list[str]) -> list:
    """One card per selected method for the bottom method-info panel, colour-matched to
    that method's chip/forecast line (see `_method_chips`)."""
    cards = []
    for i, method in enumerate(methods):
        color = METHOD_COLORS[i % len(METHOD_COLORS)]
        cards.append(
            html.Div(
                style={**_METHOD_INFO_CARD_STYLE, "borderLeft": f"3px solid {color}"},
                children=[
                    html.Div(method, style={"fontWeight": "bold", "marginBottom": "4px"}),
                    dcc.Markdown(
                        method_info(case_id, method), style={"fontSize": "0.85em", "margin": "0"}
                    ),
                    html.Details(
                        style={"marginTop": "6px"},
                        children=[
                            html.Summary(
                                "Implementation detail",
                                style={"fontSize": "0.85em", "cursor": "pointer"},
                            ),
                            dcc.Markdown(
                                method_info_detail(case_id, method),
                                style={"fontSize": "0.85em", "margin": "4px 0 0 0"},
                            ),
                        ],
                    ),
                ],
            )
        )
    return cards


EMPTY_SELECTION = {"case_id": None, "methods": []}

# Cap on methods overlaid per case: the snapshot graph and the method-info panel both
# lay their per-method content out side by side, which stops being readable much past
# this. Adding a 4th distinct method evicts the oldest rather than refusing the click.
MAX_OVERLAY_METHODS = 3


def next_selection(trigger, trigger_value, click_data: dict | None, sel: dict) -> dict:
    """Pure transition for the overlay selection — the logic behind `update_selection`,
    factored out of Dash's callback context so it can be reasoned about and tested.

    `trigger` is the triggering component id (a str, or a pattern-matching dict for a
    method chip); `trigger_value` is that input's value. Raises `PreventUpdate` when the
    event shouldn't change the selection.
    """
    sel = sel or EMPTY_SELECTION

    if trigger == "clear-btn":
        return dict(EMPTY_SELECTION)

    # Removing a method via its chip. Rebuilding chips on every render also fires this
    # callback with all-None n_clicks; act only on a genuine click (a truthy value).
    if isinstance(trigger, dict) and trigger.get("type") == "method-chip":
        if not trigger_value:
            raise PreventUpdate
        methods = [m for m in sel["methods"] if m != trigger["method"]]
        return {"case_id": sel["case_id"], "methods": methods} if methods else dict(EMPTY_SELECTION)

    points = (click_data or {}).get("points") or []
    if not points:
        raise PreventUpdate
    case_id, method = points[0]["y"], points[0]["x"]
    if case_id != sel["case_id"]:
        return {"case_id": case_id, "methods": [method]}
    if method in sel["methods"]:
        methods = [m for m in sel["methods"] if m != method]
    else:
        methods = [*sel["methods"], method][-MAX_OVERLAY_METHODS:]
    return {"case_id": case_id, "methods": methods} if methods else dict(EMPTY_SELECTION)


@app.callback(
    Output("snapshot-selection", "data"),
    Input("matrix", "clickData"),
    Input("clear-btn", "n_clicks"),
    Input({"type": "method-chip", "method": ALL}, "n_clicks"),
    State("snapshot-selection", "data"),
    prevent_initial_call=True,
)
def update_selection(click_data, clear_clicks, chip_clicks, sel):
    """Maintain the {case_id, methods} overlay selection.

    Same-row clicks accumulate methods, up to `MAX_OVERLAY_METHODS` (beyond that, the
    oldest is evicted; re-click removes, when Plotly re-fires it); a click in a
    different row resets to that case; chip clicks and Clear remove methods.
    """
    return next_selection(ctx.triggered_id, ctx.triggered[0]["value"], click_data, sel)


@app.callback(
    Output("case-panel-title", "children"),
    Output("case-training-caption", "children"),
    Output("case-training-graph", "figure"),
    Output("case-test-graph", "figure"),
    Output("case-distribution-graph", "figure"),
    Output("case-proportions-graph", "figure"),
    Output("case-summary", "children"),
    Output("case-panel", "style"),
    Input("snapshot-selection", "data"),
    Input("show-global", "value"),
    Input("reload-btn", "n_clicks"),
)
def render_case_panel(sel, show_global, n_reload):
    """Right-hand panel: training/test snapshots, a train-vs-test distribution, a
    train/val/test proportions bar, and case metadata for the clicked cell's case.
    Depends only on `case_id` (plus the global toggle), so it stays open across method
    chip clicks. When `show_global` reveals extended history for this case (see
    `data.case_has_extended_history`), the training snapshot, distribution, and
    proportions bar all swap in the extended-history train split."""
    case_id = (sel or {}).get("case_id")
    if not case_id:
        empty = go.Figure()
        return "", "", empty, empty, empty, empty, "", _CASE_PANEL_HIDDEN_STYLE

    show_global = "on" in (show_global or [])
    extended = show_global and case_has_extended_history(load_metrics(n_reload or 0), case_id)
    caption = f"Training data (last {TRAINING_TAIL_N:,} points"
    caption += ", including extended history)" if extended else ")"

    return (
        case_id.replace("_", " ").upper(),
        caption,
        render_training_snapshot(case_id, extended=extended),
        render_test_snapshot(case_id),
        render_split_distribution(case_id, extended=extended),
        render_split_proportions(case_id, extended=extended),
        dcc.Markdown(case_summary(case_id)),
        _CASE_PANEL_VISIBLE_STYLE,
    )


@app.callback(
    Output("snapshot-title", "children"),
    Output("snapshot-caption", "children"),
    Output("snapshot-graph", "figure"),
    Output("snapshot-chips", "children"),
    Output("snapshot-panel", "style"),
    Output("snapshot-hint", "style"),
    Input("snapshot-selection", "data"),
    Input("reload-btn", "n_clicks"),
    Input("show-global", "value"),
    Input("metric", "value"),
)
def render_panel(sel, n_reload, show_global, metric):
    sel = sel or {}
    case_id, methods = sel.get("case_id"), sel.get("methods") or []
    if not case_id or not methods:
        return "", "", go.Figure(), [], {"display": "none"}, {"color": "#666"}

    show_global = "on" in (show_global or [])
    title = "[Test] Prediction Snapshots"
    caption = f"Origins curated by {methods[0]}'s {metric} (best / median / worst)."
    return (
        title,
        caption,
        render_snapshots(
            case_id,
            methods,
            load_snapshots(n_reload or 0),
            show_global,
            max_cols=1,
            metric=metric,
        ),
        _method_chips(methods),
        {"display": "block"},
        {"display": "none"},
    )


@app.callback(
    Output("method-info-panel", "children"),
    Output("method-info-panel", "style"),
    Input("snapshot-selection", "data"),
)
def render_method_info_panel(sel):
    """Bottom panel: one card per selected method with a brief description and its
    exogenous-input usage for this case (see `data.method_info`), plus a collapsed
    "Implementation detail" section (library, compute cost, limitations, a code
    excerpt — see `data.method_info_detail`)."""
    sel = sel or {}
    case_id, methods = sel.get("case_id"), sel.get("methods") or []
    if not case_id or not methods:
        return [], _METHOD_INFO_HIDDEN_STYLE
    return _method_info_cards(case_id, methods), _METHOD_INFO_STYLE


if __name__ == "__main__":
    app.run(debug=True, port=8050)
