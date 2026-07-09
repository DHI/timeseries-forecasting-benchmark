"""Interactive viewer for the benchmark results matrix.

Reads `results/pilot_metrics.parquet` and renders the same case x method grid as the
static `results/pilot_matrix_*.png` plots, but with hover detail, row/column filtering
(including by domain/method-family), and a metric switcher.

Run with: uv run --extra app streamlit run app/app.py
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from snapshot import render_snapshots

from data import FAMILY_ORDER, load_case_notes, load_metrics, ordered_cases, ordered_methods
from tsforecastbench.bench.viz import METRIC_CONFIG

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


def render_matrix(df, metric: str, case_ids: list[str], methods: list[str]) -> go.Figure:
    df_metric = df[
        (df["metric"] == metric) & df["case_id"].isin(case_ids) & df["method"].isin(methods)
    ]
    row_order = ordered_cases(df_metric)
    col_order = ordered_methods(df_metric)

    values = _pivot(df_metric, row_order, col_order, "value")
    status = _pivot(df_metric, row_order, col_order, "status").fillna("—")
    n_origins = _pivot(df_metric, row_order, col_order, "n_origins")
    frac_finite = _pivot(df_metric, row_order, col_order, "frac_finite_test")
    seconds = _pivot(df_metric, row_order, col_order, "seconds")
    horizon = _pivot(df_metric, row_order, col_order, "horizon").fillna("—")

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
        hovertemplate=(
            "<b>%{y}</b> x <b>%{x}</b><br>"
            f"{metric}: %{{z:{_d3_format(fmt)}}}<br>"
            "status: %{customdata[0]}<br>"
            "n_origins: %{customdata[1]}<br>"
            "frac_finite_test: %{customdata[2]}<br>"
            "runtime: %{customdata[3]}<br>"
            "horizon: %{customdata[4]}"
            "<extra></extra>"
        ),
        colorbar={"title": metric, "len": 0.6},
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

    for pos, label in _group_labels(col_order, family_of):
        fig.add_annotation(
            x=pos,
            y=0.9,
            xref="x",
            yref="paper",
            yshift=10,
            text=f"<b>{label.replace('_', '<br>').capitalize()}</b>",
            showarrow=False,
            font={"size": 14},
        )
    for pos, label in _group_labels(row_order, domain_of):
        fig.add_annotation(
            x=0.05,
            y=pos,
            xref="paper",
            yref="y",
            xshift=-90,
            text=f"<b>{DOMAIN_KEY_DICT.get(label, label)}</b>",
            showarrow=False,
            font={"size": 14},
            textangle=-90,
        )

    fig.update_layout(
        height=max(600, 40 * n_rows + 120),
        margin={"l": 140, "t": 60, "r": 20, "b": 80},
        xaxis={"side": "bottom", "tickangle": -35, "constrain": "domain"},
        yaxis={"autorange": "reversed", "scaleanchor": "x", "scaleratio": 1},
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def main() -> None:
    st.set_page_config(page_title="Benchmark results matrix", layout="wide")
    st.title("Benchmark results matrix")

    if "reload_version" not in st.session_state:
        st.session_state.reload_version = 0

    with st.sidebar:
        st.header("Controls")
        if st.button("Reload data"):
            st.session_state.reload_version += 1
        df = load_metrics(st.session_state.reload_version)

        metric_options = [m for m in METRIC_CONFIG if m in df["metric"].unique()]
        metric = st.selectbox("Metric", metric_options, index=0)

        all_domains = sorted(df["domain"].dropna().unique())
        st.markdown(
            "Domains",
            # help="Domain abbreviations: "
            # + ", ".join(
            #     f"{DOMAIN_KEY_DICT.get(d, d)} = {d.replace('_', ' ').capitalize()}"
            #     for d in all_domains
            # ),
        )
        domains = [
            d
            for d in all_domains
            if st.checkbox(
                DOMAIN_KEY_DICT.get(d, d),
                value=True,
                help=d.replace("_", " ").capitalize(),
                key=f"domain_{d}",
            )
        ]

        all_cases = sorted(df.loc[df["domain"].isin(domains), "case_id"].unique())
        # Commenting for now, in case we want to filter by domain only, not by individual cases. Can be re-enabled if needed.
        # cases = st.multiselect("Cases", all_cases, default=all_cases)

        all_families = [f for f in FAMILY_ORDER if f in df["family"].unique()] + sorted(
            set(df["family"].unique()) - set(FAMILY_ORDER)
        )
        families = st.multiselect("Method families", all_families, default=all_families)

        all_methods = sorted(df.loc[df["family"].isin(families), "method"].unique())
        # Commenting for now, in case we want to filter by family only, not by individual methods. Can be re-enabled if needed.
        # methods = st.multiselect("Methods", all_methods, default=all_methods)

    # if not cases or not methods:
    #     st.warning("Select at least one case and one method to render the matrix.")
    #     return

    fig = render_matrix(df, metric, all_cases, all_methods)
    event = st.plotly_chart(
        fig, width="stretch", on_select="rerun", selection_mode="points", key="matrix_chart"
    )

    st.divider()
    points = (event.get("selection") or {}).get("points", []) if event else []
    if not points:
        st.caption("Click a cell in the matrix to see case notes and a forecast preview.")
        return

    point = points[0]
    case_id, method = point["y"], point["x"]
    st.subheader(f"{case_id} x {method}")

    col_chart, col_notes = st.columns([2, 1])
    with col_chart:
        st.caption(
            "Forecast preview — placeholder data. The eval pipeline doesn't persist "
            "per-origin forecasts yet, so this scaffolds the layout rather than showing "
            "real predictions."
        )
        st.plotly_chart(render_snapshots(case_id, method), width="stretch")
    with col_notes:
        st.caption("Case review notes")
        notes = load_case_notes(case_id)
        if notes:
            st.markdown(notes)
        else:
            st.info(f"No case_review notes found for `{case_id}`.")


if __name__ == "__main__":
    main()
