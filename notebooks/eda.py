import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from tsforecastbench.data.registry import discover_cases, get_case

    return discover_cases, get_case


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from statsmodels.graphics.tsaplots import plot_acf

    return go, make_subplots, pd, plot_acf, plt, px


@app.cell
def _(discover_cases, mo):
    cases = list(discover_cases())
    case_picker = mo.ui.dropdown(cases, value=cases[0], label="Case")
    mo.hstack([case_picker], justify="start")
    return (case_picker,)


@app.cell
def _(case_picker, get_case):
    case = get_case(case_picker.value)
    return (case,)


@app.cell
def _(case, mo):
    exo_parts = list(case.exogenous)
    if case.exogenous_known_future:
        exo_parts += [f"{v} (kf)" for v in case.exogenous_known_future]
    exo_display = ", ".join(exo_parts) if exo_parts else "none"
    source_link = f"[{case.source.name}]({case.source.url})" if case.source.url else case.source.name

    mo.vstack([
        mo.hstack([
            mo.stat(case.domain, label="Domain", bordered=True),
            mo.stat(f"{case.target} ({case.target_unit})", label="Target", bordered=True),
            mo.stat(case.freq, label="Frequency", bordered=True),
            mo.stat(case.horizon_primary, label="Horizon", bordered=True),
            mo.stat(exo_display, label="Exogenous (kf=known-future)", bordered=True),
        ], justify="start", gap=1),
        mo.md(f"> {case.notes}  \n"
              f"> **Source:** {source_link} · "
              f"**Splits ({case.split_mode}):** "
              f"train → {case.splits.train_end} · "
              f"val → {case.splits.val_end} · "
              f"test → {case.splits.test_end}"),
    ])
    return


@app.cell
def _(case):
    df = case.load()
    splits = case.split_frame(df)
    return df, splits


@app.cell
def _(case, go, mo, pd, splits):
    def _thin(part, max_pts=3000):
        if len(part) > max_pts:
            return part.iloc[:: len(part) // max_pts]
        return part

    fig_ts = go.Figure()
    for _part, _color, _name in [
        (_thin(splits["train"]), "#4C72B0", "train"),
        (_thin(splits["val"]),   "#DD8452", "val"),
        (_thin(splits["test"]),  "#55A868", "test"),
    ]:
        fig_ts.add_trace(go.Scatter(
            x=pd.to_datetime(_part["ds"]),
            y=_part["y"],
            mode="lines",
            name=_name,
            line=dict(color=_color, width=1),
        ))
    fig_ts.update_layout(
        title=f"{case.id} — full time series",
        xaxis_title="Date",
        yaxis_title=f"{case.target} [{case.target_unit}]",
        height=350,
        margin=dict(t=40, b=30, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    mo.ui.plotly(fig_ts)
    return


@app.cell
def _(df, mo, pd, px):
    y = pd.to_numeric(df["y"], errors="coerce")
    n_obs = int(y.notna().sum())
    span_days = (pd.to_datetime(df["ds"]).max() - pd.to_datetime(df["ds"]).min()).days

    _tmp = df[["ds"]].copy()
    _tmp["_year"] = pd.to_datetime(df["ds"]).dt.year
    _tmp["_y"] = y
    yearly = _tmp.groupby("_year")["_y"].apply(lambda s: s.notna().mean()).reset_index()
    yearly.columns = ["Year", "Completeness"]

    fig_qual = px.bar(
        yearly, x="Year", y="Completeness",
        title="Data completeness per year",
        labels={"Completeness": "Fraction observed"},
        color="Completeness",
        color_continuous_scale=["#d73027", "#fee08b", "#1a9850"],
        range_color=[0, 1],
        height=260,
    )
    fig_qual.update_layout(margin=dict(t=40, b=30), coloraxis_showscale=False)

    mo.vstack([
        mo.hstack([
            mo.stat(f"{len(df):,}", label="Total rows", bordered=True),
            mo.stat(f"{n_obs:,}", label="Observed", bordered=True),
            mo.stat(f"{y.notna().mean():.1%}", label="Overall completeness", bordered=True),
            mo.stat(f"{span_days:,}", label="Span (days)", bordered=True),
        ], justify="start", gap=1),
        mo.ui.plotly(fig_qual),
    ])
    return (y,)


@app.cell
def _(go, mo, pd, y):
    y_vals = y.dropna().values
    _q1 = float(pd.Series(y_vals).quantile(0.25))
    _med = float(pd.Series(y_vals).median())
    _q3 = float(pd.Series(y_vals).quantile(0.75))

    fig_dist = go.Figure(go.Histogram(
        x=y_vals, nbinsx=60,
        marker_color="#4C72B0", opacity=0.8, name="observed",
    ))
    fig_dist.update_layout(
        title="Target distribution",
        xaxis_title="value",
        yaxis_title="count",
        height=280,
        margin=dict(t=40, b=30),
        bargap=0.04,
    )

    mo.vstack([
        mo.hstack([
            mo.stat(f"{y_vals.mean():.4g}", label="Mean", bordered=True),
            mo.stat(f"{y_vals.std():.4g}", label="Std", bordered=True),
            mo.stat(f"{_q1:.4g}", label="Q25", bordered=True),
            mo.stat(f"{_med:.4g}", label="Median", bordered=True),
            mo.stat(f"{_q3:.4g}", label="Q75", bordered=True),
        ], justify="start", gap=1),
        mo.ui.plotly(fig_dist),
    ])
    return


@app.cell
def _(case, mo, plot_acf, plt, y):
    y_acf = y.dropna()
    y_acf = y_acf.iloc[-5000:] if len(y_acf) > 5000 else y_acf

    _freq_lags = {"15min": 96, "1h": 48, "1D": 30, "1W": 26}
    n_lags = _freq_lags.get(case.freq, 40)

    fig_acf, ax_acf = plt.subplots(figsize=(9, 3))
    plot_acf(y_acf, lags=n_lags, ax=ax_acf, alpha=0.05, zero=False)
    ax_acf.set_title(f"Autocorrelation — {case.target} (last {len(y_acf):,} obs, {n_lags} lags)")
    ax_acf.set_xlabel(f"Lag ({case.freq})")
    plt.tight_layout()
    mo.mpl.interactive(fig_acf)
    return


@app.cell
def _(case, df, go, make_subplots, mo, pd):
    exo_cols = list(dict.fromkeys([*case.exogenous, *case.exogenous_known_future]))

    if not exo_cols:
        mo.md("_No exogenous variables for this case._")
    else:
        _plot_df = df.copy()
        if len(_plot_df) > 6000:
            _plot_df = _plot_df.iloc[:: len(_plot_df) // 6000]

        _n = len(exo_cols)
        _fig = make_subplots(
            rows=_n + 1, cols=1, shared_xaxes=True,
            subplot_titles=[case.target] + exo_cols,
            vertical_spacing=0.05,
        )
        _fig.add_trace(go.Scatter(
            x=pd.to_datetime(_plot_df["ds"]), y=_plot_df["y"],
            mode="lines", name=case.target, line=dict(color="#333", width=0.8),
        ), row=1, col=1)

        _palette = ["#e6550d", "#31a354", "#3182bd", "#756bb1", "#636363"]
        for _i, _col in enumerate(exo_cols):
            if _col in _plot_df.columns:
                _fig.add_trace(go.Scatter(
                    x=pd.to_datetime(_plot_df["ds"]), y=_plot_df[_col],
                    mode="lines", name=_col,
                    line=dict(color=_palette[_i % len(_palette)], width=0.8),
                ), row=_i + 2, col=1)

        _fig.update_layout(
            height=220 * (_n + 1),
            title="Target vs exogenous variables",
            showlegend=False,
            margin=dict(t=60, b=30),
        )
        mo.ui.plotly(_fig)
    return


if __name__ == "__main__":
    app.run()
