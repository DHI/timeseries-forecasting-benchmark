import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", title="Forecast results explorer")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import sys
    import pathlib
    _root = pathlib.Path(__file__).parents[1]
    sys.path.insert(0, str(_root / "src"))
    CACHE_DIR = _root / "notebooks" / ".cache"
    CACHE_DIR.mkdir(exist_ok=True)
    from tsforecastbench.data.registry import discover_cases, get_case
    return CACHE_DIR, _root, discover_cases, get_case


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots, np, pd


@app.cell
def _(discover_cases, mo):
    cases = list(discover_cases())
    case_picker = mo.ui.dropdown(cases, value=cases[0], label="Case")
    origins_slider = mo.ui.slider(
        start=3, stop=20, step=1, value=6,
        label="Rolling origins (more = slower, more reliable)",
    )
    mo.hstack([case_picker, origins_slider], justify="start", gap=2)
    return case_picker, cases, origins_slider


@app.cell
def _(case_picker, get_case):
    case = get_case(case_picker.value)
    return (case,)


@app.cell
def _(CACHE_DIR, case, mo, origins_slider, pd):
    from tsforecastbench.bench.eval import run_case_rolling
    from tsforecastbench.bench.methods import (
        Persistence, SeasonalNaive, ThetaForecaster, DLinearForecaster,
    )

    _n = origins_slider.value
    _cache = CACHE_DIR / f"{case.id}_{_n}origins.parquet"
    _methods = [Persistence(), SeasonalNaive(), ThetaForecaster(), DLinearForecaster()]
    try:
        from tsforecastbench.bench.methods import ChronosForecaster as _CF
        _methods.append(_CF())
    except Exception:
        pass

    scores = None
    _src = "cached"
    if _cache.exists():
        scores = pd.read_parquet(_cache)
    else:
        with mo.status.spinner(title=f"Running {_n} rolling origins for {case.id} …"):
            scores = run_case_rolling(case, _methods, n_origins=_n)
        scores.to_parquet(_cache, index=False)
        _src = "computed"

    return (
        DLinearForecaster,
        Persistence,
        SeasonalNaive,
        ThetaForecaster,
        scores,
    )


@app.cell
def _(mo, scores):
    _wide = scores.pivot_table(index="method", columns="metric", values="value")
    _keep = [c for c in ["MASE", "Skill_vs_Persistence", "Skill_vs_Mean", "MAE", "RMSE"] if c in _wide.columns]
    _wide = _wide[_keep].sort_values("Skill_vs_Persistence", ascending=False)
    _fmt = _wide.copy()
    for _col in _fmt.columns:
        _fmt[_col] = _fmt[_col].map(lambda v: f"{v:+.3f}" if "Skill" in _col else f"{v:.3f}")  # noqa: B023
    _fmt = _fmt.reset_index()
    mo.vstack([
        mo.md("## Scoreboard"),
        mo.md("_Sorted by skill vs persistence. Positive = beats persistence baseline._"),
        mo.ui.table(_fmt, selection=None),
    ])
    return


@app.cell
def _(go, mo, scores):
    _skill = (
        scores[scores["metric"] == "Skill_vs_Persistence"]
        .set_index("method")["value"]
        .sort_values()
    )
    fig_bar = go.Figure(go.Bar(
        x=_skill.values,
        y=_skill.index.tolist(),
        orientation="h",
        marker_color=["#d73027" if v < 0 else "#1a9850" for v in _skill.values],
        text=[f"{v:+.3f}" for v in _skill.values],
        textposition="outside",
    ))
    fig_bar.add_vline(x=0, line_color="black", line_width=1)
    fig_bar.update_layout(
        title="Skill vs persistence (higher = better)",
        xaxis_title="Skill score",
        height=max(280, 44 * len(_skill) + 80),
        margin=dict(t=40, b=30, l=130, r=70),
    )
    mo.vstack([mo.md("## Skill ranking"), mo.ui.plotly(fig_bar)])
    return fig_bar,


@app.cell
def _(case, mo, np, pd):
    from tsforecastbench.bench.eval.loop import _complete_for_fit, _horizon_to_steps

    df_full = case.load().sort_values("ds").reset_index(drop=True)
    splits_ex = case.split_frame(df_full)
    train_val_ex = pd.concat([splits_ex["train"], splits_ex["val"]], ignore_index=True)
    full = pd.concat([train_val_ex, splits_ex["test"]], ignore_index=True)
    exo = list(dict.fromkeys([*case.exogenous, *case.exogenous_known_future]))
    full_fit = _complete_for_fit(full, ["y", *exo])
    raw_y = full["y"].to_numpy(dtype=float)
    H = _horizon_to_steps(case.horizon_primary, case.freq)
    test_start = len(train_val_ex)

    _step_ex = max(1, (len(full) - test_start - H) // 12)
    origins_ex = [
        test_start + i * _step_ex
        for i in range(12)
        if test_start + i * _step_ex + H <= len(full)
    ]
    origin_picker = mo.ui.slider(
        start=1, stop=len(origins_ex), step=1, value=1,
        label="Test origin",
    )
    mo.vstack([mo.md("## Example forecast"), origin_picker])
    return H, exo, full, full_fit, np, origin_picker, origins_ex, raw_y, test_start


@app.cell
def _(
    DLinearForecaster,
    H,
    Persistence,
    SeasonalNaive,
    ThetaForecaster,
    case,
    exo,
    full,
    full_fit,
    go,
    mo,
    origin_picker,
    origins_ex,
    pd,
    raw_y,
):
    o = origins_ex[origin_picker.value - 1]
    ctx = min(3 * H, o)
    hist_ex = full_fit.iloc[:o]
    fexog_ex = (
        full_fit.iloc[o: o + H][["ds", *[c for c in exo if c in full_fit.columns]]]
        if exo else None
    )

    _methods_ex = [Persistence(), SeasonalNaive(), ThetaForecaster(), DLinearForecaster()]
    try:
        from tsforecastbench.bench.methods import ChronosForecaster as _CF2
        _methods_ex.append(_CF2())
    except Exception:
        pass

    fig_ex = go.Figure()
    fig_ex.add_trace(go.Scatter(
        x=pd.to_datetime(full["ds"].iloc[o - ctx: o]),
        y=raw_y[o - ctx: o],
        mode="lines", name="observed (history)",
        line=dict(color="#aaa", width=1, dash="dot"),
    ))
    fig_ex.add_trace(go.Scatter(
        x=pd.to_datetime(full["ds"].iloc[o: o + H]),
        y=raw_y[o: o + H],
        mode="lines", name="observed (truth)",
        line=dict(color="#111", width=2.5),
    ))
    _palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    for _mi, _m in enumerate(_methods_ex):
        try:
            _m.fit(case, hist_ex)
            _yh = _m.predict(case, hist_ex, H, future_exog=fexog_ex)["y_hat"].to_numpy(float)
            fig_ex.add_trace(go.Scatter(
                x=pd.to_datetime(full["ds"].iloc[o: o + H]),
                y=_yh, mode="lines", name=_m.name,
                line=dict(color=_palette[_mi % len(_palette)], width=1.5),
            ))
        except Exception:
            pass

    fig_ex.add_vline(
        x=str(pd.to_datetime(full["ds"].iloc[o])),
        line_color="grey", line_dash="dash", line_width=1,
    )
    fig_ex.update_layout(
        title=f"{case.id}: {case.horizon_primary} forecast from {pd.to_datetime(full['ds'].iloc[o]).date()}",
        xaxis_title="Date",
        yaxis_title=f"{case.target} [{case.target_unit}]",
        height=380,
        margin=dict(t=50, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    mo.ui.plotly(fig_ex)
    return ctx, fig_ex, fexog_ex, hist_ex, o


@app.cell
def _(
    H,
    Persistence,
    ThetaForecaster,
    DLinearForecaster,
    case,
    exo,
    full,
    full_fit,
    go,
    make_subplots,
    mo,
    np,
    pd,
    raw_y,
    scores,
    test_start,
):
    _skill_col = scores[scores["metric"] == "Skill_vs_Persistence"].set_index("method")["value"]
    _worst = _skill_col.idxmin()

    _step_d = max(1, H // 2)
    _candidates = list(range(test_start, len(full) - H, _step_d))[:32]
    _errors = []
    _m_p = Persistence()
    for _o in _candidates:
        _hist_d = full_fit.iloc[:_o]
        try:
            _m_p.fit(case, _hist_d)
            _yh_p = _m_p.predict(case, _hist_d, H)["y_hat"].to_numpy(float)
            _errors.append((_o, float(np.sqrt(np.nanmean((raw_y[_o: _o + H] - _yh_p) ** 2)))))
        except Exception:
            _errors.append((_o, 0.0))

    _errors.sort(key=lambda x: -x[1])
    _hard = [_o for _o, _ in _errors[:4]]

    _exo_d = list(dict.fromkeys([*case.exogenous, *case.exogenous_known_future]))
    _method_objs_d = [Persistence()]
    try:
        _method_objs_d += [ThetaForecaster(), DLinearForecaster()]
    except Exception:
        pass

    _palette_d = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    _n_w = len(_hard)
    fig_diag = make_subplots(
        rows=1, cols=_n_w, shared_yaxes=True,
        subplot_titles=[str(pd.to_datetime(full["ds"].iloc[_o]).date()) for _o in _hard],
    )
    for _wi, _o in enumerate(_hard):
        _ctx_d = min(2 * H, _o)
        _hist_d2 = full_fit.iloc[:_o]
        _fexog_d = (
            full_fit.iloc[_o: _o + H][["ds", *[c for c in _exo_d if c in full_fit.columns]]]
            if _exo_d else None
        )
        fig_diag.add_trace(go.Scatter(
            x=list(range(-_ctx_d, 0)), y=raw_y[_o - _ctx_d: _o].tolist(),
            mode="lines", name="history", line=dict(color="#ccc", width=0.8),
            showlegend=(_wi == 0),
        ), row=1, col=_wi + 1)
        fig_diag.add_trace(go.Scatter(
            x=list(range(H)), y=raw_y[_o: _o + H].tolist(),
            mode="lines", name="truth", line=dict(color="#111", width=2),
            showlegend=(_wi == 0),
        ), row=1, col=_wi + 1)
        for _mi, _m_d in enumerate(_method_objs_d):
            try:
                _m_d.fit(case, _hist_d2)
                _yh_d = _m_d.predict(case, _hist_d2, H, future_exog=_fexog_d)["y_hat"].to_numpy(float)
                fig_diag.add_trace(go.Scatter(
                    x=list(range(H)), y=_yh_d.tolist(),
                    mode="lines", name=_m_d.name,
                    line=dict(color=_palette_d[_mi % len(_palette_d)], width=1.2),
                    showlegend=(_wi == 0),
                ), row=1, col=_wi + 1)
            except Exception:
                pass

    fig_diag.update_layout(
        height=320,
        title=f"4 hardest test windows — worst method: {_worst} (skill {_skill_col[_worst]:+.2f})",
        margin=dict(t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.1, x=0),
    )
    mo.vstack([
        mo.md("## Diagnostic: hardest forecast windows"),
        mo.md(f"_Showing 4 test origins with highest Persistence RMSE (= hardest to forecast). "
              f"Worst overall method: **{_worst}** (skill {_skill_col[_worst]:+.2f})._"),
        mo.ui.plotly(fig_diag),
    ])
    return fig_diag,


if __name__ == "__main__":
    app.run()
