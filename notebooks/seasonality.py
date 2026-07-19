import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.signal import periodogram
    from scipy.stats import binned_statistic

    from tsforecastbench.data.registry import discover_cases, get_case
    from tsforecastbench.data.schema import _observed_mask as observed_mask
    from tsforecastbench.data.seasonality import (
        CANDIDATE_MENU,
        detect_candidates,
        normalize_freq,
        proposed_seasonalities,
        robust_period_candidates,
    )
    from tsforecastbench.data.seasonality import (
        _acf_at as acf_at,
    )
    from tsforecastbench.data.seasonality import (
        _fill_and_detrend as fill_and_detrend,
    )
    from tsforecastbench.data.seasonality import (
        _mstl_decompose as mstl_decompose,
    )
    from tsforecastbench.data.seasonality import (
        _pacf as pacf_at_lags,
    )

    return (
        CANDIDATE_MENU,
        acf_at,
        binned_statistic,
        detect_candidates,
        discover_cases,
        fill_and_detrend,
        get_case,
        go,
        make_subplots,
        mstl_decompose,
        normalize_freq,
        np,
        observed_mask,
        pacf_at_lags,
        pd,
        periodogram,
        proposed_seasonalities,
        robust_period_candidates,
    )


@app.cell
def _(go):
    # Minimal look shared by every figure: white background, no grid, and all four
    # spines drawn in black (mirror=True closes the box). Template axis defaults
    # cascade to every axis, so make_subplots figures pick it up on all rows too.
    _axis = {
        "showgrid": False,
        "zeroline": False,
        "showline": True,
        "linecolor": "black",
        "mirror": True,
        "ticks": "outside",
    }
    plot_template = go.layout.Template(
        layout={"plot_bgcolor": "white", "paper_bgcolor": "white", "xaxis": _axis, "yaxis": _axis}
    )
    return (plot_template,)


@app.cell
def _(mo):
    mo.md("""
    # Seasonality curation

    One-time tooling for the `seasonalities` field in each case.toml — the **single
    source of truth** for seasonal periods during the benchmark train/predict cycle
    (periods in steps, dominant first, `[]` = non-seasonal).

    Candidates come from a fixed physically motivated menu per frequency; the evidence
    below (periodogram prominence, corroborated by ACF at lag and/or MSTL seasonal
    strength, computed on the **train split only**) accepts or rejects each. Review
    the evidence, then commit the proposed line — or deviate from it with a written
    justification in the case notes.
    """)
    return


@app.cell
def _(discover_cases, mo):
    cases = list(discover_cases())
    case_picker = mo.ui.dropdown(cases, value=cases[0], label="Case")
    extra_input = mo.ui.text(
        label="Extra candidate periods (steps, comma-separated; justify in case notes)",
        value="",
    )
    case_picker
    return case_picker, extra_input


@app.cell
def _(mo):
    mo.md("""
    ## RobustPeriod discovery (data-driven)

    Before the menu-based evidence below, this reads the seasonal structure **straight from the
    train split**, assuming nothing about the case — a robust, multi-period detector
    ([RobustPeriod](https://arxiv.org/abs/2002.09535), Wen et al. 2021). The series is
    detrended and Huber-cleaned (so a trend or a few spikes can't mislead it), split into dyadic
    frequency bands by a wavelet transform (so coexisting cycles separate), and each band's
    dominant period is scored by **spectral prominence** and its **robust amplitude** (the
    cycle's size in the target's own units).

    Every period found is listed, ranked by prominence — nothing is auto-accepted. `prominent`
    marks periods whose spectral ratio clears `prominence_bar` (the menu evidence's attention
    bar, raised with the width of the spectral peak window so a long series can't fake
    prominence out of noise); `harmonic_of` marks a
    shorter period that is just an integer fraction of a stronger one (e.g. 12 under 24, not a
    separate cycle); `approximate` marks a long period the window holds too few cycles of to
    pin down precisely; `menu_period` shows the nearest fixed-menu period for reference only.
    Use this to spot cycles the menu doesn't anticipate, then enter the ones worth studying in
    the final-candidate box further down.
    """)
    return


@app.cell
def _(case, mo, pd, robust_period_candidates, extra_input, train):
    robust_result = robust_period_candidates(train["y"].to_numpy(), case.freq)
    _candidates = robust_result["candidates"]
    if not _candidates:
        _discovery = mo.vstack(
            [mo.md("No periodic structure discovered (train split too short or flat)."), extra_input]
        )
    else:
        _df = pd.DataFrame(_candidates)[
            [
                "period",
                "spectral_ratio",
                "prominent",
                "harmonic_of",
                "approximate",
                "menu_period",
            ]
        ]
        _headline = [c["period"] for c in _candidates if c["prominent"] and not c["harmonic_of"]]
        _discovery = mo.vstack(
            [
                mo.ui.table(
                    _df.round(2),
                    selection=None,
                    label="RobustPeriod discovery — ranked by spectral prominence",
                ),
                mo.md(
                    f"**Prominent, non-harmonic periods:** `{_headline}`  \n"
                    f"**Menu periods not corroborated by discovery:** "
                    f"`{robust_result['menu_periods_missing']}`"
                ),
                extra_input,
            ]
        )
    _discovery
    return (robust_result,)


@app.cell
def _(mo):
    # Resolution of the uniform log-frequency grid the periodogram is resampled onto (see
    # the periodogram cell). This is the single smoothing knob: fewer bins = coarser grid =
    # only major peaks survive; more bins = finer grid = more structure.
    grid_resolution = mo.ui.slider(
        start=50,
        stop=350,
        step=50,
        value=150,
        label="Log-frequency grid resolution M (bins; higher = finer)",
        full_width=False,
    )
    return (grid_resolution,)


@app.cell
def _(
    CANDIDATE_MENU,
    binned_statistic,
    case,
    extra_candidates,
    fill_and_detrend,
    go,
    grid_resolution,
    mo,
    normalize_freq,
    pd,
    np,
    periodogram,
    plot_template,
    robust_result,
    train,
):
    _yd = fill_and_detrend(train["y"].to_numpy())
    _freqs, _power = periodogram(_yd, fs=1.0)
    _f, _p = _freqs[1:], _power[1:]  # drop the DC bin (frequency 0)

    # The periodogram is uniform in frequency, but we reason about periods on a log axis:
    # it is wildly over-resolved at short periods (noise clutter) and under-resolved at long
    # ones. Resample log-power (floor zeros so log10 stays finite) onto a uniform
    # log-frequency grid, taking the MAX per bin so sharp seasonal lines survive the
    # down-sampling (a mean averages them into the noise floor). Empty low-frequency bins —
    # where the grid is finer than the raw spectrum — are filled by interpolation. The result
    # is the smoothed curve plotted alongside the raw periodogram.
    _pos = _p[_p > 0]
    _floor = _pos.min() if _pos.size else 1e-12
    _logp = np.log10(np.maximum(_p, _floor))
    _M = int(grid_resolution.value)
    _lf = np.log10(_f)
    _edges = np.linspace(_lf.min(), _lf.max(), _M + 1)
    _centers = 0.5 * (_edges[:-1] + _edges[1:])
    _binned, _, _ = binned_statistic(_lf, _logp, statistic="max", bins=_edges)
    _empty = ~np.isfinite(_binned)
    if _empty.any():
        _binned[_empty] = np.interp(_centers[_empty], _lf, _logp)
    _grid_f = 10.0**_centers  # log-spaced frequency grid, used only for the smoothed curve

    _menu = sorted(CANDIDATE_MENU.get(normalize_freq(case.freq), ()))

    _fig = go.Figure()
    _fig.add_trace(go.Scattergl(x=_f, y=_p, mode="lines", name="periodogram", opacity=0.35))
    _fig.add_trace(
        go.Scattergl(
            x=_grid_f,
            y=10.0**_binned,
            mode="lines",
            name=f"log-resampled max (M={_M})",
            line={"width": 2.0, "color": "#1f77b4"},
        )
    )
    # Draw each marker as a vertical trace (not add_vline, whose shapes can't be hovered) so
    # hovering anywhere along the line reports its period in steps. The line spans the full
    # data range in y; the period label is kept as a paper-anchored annotation. On a log x
    # axis traces take raw frequency but annotations take log10(frequency).
    _yv = np.concatenate([_p[_p > 0], (10.0**_binned)[np.isfinite(_binned)]])
    _yline = np.geomspace(float(_yv.min()), float(_yv.max()), 50)

    def _vline(_freq, _period, _color, _dash, _pos):

        _hover_tooltip_value = np.round(
            (pd.Timedelta(case.freq) * _period).total_seconds() / 3600 / 24, 1
        )
        _fig.add_trace(
            go.Scattergl(
                x=np.full(_yline.size, _freq),
                y=_yline,
                mode="lines",
                line={"color": _color, "dash": _dash},
                showlegend=False,
                # hovertemplate=f"{_period} steps<extra></extra>",
                hovertemplate=f"{_hover_tooltip_value} days<extra></extra>",
            )
        )
        _fig.add_annotation(
            x=np.log10(_freq),
            y=1.0 if _pos == "top" else 0.0,
            yref="paper",
            yanchor="bottom" if _pos == "top" else "top",
            text=f"{_period}",
            showarrow=False,
            font={"color": _color, "size": 10},
        )

    for _period in _menu:  # grey — fixed physically-motivated candidate menu
        _vline(1.0 / _period, _period, "gray", "dot", "top")
    for _c in robust_result["candidates"]:  # gold — RobustPeriod discovery
        _dash = "solid" if _c["prominent"] else "dot"
        _vline(1.0 / _c["period"], _c["period"], "gold", _dash, "bottom")
    for _period in extra_candidates:  # orange — manually entered extra candidates
        _vline(1.0 / _period, _period, "darkorange", "dash", "top")
    _fig.update_xaxes(type="log", title="frequency (1/steps)")
    _fig.update_yaxes(type="log", title="power")
    _fig.update_layout(
        template=plot_template, title="Train-split periodogram", height=400, showlegend=True
    )

    _info = mo.callout(
        mo.md(
            "**Grey lines — the fixed candidate menu.** These are physically-motivated "
            "seasonal periods for this sampling frequency (semi-diurnal / diurnal / weekly "
            "for sub-daily data; weekly / annual for daily data). Detection only ever "
            "accepts periods from this fixed menu (plus justified per-case extras), so "
            "ordinary spectral noise can never mint a meaningless period into the benchmark "
            "metadata.  \n"
            "**Yellow lines** are periods the RobustPeriod discovery cell above found (solid = "
            "clears its prominence bar, dotted = does not); "
            "**orange lines** are the extra candidates typed above. A yellow line sitting on "
            "a grey line corroborates a menu period; a yellow line with no grey line nearby "
            "flags an off-menu cycle worth adding as an extra candidate.  \n"
            "⚠️ *Long-period peaks are approximate*: a finite-length periodogram cannot "
            "resolve low frequencies finely, so an annual cycle may read as ~360-375. Treat a "
            "long-period peak as a neighbourhood, not an exact period."
        ),
        kind="info",
    )
    mo.vstack(
        [
            grid_resolution,
            _fig,
            mo.accordion({"Plot info": _info}),
        ]
    )
    return


@app.cell
def _(case_picker, extra_input, get_case):
    case = get_case(case_picker.value)
    extra_candidates = tuple(
        int(tok) for tok in extra_input.value.replace(" ", "").split(",") if tok
    )
    return case, extra_candidates


@app.cell
def _(case, observed_mask):
    _df = case.load(curated=True)
    train = case.split_frame(_df)["train"]
    filled_frac = 1.0 - float(observed_mask(train).mean())
    return filled_frac, train


@app.cell
def _(
    case,
    detect_candidates,
    extra_candidates,
    final_candidates,
    proposed_seasonalities,
    train,
):
    evidence = detect_candidates(
        train["y"].to_numpy(), case.freq, tuple(set(extra_candidates) | set(final_candidates))
    )
    if final_candidates:
        evidence = [e for e in evidence if e["period"] in final_candidates]
    proposed = proposed_seasonalities(evidence)
    return evidence, proposed


@app.cell
def _(mo):
    mo.md("""## Final candidate study
            Insert below your chosen period candidates (steps, comma-separated)
          """)
    return


@app.cell
def _(mo):
    final_input = mo.ui.text(
        value="",
    )
    final_input
    return (final_input,)


@app.cell
def _(final_input):
    final_candidates = tuple(
        int(tok) for tok in final_input.value.replace(" ", "").split(",") if tok
    )
    return (final_candidates,)


@app.cell
def _(case, evidence, filled_frac, mo, pd, proposed):
    declared = getattr(case, "seasonalities", None)
    _table = pd.DataFrame(evidence)[
        ["period", "n_cycles", "power_ratio", "acf", "mstl_strength", "accepted"]
    ]

    _fill_note = (
        f"⚠️ {filled_frac:.1%} of the train split is gap-filled — filled stretches are "
        "interpolations, so the evidence is correspondingly weaker."
        if filled_frac > 0.01
        else f"{filled_frac:.2%} of the train split is gap-filled."
    )
    mo.vstack(
        [
            mo.md(f"**{case.id}** ({case.freq}) — {_fill_note}"),
            mo.ui.table(_table.round(3), selection=None, label="Candidate evidence"),
            mo.md(
                f"Proposed (dominant first): `seasonalities = {proposed}`  \n"
                f"Currently declared: `{declared if declared is not None else 'not declared yet'}`"
            ),
        ]
    )
    return


@app.cell
def _(
    acf_at,
    fill_and_detrend,
    final_candidates,
    go,
    np,
    plot_template,
    train,
):
    _yd = fill_and_detrend(train["y"].to_numpy())
    _menu = sorted(final_candidates)
    _max_lag = min(int(2.2 * max(_menu)) if _menu else 100, len(_yd) - 2)
    _lags = np.arange(1, _max_lag + 1)
    _acf = np.array([acf_at(_yd, int(lag)) for lag in _lags])

    _fig = go.Figure()
    _fig.add_trace(go.Scattergl(x=_lags, y=_acf, mode="lines", name="ACF"))
    _sig = 2.0 / np.sqrt(len(_yd))
    _fig.add_hline(y=_sig, line_dash="dash", line_color="gray")
    _fig.add_hline(y=-_sig, line_dash="dash", line_color="gray")
    for _p in _menu:
        if _p <= _max_lag:
            _fig.add_vline(
                x=_p,
                line_dash="dot",
                line_color="crimson",
                annotation_text=f"{_p}",
                annotation_position="top",
            )
    _fig.update_layout(
        template=plot_template,
        title="Train-split ACF (candidate lags marked)",
        xaxis_title="lag (steps)",
        yaxis_title="ACF",
        height=380,
    )
    _fig
    return


@app.cell
def _(fill_and_detrend, final_candidates, go, mo, np, pacf_at_lags, plot_template, train):
    _yd = fill_and_detrend(train["y"].to_numpy())
    _menu = sorted(final_candidates)
    _max_lag = min(int(2.2 * max(_menu)) if _menu else 100, len(_yd) // 2 - 1)
    _lags = np.arange(1, _max_lag + 1)
    _pacf = pacf_at_lags(_yd, _max_lag)

    _fig = go.Figure()
    _fig.add_trace(go.Scattergl(x=_lags, y=_pacf, mode="lines", name="PACF"))
    _sig = 2.0 / np.sqrt(len(_yd))
    _fig.add_hline(y=_sig, line_dash="dash", line_color="gray")
    _fig.add_hline(y=-_sig, line_dash="dash", line_color="gray")
    for _p in _menu:
        if _p <= _max_lag:
            _fig.add_vline(
                x=_p,
                line_dash="dot",
                line_color="crimson",
                annotation_text=f"{_p}",
                annotation_position="top",
            )
    _fig.update_layout(
        template=plot_template,
        title="Train-split PACF (candidate lags marked)",
        xaxis_title="lag (steps)",
        yaxis_title="PACF",
        height=380,
    )
    _note = mo.md(
        "Unlike ACF, each lag here nets out the shorter lags already accounted for. A "
        "candidate that's an exact multiple of a shorter, more prominent one already on "
        "the menu (e.g. 24 = 2x12 on hourly data) should only spike here if it carries "
        "seasonal information beyond what the shorter cycle already explains — otherwise "
        "the ACF peak at that lag is just the shorter cycle's own persistence bleeding "
        "through, not evidence of an independent period."
    )
    mo.accordion({"PACF": mo.vstack([_fig, _note])})
    return


@app.cell
def _(final_candidates, go, make_subplots, mo, mstl_decompose, np, plot_template, train):
    _periods = sorted(final_candidates)
    _fit = mstl_decompose(train["y"].to_numpy(), _periods) if _periods else None

    if _fit is None:
        _out = mo.md(
            "No final candidates entered above, or none fits >= 2 cycles within the "
            "sampled `MSTL_SAMPLE_ROWS` window for an MSTL decomposition."
        )
    else:
        _decomposition, _fitted = _fit
        _seasonal = np.asarray(_decomposition.seasonal)
        if _seasonal.ndim == 1:
            _seasonal = _seasonal[:, None]
        _series = (
            [("trend", np.asarray(_decomposition.trend))]
            + [(f"seasonal({_p})", _seasonal[:, _i]) for _i, _p in enumerate(_fitted)]
            + [("remainder", np.asarray(_decomposition.resid))]
        )
        _fig = make_subplots(
            rows=len(_series),
            cols=1,
            shared_xaxes=True,
            subplot_titles=[_label for _label, _ in _series],
            vertical_spacing=0.35 / len(_series),
        )
        for _row, (_label, _values) in enumerate(_series, start=1):
            _fig.add_trace(
                go.Scattergl(y=_values, mode="lines", showlegend=False),
                row=_row,
                col=1,
            )
        _fig.update_xaxes(title_text="step (within sampled window)", row=len(_series), col=1)
        _fig.update_layout(
            template=plot_template,
            title=f"MSTL decomposition (last {len(_series[0][1])} train steps, periods {_fitted})",
            height=160 * len(_series) + 100,
        )
        _out = _fig
    _out
    return


if __name__ == "__main__":
    app.run()
