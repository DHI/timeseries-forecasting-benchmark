import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import io

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.signal import periodogram
    from scipy.stats import binned_statistic

    from tsforecastbench.data.curate import ACF_THRESHOLD
    from tsforecastbench.data.curate import _fill_small_gaps as fill_small_gaps
    from tsforecastbench.data.curate import _nan_runs as contiguous_runs
    from tsforecastbench.data.seasonality import (
        CANDIDATE_MENU,
        detect_candidates,
        fill_and_detrend,
        normalize_freq,
        proposed_seasonalities,
        robust_period_candidates,
    )
    from tsforecastbench.data.seasonality import (
        _acf_at as acf_at,
    )
    from tsforecastbench.data.seasonality import (
        _mstl_decompose as mstl_decompose,
    )
    from tsforecastbench.data.seasonality import (
        _pacf as pacf_at_lags,
    )

    return (
        ACF_THRESHOLD,
        CANDIDATE_MENU,
        acf_at,
        binned_statistic,
        contiguous_runs,
        detect_candidates,
        fill_and_detrend,
        fill_small_gaps,
        go,
        io,
        make_subplots,
        mstl_decompose,
        normalize_freq,
        np,
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
    # Seasonality exploration

    Upload a timeseries and find its seasonal periods. Candidates come from a fixed
    physically motivated menu per sampling frequency; the evidence below (periodogram
    prominence, corroborated by ACF at lag and/or MSTL seasonal strength) accepts or
    rejects each — computed on a **gapless slice** you select just below the upload.
    Review the evidence, then read off the proposed periods, or add extras you spot
    yourself.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Upload a series

    Upload a CSV or Parquet file with a timestamp column and a numeric column to study
    (a file with several numeric columns is fine — you'll pick which one below). Every
    other section of this notebook waits until a gapless slice is selected here.
    """)
    return


@app.cell
def _(mo):
    file_uploader = mo.ui.file(
        filetypes=[".csv", ".parquet"],
        kind="area",
        label="Upload a timeseries (CSV or Parquet)",
    )
    file_uploader
    return (file_uploader,)


@app.cell
def _(file_uploader, io, mo, pd):
    mo.stop(not file_uploader.value, mo.md("Upload a CSV or Parquet file above to begin."))
    _upload = file_uploader.value[0]
    series_label = _upload.name

    def _read_csv_bytes(contents):
        # Mirrors the encoding/separator sniff in data/cases/_io.py's read_table, adapted
        # to bytes: try common encodings and separators, keep the first parse that yields
        # more than one column, and fall back to latin-1 (decodes any byte) as a last resort.
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            for sep in (",", ";", "\t"):
                try:
                    _df = pd.read_csv(io.BytesIO(contents), sep=sep, encoding=enc)
                    if _df.shape[1] > 1:
                        return _df
                except Exception:
                    continue
        return pd.read_csv(io.BytesIO(contents), encoding="latin-1")

    if series_label.lower().endswith(".parquet"):
        raw_df = pd.read_parquet(io.BytesIO(_upload.contents))
    else:
        raw_df = _read_csv_bytes(_upload.contents)

    if isinstance(raw_df.index, pd.DatetimeIndex):
        if raw_df.index.name is None:
            raw_df.index.name = "index"
        raw_df = raw_df.reset_index()
    return raw_df, series_label


@app.cell
def _(mo, pd, raw_df):
    _cols = list(raw_df.columns)

    def _guess_ts_column():
        for _c in _cols:
            if pd.api.types.is_datetime64_any_dtype(raw_df[_c]):
                return _c
        _common_names = {"ds", "date", "time", "timestamp", "datetime"}
        for _c in _cols:
            if str(_c).lower() in _common_names:
                return _c
        _best, _best_rate = _cols[0], 0.0
        for _c in _cols:
            _rate = pd.to_datetime(raw_df[_c], errors="coerce").notna().mean()
            if _rate > _best_rate:
                _best, _best_rate = _c, _rate
        return _best

    ts_picker = mo.ui.dropdown(_cols, value=_guess_ts_column(), label="Timestamp column")
    ts_picker
    return (ts_picker,)


@app.cell
def _(mo, pd, raw_df, ts_picker):
    _candidates = [
        c
        for c in raw_df.columns
        if c != ts_picker.value and pd.api.types.is_numeric_dtype(raw_df[c])
    ] or [c for c in raw_df.columns if c != ts_picker.value]
    value_picker = mo.ui.dropdown(
        _candidates, value=_candidates[0] if _candidates else None, label="Variable of interest"
    )
    value_picker
    return (value_picker,)


@app.cell
def _(mo, pd, raw_df, ts_picker):
    mo.stop(ts_picker.value is None, mo.md("Select a timestamp column above."))
    _ts = pd.to_datetime(raw_df[ts_picker.value], errors="coerce").dropna().sort_values()
    _ts = _ts[~_ts.duplicated(keep="first")]
    _diffs = _ts.diff().dropna()
    mo.stop(len(_diffs) < 2, mo.md("Not enough distinct timestamps to infer a sampling frequency."))
    _step = _diffs.mode().iloc[0]
    _freq_guess = pd.tseries.frequencies.to_offset(_step).freqstr
    freq_input = mo.ui.text(
        value=_freq_guess,
        label=(
            "Sampling frequency (pandas offset alias, e.g. '1h', '15min', '1D') — "
            "edit if the guess looks wrong"
        ),
    )
    freq_input
    return (freq_input,)


@app.cell
def _(freq_input, mo, pd, raw_df, ts_picker, value_picker):
    mo.stop(not freq_input.value, mo.md("Enter a sampling frequency above."))
    _ts = pd.to_datetime(raw_df[ts_picker.value], errors="coerce")
    _series = pd.Series(raw_df[value_picker.value].to_numpy(dtype=float), index=_ts)
    _series = _series[_series.index.notna()].sort_index()
    _series = _series[~_series.index.duplicated(keep="first")]
    _grid = pd.date_range(_series.index.min(), _series.index.max(), freq=freq_input.value)
    full_series = _series.reindex(_grid)
    observed = full_series.notna().to_numpy()
    _n_missing = int((~observed).sum())
    gap_note = mo.md(
        f"⚠️ {_n_missing} of {len(full_series)} steps ({_n_missing / len(full_series):.1%}) "
        "are missing on the inferred grid — some stretches have gaps."
        if _n_missing
        else f"No gaps on the inferred grid ({len(full_series)} steps)."
    )
    return full_series, gap_note, observed


@app.cell
def _(contiguous_runs, mo, observed):
    _runs = list(contiguous_runs(observed))
    mo.stop(not _runs, mo.md("No gapless stretch found in the uploaded series."))
    longest_run = max(_runs, key=lambda r: r[1] - r[0])
    return (longest_run,)


@app.cell
def _(ACF_THRESHOLD, acf_at, full_series, longest_run, mo, np):
    # n is the lag at which the longest gapless span's own ACF first drops below
    # ACF_THRESHOLD — the same "still informative" cutoff the production curation
    # pipeline uses (gap_cap() in data/curate.py), computed here on this slice instead
    # of the whole raw series.
    _start, _stop = longest_run
    mo.stop(
        _stop - _start < 3,
        mo.md("Longest gapless span is too short to estimate an ACF-based gap cap."),
    )
    _y = full_series.to_numpy()[_start:_stop]
    _max_lag = len(_y) - 2
    gap_cap_n = _max_lag
    for _lag in range(1, _max_lag + 1):
        _r = acf_at(_y, _lag)
        if not np.isfinite(_r) or _r < ACF_THRESHOLD:
            gap_cap_n = _lag
            break
    return (gap_cap_n,)


@app.cell
def _(ACF_THRESHOLD, fill_small_gaps, full_series, gap_cap_n, mo, observed):
    filled_series = fill_small_gaps(full_series, gap_cap_n)
    filled_observed = filled_series.notna().to_numpy()
    _n_missing = int((~observed).sum())
    _n_filled = int(filled_observed.sum() - observed.sum())
    fill_note = mo.md(
        f"Filled {_n_filled} of {_n_missing} missing steps by linear interpolation "
        f"(gaps ≤ {gap_cap_n} steps, the lag where the longest gapless span's ACF first "
        f"drops below {ACF_THRESHOLD:.1f}); longer gaps remain missing."
        if _n_missing
        else "No gaps to fill."
    )
    return filled_observed, filled_series, fill_note


@app.cell
def _(contiguous_runs, fill_note, filled_observed, filled_series, gap_note, mo):
    _runs = sorted(contiguous_runs(filled_observed), key=lambda r: r[1] - r[0], reverse=True)
    mo.stop(not _runs, mo.md("No gapless stretch found in the uploaded series."))
    _shown = _runs[:15]
    run_options = {
        f"{_b - _a} steps  ({filled_series.index[_a]} → {filled_series.index[_b - 1]})": (_a, _b)
        for _a, _b in _shown
    }
    _more = f"  \n_{len(_runs) - 15} shorter run(s) not shown._" if len(_runs) > 15 else ""
    slice_picker = mo.ui.dropdown(
        list(run_options.keys()),
        value=next(iter(run_options)),
        label="Gapless slice to study (default: longest)",
    )
    mo.vstack(
        [gap_note, fill_note, mo.md(f"{len(_runs)} gapless run(s) found.{_more}"), slice_picker]
    )
    return run_options, slice_picker


@app.cell
def _(filled_series, go, mo, plot_template, run_options, slice_picker):
    _start, _stop = run_options[slice_picker.value]
    _fig = go.Figure()
    _fig.add_trace(
        go.Scattergl(
            x=filled_series.index,
            y=filled_series.to_numpy(),
            mode="lines",
            name="uploaded series",
        )
    )
    _fig.add_vrect(
        x0=filled_series.index[_start],
        x1=filled_series.index[_stop - 1],
        fillcolor="crimson",
        opacity=0.15,
        line_width=0,
        annotation_text="selected slice",
        annotation_position="top left",
    )
    _fig.update_layout(
        template=plot_template,
        title="Uploaded series, small gaps filled (remaining gaps break the line; shaded = "
        "selected slice)",
        height=320,
    )
    mo.vstack([_fig])
    return


@app.cell
def _(filled_series, freq_input, mo, run_options, slice_picker):
    _start, _stop = run_options[slice_picker.value]
    mo.stop(
        _stop - _start < 3, mo.md("Selected slice is too short to study (need at least 3 points).")
    )
    y = filled_series.to_numpy()[_start:_stop]
    slice_index = filled_series.index[_start:_stop]
    freq = freq_input.value
    return freq, slice_index, y


@app.cell
def _(mo):
    extra_input = mo.ui.text(
        label="Extra candidate periods (steps, comma-separated)",
        value="",
    )
    extra_input
    return (extra_input,)


@app.cell
def _(extra_input):
    extra_candidates = tuple(
        int(tok) for tok in extra_input.value.replace(" ", "").split(",") if tok
    )
    return (extra_candidates,)


@app.cell
def _(mo):
    mo.md("""
    ## RobustPeriod discovery (data-driven)

    Before the menu-based evidence below, this reads the seasonal structure **straight from the
    selected slice**, assuming nothing about the series — a robust, multi-period detector
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
def _(extra_input, freq, mo, pd, robust_period_candidates, y):
    robust_result = robust_period_candidates(y, freq)
    _candidates = robust_result["candidates"]
    if not _candidates:
        _discovery = mo.vstack(
            [
                mo.md("No periodic structure discovered (selected slice too short or flat)."),
                extra_input,
            ]
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
    extra_candidates,
    fill_and_detrend,
    freq,
    go,
    grid_resolution,
    mo,
    normalize_freq,
    pd,
    np,
    periodogram,
    plot_template,
    robust_result,
    y,
):
    _yd = fill_and_detrend(y)
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

    _menu = sorted(CANDIDATE_MENU.get(normalize_freq(freq), ()))

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
            (pd.Timedelta(normalize_freq(freq)) * _period).total_seconds() / 3600 / 24, 1
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
        template=plot_template, title="Selected-slice periodogram", height=400, showlegend=True
    )

    _info = mo.callout(
        mo.md(
            "**Grey lines — the fixed candidate menu.** These are physically-motivated "
            "seasonal periods for this sampling frequency (semi-diurnal / diurnal / weekly "
            "for sub-daily data; weekly / annual for daily data). Detection only ever "
            "accepts periods from this fixed menu (plus justified extras), so "
            "ordinary spectral noise can never mint a meaningless period into the evidence "
            "table below.  \n"
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
def _(
    detect_candidates,
    extra_candidates,
    final_candidates,
    freq,
    proposed_seasonalities,
    y,
):
    evidence = detect_candidates(y, freq, tuple(set(extra_candidates) | set(final_candidates)))
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
def _(evidence, freq, mo, pd, proposed, series_label, slice_index):
    _table = pd.DataFrame(evidence)[
        ["period", "n_cycles", "power_ratio", "acf", "mstl_strength", "accepted"]
    ]
    mo.vstack(
        [
            mo.md(
                f"**{series_label}** ({freq}) — selected slice: {slice_index[0]} "
                f"→ {slice_index[-1]} ({len(slice_index)} steps, no gaps)"
            ),
            mo.ui.table(_table.round(3), selection=None, label="Candidate evidence"),
            mo.md(f"Proposed (dominant first): `seasonalities = {proposed}`"),
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
    y,
):
    _yd = fill_and_detrend(y)
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
        title="Selected-slice ACF (candidate lags marked)",
        xaxis_title="lag (steps)",
        yaxis_title="ACF",
        height=380,
    )
    _fig
    return


@app.cell
def _(fill_and_detrend, final_candidates, go, mo, np, pacf_at_lags, plot_template, y):
    _yd = fill_and_detrend(y)
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
        title="Selected-slice PACF (candidate lags marked)",
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
def _(final_candidates, go, make_subplots, mo, mstl_decompose, np, plot_template, y):
    _periods = sorted(final_candidates)
    _fit = mstl_decompose(y, _periods) if _periods else None

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
            title=f"MSTL decomposition (last {len(_series[0][1])} slice steps, periods {_fitted})",
            height=160 * len(_series) + 100,
        )
        _out = _fig
    _out
    return


if __name__ == "__main__":
    app.run()
