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
    from scipy.signal import find_peaks, periodogram
    from scipy.stats import binned_statistic

    from tsforecastbench.data.registry import discover_cases, get_case
    from tsforecastbench.data.schema import _observed_mask as observed_mask
    from tsforecastbench.data.seasonality import (
        CANDIDATE_MENU,
        detect_candidates,
        normalize_freq,
        proposed_seasonalities,
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

    return (
        CANDIDATE_MENU,
        acf_at,
        binned_statistic,
        detect_candidates,
        discover_cases,
        fill_and_detrend,
        find_peaks,
        get_case,
        go,
        mstl_decompose,
        normalize_freq,
        np,
        observed_mask,
        pd,
        periodogram,
        proposed_seasonalities,
    )


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
    mo.hstack([case_picker, extra_input], justify="start", gap=2)
    return case_picker, extra_input


@app.cell
def _(mo):
    # Resolution of the uniform log-frequency grid the periodogram is resampled onto (see
    # the periodogram cell). This is the single smoothing knob: fewer bins = coarser grid =
    # only major peaks survive; more bins = finer grid = more structure (and more peaks).
    grid_resolution = mo.ui.slider(
        start=50,
        stop=350,
        step=50,
        value=150,
        label="Log-frequency grid resolution M (bins; higher = finer / more peaks)",
        full_width=True,
    )
    # One text input per scipy.signal.find_peaks parameter; blank = leave unset (None).
    # On the log grid a bin is a fixed frequency ratio, so distance/width finally transfer
    # across cases. distance and prominence are pre-filled conservatively so only major
    # peaks are found out of the box; loosen them (or raise M) to surface more.
    find_peaks_ui = mo.ui.dictionary(
        {
            "height": mo.ui.text(label="height", placeholder="min power"),
            "threshold": mo.ui.text(label="threshold", placeholder="Δ to neighbours"),
            "distance": mo.ui.text(value="5", label="distance", placeholder="≥1 bin"),
            "prominence": mo.ui.text(value="0.7", label="prominence", placeholder="min prominence"),
            #            "width": mo.ui.text(label="width", placeholder="bins"),
            #            "wlen": mo.ui.text(label="wlen", placeholder="bins"),
            "rel_height": mo.ui.text(value="0.5", label="rel_height", placeholder="0-1"),
            #            "plateau_size": mo.ui.text(label="plateau_size", placeholder="bins"),
        }
    )
    return find_peaks_ui, grid_resolution


@app.cell
def _(
    CANDIDATE_MENU,
    binned_statistic,
    case,
    extra_candidates,
    fill_and_detrend,
    find_peaks,
    find_peaks_ui,
    go,
    grid_resolution,
    mo,
    normalize_freq,
    np,
    periodogram,
    train,
):
    _yd = fill_and_detrend(train["y"].to_numpy())
    _freqs, _power = periodogram(_yd, fs=1.0)
    _f, _p = _freqs[1:], _power[1:]  # drop the DC bin (frequency 0)

    # The periodogram is uniform in frequency, but we reason about periods on a log axis:
    # it is wildly over-resolved at short periods (noise clutter) and under-resolved at long
    # ones, and find_peaks works on array index so its distance/width mean different things
    # at each end. Resample log-power (floor zeros so log10 stays finite) onto a uniform
    # log-frequency grid, taking the MAX per bin so sharp seasonal lines survive the
    # down-sampling (a mean averages them into the noise floor). Empty low-frequency bins —
    # where the grid is finer than the raw spectrum — are filled by interpolation. find_peaks
    # then runs on this evenly-spaced curve, where a bin is a fixed frequency ratio.
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
    _grid_f = 10.0**_centers  # log-spaced frequency grid; find_peaks runs on _binned

    _int_params = {"distance", "wlen", "plateau_size"}
    _kwargs = {}
    for _name, _tok in find_peaks_ui.value.items():
        _tok = _tok.strip()
        if not _tok:
            continue
        _kwargs[_name] = int(float(_tok)) if _name in _int_params else float(_tok)
    try:
        _peak_idx, _ = find_peaks(_binned, **_kwargs)
        _err = None
    except Exception as _e:  # bad hyperparameter (e.g. distance < 1) — show it, keep plotting
        _peak_idx, _err = np.array([], dtype=int), str(_e)
    _peak_freqs = _grid_f[_peak_idx]
    _peak_periods = [round(1.0 / _pf) for _pf in _peak_freqs]

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
    for _period in _menu:  # grey — fixed physically-motivated candidate menu
        _fig.add_vline(
            x=1.0 / _period,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"{_period}",
            annotation_position="top",
        )
    for _period, _pf in zip(_peak_periods, _peak_freqs, strict=True):  # yellow — find_peaks peaks
        _fig.add_vline(
            x=_pf,
            line_color="gold",
            annotation_text=f"{_period}",
            annotation_position="bottom",
        )
    for _period in extra_candidates:  # orange — manually entered extra candidates
        _fig.add_vline(
            x=1.0 / _period,
            line_dash="dash",
            line_color="darkorange",
            annotation_text=f"{_period}",
            annotation_position="top",
        )
    _fig.update_xaxes(type="log", title="frequency (1/steps)")
    _fig.update_yaxes(type="log", title="power")
    _fig.update_layout(title="Train-split periodogram", height=400, showlegend=True)

    _info = mo.callout(
        mo.md(
            "**Grey lines — the fixed candidate menu.** These are physically-motivated "
            "seasonal periods for this sampling frequency (semi-diurnal / diurnal / weekly "
            "for sub-daily data; weekly / annual for daily data). Detection only ever "
            "accepts periods from this fixed menu (plus justified per-case extras), so "
            "ordinary spectral noise can never mint a meaningless period into the benchmark "
            "metadata.  \n"
            "**Yellow lines** are peaks `find_peaks` located on the log-resampled curve; "
            "**orange lines** are the extra candidates typed above. A yellow peak sitting on "
            "a grey line corroborates a menu period; a yellow peak with no grey line nearby "
            "flags an off-menu cycle worth adding as an extra candidate.  \n"
            "⚠️ *Long-period peaks are approximate*: a finite-length periodogram cannot "
            "resolve low frequencies finely, so an annual cycle may read as ~360-375. Treat a "
            "long-period peak as a neighbourhood, not an exact period."
        ),
        kind="info",
    )
    _peaks_note = (
        f"⚠️ find_peaks error: {_err}"
        if _err
        else f"Detected peak periods (steps): `{sorted(set(_peak_periods))}`"
    )
    _fp_box = mo.vstack(
        [
            mo.md(
                "**`scipy.signal.find_peaks` parameters** — blank leaves a parameter unset; "
                "edit any of them to change which yellow peaks are detected on the "
                "log-resampled curve."
            ),
            find_peaks_ui,
            mo.md(_peaks_note),
        ]
    )
    mo.vstack([_info, grid_resolution, _fig, _fp_box])
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
    proposed_seasonalities,
    train,
):
    evidence = detect_candidates(train["y"].to_numpy(), case.freq, extra_candidates)
    proposed = proposed_seasonalities(evidence)
    return evidence, proposed


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
    CANDIDATE_MENU,
    acf_at,
    case,
    extra_candidates,
    fill_and_detrend,
    go,
    normalize_freq,
    np,
    train,
):
    _yd = fill_and_detrend(train["y"].to_numpy())
    _menu = sorted(set(CANDIDATE_MENU.get(normalize_freq(case.freq), ())) | set(extra_candidates))
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
        title="Train-split ACF (candidate lags marked)",
        xaxis_title="lag (steps)",
        yaxis_title="ACF",
        height=380,
    )
    _fig
    return


@app.cell
def _(evidence, go, mo, mstl_decompose, np, train):
    _periods = sorted(e["period"] for e in evidence if np.isfinite(e["mstl_strength"]))
    _fit = mstl_decompose(train["y"].to_numpy(), _periods) if _periods else None

    if _fit is None:
        _out = mo.md(
            "No candidate is both plausible (periodogram + cycle count) and affordable "
            "(fits >= 2 cycles within the sampled `MSTL_SAMPLE_ROWS` window) for an MSTL "
            "decomposition."
        )
    else:
        _decomposition, _fitted = _fit
        _fig = go.Figure()
        _fig.add_trace(go.Scattergl(y=_decomposition["trend"], mode="lines", name="trend"))
        for _p in _fitted:
            _col = "seasonal" if len(_fitted) == 1 else f"seasonal{_p}"
            _fig.add_trace(
                go.Scattergl(y=_decomposition[_col], mode="lines", name=f"seasonal({_p})")
            )
        _fig.add_trace(
            go.Scattergl(y=_decomposition["remainder"], mode="lines", name="remainder", opacity=0.4)
        )
        _fig.update_layout(
            title=f"MSTL decomposition (last {len(_decomposition)} train steps, periods {_fitted})",
            xaxis_title="step (within sampled window)",
            height=420,
        )
        _out = _fig
    _out
    return


if __name__ == "__main__":
    app.run()
