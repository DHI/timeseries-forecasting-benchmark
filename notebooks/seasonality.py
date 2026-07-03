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
    from scipy.signal import periodogram

    from tsforecastbench.data.registry import discover_cases, get_case
    from tsforecastbench.data.schema import _observed_mask as observed_mask
    from tsforecastbench.data.seasonality import (
        CANDIDATE_MENU,
        detect_candidates,
        normalize_freq,
        proposed_seasonalities,
        suggest_peak_candidates,
    )
    from tsforecastbench.data.seasonality import (
        _acf_at as acf_at,
    )
    from tsforecastbench.data.seasonality import (
        _fill_and_detrend as fill_and_detrend,
    )

    return (
        CANDIDATE_MENU,
        acf_at,
        detect_candidates,
        discover_cases,
        fill_and_detrend,
        get_case,
        go,
        normalize_freq,
        np,
        observed_mask,
        pd,
        periodogram,
        proposed_seasonalities,
        suggest_peak_candidates,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # Seasonality curation

        One-time tooling for the `seasonalities` field in each case.toml — the **single
        source of truth** for seasonal periods during the benchmark train/predict cycle
        (periods in steps, dominant first, `[]` = non-seasonal).

        Candidates come from a fixed physically motivated menu per frequency; the evidence
        below (periodogram prominence + ACF at lag, computed on the **train split only**)
        accepts or rejects each. Review the evidence, then commit the proposed line — or
        deviate from it with a written justification in the case notes.
        """
    )
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
def _(case, detect_candidates, extra_candidates, proposed_seasonalities, train):
    evidence = detect_candidates(train["y"].to_numpy(), case.freq, extra_candidates)
    proposed = proposed_seasonalities(evidence)
    return evidence, proposed


@app.cell
def _(case, extra_candidates, suggest_peak_candidates, train):
    peak_suggestions = suggest_peak_candidates(
        train["y"].to_numpy(), case.freq, exclude=extra_candidates
    )
    return (peak_suggestions,)


@app.cell
def _(mo, peak_suggestions):
    mo.md(
        f"**Periodogram peak suggestions** (outside the fixed menu, not yet evidenced): "
        f"`{peak_suggestions}`  \n"
        "Copy a promising one into the extra-candidates box above to run it through the "
        "full ACF / power-ratio evidence check."
        if peak_suggestions
        else "No periodogram peaks found outside the fixed menu."
    )
    return


@app.cell
def _(case, evidence, filled_frac, mo, pd, proposed):
    declared = getattr(case, "seasonalities", None)
    _table = pd.DataFrame(evidence)[["period", "n_cycles", "power_ratio", "acf", "accepted"]]

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
    CANDIDATE_MENU, case, extra_candidates, fill_and_detrend, go, normalize_freq, periodogram, train
):
    _yd = fill_and_detrend(train["y"].to_numpy())
    _freqs, _power = periodogram(_yd, fs=1.0)
    _menu = sorted(set(CANDIDATE_MENU.get(normalize_freq(case.freq), ())) | set(extra_candidates))

    _fig = go.Figure()
    _fig.add_trace(go.Scattergl(x=_freqs[1:], y=_power[1:], mode="lines", name="periodogram"))
    for _p in _menu:
        _fig.add_vline(
            x=1.0 / _p,
            line_dash="dot",
            line_color="crimson",
            annotation_text=f"{_p}",
            annotation_position="top",
        )
    _fig.update_xaxes(type="log", title="frequency (1/steps)")
    _fig.update_yaxes(type="log", title="power")
    _fig.update_layout(title="Train-split periodogram (candidate periods marked)", height=380)
    _fig
    return


@app.cell
def _(
    CANDIDATE_MENU, acf_at, case, extra_candidates, fill_and_detrend, go, normalize_freq, np, train
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


if __name__ == "__main__":
    app.run()
