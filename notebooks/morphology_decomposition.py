import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Morphology features — pick a case and build intuition

    This notebook turns a raw series into its **morphology feature vector** by
    calling the *actual* functions in `tsforecastbench.morphology.features`, so
    what you see is the real implementation, not a re-derivation.

    Pick any case from the dropdown below: every plot, the structural
    decomposition, and the full feature table re-compute for that case. The goal
    is to *see* what each feature is measuring so the numbers in
    `results/morphology_features_all.csv` stop being opaque.

    ## The trend / seasonal features in one line

    Fit a single ordinary-least-squares model

    $$y(t) \;=\; \underbrace{T(t)}_{\text{polynomial trend}} \;+\;
    \underbrace{\sum_{p}\sum_{h=1}^{K} \big[a_{p,h}\cos\tfrac{2\pi h t}{p}
    + b_{p,h}\sin\tfrac{2\pi h t}{p}\big]}_{\text{Fourier seasonality, one block per detected period }p}
    \;+\; e(t)$$

    and read every trend/seasonal feature off the fitted pieces $T$, $S=\sum_p S_p$
    and the residual $e$. The remaining nine features (distribution shape,
    complexity, memory) are computed directly from the series and do **not** need
    the decomposition.

    ## Why this and not MSTL?

    The goal is **not** a physically perfect decomposition — it is to map each case
    into a feature space that **separates cases consistently** and is
    **explainable**. A single linear solve gives us that:

    | | MSTL (previous) | Structural OLS (this) |
    |---|---|---|
    | Multiple seasonalities | yes (slow, iterative) | yes (native, one solve) |
    | Cost on 350k samples | minutes → needed downsampling | sub-second, no downsampling |
    | Long period on fine data (365 d on 15-min) | intractable | one more Fourier block |
    | Knobs to explain | STL smoother windows, robust iters | trend degree, `K=3` harmonics |
    | Determinism | iterative | closed-form |
    """)
    return


@app.cell
def _():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from tsforecastbench.data.registry import discover_cases, get_case
    from tsforecastbench.morphology import features as F

    return F, discover_cases, get_case, np, pd, plt


@app.cell
def _(discover_cases, mo):
    import os

    cases = list(discover_cases())
    # EDA_CASE lets a static-site build select a case; interactively it just sets
    # which case loads first in the dropdown.
    _default = os.environ.get("EDA_CASE")
    _value = _default if _default in cases else cases[0]
    case_picker = mo.ui.dropdown(cases, value=_value, label="Case")
    mo.hstack([case_picker], justify="start")
    return (case_picker,)


@app.cell
def _(case_picker, get_case):
    case = get_case(case_picker.value)
    return (case,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1  The series

    We fit on the **full** record (period detection is a cheap FFT, the fit is one
    linear solve). Time-domain plots show a window of roughly eight dominant cycles
    so the structure stays legible regardless of how long the series is.
    """)
    return


@app.cell
def _(F, case, mo, np):
    _df = case.load().dropna(subset=["y"]).reset_index(drop=True)

    y = _df["y"].to_numpy().astype(float)
    t = np.arange(len(y), dtype=float)

    # Run the structural decomposition ONCE here; every section below reads off it.
    d = F._decompose(y)

    if d.periods:
        dom_p = d.periods[d.dom_idx]
        win_len = int(min(len(y), max(dom_p * 8, 200)))
    else:
        dom_p = None
        win_len = int(min(len(y), 1000))
    WIN = slice(0, win_len)

    _span = ""
    try:
        _ds = case.load()["ds"]
        import pandas as _pd

        _days = (_pd.to_datetime(_ds).max() - _pd.to_datetime(_ds).min()).days
        _span = f" · span ≈ {_days:,} days"
    except Exception:
        pass

    mo.vstack([
        mo.hstack([
            mo.stat(case.id, label="Case", bordered=True),
            mo.stat(case.freq, label="Frequency", bordered=True),
            mo.stat(f"{len(y):,}", label="Samples", bordered=True),
            mo.stat(str(dom_p) if dom_p else "none", label="Dominant period (steps)", bordered=True),
        ], justify="start", gap=1),
        mo.md(
            f"> **{case.target} ({case.target_unit})** at `{case.freq}`{_span}. "
            f"y: min={y.min():.3g}, max={y.max():.3g}, mean={y.mean():.3g}. "
            f"Detected periods (steps): **{d.periods or 'none'}**. "
            f"Plot window: first {win_len:,} steps."
        ),
    ])
    return WIN, d, dom_p, t, y


@app.cell
def _(WIN, case, mo, plt, t, y):
    _fig, _ax = plt.subplots(figsize=(13, 3.0))
    _ax.plot(t[WIN], y[WIN], lw=0.8, color="steelblue")
    _ax.set_xlabel(f"step ({case.freq})")
    _ax.set_ylabel(case.target)
    _ax.set_title(f"{case.id} — first {WIN.stop:,} steps")
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2  Period detection  → drives `seasonal_strength`

    Periods are found by `detect_periods`, a periodogram method that runs at full
    resolution:

    1. **Detrend** with an AIC-chosen polynomial (degree 1–3) so trend power does
       not swamp the seasonal peaks.
    2. **Periodogram**; keep peaks that are both **tall** (height
       $> `_PEAK_THRESHOLD`\times$ median PSD) and **locally distinct**
       (prominence $> `_PEAK_PROMINENCE`\times$ median PSD). The prominence gate
       keeps a sharp seasonal peak alive on a *red* ($1/f$) spectrum where a global
       height threshold alone would admit the low-frequency slope itself.
    3. Require at least `_MIN_CYCLES` complete cycles and a period $\ge$ `_MIN_PERIOD`.
    4. **Rank by prominence**, suppress spectral-leakage side-bands within
       `_CLUSTER_TOL`, and keep at most **`_MAX_PERIODS`** periods.

    The red lines below are the periods that survived. If none survive, the series
    has no detectable seasonality and `seasonal_strength` is 0.
    """)
    return


@app.cell
def _(F, d, mo, np, plt, y):
    from scipy.signal import find_peaks, periodogram

    _y_det, _ = F._poly_detrend(y)
    _freq, _psd = periodogram(_y_det)
    _psd = _psd[1:]
    _freq = _freq[1:]
    _med = np.median(_psd)

    _fig, _ax = plt.subplots(figsize=(13, 3.6))
    _per = 1.0 / _freq  # period in steps
    _ax.semilogx(_per, _psd, lw=0.7, color="slategray", label="periodogram (detrended)")
    _ax.axhline(F._PEAK_THRESHOLD * _med, color="orange", ls="--", lw=1.0, label="height threshold")
    for _p in d.periods:
        _ax.axvline(_p, color="crimson", ls=":", lw=1.2)
        _ax.text(_p, _ax.get_ylim()[1] * 0.6, f"{_p}", rotation=90,
                 color="crimson", fontsize=8, ha="right", va="top")
    _ax.set_xlabel("period (steps, log scale)")
    _ax.set_ylabel("power")
    _ax.set_title("Periodogram with detected periods (red)")
    _ax.legend(loc="upper right", fontsize=8)
    _ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    print("detected periods (steps):", d.periods)
    print(
        "  thresholds — height ×median:", F._PEAK_THRESHOLD,
        "| prominence ×median:", F._PEAK_PROMINENCE,
        "| max periods:", F._MAX_PERIODS,
    )
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3  The structural fit  → `y = T + S + e`

    `_decompose(y)` builds the design matrix and solves it once with `lstsq`:

    - **Trend columns** — powers of a $[-1, 1]$-scaled time axis, to the
      AIC-selected degree.
    - **Seasonal columns** — `_N_HARMONICS` ($K=3$) sine/cosine pairs per period.
      Frequencies are de-duplicated across periods so the design stays full rank.

    The four panels below are the decomposition. Eyeballing them tells you what the
    strength features *should* report: a clear trend panel → high `trend_strength`;
    a clean repeating seasonal panel → high `seasonal_strength`; a residual that
    still looks structured → those features are honestly reporting unexplained
    variance.
    """)
    return


@app.cell
def _(WIN, case, d, mo, plt, t, y):
    _fig, _ax = plt.subplots(4, 1, figsize=(13, 8.5), sharex=True)
    _ax[0].plot(t[WIN], y[WIN], lw=0.8, color="steelblue")
    _ax[0].set_ylabel("y")
    _ax[0].set_title(f"{case.id} — structural decomposition  y = T + S + e")

    _ax[1].plot(t[WIN], d.trend[WIN], lw=1.6, color="darkred")
    _ax[1].set_ylabel("trend T")

    _seasonal = d.seasonal_total
    if hasattr(_seasonal, "__len__"):
        _ax[2].plot(t[WIN], _seasonal[WIN], lw=0.9, color="seagreen")
    _ax[2].set_ylabel("seasonal S")

    _ax[3].plot(t[WIN], d.resid[WIN], lw=0.6, color="gray")
    _ax[3].axhline(0, color="k", lw=0.4)
    _ax[3].set_ylabel("residual e")
    _ax[3].set_xlabel(f"step ({case.freq})")

    for _a in _ax:
        _a.grid(True, alpha=0.3)
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(WIN, d, mo, np, plt):
    if not d.periods:
        _out = mo.md("_No seasonal periods detected — nothing to break down per period._")
    else:
        _fig, _ax = plt.subplots(figsize=(13, 3.4))
        for _p, _s in zip(d.periods, d.seasonal_arrays):
            _ax.plot(np.arange(WIN.stop), _s[WIN], lw=1.0, label=f"p = {_p}")
        _ax.axhline(0, color="k", lw=0.4)
        _ax.set_xlabel("step")
        _ax.set_ylabel("amplitude")
        _ax.set_title("Per-period seasonal components $S_p$ (frequencies de-duplicated across periods)")
        _ax.legend(loc="upper right", fontsize=8, ncol=len(d.periods))
        _ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _out = mo.mpl.interactive(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4  Strength features (variance ratios)

    Both are shares of variance explained, read directly off the one fit:

    $$\text{trend\_strength} = \max\!\Big(0,\; 1 - \tfrac{\operatorname{Var}(e)}{\operatorname{Var}(T+e)}\Big),
    \qquad
    \text{seasonal\_strength} = \max\!\Big(0,\; 1 - \tfrac{\operatorname{Var}(e)}{\operatorname{Var}(S+e)}\Big)$$

    where $T+e = y - S$ (the series with seasonality removed) and $S+e = y - T$
    (the de-trended series).
    """)
    return


@app.cell
def _(F, d, np, y):
    _vr = np.var(d.resid, ddof=1)
    _vt = np.var(y - d.seasonal_total, ddof=1)
    _vs = np.var(y - d.trend, ddof=1)
    print(f"Var(e)            = {_vr:.4g}")
    print(f"Var(T+e)=Var(y-S) = {_vt:.4g}   →  trend_strength    = {max(0, 1 - _vr / _vt):.4f}")
    print(f"Var(S+e)=Var(y-T) = {_vs:.4g}   →  seasonal_strength = {max(0, 1 - _vr / _vs) if d.periods else 0.0:.4f}")
    print()
    print("via public API:")
    print("  trend_strength   :", round(F.trend_strength(y), 4))
    print("  seasonal_strength:", round(F.seasonal_strength(y), 4))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5  Evolution features (how the dominant cycle *changes*)

    The strength features assume a fixed seasonal shape. Two more features measure
    how the **dominant** cycle drifts across its repetitions.

    We take $g = e + S_{\text{dom}}$, fold it into complete cycles of the dominant
    period, and project each cycle onto the fundamental sinusoid to get a per-cycle
    amplitude $A_k=\sqrt{a_k^2+b_k^2}$ and phase $\varphi_k=\operatorname{atan2}(b_k,a_k)$:

    - `amplitude_dispersion` $= c/(1+c)$ with $c=\operatorname{std}(A_k)/\operatorname{mean}(A_k)$ — does the cycle keep its **size**? (low = steady)
    - `phase_dispersion` $= 1 - \big|\operatorname{mean}(e^{i\varphi_k})\big|$ — how much its **timing** drifts (low = phase-locked)

    The left panel overlays every dominant cycle (tight overlay = low dispersion);
    the right panel shows the per-cycle amplitude.
    """)
    return


@app.cell
def _(F, d, dom_p, mo, np, plt):
    if dom_p is None:
        _out = mo.md("_No dominant cycle — amplitude/phase dispersion fall back to the 1.0 sentinel._")
    else:
        _g = d.resid + d.seasonal_arrays[d.dom_idx]
        _res = F._cycle_amplitude_phase(_g, dom_p)
        if _res is None:
            _out = mo.md("_Fewer than two complete cycles — dispersion features use the 1.0 sentinel._")
        else:
            _amp, _phase = _res
            _n_cyc = len(_g) // dom_p
            _chunks = _g[: _n_cyc * dom_p].reshape(_n_cyc, dom_p)

            _cv = _amp.std(ddof=1) / _amp.mean()
            _R = np.abs(np.exp(1j * _phase).mean())
            print(f"dominant period      : {dom_p}")
            print(f"complete cycles      : {len(_amp)}")
            print(f"amplitude_dispersion : {_cv / (1 + _cv):.4f}  (bounded CV of cycle amplitude)")
            print(f"phase_dispersion     : {np.clip(1 - _R, 0, 1):.4f}  (1 - circular concentration)")

            _fig, _ax = plt.subplots(1, 2, figsize=(13, 3.6))
            for _c in _chunks[:: max(1, _n_cyc // 60)]:
                _ax[0].plot(np.arange(dom_p), _c, lw=0.4, color="steelblue", alpha=0.25)
            _ax[0].plot(np.arange(dom_p), _chunks.mean(axis=0), lw=2.0, color="crimson", label="mean cycle")
            _ax[0].set_title(f"All cycles of dominant period p={dom_p}, overlaid")
            _ax[0].set_xlabel("phase within cycle (steps)")
            _ax[0].legend(fontsize=8)
            _ax[0].grid(True, alpha=0.3)

            _ax[1].plot(_amp, lw=1.0, color="darkorange")
            _ax[1].set_title("Per-cycle amplitude $A_k$")
            _ax[1].set_xlabel("cycle index")
            _ax[1].axhline(_amp.mean(), color="gray", ls="--", lw=1.0)
            _ax[1].grid(True, alpha=0.3)
            plt.tight_layout()
            _out = mo.mpl.interactive(_fig)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6  The full feature vector — what each number means

    `compute_features_all` runs the decomposition once and returns all 13 candidate
    features. The four trend/seasonal ones come from the fit above; the other nine
    describe the series' **distribution shape, complexity, and memory** directly.

    The working 4-feature taxonomy vector is `seasonal_strength`, `hurst`,
    `amplitude_dispersion`, `l_skewness`.
    """)
    return


@app.cell
def _(F, mo, y):
    # One-line, plain-language meaning for every feature. Keep keys in sync with
    # compute_features_all().
    _EXPLAIN = {
        "acf1": "Lag-1 autocorrelation — how strongly each value follows the previous. High → smooth/persistent.",
        "trend_strength": "Variance share of a smooth polynomial trend. High → strong deterministic drift.",
        "seasonal_strength": "De-trended variance share of the fitted seasonal cycles. High → dominant periodicity.",
        "spectral_entropy": "Spectrum flatness [0,1]. Low → energy in a few frequencies (periodic); high → broadband/noisy.",
        "boxcox_lambda": "Box-Cox power best symmetrising the values. ≈1 → already symmetric; <1 → right-skew (log-like).",
        "kurtosis": "Excess kurtosis — tail heaviness vs normal. High → spiky, extreme-event-prone.",
        "hurst": "Hurst exponent. ≈0.5 → random walk; >0.5 → long-range persistence / slow wander.",
        "perm_entropy": "Permutation entropy [0,1] — ordinal complexity. Low → structured/predictable; high → complex.",
        "amplitude_dispersion": "Variability of dominant-cycle amplitude [0,1). 0 → identical swing each cycle; 1 → erratic/none.",
        "phase_dispersion": "Drift of dominant-cycle timing [0,1]. 0 → phase-locked; 1 → timing wanders.",
        "asymmetry": "Rise/fall asymmetry (−1,1). Positive → faster rises (flashy); ≈0 → symmetric dynamics.",
        "l_skewness": "Robust (L-moment) skewness. 0 → symmetric; positive → heavy upper tail.",
        "zero_proportion": "Fraction of near-zero values. High → intermittent / zero-inflated regime.",
    }

    _feats = F.compute_features_all(y)
    _core = {"trend_strength", "seasonal_strength", "amplitude_dispersion", "phase_dispersion"}
    _rows = "\n".join(
        f"| `{_k}` | {_feats[_k]:+.4f} | {'fit' if _k in _core else 'direct'} | {_EXPLAIN[_k]} |"
        for _k in _EXPLAIN
    )
    mo.md(f"""
    | feature | value | source | what it measures |
    |---|---|---|---|
    {_rows}

    *source: **fit** = read off the `y = T + S + e` decomposition; **direct** = computed straight from the series.*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Where does this case sit?

    Each panel below is one feature. The box shows the spread of that feature
    **across all cases** in `results/morphology_features_all.csv`; the red line is
    the **selected case**. A red line far to one side of the box is what makes this
    case morphologically distinctive on that axis — the same intuition the taxonomy
    clustering runs on.
    """)
    return


@app.cell
def _(F, case, mo, np, pd, plt, y):
    from pathlib import Path

    _feats = F.compute_features_all(y)
    _cols = list(_feats.keys())

    _csv = Path(F.__file__).resolve().parents[3] / "results" / "morphology_features_all.csv"
    if _csv.exists():
        _pop = pd.read_csv(_csv)
    else:
        _pop = pd.DataFrame(columns=_cols)

    _ncol = 3
    _nrow = int(np.ceil(len(_cols) / _ncol))
    _fig, _axes = plt.subplots(_nrow, _ncol, figsize=(13, 2.0 * _nrow))
    _axes = np.atleast_1d(_axes).ravel()

    for _i, _name in enumerate(_cols):
        _ax = _axes[_i]
        _vals = pd.to_numeric(_pop[_name], errors="coerce").dropna().to_numpy() if _name in _pop else np.array([])
        if _vals.size:
            _ax.boxplot(
                _vals, vert=False, widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor="#dfe7ef", edgecolor="slategray"),
                medianprops=dict(color="slategray"),
                flierprops=dict(marker="o", markersize=3, alpha=0.4),
            )
            _ax.scatter(_vals, np.ones_like(_vals), color="slategray", s=10, alpha=0.5, zorder=3)
        _cur = _feats[_name]
        _ax.axvline(_cur, color="crimson", lw=1.8, zorder=4)
        _ax.set_title(f"{_name}\n{case.id} = {_cur:+.3g}", fontsize=8)
        _ax.set_yticks([])
        _ax.grid(True, axis="x", alpha=0.3)

    for _j in range(len(_cols), len(_axes)):
        _axes[_j].axis("off")

    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7  Design choices & known trade-offs

    - **Polynomial trend, not a spline.** Slow stochastic wander (e.g. groundwater)
      is captured by `hurst`, a separate feature. `trend_strength` is meant to mean
      *smooth deterministic drift*, so a rigid polynomial is the correct tool.
    - **`K = 3` harmonics per period, fixed.** Small enough to avoid fitting noise
      and stay collision-free, large enough to register non-sinusoidal shape.
    - **Peaky seasonals are under-scored.** A sharp, perfectly-regular freshet needs
      many harmonics to reconstruct; truncating at `K=3` lowers its
      `seasonal_strength`. This bias is *consistent*, so such cases still separate
      cleanly — which is the objective.
    - **Period ranking favours dominant energy.** On long red-spectrum series
      (waves, surge) the strongest cycles sit at multi-day scales, so those — not a
      lower-energy diurnal — are detected. `seasonal_strength` then reads as the
      *variance share* of the dominant cycles.
    """)
    return


if __name__ == "__main__":
    app.run()
