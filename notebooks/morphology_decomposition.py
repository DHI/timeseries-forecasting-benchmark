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
    # Trend & Seasonal Features via a Structural Least-Squares Decomposition

    This notebook documents how the morphology module turns a raw series into its
    **trend / seasonal** features. It calls the *actual* functions in
    `tsforecastbench.morphology.features`, so what you see here is the real
    implementation, not a re-derivation.

    ## The idea in one line

    Fit a single ordinary-least-squares model

    $$y(t) \;=\; \underbrace{T(t)}_{\text{polynomial trend}} \;+\;
    \underbrace{\sum_{p}\sum_{h=1}^{K} \big[a_{p,h}\cos\tfrac{2\pi h t}{p}
    + b_{p,h}\sin\tfrac{2\pi h t}{p}\big]}_{\text{Fourier seasonality, one block per detected period }p}
    \;+\; e(t)$$

    and read every trend/seasonal feature off the fitted pieces $T$, $S=\sum_p S_p$
    and the residual $e$.

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

    The trade we accept on purpose: a fixed-shape Fourier seasonal under-represents
    *peaky* cycles (a sharp snowmelt freshet) — but it does so **consistently**, so
    those cases still cluster together. Stochastic wander that a polynomial trend
    can't follow is captured by a *separate* feature (`hurst`), so the trend model
    is deliberately kept rigid.
    """)
    return


@app.cell
def _():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from tsforecastbench.data.registry import get_case
    from tsforecastbench.morphology import features as F

    return F, get_case, np, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## 1  Sample series

    **`demand_ltown`** — hourly water demand, and the one case in the suite with
    genuinely *distinct* multiple seasonalities: a **diurnal** cycle (24 h) and a
    **weekly** cycle (168 h). That makes it the natural showcase for a method built
    to handle more than one period at once.

    It is also a realistic, **noisy** operational series — much of its variance is
    irregular, so its `seasonal_strength` comes out modest. That is the features
    reporting honestly, not a failure of the fit: the diurnal/weekly *structure* is
    clearly present (see the per-period and folded-cycle plots below), it simply
    does not dominate the variance. We fit on the full record; time-domain plots
    show a 3-week window so the cycles are legible.
    """)
    return


@app.cell
def _(get_case, np):
    _case = get_case("demand_ltown")
    _df = _case.load().dropna(subset=["y"]).reset_index(drop=True)

    y = _df["y"].to_numpy().astype(float)
    t = np.arange(len(y), dtype=float)

    # window for legible time-domain plots (3 weeks of hourly data)
    WIN = slice(0, 24 * 21)

    print(f"n = {len(y)} hourly samples  (~{len(y) / 24 / 365:.1f} years)")
    print(f"y: min={y.min():.2f}  max={y.max():.2f}  mean={y.mean():.2f}")
    return WIN, t, y


@app.cell
def _(WIN, mo, plt, t, y):
    _fig, _ax = plt.subplots(figsize=(13, 3.0))
    _ax.plot(t[WIN], y[WIN], lw=0.8, color="steelblue")
    _ax.set_xlabel("hour")
    _ax.set_ylabel("demand")
    _ax.set_title("demand_ltown — first 3 weeks")
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2  Period detection

    Periods are found by `detect_periods`, a periodogram method that runs at full
    resolution (a cheap FFT, even on hundreds of thousands of samples):

    1. **Detrend** with an AIC-chosen polynomial (degree 1–3) so trend power does
       not swamp the seasonal peaks.
    2. **Periodogram**; keep peaks that are both **tall** (height
       $> `_PEAK_THRESHOLD`\times$ median PSD) and **locally distinct**
       (prominence $> `_PEAK_PROMINENCE`\times$ median PSD). The prominence gate is
       what keeps a sharp seasonal peak alive on a *red* ($1/f$) spectrum where a
       global height threshold alone would admit the low-frequency slope itself.
    3. Require at least `_MIN_CYCLES` complete cycles and a period $\ge$ `_MIN_PERIOD`.
    4. **Rank by prominence**, suppress spectral-leakage side-bands within
       `_CLUSTER_TOL`, and keep at most **`_MAX_PERIODS`** periods.

    Note we deliberately do **not** suppress sub-harmonics: a genuine 12 h
    semidiurnal and a 24 h diurnal cycle must both survive. Any *redundant
    frequency* they share (the 24 h cycle's 2nd harmonic is the 12 h fundamental)
    is removed later, when the Fourier design is built.
    """)
    return


@app.cell
def _(F, y):
    periods = F.detect_periods(y)
    # hourly data, so 1 step = 1 hour: 24 = daily, 168 = weekly, 84 ≈ half-week
    print("detected periods (steps = hours):", periods)
    print(
        "  thresholds — height ×median:", F._PEAK_THRESHOLD,
        "| prominence ×median:", F._PEAK_PROMINENCE,
        "| max periods:", F._MAX_PERIODS,
    )
    return (periods,)


@app.cell
def _(F, mo, np, periods, plt, y):
    from scipy.signal import find_peaks, periodogram

    _y_det, _ = F._poly_detrend(y)
    _freq, _psd = periodogram(_y_det)
    _psd = _psd[1:]
    _freq = _freq[1:]
    _med = np.median(_psd)
    _pk, _ = find_peaks(
        _psd, height=F._PEAK_THRESHOLD * _med, prominence=F._PEAK_PROMINENCE * _med
    )

    _fig, _ax = plt.subplots(figsize=(13, 3.6))
    _per = 1.0 / _freq  # period in steps
    _ax.semilogx(_per, _psd, lw=0.7, color="slategray", label="periodogram (detrended)")
    _ax.axhline(F._PEAK_THRESHOLD * _med, color="orange", ls="--", lw=1.0, label="height threshold")
    for _p in periods:
        _ax.axvline(_p, color="crimson", ls=":", lw=1.2)
        _ax.text(_p, _ax.get_ylim()[1] * 0.6, f"{_p}", rotation=90,
                 color="crimson", fontsize=8, ha="right", va="top")
    _ax.set_xlabel("period (steps, log scale)")
    _ax.set_ylabel("power")
    _ax.set_title("Periodogram with detected periods (red)")
    _ax.legend(loc="upper right", fontsize=8)
    _ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3  The structural fit

    `_decompose(y)` builds the design matrix and solves it once with `lstsq`:

    - **Trend columns** — powers of a $[-1, 1]$-scaled time axis (well conditioned
      even for very long series), to the AIC-selected degree.
    - **Seasonal columns** — `_N_HARMONICS` ($K=3$) sine/cosine pairs per period.
      Frequencies are de-duplicated across periods via reduced fractions $h/p$: a
      shared frequency is fitted **once** and attributed to the *shortest* period
      that owns it (its most fundamental generator). This keeps the design full
      rank without dropping any physically distinct period.

    The result object exposes `trend` ($T$), `seasonal_arrays` (the per-period
    $S_p$), `resid` ($e$), and `dom_idx` — the dominant period, chosen as the one
    explaining the most variance.
    """)
    return


@app.cell
def _(F, np, y):
    d = F._decompose(y)
    _recon = d.trend + d.seasonal_total + d.resid
    print("periods          :", d.periods)
    print("dominant period  :", d.periods[d.dom_idx])
    print("reconstruction err:", float(np.max(np.abs(y - _recon))), "(≈ machine precision)")
    print("per-period variance share of S_p:")
    for _p, _s in zip(d.periods, d.seasonal_arrays):
        print(f"   p={_p:5d}   Var(S_p)={np.var(_s, ddof=1):.4g}")
    return (d,)


@app.cell
def _(WIN, d, mo, plt, t, y):
    _fig, _ax = plt.subplots(4, 1, figsize=(13, 8.5), sharex=True)
    _ax[0].plot(t[WIN], y[WIN], lw=0.8, color="steelblue")
    _ax[0].set_ylabel("y")
    _ax[0].set_title("Structural decomposition  y = T + S + e  (first 3 weeks)")

    _ax[1].plot(t[WIN], d.trend[WIN], lw=1.6, color="darkred")
    _ax[1].set_ylabel("trend T")

    _ax[2].plot(t[WIN], d.seasonal_total[WIN], lw=0.9, color="seagreen")
    _ax[2].set_ylabel("seasonal S")

    _ax[3].plot(t[WIN], d.resid[WIN], lw=0.6, color="gray")
    _ax[3].axhline(0, color="k", lw=0.4)
    _ax[3].set_ylabel("residual e")
    _ax[3].set_xlabel("hour")

    for _a in _ax:
        _a.grid(True, alpha=0.3)
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(WIN, d, mo, plt, t):
    _fig, _ax = plt.subplots(figsize=(13, 3.4))
    for _p, _s in zip(d.periods, d.seasonal_arrays):
        _ax.plot(t[WIN], _s[WIN], lw=1.0, label=f"p = {_p}")
    _ax.axhline(0, color="k", lw=0.4)
    _ax.set_xlabel("hour")
    _ax.set_ylabel("amplitude")
    _ax.set_title("Per-period seasonal components $S_p$ (frequencies de-duplicated across periods)")
    _ax.legend(loc="upper right", fontsize=8, ncol=len(d.periods))
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mo.mpl.interactive(_fig)
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
    (the de-trended series). `seasonal_strength` therefore reads as *"the share of
    de-trended variance explained by the fitted seasonality across all detected
    periods."*
    """)
    return


@app.cell
def _(F, d, np, y):
    _vr = np.var(d.resid, ddof=1)
    _vt = np.var(y - d.seasonal_total, ddof=1)
    _vs = np.var(y - d.trend, ddof=1)
    print(f"Var(e)        = {_vr:.4g}")
    print(f"Var(T+e)=Var(y-S) = {_vt:.4g}   →  trend_strength    = {max(0,1-_vr/_vt):.4f}")
    print(f"Var(S+e)=Var(y-T) = {_vs:.4g}   →  seasonal_strength = {max(0,1-_vr/_vs):.4f}")
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
    how the **dominant** cycle drifts across its repetitions — and only these need
    a time-varying view.

    We take $g = e + S_{\text{dom}}$ (residual plus the dominant seasonal), fold it
    into complete cycles of the dominant period, and **project each cycle onto the
    fundamental sinusoid** to get a per-cycle amplitude $A_k=\sqrt{a_k^2+b_k^2}$ and
    phase $\varphi_k=\operatorname{atan2}(b_k,a_k)$:

    - `seasonal_stability` $= \operatorname{std}(A_k)/\operatorname{mean}(A_k)$ — does the cycle keep its **size**?
    - `phase_consistency` $= 1 - \big|\operatorname{mean}(e^{i\varphi_k})\big|$ — does it keep its **timing**?

    (Projection onto the fundamental is sturdier than reading an `argmax`, which is
    noisy and quantised to integer lags.)
    """)
    return


@app.cell
def _(F, d, np):
    _dom_p = d.periods[d.dom_idx]
    _g = d.resid + d.seasonal_arrays[d.dom_idx]
    _amp, _phase = F._cycle_amplitude_phase(_g, _dom_p)
    print(f"dominant period   : {_dom_p}")
    print(f"complete cycles    : {len(_amp)}")
    print(f"mean amplitude     : {_amp.mean():.4g}")
    print(f"seasonal_stability : {_amp.std(ddof=1)/_amp.mean():.4f}  (CV of cycle amplitude)")
    _R = np.abs(np.exp(1j * _phase).mean())
    print(f"phase_consistency  : {np.clip(1-_R,0,1):.4f}  (1 - circular concentration)")
    return


@app.cell
def _(F, WIN, d, mo, np, plt):
    _dom_p = d.periods[d.dom_idx]
    _g = d.resid + d.seasonal_arrays[d.dom_idx]
    _n_cyc = len(_g) // _dom_p
    _chunks = _g[: _n_cyc * _dom_p].reshape(_n_cyc, _dom_p)
    _amp, _phase = F._cycle_amplitude_phase(_g, _dom_p)

    _fig, _ax = plt.subplots(1, 2, figsize=(13, 3.6))
    # overlay every cycle of the dominant period
    for _c in _chunks[::max(1, _n_cyc // 60)]:
        _ax[0].plot(np.arange(_dom_p), _c, lw=0.4, color="steelblue", alpha=0.25)
    _ax[0].plot(np.arange(_dom_p), _chunks.mean(axis=0), lw=2.0, color="crimson", label="mean cycle")
    _ax[0].set_title(f"All cycles of dominant period p={_dom_p}, overlaid")
    _ax[0].set_xlabel("phase within cycle (steps)")
    _ax[0].legend(fontsize=8)
    _ax[0].grid(True, alpha=0.3)

    _ax[1].plot(_amp, lw=1.0, color="darkorange")
    _ax[1].set_title("Per-cycle amplitude $A_k$")
    _ax[1].set_xlabel("cycle index")
    _ax[1].axhline(_amp.mean(), color="gray", ls="--", lw=1.0)
    _ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6  The full feature vector

    `compute_features_all` runs the decomposition **once** and reads the four
    trend/seasonal features off it, alongside the nine distribution/complexity
    features (which do not use the decomposition).
    """)
    return


@app.cell
def _(F, mo, pd, y):
    _feats = F.compute_features_all(y)
    _rows = "\n".join(f"| `{_k}` | {_v:+.4f} |" for _k, _v in _feats.items())
    mo.md(f"""
    | feature | value |
    |---|---|
    {_rows}

    *(The working 5-feature taxonomy vector is `seasonal_strength`, `hurst`,
    `trend_strength`, `phase_consistency`, `l_skewness`.)*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7  Design choices & known trade-offs

    - **Polynomial trend, not a spline.** Slow stochastic wander (e.g. groundwater)
      is captured by `hurst`, a separate feature. `trend_strength` is meant to mean
      *smooth deterministic drift*, so a rigid polynomial is the correct tool, not a
      limitation.
    - **`K = 3` harmonics per period, fixed.** Small enough to avoid fitting noise
      and to stay collision-free (the daily/weekly frequency clash is at harmonic 7),
      large enough to register non-sinusoidal shape. No per-case selection to explain.
    - **Peaky seasonals are under-scored.** A sharp, perfectly-regular freshet needs
      many harmonics to reconstruct; truncating at `K=3` lowers its
      `seasonal_strength`. This bias is *consistent* across such cases, so they still
      separate cleanly — which is the objective.
    - **Period ranking favours dominant energy.** On long red-spectrum series
      (waves, surge), the strongest cycles sit at multi-day scales, so those — not a
      lower-energy diurnal — are detected. `seasonal_strength` then correctly reads
      as the *variance share* of the dominant cycles. If physical identification of a
      specific (e.g. tidal) period ever matters more than variance-share consistency,
      that would require spectral whitening at detection time — added complexity we
      have deliberately not taken on.
    """)
    return


if __name__ == "__main__":
    app.run()
