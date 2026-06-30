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
    # Wavelet-Based Seasonality Detection with PyWavelets

    **Dataset:** Hourly significant wave height (*Hₛ*, metres) at NOAA buoy 46050,
    Stonewall Bank, Oregon — a real-world signal with overlapping periodic components
    buried in broadband stochastic noise.

    **Goal:** Use the `pywt` library to *discover* those seasonal cycles without knowing
    their frequencies in advance, and then isolate each component.

    > **Install:** `uv pip install pywavelets scipy`
    """)
    return


@app.cell
def _():
    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pywt
    from scipy.signal import find_peaks

    from tsforecastbench.data.registry import get_case

    return find_peaks, get_case, matplotlib, np, pd, plt, pywt


@app.cell
def _(mo):
    mo.md(r"""
    ## 1  Load Data & Select Subseries

    We load the pre-processed Parquet and take **2 000 consecutive hours** (≈ 83 days)
    starting from index 200 to skip sensor warm-up at the record start.
    That window is long enough to capture diurnal (24 h), semi-diurnal (12 h),
    and multi-day storm cycles while keeping the CWT fast.
    """)
    return


@app.cell
def _(get_case, np, pd):
    _case = get_case("waves_oregon")
    _df = _case.load().dropna(subset=["wvht"]).reset_index(drop=True)

    N = 2000
    _start = 3000
    sub = _df.iloc[_start : _start + N].copy()
    sub["ds"] = pd.to_datetime(sub["ds"], utc=True)

    # y: raw target signal (significant wave height in metres)
    # hours: integer hour axis 0..1999, used as x-axis throughout
    y = sub["wvht"].values.astype(float)
    hours = np.arange(N, dtype=float)

    print(f"Window : {sub['ds'].iloc[0]}  →  {sub['ds'].iloc[-1]}")
    print(f"min={y.min():.2f} m   max={y.max():.2f} m   mean={y.mean():.2f} m")
    return N, hours, sub, y


@app.cell
def _(mo, pd, plt, hours, y):
    _fig, _ax = plt.subplots(figsize=(14, 3.2))
    _ma = pd.Series(y).rolling(24, center=True).mean().values
    _ax.plot(hours, y, lw=0.7, color="steelblue", alpha=0.75, label="Hₛ raw (1 h)")
    _ax.plot(hours, _ma, lw=1.8, color="crimson", label="24 h moving average")
    _ax.set_xlabel("Time (hours from start)")
    _ax.set_ylabel("Hₛ (m)")
    _ax.set_title("Waves Oregon — 2 000-sample subseries (raw signal)")
    _ax.legend(loc="upper right")
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2  Wavelet Transforms — Core Concepts

    ### Why not Fourier?

    The **Fourier Transform** decomposes a signal into globally-defined sinusoids.
    It gives perfect *frequency* resolution but zero *time* resolution: you cannot tell
    *when* a given frequency is active, only that it is present somewhere.

    For ocean waves this matters — storm-driven swell arrives intermittently; a tidal
    signal can be amplitude-modulated by lunar declination; the "24 h" component switches
    between sea-breeze and tidal forcing depending on season.

    ### Enter Wavelets

    A **wavelet** ψ is a short, oscillatory function that is *localised in both time and
    frequency*.  The wavelet transform correlates the signal against scaled and translated
    copies of ψ:

    $$W(s,\,\tau) = \frac{1}{\sqrt{s}}\int_{-\infty}^{\infty} x(t)\;\psi^*\!\!\left(\frac{t-\tau}{s}\right)dt$$

    * **Scale** *s* stretches/compresses ψ → large *s* ↔ low frequency (long period).
    * **Translation** *τ* slides the window along time.

    The result is a **2-D time–scale map** (scalogram) showing *where in time* each
    frequency band is energetic — ideal for non-stationary geophysical signals.

    ### Two Transforms Used Here

    | | CWT (Continuous) | DWT (Discrete) |
    |---|---|---|
    | Output | Dense 2-D time–scale matrix | Small set of additive scale bands |
    | Use here | Visualise & detect periodicities | Separate signal into labelled components |
    | PyWavelets call | `pywt.cwt()` | `pywt.wavedec()` + reconstruction |

    ### Wavelet Choice

    **Complex Morlet (`cmor1.5-1.0`)** for the CWT — complex sinusoid inside a Gaussian
    envelope; complex output gives instantaneous *amplitude and phase*:

    $$\psi(t)=\frac{1}{\sqrt{\pi B}}\,e^{2\pi i f_c t}\;e^{-t^2/B}, \quad B=1.5,\;f_c=1.0$$

    **Daubechies-8 (`db8`)** for the DWT — smooth, compact support, widely used in geophysics.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3  Continuous Wavelet Transform (CWT) — Scalogram

    ### Scale ↔ Period Conversion

    `pywt.cwt()` maps each scale *s* to a pseudo-frequency:

    $$f_{\text{pseudo}} = \frac{f_c}{s \cdot \Delta t}$$

    Inverting: $\text{period (hours)} = \frac{s}{f_c}$ (since $\Delta t = 1\,\text{h}$).

    To target specific periods we pre-compute the required scales:

    ```python
    fc     = pywt.central_frequency('cmor1.5-1.0')   # → 1.0
    scales = fc * target_periods                       # scale = fc × period_in_samples
    coef, freqs = pywt.cwt(y, scales, 'cmor1.5-1.0', sampling_period=1.0)
    periods = 1.0 / freqs                              # convert cycles/h → hours/cycle
    ```

    We log-space 200 scales to span **6 h → 500 h** (≈ 3 weeks).
    """)
    return


@app.cell
def _(np, pywt, y):
    _WAVELET = "cmor1.5-1.0"
    _P_MIN, _P_MAX = 6.0, 500.0

    _fc = pywt.central_frequency(_WAVELET)
    _tgt_p = np.logspace(np.log10(_P_MIN), np.log10(_P_MAX), 200)
    _scales = _fc * _tgt_p  # scale = fc × period (sampling_period = 1 h)

    # Zero-mean before CWT: removes DC offset and reduces edge-artefacts
    _y_c = y - y.mean()

    coef_cwt, freqs_cwt = pywt.cwt(_y_c, _scales, _WAVELET, sampling_period=1.0)
    periods_cwt = 1.0 / freqs_cwt  # hours per cycle  (shape: 200,)
    power_cwt = np.abs(coef_cwt) ** 2

    print(f"CWT array shape : {coef_cwt.shape}  (n_scales × n_time)")
    print(f"Period range    : {periods_cwt[-1]:.0f} h → {periods_cwt[0]:.0f} h")
    return coef_cwt, freqs_cwt, periods_cwt, power_cwt


@app.cell
def _(mo, np, periods_cwt, plt, power_cwt, hours, y):
    _fig = plt.figure(figsize=(15, 8))
    _gs = _fig.add_gridspec(
        2, 2, height_ratios=[1, 3], width_ratios=[5, 1], hspace=0.05, wspace=0.04
    )
    _ax_sig = _fig.add_subplot(_gs[0, 0])
    _ax_scal = _fig.add_subplot(_gs[1, 0])
    _ax_gws = _fig.add_subplot(_gs[1, 1])

    # ── raw signal ─────────────────────────────────────────────────────────
    _ax_sig.plot(hours, y, lw=0.7, color="steelblue", alpha=0.8)
    _ax_sig.set_ylabel("Hₛ (m)")
    _ax_sig.set_xlim(0, len(y) - 1)
    _ax_sig.set_xticklabels([])
    _ax_sig.grid(True, alpha=0.3)
    _ax_sig.set_title("CWT Scalogram — Waves Oregon (2 000 h subseries, Morlet wavelet)")

    # ── scalogram (log₂-normalised power) ──────────────────────────────────
    _logp = np.log2(power_cwt + 1e-10)
    _vmin, _vmax = np.percentile(_logp, [5, 99])
    _im = _ax_scal.contourf(
        hours,
        np.log2(periods_cwt),
        _logp,
        levels=60,
        cmap="inferno",
        vmin=_vmin,
        vmax=_vmax,
    )

    # Guide lines at expected physical periods
    for _lbl, _p in [("12 h (semi-diurnal)", 12), ("24 h (diurnal)", 24), ("7 d", 168)]:
        if periods_cwt.min() <= _p <= periods_cwt.max():
            _ax_scal.axhline(np.log2(_p), color="lime", lw=1.1, ls="--", alpha=0.8)
            _ax_scal.text(30, np.log2(_p) + 0.07, _lbl, color="lime", fontsize=8, va="bottom")

    _yticks = [
        p for p in [6, 12, 24, 48, 96, 168, 336, 500] if periods_cwt.min() <= p <= periods_cwt.max()
    ]
    _ax_scal.set_yticks(np.log2(_yticks))
    _ax_scal.set_yticklabels([f"{p} h" if p < 48 else f"{p // 24} d" for p in _yticks])
    _ax_scal.invert_yaxis()
    _ax_scal.set_xlabel("Time (hours from start)")
    _ax_scal.set_ylabel("Period (log₂ scale)")
    _ax_scal.set_xlim(0, len(y) - 1)
    plt.colorbar(_im, ax=_ax_scal, label="log₂ power", pad=0.01, shrink=0.9)

    # ── global wavelet spectrum (time-averaged power) ──────────────────────
    _gws_plot = power_cwt.mean(axis=1)
    _ax_gws.plot(_gws_plot, np.log2(periods_cwt), color="darkorange", lw=1.5)
    _ax_gws.set_yticks(np.log2(_yticks))
    _ax_gws.set_yticklabels([])
    _ax_gws.invert_yaxis()
    _ax_gws.set_xlabel("Mean\npower", fontsize=8)
    _ax_gws.set_title("GWS", fontsize=9)
    _ax_gws.grid(True, alpha=0.3)

    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **How to read the scalogram:**

    * **Bright horizontal bands** — persistent periodicities active throughout the window
      (look for the lime guide lines at 12 h and 24 h).
    * **Bright vertical patches** — sudden broadband energy bursts, typically storm arrivals.
    * **Global Wavelet Spectrum (GWS, right panel)** — time-averaged power per period bin;
      analogous to a Fourier power spectrum. Peaks in the GWS are the dominant periods.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4  Automatic Period Detection (GWS Peaks)

    We find local maxima in the time-averaged (Global) Wavelet Spectrum to identify
    dominant periods *without assuming their values upfront*.
    The GWS is first lightly smoothed to suppress noise before peak-finding.
    """)
    return


@app.cell
def _(find_peaks, np, periods_cwt, power_cwt):
    # Time-averaged power (Global Wavelet Spectrum)
    gws = power_cwt.mean(axis=1)

    # Light smoothing (7-point boxcar) before peak detection
    _smooth = np.convolve(gws, np.ones(7) / 7, mode="same")

    _thresh = np.percentile(_smooth, 25)
    _peak_idx, _ = find_peaks(
        _smooth,
        height=_thresh,
        distance=4,
        prominence=_thresh * 0.4,
    )

    _order = np.argsort(gws[_peak_idx])[::-1]
    dominant_periods = periods_cwt[_peak_idx[_order]]
    dominant_power = gws[_peak_idx[_order]]

    print("Dominant periods (Global Wavelet Spectrum peaks):")
    for _i, (_p, _pw) in enumerate(zip(dominant_periods[:10], dominant_power[:10])):
        _tag = f"= {_p / 24:.1f} days" if _p >= 24 else ""
        print(f"  [{_i + 1:2d}]  {_p:6.1f} h  {_tag:<18}  power={_pw:.4f}")
    return dominant_periods, dominant_power, gws


@app.cell
def _(dominant_periods, dominant_power, gws, mo, np, periods_cwt, plt):
    _ymax = np.max(gws) * 1.08
    _smooth = np.convolve(gws, np.ones(7) / 7, mode="same")

    _fig, _ax = plt.subplots(figsize=(11, 4))
    _ax.semilogx(periods_cwt, gws, color="darkorange", lw=1.5, label="GWS (raw)")
    _ax.semilogx(
        periods_cwt, _smooth, color="navy", lw=1.0, ls="--", alpha=0.7, label="GWS (smoothed)"
    )
    _ax.scatter(
        dominant_periods,
        dominant_power,
        color="red",
        zorder=6,
        s=70,
        label="Detected peaks",
        clip_on=False,
    )

    for _p, _pw in zip(dominant_periods[:8], dominant_power[:8]):
        _lbl = f"{_p:.0f} h" if _p < 36 else f"{_p / 24:.1f} d"
        _ax.annotate(
            _lbl,
            (_p, _pw),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="red",
        )

    for _ref, _name in [(12, "12 h"), (24, "24 h"), (168, "7 d")]:
        if periods_cwt.min() <= _ref <= periods_cwt.max():
            _ax.axvline(_ref, color="gray", ls=":", lw=0.9, alpha=0.7)
            _ax.text(_ref, _ymax * 0.97, _name, fontsize=8, color="gray", ha="center", va="top")

    _ax.set_xlim(periods_cwt.min(), periods_cwt.max())
    _ax.set_ylim(0, _ymax)
    _ax.set_xlabel("Period (hours, log scale)")
    _ax.set_ylabel("Mean wavelet power")
    _ax.set_title("Global Wavelet Spectrum — dominant period detection")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5  Discrete Wavelet Transform — Multi-Resolution Analysis

    ### How DWT multi-resolution works

    The DWT applies a bank of **high-pass** (detail *D*) and **low-pass** (approximation *A*)
    filters at successively coarser resolutions.
    Each detail level *j* captures oscillations in a dyadic period band:

    $$\text{period band at level }j \;=\; [2^j,\; 2^{j+1}]\times\Delta t \text{ hours}$$

    For hourly data with 7 levels:

    | Level | Period band | Physical signal |
    |-------|-------------|-----------------|
    | D1 | 2–4 h | Instrument noise, wave chop |
    | D2 | 4–8 h | Short-period swell tails |
    | D3 | **8–16 h** | **Semi-diurnal tidal (≈ 12 h)** |
    | D4 | **16–32 h** | **Diurnal cycle (≈ 24 h)** |
    | D5 | 32–64 h | Weather-system transitions |
    | D6 | 64–128 h | Storm cycle envelope |
    | D7 | 128–256 h | Synoptic swell propagation |
    | A7 | > 256 h | Slow seasonal trend |

    Each component is reconstructed back to the original length by zeroing all other
    coefficient arrays and calling `pywt.waverec`:

    ```python
    coeffs = pywt.wavedec(y, 'db8', level=7, mode='periodization')
    # coeffs = [cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD1]

    def reconstruct_level(coeffs, j):
        zeros = [np.zeros_like(c) for c in coeffs]
        zeros[len(coeffs) - j] = coeffs[len(coeffs) - j]
        return pywt.waverec(zeros, 'db8', mode='periodization')[:len(y)]
    ```

    > **Additivity:** `y ≈ D1 + D2 + … + D7 + A7` (exact up to numerical precision).
    """)
    return


@app.cell
def _(np, pywt, y):
    _LEVEL = 7

    dwt_coeffs = pywt.wavedec(y, "db8", level=_LEVEL, mode="periodization")
    # dwt_coeffs[0] = cA7, dwt_coeffs[1] = cD7, ..., dwt_coeffs[7] = cD1

    def _reconstruct(coeffs, j):
        n = len(coeffs)
        zeros = [np.zeros_like(c) for c in coeffs]
        idx = 0 if j == 0 else n - j
        zeros[idx] = coeffs[idx]
        return pywt.waverec(zeros, "db8", mode="periodization")[: len(y)]

    details = []
    for _j in range(1, _LEVEL + 1):
        _sig = _reconstruct(dwt_coeffs, _j)
        details.append(
            {
                "level": _j,
                "label": f"D{_j}: {2**_j}–{2 ** (_j + 1)} h",
                "signal": _sig,
                "energy": float(np.sum(_sig**2)),
            }
        )

    trend = _reconstruct(dwt_coeffs, 0)

    _recon = sum(d["signal"] for d in details) + trend
    print(f"Max reconstruction error: {np.max(np.abs(_recon - y)):.2e} m")
    print(f"Coefficient lengths (cA7…cD1): {[c.shape[0] for c in dwt_coeffs]}")
    return details, dwt_coeffs, trend


@app.cell
def _(details, mo, np, plt, hours, trend, y):
    _n = len(details)
    _fig, _axes = plt.subplots(_n + 2, 1, figsize=(14, 16), sharex=True)
    _colors = plt.cm.tab10(np.linspace(0, 1, _n))

    _axes[0].plot(hours, y, lw=0.7, color="steelblue")
    _axes[0].set_ylabel("Raw", fontsize=8)
    _axes[0].set_title("DWT Multi-Resolution Analysis (db8, 7 levels)")
    _axes[0].grid(True, alpha=0.3)

    for _i, _d in enumerate(details):
        _axes[_i + 1].plot(hours, _d["signal"], lw=0.8, color=_colors[_i])
        _axes[_i + 1].axhline(0, color="k", lw=0.4)
        _axes[_i + 1].set_ylabel(_d["label"], fontsize=7)
        _axes[_i + 1].grid(True, alpha=0.3)

    _axes[-1].plot(hours, trend, lw=1.5, color="darkred")
    _axes[-1].set_ylabel("A7: trend\n(> 256 h)", fontsize=7)
    _axes[-1].set_xlabel("Time (hours from start)")
    _axes[-1].grid(True, alpha=0.3)

    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reading the MRA plot:**

    * **D1–D2**: High-frequency chop and noise — no clear periodicity.
    * **D3 (8–16 h)**: Regular oscillations with a cycle length ≈ 12 h — the
      **semi-diurnal tidal signature**.
    * **D4 (16–32 h)**: Slower oscillations with ≈ 24 h period — the **diurnal component**
      (sea-breeze + diurnal tidal constituent). Amplitude modulates with time,
      reflecting the spring/neap tidal cycle.
    * **D5–D6**: Multi-day variability driven by passing storm systems.
    * **D7 + A7**: Slow swell envelope and seasonal trend.
    """)
    return


@app.cell
def _(mo):
    mo.md("## 6  Energy by Scale Level")
    return


@app.cell
def _(details, mo, np, plt, trend, y):
    _total = float(np.sum(y**2))
    _labels = [d["label"] for d in details] + ["A7: trend\n(> 256 h)"]
    _fracs = [d["energy"] / _total for d in details] + [float(np.sum(trend**2)) / _total]

    _fig, _ax = plt.subplots(figsize=(11, 4))
    _bars = _ax.bar(
        range(len(_labels)), _fracs, color=plt.cm.viridis(np.linspace(0.15, 0.9, len(_labels)))
    )
    for _b, _f in zip(_bars, _fracs):
        _ax.text(
            _b.get_x() + _b.get_width() / 2,
            _f + 0.002,
            f"{_f:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    _ax.set_xticks(range(len(_labels)))
    _ax.set_xticklabels(_labels, rotation=25, ha="right", fontsize=9)
    _ax.set_ylabel("Fraction of total signal energy")
    _ax.set_title("Energy distribution across DWT levels")
    _ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7  Isolated Seasonal Components (Overlay)

    Overlay the raw signal with the most physically meaningful extracted bands
    to confirm the wavelet decomposition captures what we expect.
    """)
    return


@app.cell
def _(details, mo, plt, hours, trend, y):
    _d3 = details[2]["signal"]  # 8–16 h  (semi-diurnal)
    _d4 = details[3]["signal"]  # 16–32 h (diurnal)
    _d5 = details[4]["signal"]  # 32–64 h (weather envelope)

    _fig, _axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    _axes[0].plot(hours, y, lw=0.7, color="steelblue", alpha=0.5, label="Raw Hₛ")
    _axes[0].plot(hours, trend, lw=2.0, color="darkred", label="A7 trend (> 256 h)")
    _axes[0].set_ylabel("Hₛ (m)")
    _axes[0].legend(loc="upper right", fontsize=9)
    _axes[0].set_title("Extracted seasonal components (DWT reconstruction)")
    _axes[0].grid(True, alpha=0.3)

    _axes[1].plot(hours, _d4, lw=1.2, color="darkorange", label="D4 (16–32 h, diurnal ≈ 24 h)")
    _axes[1].plot(
        hours, _d3, lw=0.8, color="purple", alpha=0.75, label="D3 (8–16 h, semi-diurnal ≈ 12 h)"
    )
    _axes[1].axhline(0, color="k", lw=0.4)
    _axes[1].set_ylabel("Amplitude (m)")
    _axes[1].legend(loc="upper right", fontsize=9)
    _axes[1].grid(True, alpha=0.3)

    _axes[2].plot(hours, _d5, lw=1.2, color="forestgreen", label="D5 (32–64 h, storm envelope)")
    _axes[2].axhline(0, color="k", lw=0.4)
    _axes[2].set_xlabel("Time (hours from start)")
    _axes[2].set_ylabel("Amplitude (m)")
    _axes[2].legend(loc="upper right", fontsize=9)
    _axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(dominant_periods, mo):
    _rows = "\n".join(
        f"| {i + 1} | {p:.1f} h | {p / 24:.2f} d |" for i, p in enumerate(dominant_periods[:8])
    )
    mo.md(f"""
    ## 8  Summary

    ### Periods Detected by the Global Wavelet Spectrum

    | Rank | Period (hours) | Period (days) |
    |------|---------------|---------------|
    {_rows}

    ### Physical Interpretation (Waves Oregon)

    | Period | Origin |
    |--------|--------|
    | **≈ 12 h** | Semi-diurnal tidal modulation (M2 tidal constituent) |
    | **≈ 24 h** | Diurnal sea-breeze cycle + K1/O1 tidal constituents |
    | **≈ 3–7 days** | Synoptic Pacific storm tracks; swell propagation |
    | **D1–D2 energy** | Broadband wind-sea noise — not a true periodicity |

    ### Key PyWavelets API Reference

    ```python
    import pywt

    # Explore available wavelets
    pywt.families()                          # ['haar', 'db', 'sym', 'coif', 'cmor', ...]
    pywt.wavelist(kind='continuous')         # CWT-compatible wavelets
    pywt.central_frequency('cmor1.5-1.0')   # → 1.0  (used for scale ↔ period)

    # Continuous Wavelet Transform (scalogram)
    fc     = pywt.central_frequency('cmor1.5-1.0')
    scales = fc * target_periods             # target_periods in hours
    coef, freqs = pywt.cwt(y, scales, 'cmor1.5-1.0', sampling_period=1.0)
    periods = 1.0 / freqs                    # hours per cycle
    power   = np.abs(coef) ** 2             # power spectrum

    # Discrete Wavelet Transform (multi-resolution)
    coeffs = pywt.wavedec(y, 'db8', level=7, mode='periodization')
    # returns [cA7, cD7, cD6, ..., cD1]  — level j detail at index (n - j)

    # Reconstruct a single level back to original length
    zeros = [np.zeros_like(c) for c in coeffs]
    zeros[len(coeffs) - j] = coeffs[len(coeffs) - j]
    d_j = pywt.waverec(zeros, 'db8', mode='periodization')[:len(y)]
    # Level j captures periods in [2^j, 2^(j+1)] * dt hours
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
