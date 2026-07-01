import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import math
    import sys
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT / "src"))

    from tsforecastbench.data.registry import discover_cases, get_case
    from tsforecastbench.morphology import features as F

    return (
        F,
        PCA,
        REPO_ROOT,
        StandardScaler,
        dendrogram,
        discover_cases,
        get_case,
        go,
        linkage,
        np,
        os,
        pd,
        plt,
        px,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Morphological Feature Selection

    The morphological taxonomy uses a feature vector to place each water-domain
    time series in a latent feature space, then visualises the result via PCA.

    **The selection problem.** We have 13 candidate features (6 original + 7 extended
    from domain analysis). With only 12 benchmark series, PCA is reliable only when
    features are genuinely uncorrelated — correlated features inflate apparent dimensions
    without adding information. The usual rule of thumb is N/p ≥ 3, so with N = 12
    we should target **4–6 features** with pairwise |r| < 0.65.

    This notebook:
    1. Shows the full candidate feature matrix
    2. Identifies redundant feature groups via correlation and clustering
    3. Analyses which features vary meaningfully across series
    4. Lets you interactively pick the final set and preview the PCA biplot
    """)
    return


@app.cell
def _(REPO_ROOT, discover_cases, mo, pd):
    _CSV = REPO_ROOT / "results" / "morphology_features.csv"

    ALL_FEATURE_COLS = [
        "acf1", "trend_strength", "seasonal_strength", "spectral_entropy",
        "boxcox_lambda", "kurtosis",
        "hurst", "perm_entropy", "amplitude_dispersion", "phase_dispersion",
        "asymmetry", "l_skewness", "zero_proportion",
    ]

    FEATURE_LABELS = {
        "acf1":               "ACF₁",
        "trend_strength":     "Trend strength",
        "seasonal_strength":  "Seasonal strength",
        "spectral_entropy":   "Spectral entropy",
        "boxcox_lambda":      "Box-Cox λ",
        "kurtosis":           "Kurtosis",
        "hurst":              "Hurst exponent",
        "perm_entropy":       "Permutation entropy",
        "amplitude_dispersion": "Amplitude dispersion",
        "phase_dispersion":   "Phase dispersion",
        "asymmetry":          "Asymmetry index",
        "l_skewness":         "L-skewness",
        "zero_proportion":    "Zero proportion",
    }

    if _CSV.exists():
        features_df = pd.read_csv(_CSV, index_col="case_id")
        domains = features_df["domain"].to_dict() if "domain" in features_df.columns else {}
        features_df = features_df[ALL_FEATURE_COLS]
        _source = "precomputed CSV"
    else:
        # Compute live — slow but self-contained
        from tsforecastbench.morphology.features import compute_features_all

        _cases = discover_cases()
        _records: dict = {}
        domains: dict = {}
        for _case_id, _case in _cases.items():
            if not _case.parquet_path.exists():
                continue
            _df = _case.load()
            _train = _case.split_frame(_df)["train"]
            _mask = _train["y"].notna()
            _y = _train.loc[_mask, "y"].to_numpy(dtype=float)
            _records[_case_id] = compute_features_all(_y)
            domains[_case_id] = _case.domain
        features_df = pd.DataFrame.from_dict(_records, orient="index")[ALL_FEATURE_COLS]
        features_df.index.name = "case_id"
        _source = "live computation"

    mo.callout(
        mo.md(f"Features loaded from **{_source}** · {len(features_df)} series × {len(ALL_FEATURE_COLS)} features"),
        kind="info",
    )
    return ALL_FEATURE_COLS, FEATURE_LABELS, domains, features_df


@app.cell
def _(ALL_FEATURE_COLS, FEATURE_LABELS, features_df, mo):
    _display = features_df[ALL_FEATURE_COLS].copy()
    _display.columns = [FEATURE_LABELS[c] for c in ALL_FEATURE_COLS]
    _display = _display.round(3).reset_index()
    _display = _display.rename(columns={"case_id": "Case"})

    mo.vstack([
        mo.md("## Candidate feature matrix\n\nAll 13 features computed on the training split of each case."),
        mo.ui.table(_display, selection=None),
    ])
    return


@app.cell
def _(ALL_FEATURE_COLS, FEATURE_LABELS, features_df, mo, np, px):
    _X = features_df[ALL_FEATURE_COLS].to_numpy()
    _n = len(ALL_FEATURE_COLS)
    _corr = np.corrcoef(_X, rowvar=False)
    _labels = [FEATURE_LABELS[c] for c in ALL_FEATURE_COLS]

    fig_corr = px.imshow(
        _corr,
        x=_labels,
        y=_labels,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        aspect="auto",
        title="Pairwise Pearson correlation between candidate features",
    )
    fig_corr.update_layout(
        height=520,
        margin=dict(t=50, b=20, l=160, r=20),
        coloraxis_colorbar=dict(title="r", thickness=12),
        xaxis=dict(tickangle=-40),
    )
    fig_corr.update_traces(textfont_size=9)

    mo.vstack([
        mo.md("## Correlation matrix\n\nHigh |r| (dark red or blue) indicates redundant feature pairs. Target: no selected pair above |r| ≈ 0.65."),
        mo.ui.plotly(fig_corr),
    ])
    return


@app.cell
def _(
    ALL_FEATURE_COLS,
    FEATURE_LABELS,
    dendrogram,
    features_df,
    linkage,
    mo,
    np,
    plt,
):
    _X = features_df[ALL_FEATURE_COLS].to_numpy()
    _corr = np.corrcoef(_X, rowvar=False)
    _dist = 1.0 - np.abs(_corr)
    _dist = np.clip(_dist, 0, None)
    _condensed = _dist[np.triu_indices(len(ALL_FEATURE_COLS), k=1)]
    _Z = linkage(_condensed, method="ward")
    _labels = [FEATURE_LABELS[c] for c in ALL_FEATURE_COLS]

    dendro_threshold = 0.5

    fig_dendro, ax_dendro = plt.subplots(figsize=(9, 3.5))
    dendrogram(
        _Z,
        labels=_labels,
        ax=ax_dendro,
        color_threshold=dendro_threshold,
        above_threshold_color="#aaa",
        leaf_rotation=40,
        leaf_font_size=9,
    )
    ax_dendro.axhline(dendro_threshold, color="#e63946", linewidth=1, linestyle="--", label=f"grouping threshold (1 − |r| = {dendro_threshold}, i.e. |r| ≥ {round(1- dendro_threshold, 2)})")
    ax_dendro.set_ylabel("1 − |r|  (distance)", fontsize=9)
    ax_dendro.set_title("Hierarchical clustering of candidate features", fontsize=10)
    ax_dendro.legend(fontsize=8, frameon=False)
    plt.tight_layout()

    mo.vstack([
        mo.md(
            "## Feature clustering\n\n"
            "Features linked below the red dashed line form a **redundancy group** — "
            "pick one representative per group. Features far apart are genuinely independent."
        ),
        mo.mpl.interactive(fig_dendro),
    ])
    return


@app.cell
def _(ALL_FEATURE_COLS, FEATURE_LABELS, features_df, mo, np, pd, px):
    _X = features_df[ALL_FEATURE_COLS].to_numpy()
    _q25 = np.percentile(_X, 25, axis=0)
    _q75 = np.percentile(_X, 75, axis=0)
    _range = _X.max(axis=0) - _X.min(axis=0)
    _spread = np.where(_range > 1e-10, (_q75 - _q25) / _range, 0.0)

    _spread_df = pd.DataFrame({
        "Feature": [FEATURE_LABELS[c] for c in ALL_FEATURE_COLS],
        "IQR / range": _spread.tolist(),
    }).sort_values("IQR / range", ascending=True)

    fig_var = px.bar(
        _spread_df,
        x="IQR / range",
        y="Feature",
        orientation="h",
        title="Feature discriminating power — IQR / range across series  (higher = more spread between series)",
        height=420,
    )
    fig_var.update_traces(marker_color="#4C72B0")
    fig_var.update_layout(
        margin=dict(t=50, b=20, l=200, r=40),
        xaxis=dict(range=[0, 1], tickformat=".0%"),
    )

    mo.vstack([
        mo.md(
            "## Feature discriminating power\n\n"
            "IQR / range: the interquartile spread of each feature across the 12 series, "
            "normalised by the feature's full range. A value near 1 means the middle 50 % "
            "of series span most of the feature's scale — high separation. "
            "Near 0 means most series cluster together — low discriminating power."
        ),
        mo.ui.plotly(fig_var),
    ])
    return


@app.cell
def _(ALL_FEATURE_COLS, FEATURE_LABELS, mo):
    label_to_key = {FEATURE_LABELS[c]: c for c in ALL_FEATURE_COLS}
    _default_keys = ["spectral_entropy", "hurst", "phase_dispersion", "l_skewness"]
    _default_labels = [FEATURE_LABELS[k] for k in _default_keys]

    feature_selector = mo.ui.multiselect(
        options=list(label_to_key.keys()),
        value=_default_labels,
        label="Select features for PCA",
    )

    mo.vstack([
        mo.md("## Interactive feature selector"),
        mo.md("Choose which features to include. The PCA biplot below updates immediately."),
        feature_selector,
    ])
    return feature_selector, label_to_key


@app.cell
def _(feature_selector, mo):
    _n_series = 12
    _n_features = len(feature_selector.value)
    _ratio = _n_series / _n_features if _n_features > 0 else float("inf")

    if _n_features == 0:
        _np_out = mo.callout(mo.md("Select at least one feature."), kind="danger")
    elif _ratio < 2.5:
        _np_out = mo.callout(
            mo.md(f"**N/p = {_ratio:.1f}** — too many features for {_n_series} series. PCA loadings will be unreliable. Target N/p ≥ 3."),
            kind="warn",
        )
    elif _ratio < 3:
        _np_out = mo.callout(
            mo.md(f"**N/p = {_ratio:.1f}** — borderline. Acceptable for a descriptive projection, but interpret loading magnitudes cautiously."),
            kind="warn",
        )
    else:
        _np_out = mo.callout(
            mo.md(f"**N/p = {_ratio:.1f}** — {_n_features} features, good for a descriptive PCA with {_n_series} series."),
            kind="success",
        )
    _np_out
    return


@app.cell
def _(
    PCA,
    StandardScaler,
    domains,
    feature_selector,
    features_df,
    go,
    label_to_key,
    mo,
    np,
):
    _selected_labels = feature_selector.value
    _selected_keys = [label_to_key[lbl] for lbl in _selected_labels]

    if len(_selected_keys) < 2:
        _biplot_out = mo.md("_Select at least 2 features to show the PCA biplot._")
    else:
        _X = features_df[_selected_keys].to_numpy()
        _X_scaled = StandardScaler().fit_transform(_X)
        _pca = PCA(n_components=2)
        _scores = _pca.fit_transform(_X_scaled)
        _loadings = _pca.components_.T  # (n_features × 2)
        _ev = _pca.explained_variance_ratio_

        _case_ids = list(features_df.index)
        _domain_list = [domains.get(cid, "unknown") for cid in _case_ids]

        _unique_domains = sorted(set(_domain_list))
        _palette = [
            "#4C72B0", "#DD8452", "#55A868", "#C44E52",
            "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
        ]
        _domain_colour = {d: _palette[i % len(_palette)] for i, d in enumerate(_unique_domains)}

        _fig_pca = go.Figure()

        for _uid in _unique_domains:
            _idx = [i for i, d in enumerate(_domain_list) if d == _uid]
            _fig_pca.add_trace(go.Scatter(
                x=_scores[_idx, 0].tolist(),
                y=_scores[_idx, 1].tolist(),
                mode="markers+text",
                name=_uid,
                text=[_case_ids[i] for i in _idx],
                textposition="top center",
                textfont=dict(size=8, color=_domain_colour[_uid]),
                marker=dict(size=10, color=_domain_colour[_uid], line=dict(width=1, color="white")),
            ))

        _score_range = max(np.abs(_scores).max(), 1e-9)
        _load_max = max(np.abs(_loadings).max(), 1e-9)
        _arrow_scale = 0.75 * _score_range / _load_max

        for _fi, _lbl in enumerate(_selected_labels):
            _l1 = float(_loadings[_fi, 0]) * _arrow_scale
            _l2 = float(_loadings[_fi, 1]) * _arrow_scale
            _fig_pca.add_annotation(
                x=_l1, y=_l2, ax=0, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#333",
            )
            _fig_pca.add_trace(go.Scatter(
                x=[_l1 * 1.12], y=[_l2 * 1.12],
                mode="text",
                text=[_lbl],
                textfont=dict(size=8, color="#222"),
                showlegend=False,
            ))

        _fig_pca.update_layout(
            title=f"PCA biplot — {len(_selected_keys)} features · PC1 {_ev[0]:.1%} + PC2 {_ev[1]:.1%} = {sum(_ev):.1%} variance",
            xaxis_title=f"PC1  ({_ev[0]:.1%})",
            yaxis_title=f"PC2  ({_ev[1]:.1%})",
            height=520,
            margin=dict(t=60, b=40, l=60, r=20),
            legend=dict(title="Domain", orientation="h", x=0.5, xanchor="center", y=-0.12),
            xaxis=dict(zeroline=True, zerolinecolor="#ddd", zerolinewidth=1),
            yaxis=dict(zeroline=True, zerolinecolor="#ddd", zerolinewidth=1, scaleanchor="x", scaleratio=1),
        )

        _biplot_out = mo.vstack([
            mo.md("## PCA biplot"),
            mo.ui.plotly(_fig_pca),
        ])

    _biplot_out
    return


@app.cell
def _(
    PCA,
    StandardScaler,
    feature_selector,
    features_df,
    label_to_key,
    mo,
    np,
    pd,
):
    _selected_labels = feature_selector.value
    _selected_keys = [label_to_key[lbl] for lbl in _selected_labels]

    if len(_selected_keys) >= 2:
        _X = features_df[_selected_keys].to_numpy()
        _X_scaled = StandardScaler().fit_transform(_X)
        _pca = PCA(n_components=min(2, len(_selected_keys)))
        _pca.fit(_X_scaled)
        _loadings = _pca.components_.T

        _tbl = pd.DataFrame(
            np.round(_loadings, 3),
            index=_selected_labels,
            columns=[f"PC{i+1}" for i in range(_loadings.shape[1])],
        ).reset_index().rename(columns={"index": "Feature"})

        _loadings_out = mo.vstack([
            mo.md("### PC loadings\n\nLarge absolute values indicate that a feature strongly defines that principal component."),
            mo.ui.table(_tbl, selection=None),
        ])
    else:
        _loadings_out = mo.md("")

    _loadings_out
    return


@app.cell
def _(feature_selector):
    _selected = feature_selector.value
    _n = len(_selected)
    _ratio = 12 / _n if _n > 0 else float("inf")
    return


@app.cell
def _(ALL_FEATURE_COLS, mo):
    _EXPLAIN = {
        "acf1":               "Lag-1 autocorrelation — persistence from one step to the next.",
        "trend_strength":     "Variance share of the polynomial trend. High → strong deterministic drift.",
        "seasonal_strength":  "De-trended variance share of the seasonal fit. High → dominant periodicity.",
        "spectral_entropy":   "Spectrum flatness [0,1]. Low → energy in a few frequencies; high → broadband noise.",
        "boxcox_lambda":      "Box-Cox power that best symmetrises the distribution.",
        "kurtosis":           "Excess kurtosis — tail heaviness. High → spiky, extreme-event-prone.",
        "hurst":              "Hurst exponent. ≈0.5 → random-walk; >0.5 → long-range persistence / slow wander.",
        "perm_entropy":       "Permutation entropy [0,1]. Low → structured/predictable; high → complex.",
        "amplitude_dispersion": "Cycle-to-cycle amplitude variability [0,1). 0 → identical swing each cycle.",
        "phase_dispersion":   "Drift of dominant-cycle timing [0,1]. 0 → phase-locked; 1 → timing wanders.",
        "asymmetry":          "Rise/fall asymmetry (−1,1). Positive → faster rises (flashy hydrograph).",
        "l_skewness":         "Robust (L-moment) skewness. 0 → symmetric; positive → heavy upper tail.",
        "zero_proportion":    "Fraction of near-zero values. High → intermittent / zero-inflated regime.",
    }
    _fit = {"trend_strength", "seasonal_strength", "amplitude_dispersion", "phase_dispersion"}
    _rows = "\n".join(
        f"| `{k}` | {'fit' if k in _fit else 'direct'} | {_EXPLAIN[k]} |"
        for k in ALL_FEATURE_COLS
    )
    _table = "| Feature | Source | What it measures |\n|---|---|---|\n" + _rows
    mo.vstack([
        mo.md(
            r"""
            ---

            ## Feature reference

            All 13 candidates derive from a single structural fit
            $y(t) = T(t) + S(t) + e(t)$, where $T$ is a polynomial trend
            (AIC-selected degree 1–3), $S = \sum_p S_p$ sums Fourier blocks per detected
            period ($K = 3$ harmonics each), and $e$ is the residual.
            **Source — fit**: read directly off the decomposition;
            **direct**: computed from the raw series.

            **Strength features** (variance ratios):
            $$\text{trend\_strength} = \max\!\Big(0,\;1 - \tfrac{\operatorname{Var}(e)}{\operatorname{Var}(T+e)}\Big),
            \qquad
            \text{seasonal\_strength} = \max\!\Big(0,\;1 - \tfrac{\operatorname{Var}(e)}{\operatorname{Var}(S+e)}\Big)$$

            **Dispersion features** — per dominant cycle $k$, project onto the fundamental sinusoid
            to get amplitude $A_k$ and phase $\varphi_k$:
            $$\text{amplitude\_dispersion} = \frac{c}{1+c},\quad
            c = \frac{\operatorname{std}(A_k)}{\operatorname{mean}(A_k)}; \qquad
            \text{phase\_dispersion} = 1 - \big|\operatorname{mean}(e^{i\varphi_k})\big|$$
            """
        ),
        mo.md(_table),
    ])
    return


@app.cell
def _(discover_cases, mo, os):
    _cases = list(discover_cases())
    _default = os.environ.get("EDA_CASE")
    _init = _default if _default in _cases else _cases[0]
    case_picker = mo.ui.dropdown(_cases, value=_init, label="Case")
    mo.vstack([
        mo.md(
            "---\n\n"
            "## Case explorer\n\n"
            "Pick any benchmark case to see its structural decomposition "
            "and where it sits in the morphological feature space."
        ),
        case_picker,
    ])
    return (case_picker,)


@app.cell
def _(F, case_picker, get_case, mo, np):
    mo.stop(not case_picker.value, mo.md("_Select a case above._"))
    case = get_case(case_picker.value)
    _df_exp = case.load().dropna(subset=["y"]).reset_index(drop=True)
    y = _df_exp["y"].to_numpy().astype(float)
    t = np.arange(len(y), dtype=float)
    d = F._decompose(y)
    if d.periods:
        dom_p = d.periods[d.dom_idx]
        _win_len = int(min(len(y), max(dom_p * 8, 200)))
    else:
        dom_p = None
        _win_len = int(min(len(y), 1000))
    WIN = slice(0, _win_len)
    return WIN, case, d, dom_p, t, y


@app.cell
def _(WIN, case, d, mo, plt, t, y):
    _fig, _ax = plt.subplots(4, 1, figsize=(10, 8.5), sharex=True)
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
def _(F, d, dom_p, mo, np, plt):
    if dom_p is None:
        _cyc_out = mo.md(
            "_No dominant cycle detected — `amplitude_dispersion` and `phase_dispersion` fall back to the 1.0 sentinel._"
        )
    else:
        _g = d.resid + d.seasonal_arrays[d.dom_idx]
        _res = F._cycle_amplitude_phase(_g, dom_p)
        if _res is None:
            _cyc_out = mo.md("_Fewer than two complete cycles — dispersion features use the 1.0 sentinel._")
        else:
            _amp, _phase = _res
            _n_cyc = len(_g) // dom_p
            _chunks = _g[: _n_cyc * dom_p].reshape(_n_cyc, dom_p)
            _cv = _amp.std(ddof=1) / _amp.mean()
            _R = np.abs(np.exp(1j * _phase).mean())

            _fig, _ax = plt.subplots(1, 2, figsize=(10, 3.6))
            for _c in _chunks[:: max(1, _n_cyc // 60)]:
                _ax[0].plot(np.arange(dom_p), _c, lw=0.4, color="steelblue", alpha=0.25)
            _ax[0].plot(np.arange(dom_p), _chunks.mean(axis=0), lw=2.0, color="crimson", label="mean cycle")
            _ax[0].set_title(f"Dominant cycle overlaid  (p = {dom_p} steps)")
            _ax[0].set_xlabel("phase within cycle (steps)")
            _ax[0].legend(fontsize=8)
            _ax[0].grid(True, alpha=0.3)

            _ax[1].plot(_amp, lw=1.0, color="darkorange")
            _ax[1].set_title(
                f"Per-cycle amplitude  ·  "
                f"amplitude_dispersion = {_cv / (1 + _cv):.3f}  ·  "
                f"phase_dispersion = {float(np.clip(1 - _R, 0, 1)):.3f}"
            )
            _ax[1].set_xlabel("cycle index")
            _ax[1].axhline(_amp.mean(), color="gray", ls="--", lw=1.0)
            _ax[1].grid(True, alpha=0.3)
            plt.tight_layout()
            _cyc_out = mo.mpl.interactive(_fig)
    _cyc_out
    return


@app.cell
def _(ALL_FEATURE_COLS, F, case, features_df, mo, np, pd, plt, y):
    _feats = F.compute_features_all(y)
    _ncol = 4
    _nrow = int(np.ceil(len(ALL_FEATURE_COLS) / _ncol))
    _fig, _axes = plt.subplots(_nrow, _ncol, figsize=(10, 2.2 * _nrow))
    _axes = np.atleast_1d(_axes).ravel()

    for _i, _name in enumerate(ALL_FEATURE_COLS):
        _ax = _axes[_i]
        _vals = (
            pd.to_numeric(features_df[_name], errors="coerce").dropna().to_numpy()
            if _name in features_df.columns else np.array([])
        )
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

    for _j in range(len(ALL_FEATURE_COLS), len(_axes)):
        _axes[_j].axis("off")

    plt.tight_layout()
    mo.vstack([
        mo.md(
            "### Feature profile — where this case sits\n\n"
            "Red line = this case. Box = spread across all benchmark series. "
            "Features recomputed live from the full series."
        ),
        mo.mpl.interactive(_fig),
    ])
    return


if __name__ == "__main__":
    app.run()
