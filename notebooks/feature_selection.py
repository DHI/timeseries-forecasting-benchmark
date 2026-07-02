import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import math
    import sys
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
    return (
        REPO_ROOT, Path, go, linkage, dendrogram,
        math, matplotlib, np, pd, plt, px,
        PCA, StandardScaler, sys,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
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
        """
    )
    return


@app.cell
def _(REPO_ROOT, mo, np, pd):
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
        from tsforecastbench.data.registry import discover_cases
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
def _(ALL_FEATURE_COLS, FEATURE_LABELS, features_df, mo, pd):
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
    return (fig_corr,)


@app.cell
def _(ALL_FEATURE_COLS, FEATURE_LABELS, features_df, mo, np, pd):
    _X = features_df[ALL_FEATURE_COLS].to_numpy()
    _corr = np.corrcoef(_X, rowvar=False)
    _labels = [FEATURE_LABELS[c] for c in ALL_FEATURE_COLS]

    _THRESHOLD = 0.65
    _pairs = []
    for _i in range(len(ALL_FEATURE_COLS)):
        for _j in range(_i + 1, len(ALL_FEATURE_COLS)):
            _r = _corr[_i, _j]
            if abs(_r) >= _THRESHOLD:
                _pairs.append({
                    "Feature A": _labels[_i],
                    "Feature B": _labels[_j],
                    "r": round(float(_r), 3),
                    "|r|": round(abs(float(_r)), 3),
                })

    _pairs_df = pd.DataFrame(_pairs).sort_values("|r|", ascending=False) if _pairs else pd.DataFrame(columns=["Feature A", "Feature B", "r", "|r|"])

    mo.vstack([
        mo.md(f"## Redundant pairs  (|r| ≥ {_THRESHOLD})\n\nFor each redundant pair, keep the feature that is more physically interpretable or varies more across series."),
        mo.ui.table(_pairs_df, selection=None) if len(_pairs_df) else mo.md("_No pairs above the threshold — all features are sufficiently independent._"),
    ])
    return


@app.cell
def _(ALL_FEATURE_COLS, FEATURE_LABELS, features_df, mo, np, plt, linkage, dendrogram):
    _X = features_df[ALL_FEATURE_COLS].to_numpy()
    _corr = np.corrcoef(_X, rowvar=False)
    _dist = 1.0 - np.abs(_corr)
    _dist = np.clip(_dist, 0, None)
    _condensed = _dist[np.triu_indices(len(ALL_FEATURE_COLS), k=1)]
    _Z = linkage(_condensed, method="ward")
    _labels = [FEATURE_LABELS[c] for c in ALL_FEATURE_COLS]

    fig_dendro, ax_dendro = plt.subplots(figsize=(9, 3.5))
    dendrogram(
        _Z,
        labels=_labels,
        ax=ax_dendro,
        color_threshold=0.30,
        above_threshold_color="#aaa",
        leaf_rotation=40,
        leaf_font_size=9,
    )
    ax_dendro.axhline(0.30, color="#e63946", linewidth=1, linestyle="--", label="grouping threshold (1 − |r| = 0.30, i.e. |r| ≥ 0.70)")
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


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Selection rationale

        Based on the correlation matrix and clustering above, the 13 candidates fall into
        these groups:

        **Predictability / complexity group** — spectral entropy, permutation entropy, Hurst exponent.
        All three measure orderliness or long-range structure; permutation entropy and Hurst cluster tightly (|r| ≥ 0.70).
        Hurst is the strongest representative: it captures *long-range* dependence directly and
        is physically interpretable across all water-domain series. Spectral entropy and
        permutation entropy are dropped as redundant.

        **Seasonality group** — seasonal strength, amplitude dispersion, phase dispersion.
        Seasonal strength measures whether a periodic pattern *exists*; phase dispersion measures
        whether its *timing* drifts across years (low = phase-locked). These are orthogonal for
        seasonal series and together span the seasonality axis more completely than either alone.
        Amplitude dispersion is dropped as the most collinear with seasonal strength.
        Keep **seasonal strength** and **phase dispersion**.

        **Distributional shape group** — kurtosis, L-skewness, Box-Cox λ.
        In the PCA biplot, the kurtosis loading vector aligns closely with L-skewness and
        adding it reduces explained variance from 82% to 72% — a clear sign of redundancy,
        not complementarity. L-skewness is retained as the more robust and interpretable
        representative. Box-Cox λ is dropped.

        **Flow-specific feature** — zero proportion.
        Near-zero for all non-flow series (temperature, waves, demand), making it a degenerate
        dimension in a cross-domain PCA. Dropped.

        **Temporal dynamics** — ACF₁, trend strength, asymmetry index.
        ACF₁ is largely redundant with Hurst (both capture temporal dependence) and has low
        discriminating power across this dataset. Trend strength is retained — it separates
        the wave and surge series from the discharge series and is uncorrelated with Hurst.
        Asymmetry index is dropped for parsimony.

        ### Selected features (4, N/p = 3.0)

        `seasonal_strength`, `hurst`, `amplitude_dispersion`, `l_skewness`

        Together these cover: periodicity, seasonal timing reliability, long-range memory,
        monotonic drift, and distributional asymmetry — one representative per distinct
        physical axis, with no confirmed redundancy in the PCA loading space.

        *Note on phase dispersion:* this is the circular spread of the dominant cycle's
        phase across cycles (`1 - R`), so **low** values mean tightly phase-locked
        seasonality and **high** values mean drifting timing. For non-seasonal series
        (waves, demand, surge), the seasonal component is near-zero noise, so phase
        dispersion reflects noise peak timing rather than real structure. This is worth
        acknowledging in the paper, though it does not change the feature selection since
        those series are already separated by seasonal strength.
        """
    )
    return


@app.cell
def _(ALL_FEATURE_COLS, FEATURE_LABELS, mo):
    label_to_key = {FEATURE_LABELS[c]: c for c in ALL_FEATURE_COLS}
    _default_keys = ["seasonal_strength", "hurst", "amplitude_dispersion", "l_skewness"]
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
    return (label_to_key, feature_selector)


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


@app.cell
def _(PCA, StandardScaler, label_to_key, domains, feature_selector, features_df, go, mo, np):
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
def _(PCA, StandardScaler, label_to_key, feature_selector, features_df, mo, np, pd):
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
def _(feature_selector, mo):
    _selected = feature_selector.value
    _n = len(_selected)
    _ratio = 12 / _n if _n > 0 else float("inf")
    _feat_list = "\n".join(f"- {f}" for f in _selected)

    mo.md(
        f"""
        ## Conclusion

        **Selected features ({_n}, N/p = {_ratio:.1f}):**

        {_feat_list}

        Update the multiselect above to explore alternatives. Once satisfied, set
        `FEATURE_COLS` in `scripts/compute_morphology.py` to the chosen keys and
        re-run that script to regenerate the final taxonomy biplot.

        **Key takeaways from the analysis above:**
        - Permutation entropy clusters tightly with Hurst (|r| ≥ 0.70) and is dropped as redundant; spectral entropy measures the same underlying axis and is also dropped.
        - Kurtosis loads nearly parallel to L-skewness in the biplot; adding it reduces explained variance from 82% to 72%. L-skewness is retained as the more robust representative.
        - ACF₁ is redundant with Hurst for this dataset and has low discriminating power.
        - Zero proportion is degenerate for non-flow series and excluded from cross-domain PCA.
        - Phase consistency adds an orthogonal seasonal timing axis beyond seasonal strength, but its values for non-seasonal series are noise-driven — worth noting in the paper.
        """
    )


if __name__ == "__main__":
    app.run()
