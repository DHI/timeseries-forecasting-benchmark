"""Per-case printable report (Phase F): get a feel for a case without running anything.

For one case, writes a self-contained Markdown report under docs/reports/<id>.md with:
  - the case header (domain, target, horizon, splits, source);
  - a data summary (span, observed fraction, gaps);
  - rolling-origin scores for a representative method set (MASE + skill);
  - an example-forecast figure at one test origin (observed vs each method);
  - the error-vs-lead-time curve.

Figures go to docs/figures/reports/. Reuses the bench eval + viz code.

    uv run python scripts/case_report.py --case surge_cuxhaven
    uv run python scripts/case_report.py            # all cases
    uv run python scripts/case_report.py --case waves_oregon --with-fm
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.bench.eval import (  # noqa: E402
    DEFAULT_ORIGINS,
    run_case_rolling,
    run_leadtime_curve,
)
from tsforecastbench.bench.eval.loop import _horizon_to_steps, _raw_y  # noqa: E402
from tsforecastbench.bench.methods import (  # noqa: E402
    DLinearForecaster,
    Persistence,
    SeasonalNaive,
    ThetaForecaster,
)
from tsforecastbench.bench.viz.leadtime import plot_leadtime_curve  # noqa: E402
from tsforecastbench.data.registry import discover_cases, get_case  # noqa: E402

REPORT_DIR = REPO_ROOT / "docs" / "reports"
FIG_DIR = REPO_ROOT / "docs" / "figures" / "reports"
FREQ_LABEL = {"1D": "days", "1h": "hours", "15min": "15-min steps", "1W": "weeks"}


def _methods(with_fm: bool):
    # LightGBM excluded: builds huge lag-feature matrices at high freq (e.g. bellinge 15-min),
    # making per-case reports impractically slow. Full LightGBM results live in RESULTS.md.
    ms = [Persistence(), SeasonalNaive(), ThetaForecaster(), DLinearForecaster()]
    if with_fm:
        try:
            from tsforecastbench.bench.methods import ChronosForecaster

            ms.append(ChronosForecaster())
        except Exception:
            pass
    return ms


def _example_figure(case, methods, out_path: Path) -> str:
    """Fit each method at one observed test origin and plot the forecast vs the truth."""
    df = case.load(curated=True).sort_values("ds").reset_index(drop=True)
    splits = case.split_frame(df)
    train_val = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    full = pd.concat([train_val, splits["test"]], ignore_index=True)
    exo = list(dict.fromkeys([*case.exogenous, *case.exogenous_known_future]))
    full_fit = full  # curated: the target is already complete within the span
    raw_y = _raw_y(full)
    H = _horizon_to_steps(case.horizon_primary, case.freq)

    # pick a dense, dynamic origin in the test span (max std of the horizon window)
    test_start = len(train_val)
    cands = np.linspace(test_start, len(full) - H, 40).astype(int)
    best_o, best_s = test_start, -1.0
    for o in cands:
        w = raw_y[o : o + H]
        if np.isfinite(w).mean() >= 0.8 and np.nanstd(w) > best_s:
            best_s, best_o = np.nanstd(w), int(o)
    o = best_o
    ctx = min(3 * H, o)
    hist = full_fit.iloc[:o]
    fexog = (
        full_fit.iloc[o : o + H][["ds", *[c for c in exo if c in full_fit.columns]]]
        if exo
        else None
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        full["ds"].iloc[o - ctx : o],
        raw_y[o - ctx : o],
        color="#333",
        lw=1.0,
        label="observed (history)",
    )
    ax.plot(
        full["ds"].iloc[o : o + H], raw_y[o : o + H], color="#333", lw=1.8, label="observed (truth)"
    )
    for m in methods:
        try:
            m.fit(case, hist)
            yh = m.predict(case, hist, H, future_exog=fexog)["y_hat"].to_numpy(float)
            ax.plot(full["ds"].iloc[o : o + H], yh, lw=1.2, alpha=0.9, label=m.name)
        except Exception:
            continue
    ax.axvline(full["ds"].iloc[o], color="grey", ls="--", lw=0.8)
    ax.set_title(
        f"{case.id}: example {case.horizon_primary} forecast from {full['ds'].iloc[o].date()}"
    )
    ax.set_ylabel(case.target)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out_path.relative_to(REPORT_DIR.parent))


def make_report(case_id: str, with_fm: bool, origins: int) -> Path:
    case = get_case(case_id)
    methods = _methods(with_fm)
    print(f"[report] {case_id}: scoring {len(methods)} methods over {origins} origins ...")
    scores, _ = run_case_rolling(case, methods, n_origins=origins)
    wide = scores.pivot_table(index="method", columns="metric", values="value")

    print(f"[report] {case_id}: lead-time curve ...")
    curve = run_leadtime_curve(case, methods, n_origins=origins)
    lt_metric = "RMSE"
    lt_fig = FIG_DIR / f"{case_id}_leadtime.png"
    plot_leadtime_curve(
        curve,
        metric=lt_metric,
        out_path=lt_fig,
        title=f"{case_id}: {lt_metric} vs lead time",
        freq_label=FREQ_LABEL.get(case.freq, "steps"),
    )
    _example_figure(case, methods, FIG_DIR / f"{case_id}_example.png")

    df = case.load()
    y = pd.to_numeric(df["y"], errors="coerce")
    span_days = (pd.to_datetime(df["ds"]).max() - pd.to_datetime(df["ds"]).min()).days

    order = [
        "Persistence",
        "SeasonalNaive",
        "Theta",
        "DLinear",
        "LightGBM",
        "LightGBM+exo",
        "Chronos-Bolt",
    ]
    rows = [m for m in order if m in wide.index]
    lines = [
        f"# Case report — `{case_id}`",
        "",
        f"**Domain** {case.domain} · **Target** `{case.target}` ({case.target_unit}) · "
        f"**Freq** `{case.freq}` · **Horizon** `{case.horizon_primary}`",
        "",
        f"*{case.notes}*",
        "",
        "## Data",
        "",
        f"- Source: {case.source.name}" + (f" — {case.source.url}" if case.source.url else ""),
        f"- Record span: {span_days} days, {int(y.notna().sum()):,} observed of {len(df):,} rows "
        f"({y.notna().mean():.0%})",
        f"- Splits ({case.split_mode}): train→{case.splits.train_end}, val→{case.splits.val_end}, "
        f"test→{case.splits.test_end}",
        f"- Exogenous: {', '.join(case.exogenous) or 'none'}"
        + (
            f" (known-future: {', '.join(case.exogenous_known_future)})"
            if case.exogenous_known_future
            else ""
        ),
        "",
        "## Scores (rolling-origin, pooled over origins x leads)",
        "",
        "| method | MASE | skill vs persistence | skill vs mean |",
        "|---|---|---|---|",
    ]
    for m in rows:
        r = wide.loc[m]
        lines.append(
            f"| {m} | {r.get('MASE', float('nan')):.3f} | "
            f"{r.get('Skill_vs_Persistence', float('nan')):+.3f} | "
            f"{r.get('Skill_vs_Mean', float('nan')):+.3f} |"
        )
    # report lives in docs/reports/, figures in docs/figures/reports/ — link is file-relative
    rel_ex = Path("../figures/reports") / f"{case_id}_example.png"
    rel_lt = Path("../figures/reports") / f"{case_id}_leadtime.png"
    lines += [
        "",
        "## Example forecast",
        "",
        f"![example]({rel_ex.as_posix()})",
        "",
        "## Error vs lead time",
        "",
        f"![leadtime]({rel_lt.as_posix()})",
        "",
        "_Generated by `scripts/case_report.py`._",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"{case_id}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {out.relative_to(REPO_ROOT)}")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case", help="single case id; default: all")
    p.add_argument("--with-fm", action="store_true", help="include Chronos-Bolt")
    p.add_argument(
        "--origins",
        type=int,
        default=DEFAULT_ORIGINS,
        help=f"rolling origins (default {DEFAULT_ORIGINS}; override with TSFB_ORIGINS)",
    )
    args = p.parse_args()
    ids = [args.case] if args.case else list(discover_cases())
    for cid in ids:
        make_report(cid, args.with_fm, args.origins)
    return 0


if __name__ == "__main__":
    sys.exit(main())
