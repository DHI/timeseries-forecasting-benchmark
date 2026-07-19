"""Automated quality review for a single benchmark case.

Answers the four questions we keep re-asking every time a new case is built:

  1. Is the data record sufficiently FULL?            (coverage, gaps, grid regularity)
  2. Is the EXOGENOUS data present and informative?   (coverage + predictive-skill probe)
  3. Is the case actually FORECASTABLE?               (run trivial baselines + one ML method)
  4. Is the configuration internally CONSISTENT?      (toml vs data vs splits)

It computes everything measurable, prints a human-readable report, writes a JSON +
Markdown report under results/case_review/, and exits non-zero when the case FAILs
(``--strict`` also fails on WARN) so it can gate CI or a pre-add checklist.

The *judgement* part — is the target operationally meaningful, is the horizon natural,
does the exogenous set match the domain story — is deliberately left to the reviewer
(human or the review-case skill that wraps this script). This script gives them the
evidence; it does not pretend to know hydrology.

Run:
    uv run python scripts/review_case.py --case sewerlevel_bellinge
    uv run python scripts/review_case.py --case waves_oregon --json-only
    uv run python scripts/review_case.py --case wwtp_influent_n2o_350k --max-gap 24h --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.bench.eval.loop import _horizon_to_steps, run_case  # noqa: E402
from tsforecastbench.bench.methods.naive import (  # noqa: E402
    Climatology,
    MeanBaseline,
    Persistence,
    SeasonalNaive,
    _default_season_length,
)
from tsforecastbench.data.registry import discover_cases  # noqa: E402
from tsforecastbench.data.schema import Case  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "case_review"

# ---- verdict levels -------------------------------------------------------------
PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_RANK = {PASS: 0, WARN: 1, FAIL: 2}


@dataclass
class Check:
    """One named finding with a level (PASS/WARN/FAIL) and a one-line message."""

    name: str
    level: str
    message: str
    detail: dict = field(default_factory=dict)


def _worst(checks: list[Check]) -> str:
    return max((c.level for c in checks), key=lambda lvl: _RANK[lvl], default=PASS)


# ---------------------------------------------------------------------------------
# 1. Fullness: coverage, gaps, grid regularity
# ---------------------------------------------------------------------------------
def _step(freq: str) -> pd.Timedelta:
    return pd.Timedelta(pd.tseries.frequencies.to_offset(freq))


def check_fullness(case: Case, df: pd.DataFrame, max_gap: pd.Timedelta) -> list[Check]:
    checks: list[Check] = []
    ds_full = pd.to_datetime(df["ds"], utc=True).sort_values()
    y_full = pd.to_numeric(df["y"], errors="coerce").to_numpy()
    n = len(df)
    n_obs = int(np.isfinite(y_full).sum())

    # Trim to the OBSERVED SPAN [first finite, last finite]. Leading/trailing NaN means
    # the target record is shorter than the forcing record (normal for CAMELS-DK-style
    # cases with a long forcing window and a short gauge record) — that is not a gap, and
    # the fraction-split handles it. We judge fullness *within* the usable span.
    fin = np.flatnonzero(np.isfinite(y_full))
    if fin.size == 0:
        checks.append(Check("observed_fraction", FAIL, "Target has NO finite values at all.", {}))
        return checks
    lo, hi = int(fin[0]), int(fin[-1])
    y = y_full[lo : hi + 1]
    n_span = len(y)
    n_span_obs = int(np.isfinite(y).sum())
    frac = n_span_obs / n_span if n_span else 0.0
    lead_trail = n - n_span

    lvl = PASS if frac >= 0.8 else WARN if frac >= 0.5 else FAIL
    span_note = (
        f" (+{lead_trail:,} forcing-only rows outside the gauge record)" if lead_trail else ""
    )
    checks.append(
        Check(
            "observed_fraction",
            lvl,
            f"{frac:.0%} observed within the {n_span:,}-row record span "
            f"({n_span_obs:,} finite){span_note}.",
            {
                "observed_fraction_in_span": frac,
                "n_rows_span": n_span,
                "n_obs_span": n_span_obs,
                "lead_trail_nan_rows": lead_trail,
                "observed_fraction_all_rows": n_obs / n if n else 0.0,
            },
        )
    )

    # grid regularity: how many timestamps are missing vs a full range at case.freq.
    # Checked over the FULL frame (the canonical Parquet should be on a complete grid).
    step = _step(case.freq)
    if n >= 2:
        full = pd.date_range(ds_full.iloc[0], ds_full.iloc[-1], freq=case.freq)
        present = pd.DatetimeIndex(ds_full.unique())
        missing = len(full) - len(present.intersection(full))
        dup = int(ds_full.duplicated().sum())
        off_grid = int((~present.isin(full)).sum())
        reg_lvl = PASS if (missing == 0 and off_grid == 0 and dup == 0) else WARN
        checks.append(
            Check(
                "grid_regularity",
                reg_lvl,
                f"{missing:,} timestamps missing from the {case.freq} grid; "
                f"{off_grid:,} off-grid, {dup:,} duplicate.",
                {"missing_rows": missing, "off_grid": off_grid, "duplicates": dup},
            )
        )

    # gap runs in the target (consecutive NaN)
    nan = ~np.isfinite(y)
    gap_lens = _run_lengths(nan)
    if gap_lens:
        largest = max(gap_lens) * step
        n_gaps = len(gap_lens)
        gap_lvl = PASS if largest <= max_gap else WARN
        checks.append(
            Check(
                "target_gaps",
                gap_lvl,
                f"{n_gaps:,} gap(s) in the target; largest {largest} (threshold {max_gap}).",
                {
                    "n_gaps": n_gaps,
                    "largest_gap": str(largest),
                    "largest_gap_steps": int(max(gap_lens)),
                    "max_gap_threshold": str(max_gap),
                },
            )
        )
    else:
        checks.append(Check("target_gaps", PASS, "No gaps in the target series.", {}))

    # per-split coverage and test sufficiency
    try:
        splits = case.split_frame(df)
        horizon_steps = _horizon_to_steps(case.horizon_primary, case.freq)
        per_split = {}
        for name, part in splits.items():
            yy = pd.to_numeric(part["y"], errors="coerce").to_numpy()
            per_split[name] = {
                "rows": len(part),
                "obs_frac": float(np.isfinite(yy).mean()) if len(yy) else 0.0,
            }
        test_obs = int(np.isfinite(pd.to_numeric(splits["test"]["y"], errors="coerce")).sum())
        if test_obs == 0:
            tlvl, tmsg = FAIL, "Test split has NO finite target points — case cannot be scored."
        elif test_obs < horizon_steps:
            tlvl, tmsg = (
                FAIL,
                f"Test split has only {test_obs} finite points < one horizon ({horizon_steps}).",
            )
        elif per_split["train"]["obs_frac"] < 0.5:
            tlvl, tmsg = WARN, f"Train split only {per_split['train']['obs_frac']:.0%} observed."
        else:
            tlvl, tmsg = (
                PASS,
                (
                    f"train/val/test obs: "
                    f"{per_split['train']['obs_frac']:.0%}/"
                    f"{per_split['val']['obs_frac']:.0%}/"
                    f"{per_split['test']['obs_frac']:.0%}; "
                    f"test has {test_obs:,} finite points."
                ),
            )
        checks.append(Check("split_coverage", tlvl, tmsg, per_split))
    except Exception as exc:
        checks.append(Check("split_coverage", FAIL, f"split_frame failed: {exc!r}", {}))

    return checks


def _run_lengths(mask: np.ndarray) -> list[int]:
    """Lengths of consecutive True runs in a boolean array."""
    if mask.size == 0:
        return []
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return [int(b - a) for a, b in zip(idx[::2], idx[1::2], strict=False)]


# ---------------------------------------------------------------------------------
# 2. Exogenous: presence + predictive-skill probe
# ---------------------------------------------------------------------------------
def check_exogenous(case: Case, df: pd.DataFrame) -> list[Check]:
    exo = [*case.exogenous, *case.exogenous_known_future]
    if not exo:
        return [Check("exogenous", PASS, "Case declares no exogenous inputs (univariate).", {})]

    checks: list[Check] = []
    y = pd.to_numeric(df["y"], errors="coerce")
    for col in exo:
        if col not in df.columns:
            checks.append(
                Check(
                    f"exo[{col}]", FAIL, f"Declared exogenous '{col}' is MISSING from the data.", {}
                )
            )
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        finite = float(v.notna().mean())
        nonzero = float((v.abs() > 1e-12).mean())
        std = float(v.std(skipna=True))
        if finite == 0.0 or std == 0.0:
            checks.append(
                Check(
                    f"exo[{col}]",
                    FAIL,
                    f"'{col}' is empty or constant (finite={finite:.0%}, std={std:.3g}).",
                    {"finite_frac": finite, "nonzero_frac": nonzero, "std": std},
                )
            )
            continue
        # Spearman (rank) corr is robust to nonlinearity/scale. Use contemporaneous and a
        # short lead/lag scan to catch a delayed driver (e.g. rain -> level after Δt).
        best_corr, best_lag = _best_lagged_corr(y, v, max_lag=_corr_lag_scan(case.freq))
        acorr = abs(best_corr)
        # mostly-zero drivers (rain) are weak contemporaneously; flag low signal but not fatal
        clvl = PASS if acorr >= 0.2 else WARN
        checks.append(
            Check(
                f"exo[{col}]",
                clvl,
                f"'{col}': finite {finite:.0%}, nonzero {nonzero:.0%}, "
                f"best |Spearman| {acorr:.2f} at lag {best_lag} step(s).",
                {
                    "finite_frac": finite,
                    "nonzero_frac": nonzero,
                    "std": std,
                    "best_abs_spearman": acorr,
                    "best_lag_steps": best_lag,
                },
            )
        )

    # joint predictive-skill probe: does adding exogenous improve a one-step lag model?
    probe = _exo_skill_probe(case, df)
    if probe is not None:
        uplift = probe["rmse_uplift_pct"]
        plvl = PASS if uplift >= 5 else WARN
        checks.append(
            Check(
                "exo_skill_probe",
                plvl,
                f"Adding exogenous changes 1-step OOS RMSE by {uplift:+.1f}% "
                f"(lags-only {probe['rmse_base']:.4g} -> +exo {probe['rmse_exo']:.4g}). "
                "Information probe, not the operational model.",
                probe,
            )
        )
    return checks


def _corr_lag_scan(freq: str) -> int:
    f = freq.lower()
    if f in {"1d", "d"}:
        return 14
    if f in {"1h", "h"}:
        return 48
    if f in {"15min", "15t"}:
        return 96
    return 12


def _best_lagged_corr(y: pd.Series, v: pd.Series, max_lag: int) -> tuple[float, int]:
    """Largest |Spearman| between y(t) and v(t-lag) over lag in [0, max_lag]."""
    best_c, best_l = 0.0, 0
    yv = y.to_numpy()
    for lag in range(0, max_lag + 1):
        vv = v.shift(lag).to_numpy()
        m = np.isfinite(yv) & np.isfinite(vv)
        if m.sum() < 30:
            continue
        c = _spearman(yv[m], vv[m])
        if abs(c) > abs(best_c):
            best_c, best_l = c, lag
    return best_c, best_l


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _exo_skill_probe(case: Case, df: pd.DataFrame) -> dict | None:
    """Fit Ridge one-step-ahead with lags+calendar, with and without exogenous, on the
    train span; score OOS RMSE on the test span. Returns the % RMSE reduction from adding
    exogenous. This is an *upper-bound information probe* (uses contemporaneous exogenous),
    not the benchmark's operational forecast.
    """
    try:
        from sklearn.linear_model import Ridge
    except Exception:
        return None
    exo = [c for c in (*case.exogenous, *case.exogenous_known_future) if c in df.columns]
    if not exo:
        return None
    try:
        splits = case.split_frame(df)
        train = pd.concat([splits["train"], splits["val"]], ignore_index=True)
        test = splits["test"]
        lags = [1, 2, 3] + ([24] if case.freq.lower() in {"1h", "h"} else [])

        def design(part: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
            d = part.copy()
            d["y"] = pd.to_numeric(d["y"], errors="coerce")
            feats = {}
            for lg in lags:
                feats[f"ylag{lg}"] = d["y"].shift(lg)
            for c in cols:
                feats[c] = pd.to_numeric(d[c], errors="coerce")
            X = pd.DataFrame(feats)
            target = d["y"]
            keep = X.notna().all(axis=1) & target.notna()
            return X[keep].to_numpy(), target[keep].to_numpy()

        def rmse_for(cols: list[str]) -> float:
            Xtr, ytr = design(train, cols)
            Xte, yte = design(test, cols)
            if len(ytr) < 50 or len(yte) < 10:
                return float("nan")
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd[sd == 0] = 1.0
            model = Ridge(alpha=1.0).fit((Xtr - mu) / sd, ytr)
            pred = model.predict((Xte - mu) / sd)
            return float(np.sqrt(np.mean((yte - pred) ** 2)))

        rmse_base = rmse_for([])
        rmse_exo = rmse_for(exo)
        if not (np.isfinite(rmse_base) and np.isfinite(rmse_exo)) or rmse_base == 0:
            return None
        return {
            "rmse_base": rmse_base,
            "rmse_exo": rmse_exo,
            "rmse_uplift_pct": 100.0 * (rmse_base - rmse_exo) / rmse_base,
            "exogenous_used": exo,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------------
# 3. Forecastability: run trivial baselines + one ML method
# ---------------------------------------------------------------------------------
def check_forecastable(case: Case, prefer_synth: bool) -> tuple[list[Check], pd.DataFrame]:
    methods = [MeanBaseline(), Persistence(), SeasonalNaive(), Climatology()]
    # Add one cheap ML method if mlforecast/lightgbm import; skip silently otherwise.
    try:
        from tsforecastbench.bench.methods.ml import LinearLagsForecaster

        methods.append(LinearLagsForecaster())
    except Exception:
        pass

    try:
        res = run_case(case, methods, prefer_synth=prefer_synth)
    except Exception as exc:
        return [
            Check("forecastable", FAIL, f"Evaluation loop crashed: {exc!r}", {})
        ], pd.DataFrame()

    checks: list[Check] = []
    wide = res.pivot_table(index="method", columns="metric", values="value", aggfunc="first")
    status = res.groupby("method")["status"].first()
    n_ok = int((status == "ok").sum())
    if n_ok == 0:
        checks.append(
            Check(
                "forecastable",
                FAIL,
                "Every method errored — case is not forecastable as configured.",
                {"statuses": status.to_dict()},
            )
        )
        return checks, res

    # Does any real method beat persistence (positive skill) and seasonal-naive (MASE<1)?
    best_skill = float(wide.get("Skill_vs_Persistence", pd.Series(dtype=float)).max())
    best_mase = float(wide.get("MASE", pd.Series(dtype=float)).min())
    if np.isfinite(best_skill) and best_skill > 0.05:
        flvl, fmsg = (
            PASS,
            f"{n_ok}/{len(methods)} methods ran; best skill-vs-persistence {best_skill:+.2f}, best MASE {best_mase:.2f}.",
        )
    elif np.isfinite(best_mase) and best_mase < 1.0:
        flvl, fmsg = (
            PASS,
            f"{n_ok}/{len(methods)} methods ran; best MASE {best_mase:.2f} (beats seasonal-naive).",
        )
    else:
        flvl, fmsg = (
            WARN,
            (
                f"{n_ok}/{len(methods)} methods ran but none clearly beat the naive floor "
                f"(best skill {best_skill:+.2f}, best MASE {best_mase:.2f}) — check the target choice."
            ),
        )
    checks.append(
        Check(
            "forecastable",
            flvl,
            fmsg,
            {"n_methods_ok": n_ok, "best_skill_vs_persistence": best_skill, "best_mase": best_mase},
        )
    )

    # surface the observed fraction of the scored test window
    frac = float(res["frac_finite_test"].iloc[0]) if len(res) else 0.0
    if frac < 1.0:
        checks.append(
            Check(
                "test_window_observed",
                WARN if frac >= 0.5 else FAIL,
                f"Scored test window is only {frac:.0%} observed.",
                {"frac_finite_test": frac},
            )
        )
    return checks, res


# ---------------------------------------------------------------------------------
# 4. Config consistency
# ---------------------------------------------------------------------------------
def check_config(case: Case, df: pd.DataFrame) -> list[Check]:
    checks: list[Check] = []
    # horizon must fit the test window
    horizon_steps = _horizon_to_steps(case.horizon_primary, case.freq)
    season = _default_season_length(case.freq)
    checks.append(
        Check(
            "horizon",
            PASS,
            f"Primary horizon {case.horizon_primary} = {horizon_steps} steps; "
            f"default season {season} steps.",
            {"horizon_steps": horizon_steps, "season": season},
        )
    )
    # required columns
    missing = [c for c in ("unique_id", "ds", "y") if c not in df.columns]
    if missing:
        checks.append(Check("schema", FAIL, f"Missing required columns {missing}.", {}))
    # single series expectation (pilot is one series per case)
    n_series = df["unique_id"].nunique() if "unique_id" in df.columns else 0
    if n_series != 1:
        checks.append(
            Check(
                "series_count",
                WARN,
                f"{n_series} unique_id series (pilot expects 1).",
                {"n_series": n_series},
            )
        )
    # doc presence
    doc = REPO_ROOT / "docs" / "cases" / f"{case.id}.md"
    checks.append(
        Check(
            "case_doc",
            PASS if doc.exists() else WARN,
            f"Case doc {'found' if doc.exists() else 'MISSING'} at docs/cases/{case.id}.md.",
            {"doc_path": str(doc.relative_to(REPO_ROOT)) if doc.exists() else None},
        )
    )
    return checks


# ---------------------------------------------------------------------------------
# Orchestration + reporting
# ---------------------------------------------------------------------------------
def review(case: Case, max_gap: pd.Timedelta) -> dict:
    has_parquet = case.parquet_path.exists()
    df = case.load(prefer_synth=not has_parquet)
    df = df.sort_values("ds").reset_index(drop=True)

    sections: dict[str, list[Check]] = {
        "config": check_config(case, df),
        "fullness": check_fullness(case, df, max_gap),
        "exogenous": check_exogenous(case, df),
    }
    fc_checks, fc_res = check_forecastable(case, prefer_synth=not has_parquet)
    sections["forecastable"] = fc_checks

    all_checks = [c for cs in sections.values() for c in cs]
    verdict = _worst(all_checks)
    if not has_parquet:
        all_checks.append(
            Check(
                "data_source",
                WARN,
                "No processed Parquet — reviewed on SYNTHETIC data; rerun after download.",
                {},
            )
        )
        sections.setdefault("config", []).append(all_checks[-1])
        verdict = _worst(all_checks)

    return {
        "case_id": case.id,
        "domain": case.domain,
        "target": case.target,
        "freq": case.freq,
        "horizon_primary": case.horizon_primary,
        "data_source": "parquet" if has_parquet else "synthetic",
        "verdict": verdict,
        "sections": {k: [c.__dict__ for c in v] for k, v in sections.items()},
        "metrics_preview": (
            fc_res.pivot_table(index="method", columns="metric", values="value", aggfunc="first")
            .round(4)
            .reset_index()
            .to_dict(orient="records")
            if len(fc_res)
            else []
        ),
    }


_ICON = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL"}


def print_report(result: dict) -> None:
    print("=" * 78)
    print(f"CASE REVIEW · {result['case_id']}  [{result['verdict']}]")
    print(
        f"  {result['domain']} · target={result['target']} · {result['freq']} · "
        f"horizon {result['horizon_primary']} · data={result['data_source']}"
    )
    print("=" * 78)
    for checks in result["sections"].values():
        seen = set()
        for c in checks:
            key = (c["name"], c["message"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  [{_ICON[c['level']]}] {c['name']:<22s} {c['message']}")
    if result["metrics_preview"]:
        print("-" * 78)
        print("  Baseline metrics (single hold-out):")
        for row in result["metrics_preview"]:
            m = row.get("method", "?")
            print(
                f"    {m:>14s}  "
                f"MASE={_fmt(row.get('MASE'))}  "
                f"skill_vs_pers={_fmt(row.get('Skill_vs_Persistence'))}  "
                f"RMSE={_fmt(row.get('RMSE'))}"
            )
    print("=" * 78)
    print(f"  VERDICT: {result['verdict']}")
    print("=" * 78)


def _fmt(x) -> str:
    try:
        return f"{float(x):+.3f}" if x is not None and np.isfinite(float(x)) else "  nan "
    except Exception:
        return "  nan "


def write_reports(result: dict) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jpath = OUT_DIR / f"{result['case_id']}.json"
    jpath.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    lines = [f"# Case review — `{result['case_id']}`  ·  **{result['verdict']}**", ""]
    lines.append(f"- domain: {result['domain']}")
    lines.append(
        f"- target: `{result['target']}` · freq: `{result['freq']}` · horizon: `{result['horizon_primary']}`"
    )
    lines.append(f"- data source: **{result['data_source']}**")
    lines.append("")
    for section, checks in result["sections"].items():
        lines.append(f"## {section}")
        seen = set()
        for c in checks:
            key = (c["name"], c["message"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- **[{c['level']}]** `{c['name']}` — {c['message']}")
        lines.append("")
    if result["metrics_preview"]:
        lines.append("## baseline metrics (single hold-out)")
        lines.append("")
        lines.append("| method | MASE | skill_vs_persistence | RMSE | MAE |")
        lines.append("|---|---|---|---|---|")
        for row in result["metrics_preview"]:
            lines.append(
                f"| {row.get('method', '?')} | {_fmt(row.get('MASE'))} | "
                f"{_fmt(row.get('Skill_vs_Persistence'))} | {_fmt(row.get('RMSE'))} | {_fmt(row.get('MAE'))} |"
            )
        lines.append("")
    mpath = OUT_DIR / f"{result['case_id']}.md"
    mpath.write_text("\n".join(lines), encoding="utf-8")
    return jpath, mpath


def main() -> int:
    p = argparse.ArgumentParser(description="Automated quality review for one benchmark case.")
    p.add_argument("--case", required=True, help="case id (see src/tsforecastbench/cases/)")
    p.add_argument(
        "--max-gap",
        default=None,
        help="largest tolerable target gap, e.g. '24h', '7D'. Default scales with horizon.",
    )
    p.add_argument("--strict", action="store_true", help="exit non-zero on WARN as well as FAIL.")
    p.add_argument(
        "--json-only", action="store_true", help="suppress the text report; just write files."
    )
    args = p.parse_args()

    cases = discover_cases()
    if args.case not in cases:
        print(f"Unknown case {args.case!r}. Known: {sorted(cases)}", file=sys.stderr)
        return 2
    case = cases[args.case]

    # default: tolerate a gap up to one primary horizon
    max_gap = pd.Timedelta(args.max_gap) if args.max_gap else pd.Timedelta(case.horizon_primary)

    result = review(case, max_gap=max_gap)
    if not args.json_only:
        print_report(result)
    jpath, mpath = write_reports(result)
    print(f"\nwrote {jpath.relative_to(REPO_ROOT)} and {mpath.relative_to(REPO_ROOT)}")

    if result["verdict"] == FAIL:
        return 1
    if result["verdict"] == WARN and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
