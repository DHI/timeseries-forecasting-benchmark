"""Measure fit cost against training-history length, to choose the benchmark's training cap.

Issue #95: every method used to truncate training history to its own hand-picked cap — 3,000
points for TBATS, 30,000 for the neuralforecast zoo, no cap at all for the mlforecast family
— so a column of the results matrix compared methods at unequal data budgets. This script
measured what removing those caps would cost; the answer (`docs/results/train_cap_calibration.csv`)
was that even the expensive classical methods reach the full 182,304-point record in minutes
per fit, so the caps were removed outright rather than equalised at a shared value.

It stays useful for re-checking that decision: on new hardware, after a method's internals
change, or when a case arrives that is much larger than anything here. What it answers is
"how fast does this method's fit cost grow with history, and where does that stop being
affordable" — the question behind `eval.budget.TRAIN_POINT_CEILING`.

What this measures: `fit` only — one call per (method, size), with no rolling origins, no
predict and no scoring. Fit is where the cap bites. Predict is deliberately left uncapped and
is already reported (see `methods/classical.py`, choice 3 in the module docstring).

How it stays cheap. Sizes escalate along a ladder and a method is dropped as soon as it
crosses `--budget`, so nothing is ever asked for the size after the one that was already too
slow. A second guard projects the next size's cost from the last measurement, assuming
quadratic growth, and stops early when that projection exceeds `--hard-limit` x `--budget`.
That guard is what keeps TBATS from disappearing into a multi-hour fit at 100,000 points.
Every measurement is written to the output CSV as it completes, so an interrupted run keeps
whatever it had already measured.

Any surviving history cap is lifted for the measurement, so the curve reflects the method
rather than its cap. No method carries one any more — MLP's went with #97, which traced its
apparent per-horizon-step cost to BLAS thread oversubscription rather than to the strategy.
Window sub-sampling (`DLinear.max_samples`, `torchdl` window striding) is left at its
default: it strides windows across the whole history rather than discarding the front of the
record, so it is a different knob and would confound the curve.

The deep-learning methods are off by default. Their fit cost is set by the gradient-step
budget rather than by how much history they are handed — the audit in research log entry 89
measured TFT's fit time as flat across an 8x difference in training size — so the curve is
near-flat and costs minutes per point to trace. Use `--with-dl` / `--with-nf` to confirm that
it is flat, not to find a cap.

`--methods` narrows a run to one method, for the "a method's internals changed" case above.
It writes only what it measured, so pair it with `--out`: the committed CSV is replaced only
by a full run, keeping every row in that file from one revision rather than mixing
measurements taken before and after a change.

Usage:
    uv run python scripts/calibrate_train_caps.py
    uv run python scripts/calibrate_train_caps.py --cases surge_cuxhaven --budget 300
    uv run python scripts/calibrate_train_caps.py --methods MLP --out /tmp/mlp_caps.csv
    uv run python scripts/calibrate_train_caps.py --with-dl --with-nf
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import pandas as pd

# Allow running from a clone without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsforecastbench.bench.methods import (
    AutoARIMAForecaster,
    AutoETSForecaster,
    CatBoostForecaster,
    DLinearForecaster,
    LightGBMForecaster,
    LinearLagsForecaster,
    MLPForecaster,
    MSTLAutoARIMAForecaster,
    MSTLAutoETSForecaster,
    TBATSForecaster,
    ThetaForecaster,
)
from tsforecastbench.data.registry import discover_cases

REPO_ROOT = Path(__file__).resolve().parents[1]
# docs/results/ is the committed, distilled corner of the results tree (results/ itself is
# gitignored). The caps derived here are meant to be frozen into the code, so the evidence
# behind them belongs in git next to the scores it explains.
DEFAULT_OUT = REPO_ROOT / "docs" / "results" / "train_cap_calibration.csv"

# The two most demanding cases: the longest record, and the highest-frequency one (15min,
# whose large seasonal periods drive STL cost for the MSTL pair — cost grows with n x period,
# so a cap derived on an hourly case does not transfer).
DEFAULT_CASES = ("surge_cuxhaven", "sewerlevel_bellinge")

LADDER = (2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000)

# Large enough to disable every history cap in the codebase without special-casing the
# `None`-means-uncapped convention in ml.py against the plain ints used elsewhere.
_NO_CAP = 10**9


def build_methods():
    """Every CPU method whose fit cost depends on how much history it is handed.

    The mlforecast trio (LinearLags, LightGBM, CatBoost) is currently uncapped, so it is
    here to establish what the top of the ladder costs, not to find a cap it is missing.
    """
    return [
        AutoARIMAForecaster(),
        AutoETSForecaster(),
        MSTLAutoETSForecaster(),
        MSTLAutoARIMAForecaster(),
        TBATSForecaster(),
        ThetaForecaster(),
        LinearLagsForecaster(),
        LightGBMForecaster(),
        CatBoostForecaster(),
        MLPForecaster(),
        DLinearForecaster(),
    ]


def uncap(method) -> None:
    """Lift a method's own history cap so the fit sees exactly the frame we hand it.

    Since #95 the shared default is `eval.budget`'s ceiling, which never binds, and #97
    removed the last per-method cap. This exists so the script still measures the method
    rather than a cap when run against an older revision.
    """
    if hasattr(method, "max_train_steps"):
        method.max_train_steps = _NO_CAP


def fit_frames(case, prefer_synth: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The frames a global method is fit on in a rolling run, test excluded.

    The train split, whose length is what the ladder walks, and the declared val split,
    handed over for the iteratively trained methods to stop on (gh #99). Keeping val out of
    the ladder is what makes the measured cost the cost of the real fit path.
    """
    df = case.load(prefer_synth=prefer_synth, curated=True)
    splits = case.split_frame(df)
    return splits["train"], splits["val"]


def sizes_for(n_available: int) -> list[int]:
    """Ladder rungs that fit the record, plus the full length when it clears the last rung."""
    rungs = [n for n in LADDER if n <= n_available]
    if not rungs or rungs[-1] < n_available:
        rungs.append(n_available)
    return rungs


def measure_method(case, method, train, val, sizes, budget, hard_limit, records, out_path):
    """Walk one method up the ladder, stopping at the first size past the budget.

    Appends one record per completed measurement and flushes the CSV after each, so a
    killed run keeps everything it had already paid for.
    """
    label = method.name
    # One throwaway fit first: the first call into a library pays numba/import warmup that
    # belongs to the process, not to the method. Left in, it inflated the smallest rung by
    # 30x (LinearLags read 1.53s then 0.05s at the same size).
    with contextlib.suppress(Exception):  # failing on a tiny frame is measured below anyway
        method.fit(case, train.iloc[-min(len(train), 512) :].copy(), val)
    for i, n in enumerate(sizes):
        frame = train.iloc[-n:].copy()
        try:
            t0 = time.perf_counter()
            method.fit(case, frame, val)
            dt = time.perf_counter() - t0
            status = "ok"
        except Exception as exc:  # a method that cannot fit at this size is a result too
            dt = float("nan")
            status = f"error: {type(exc).__name__}: {exc}"
        records.append(
            {
                "case_id": case.id,
                "freq": case.freq,
                "method": label,
                "n_points": n,
                "fit_seconds": dt,
                "status": status,
            }
        )
        pd.DataFrame(records).to_csv(out_path, index=False)

        if status != "ok":
            print(f"    {label:<12} n={n:>7,}  {status}")
            return
        print(f"    {label:<12} n={n:>7,}  {dt:8.2f}s")

        if dt > budget:
            print(f"      -> over budget ({budget}s); stopping this method")
            return
        if i + 1 < len(sizes):
            # Quadratic is the pessimistic shape among the methods here (STL grows with
            # n x period, TBATS worse); projecting on it means the guard fires before the
            # expensive measurement rather than after it.
            projected = dt * (sizes[i + 1] / n) ** 2
            if projected > hard_limit * budget:
                print(
                    f"      -> next rung projected at ~{projected:,.0f}s "
                    f"(> {hard_limit}x budget); stopping this method"
                )
                return


def summarize(df: pd.DataFrame, budget: float) -> None:
    """Print the cost curves, then what each candidate cap would cost per case."""
    ok = df[df["status"] == "ok"]
    if ok.empty:
        print("\nNo successful measurements.")
        return

    pd.set_option("display.width", 200, "display.max_columns", 50)
    for case_id, sub in ok.groupby("case_id"):
        print(f"\n=== {case_id}: fit seconds by training size ===")
        print(sub.pivot_table(index="method", columns="n_points", values="fit_seconds").round(2))

        print(f"\n=== {case_id}: what each candidate cap costs ===")
        rows = []
        for n in sorted(sub["n_points"].unique()):
            at_n = sub[sub["n_points"] == n]
            reached = set(at_n["method"])
            missing = sorted(set(sub["method"]) - reached)
            rows.append(
                {
                    "cap": n,
                    "methods_reaching": len(reached),
                    "total_fit_s": at_n["fit_seconds"].sum().round(1),
                    "slowest": at_n.loc[at_n["fit_seconds"].idxmax(), "method"],
                    "slowest_s": at_n["fit_seconds"].max().round(1),
                    "did_not_reach": ", ".join(missing) if missing else "-",
                }
            )
        print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n=== largest size measured within {budget}s, per method ===")
    within = ok[ok["fit_seconds"] <= budget]
    if within.empty:
        print("(none)")
        return
    # Reindex over every method measured, so one that never came in under budget shows a
    # 0 rather than vanishing from the table — "cannot reach even the smallest rung" is the
    # single most important thing this script can tell you.
    print(
        within.groupby(["method", "case_id"])["n_points"]
        .max()
        .unstack()
        .reindex(index=sorted(ok["method"].unique()), columns=sorted(ok["case_id"].unique()))
        .fillna(0)
        .astype(int)
        .to_string()
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cases",
        nargs="*",
        default=list(DEFAULT_CASES),
        help=f"Case ids to measure on (default: {', '.join(DEFAULT_CASES)}).",
    )
    p.add_argument(
        "--budget",
        type=float,
        default=150.0,
        help="Fit-seconds a method may spend before it is dropped from the ladder "
        "(default 150, roughly the heaviest current fit in the pilot).",
    )
    p.add_argument(
        "--hard-limit",
        type=float,
        default=4.0,
        help="Stop before a rung whose projected cost exceeds this multiple of --budget "
        "(default 4). This is what bounds the worst single measurement.",
    )
    p.add_argument("--synth", action="store_true", help="Force synthetic data generators")
    p.add_argument(
        "--with-dl",
        action="store_true",
        help="Also measure the torch models (N-HiTS, PatchTST). Their cost is set by the "
        "step budget, not by history length — expect a flat curve.",
    )
    p.add_argument(
        "--with-nf",
        action="store_true",
        help="Also measure the neuralforecast zoo (TFT, iTransformer, TimesNet). Slow, and "
        "flat for the same reason as --with-dl.",
    )
    p.add_argument(
        "--methods",
        nargs="*",
        help="Method names to measure (default: all). Case-insensitive. Use with --out to "
        "re-check one method without overwriting the committed CSV.",
    )
    p.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help=f"CSV path (default {DEFAULT_OUT})"
    )
    args = p.parse_args()

    all_cases = discover_cases()
    cases = [all_cases[cid] for cid in args.cases if cid in all_cases]
    if not cases:
        print(f"No matching cases. Available: {list(all_cases)}")
        return 1

    methods = build_methods()
    if args.with_dl:
        from tsforecastbench.bench.methods import NHITSForecaster, PatchTSTForecaster

        methods += [NHITSForecaster(), PatchTSTForecaster()]
    if args.with_nf:
        from tsforecastbench.bench.methods import (
            ITransformerForecaster,
            TFTForecaster,
            TimesNetForecaster,
        )

        methods += [TFTForecaster(), ITransformerForecaster(), TimesNetForecaster()]
    if args.methods:
        available = [m.name for m in methods]
        wanted = {n.lower() for n in args.methods}
        methods = [m for m in methods if m.name.lower() in wanted]
        if not methods:
            print(f"No matching methods. Available: {', '.join(available)}")
            return 1
    for m in methods:
        uncap(m)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Calibrating {len(methods)} method(s) on {len(cases)} case(s); "
        f"budget {args.budget}s, hard limit {args.hard_limit}x\n"
    )

    records: list[dict] = []
    for case in cases:
        train, val = fit_frames(case, prefer_synth=args.synth)
        sizes = sizes_for(len(train))
        print(
            f"=== {case.id} (freq={case.freq}, train={len(train):,} points, "
            f"val={len(val):,} held out) ==="
        )
        print(f"  sizes: {', '.join(f'{n:,}' for n in sizes)}")
        for method in methods:
            measure_method(
                case, method, train, val, sizes, args.budget, args.hard_limit, records, args.out
            )
        print()

    if not records:
        print("Nothing measured.")
        return 1
    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)
    summarize(df, args.budget)
    out = args.out.resolve()
    shown = out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out
    print(f"\nWrote {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
