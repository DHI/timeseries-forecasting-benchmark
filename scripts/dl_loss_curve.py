"""Record the validation loss curve of the torch DL methods with early stopping turned off.

Diagnostic for gh #153: is validation loss still falling where the step-based patience
stops the fit? Runs the same fit the pilot runs (same train/val frames, curated and
extended), but with patience disabled and a ceiling of `--passes` passes over the
training windows. Every validation check is written to a CSV, together with the mean
training loss since the previous check and the step where today's rule would have stopped.

    uv run python scripts/dl_loss_curve.py --case sewerlevel_bellinge \
        --methods N-HiTS PatchTST --seeds 0 1 --passes 5 --out dl_loss_curve.csv
"""

from __future__ import annotations

import argparse
import math
import time
from typing import ClassVar

import pandas as pd
import torch

from tsforecastbench.bench.eval import budget
from tsforecastbench.bench.eval.loop import (
    _horizon_to_steps,
    _segment_extended,
    _split_segments_at_train_end,
)
from tsforecastbench.bench.methods import torchdl
from tsforecastbench.data.registry import get_case

METHODS = {
    "N-HiTS": torchdl.NHITSForecaster,
    "N-HiTS+exo": torchdl.NHITSExoForecaster,
    "PatchTST": torchdl.PatchTSTForecaster,
    "PatchTST+exo": torchdl.PatchTSTExoForecaster,
}


class _RecordingMSE(torch.nn.MSELoss):
    """MSELoss that keeps every value it returns: training calls run with gradients on,
    the validation check runs under `torch.no_grad()`."""

    train: ClassVar[list[float]] = []
    val: ClassVar[list[float]] = []

    def forward(self, a, b):
        out = super().forward(a, b)
        (self.train if torch.is_grad_enabled() else self.val).append(float(out))
        return out


def _frames(case, extended: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The (train, val) pair `run_case` hands to `fit` for this variant."""
    df = case.load(curated=True)
    splits = case.split_frame(df)
    if not extended:
        return splits["train"].reset_index(drop=True), splits["val"].reset_index(drop=True)
    train_end, _, _ = case.split_bounds(df)
    h = _horizon_to_steps(case.horizon_primary, case.freq)
    segments = _segment_extended(case.load_extended(), case, h)
    if len(segments) < 2:
        raise SystemExit(f"{case.id} has no history beyond the curated span")
    return _split_segments_at_train_end(segments, train_end)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="sewerlevel_bellinge")
    p.add_argument("--methods", nargs="+", default=["N-HiTS", "PatchTST"], choices=METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--variants", nargs="+", default=["curated", "extended"])
    p.add_argument("--passes", type=float, default=5.0, help="step ceiling, in passes")
    p.add_argument("--out", default="dl_loss_curve.csv")
    args = p.parse_args()

    # torchdl imports torch inside its functions, so the class is swapped on torch itself.
    torch.nn.MSELoss = _RecordingMSE
    torchdl.DL_PATIENCE_CHECKS = 10**9
    every = budget.DL_VAL_CHECK_STEPS
    old_patience = budget.DL_PATIENCE_CHECKS

    case = get_case(args.case)
    rows = []
    for variant in args.variants:
        train, val = _frames(case, variant == "extended")
        for name in args.methods:
            for seed in args.seeds:
                method = METHODS[name]()
                method.seed = seed
                seen = {}
                torchdl.dl_step_ceiling = lambda n, b, seen=seen: (
                    seen.update(n=n, b=b) or max(1, math.ceil(args.passes * n / b))
                )
                _RecordingMSE.train, _RecordingMSE.val = [], []
                t0 = time.monotonic()
                method.fit(case, train, val)
                secs = time.monotonic() - t0
                steps_per_pass = math.ceil(seen["n"] / seen["b"])
                vals = _RecordingMSE.val
                # Where today's rule (stop after `old_patience` checks with no new best)
                # would have ended the fit.
                best, stale, old_stop = math.inf, 0, None
                for k, v in enumerate(vals, start=1):
                    if v < best:
                        best, stale = v, 0
                    else:
                        stale += 1
                        if stale >= old_patience and old_stop is None:
                            old_stop = k * every
                for k, v in enumerate(vals, start=1):
                    tr = _RecordingMSE.train[(k - 1) * every : k * every]
                    rows.append(
                        {
                            "case": case.id,
                            "variant": variant,
                            "method": name,
                            "seed": seed,
                            "step": k * every,
                            "passes": k * every / steps_per_pass,
                            "val_mse": v,
                            "train_mse": sum(tr) / len(tr),
                            "steps_per_pass": steps_per_pass,
                            "train_windows": seen["n"],
                            "old_rule_stop_step": old_stop,
                        }
                    )
                best_k = min(range(len(vals)), key=vals.__getitem__) + 1
                print(
                    f"{variant:9s} {name:13s} seed {seed}: {len(vals) * every} steps "
                    f"({len(vals) * every / steps_per_pass:.2f} passes, {secs:.0f}s), "
                    f"best at step {best_k * every} ({best_k * every / steps_per_pass:.2f} "
                    f"passes), old rule stops at step {old_stop}",
                    flush=True,
                )
                pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
