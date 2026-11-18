"""Bootstrap intervals for the matrix scores, recomputed from the forecast snapshots.

Each cell's score pools the errors of every forecast start. Neighbouring starts share
part of their forecast window, so their errors are correlated. A circular block bootstrap
resamples runs of consecutive starts together, which keeps that correlation inside each
resample. A block covers at least one horizon of time, so starts whose windows overlap
always travel together.

The interval is the score plus or minus a t-quantile times the bootstrap standard
deviation. The t-quantile has one degree of freedom fewer than the number of blocks the
test span holds. A test span of few blocks gives few independent pieces of evidence, and
plain bootstrap percentiles came out too narrow there: on synthetic AR(1) errors they
covered the true value 70-90% of the time where 95% was asked for. The t-quantile brings
coverage to 87-96% in the same setups. It still falls short, at about 75%, when the
errors stay correlated for several horizons and the test span holds few of them.

Every cell of a case is resampled with the same draws. The difference between a cell and
the case's best method is taken resample by resample, which gives an interval on that
difference. A cell whose difference interval contains 0 cannot be told apart from the
best.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from tsforecastbench.bench.eval.metrics import POOLED_METRICS, pooled_metrics

N_BOOT = 1000
LEVEL = 0.95
CELL_KEY = ["case_id", "method", "used_extended_history"]


def block_length(origin_ds: pd.Series, horizon: pd.Timedelta) -> int:
    """Consecutive starts per block: enough to span one horizon, and at least the usual
    n^(1/3) for a block bootstrap, capped at the number of starts."""
    n = len(origin_ds)
    if n < 2:
        return 1
    spacing = pd.Series(origin_ds).sort_values().diff().dropna().median()
    overlap = math.ceil(horizon / spacing) if spacing > pd.Timedelta(0) else n
    return int(min(n, max(overlap, math.ceil(n ** (1 / 3)))))


def resample_indices(n: int, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """`(n_boot, n)` start positions: each row joins blocks of `block` consecutive starts,
    beginning at random places and wrapping from the last start to the first, and cuts
    the result to `n`."""
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    rows = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)
    return rows[:, :n] % n


def _per_origin(cell: pd.DataFrame, origins: tuple) -> list[tuple[np.ndarray, ...]]:
    """Each start's observed points as (y, y_hat, y_persistence, y_mean)."""
    by_origin = dict(tuple(cell.groupby("origin_ds", sort=False)))
    cols = ["y", "y_hat", "y_persistence", "y_mean"]
    return [tuple(by_origin[o][c].to_numpy(dtype=float) for c in cols) for o in origins]


def _scores(parts: list[tuple[np.ndarray, ...]], mase_scale: float) -> np.ndarray:
    pooled = [np.concatenate(arrays) for arrays in zip(*parts, strict=True)]
    metrics = pooled_metrics(*pooled, mase_scale=mase_scale)
    return np.array([metrics[m] for m in POOLED_METRICS])


def _interval(point: float, samples: np.ndarray, t_quantile: float) -> tuple[float, float]:
    spread = t_quantile * float(np.nanstd(samples, ddof=1))
    return point - spread, point + spread


def case_intervals(
    snaps: pd.DataFrame,
    mase_scale: float,
    horizon: pd.Timedelta,
    lower_is_better: Mapping[str, bool],
    reference_methods: Iterable[str],
    n_boot: int = N_BOOT,
    level: float = LEVEL,
    seed: int = 0,
) -> pd.DataFrame:
    """Interval and difference from the best for every cell of one case.

    `snaps` holds one case's snapshots. The best method of each metric is picked from the
    local fits of `reference_methods` by point score.

    Columns: the cell key, `metric`, `value` (the point score), `ci_low`, `ci_high`,
    `best_method`, `diff_low` and `diff_high` (the cell minus the best). The interval is
    NaN when the test span holds fewer than 2 blocks. The difference is also NaN for a
    cell forecast from other starts than the best, since the two cannot be resampled
    together.
    """
    reference_methods = set(reference_methods)
    observed = snaps[snaps["y"].notna()]
    cells = {key: cell for key, cell in observed.groupby(CELL_KEY, sort=True)}
    origin_sets = {key: tuple(sorted(cell["origin_ds"].unique())) for key, cell in cells.items()}

    # Cells forecast from the same starts share one set of draws.
    draws: dict[tuple, np.ndarray] = {}
    t_quantiles: dict[tuple, float] = {}
    for origins in set(origin_sets.values()):
        block = block_length(pd.Series(origins), horizon)
        n_blocks = len(origins) // block
        draws[origins] = resample_indices(len(origins), block, n_boot, np.random.default_rng(seed))
        t_quantiles[origins] = (
            float(stats.t.ppf(0.5 + level / 2.0, n_blocks - 1)) if n_blocks >= 2 else math.nan
        )

    point: dict[tuple, np.ndarray] = {}
    boot: dict[tuple, np.ndarray] = {}
    for key, cell in cells.items():
        origins = origin_sets[key]
        parts = _per_origin(cell, origins)
        point[key] = _scores(parts, mase_scale)
        boot[key] = np.array(
            [_scores([parts[i] for i in row], mase_scale) for row in draws[origins]]
        )

    rows = []
    for j, metric in enumerate(POOLED_METRICS):
        candidates = [
            k for k in cells if k[1] in reference_methods and not k[2] and np.isfinite(point[k][j])
        ]
        best = None
        if candidates:
            sign = 1.0 if lower_is_better[metric] else -1.0
            best = min(candidates, key=lambda k: sign * point[k][j])
        for key in cells:
            q = t_quantiles[origin_sets[key]]
            low, high = _interval(point[key][j], boot[key][:, j], q)
            diff_low = diff_high = math.nan
            if best is not None and origin_sets[key] == origin_sets[best]:
                diff_low, diff_high = _interval(
                    point[key][j] - point[best][j], boot[key][:, j] - boot[best][:, j], q
                )
            rows.append(
                {
                    **dict(zip(CELL_KEY, key, strict=True)),
                    "metric": metric,
                    "value": float(point[key][j]),
                    "ci_low": low,
                    "ci_high": high,
                    "best_method": best[1] if best is not None else None,
                    "diff_low": diff_low,
                    "diff_high": diff_high,
                }
            )
    return pd.DataFrame(rows)


INTERVAL_COLUMNS = ["ci_low", "ci_high", "best_method", "diff_low", "diff_high"]


def matrix_intervals(
    results: pd.DataFrame,
    snapshots: pd.DataFrame,
    lower_is_better: Mapping[str, bool],
    n_boot: int = N_BOOT,
) -> pd.DataFrame:
    """`results` with `INTERVAL_COLUMNS` added from the bootstrap of each case's snapshots.

    The best method of a case is picked among the local fits of the methods that use no
    exogenous inputs, which is the matrix's default view. Each case's MASE scale is read
    off its rows (MAE over MASE) and its horizon from the `horizon` column, so the
    intervals use the same denominators as the scores. A cell with no snapshots, such as
    a failed run, keeps NaN intervals.
    """
    reference = set(results.loc[~results["uses_exogenous"], "method"])
    frames = []
    for case_id, case_snaps in snapshots.groupby("case_id", sort=True):
        rows = results[(results["case_id"] == case_id) & ~results["used_extended_history"]]
        mae = rows[rows["metric"] == "MAE"].set_index("method")["value"]
        mase = rows[rows["metric"] == "MASE"].set_index("method")["value"]
        ratio = (mae / mase).dropna()
        if ratio.empty:
            continue
        horizon = pd.Timedelta(rows["horizon"].iloc[0])
        print(f"[intervals] {case_id}: {case_snaps['method'].nunique()} method(s)")
        frames.append(
            case_intervals(
                case_snaps,
                mase_scale=float(ratio.iloc[0]),
                horizon=horizon,
                lower_is_better=lower_is_better,
                reference_methods=reference,
                n_boot=n_boot,
            )
        )
    if not frames:
        return results.assign(**{c: np.nan for c in INTERVAL_COLUMNS})
    intervals = pd.concat(frames, ignore_index=True)
    key = [*CELL_KEY, "metric"]
    merged = results.merge(intervals[[*key, *INTERVAL_COLUMNS]], on=key, how="left")
    return merged
