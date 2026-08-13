"""Training budgets — how much data and how much compute a single fit may spend.

Every method used to carry its own hand-picked training cap: 3,000 points for TBATS, 5,000
for AutoARIMA, 8,000 for Theta and MSTL-ETS, 20,000 for the torch models and MLP, 30,000 for
the neuralforecast zoo, and none at all for the mlforecast family. A column of the results
matrix therefore compared methods at budgets that differed by two orders of magnitude, which
is issue #95: on `surge_cuxhaven` TBATS saw 1.6% of the record while LightGBM saw all of it,
so a win could not be attributed to the method rather than to its ration of data.

Those caps are gone. Every method now trains on the whole record, and the cost of doing so is
reported rather than engineered around — the same stance `methods/classical.py` already takes
for predict (choice 3 in its module docstring). The measured curves behind that decision are
in `docs/results/train_cap_calibration.csv`: extrapolated to the full 182,304-point
`surge_cuxhaven` record, the four expensive classical methods cost roughly 7 minutes
(AutoARIMA), 10 (MSTL-ARIMA), 6-12 (TBATS) and 47-66 (Theta) per fit. Fits happen once per
case, not once per origin, so that is an affordable one-off.

What remains here is not a comparability device. `TRAIN_POINT_CEILING` is a guard against a
future case far larger than anything in the benchmark today, deliberately set well above the
longest current record so it never binds and never silently reintroduces the problem above.
The other constants bound memory and wall-clock for the windowed methods, where an unbounded
fit would materialise a matrix of overlapping windows rather than merely take a long time.

Values here are the single source of truth: methods read them, they are not duplicated as
per-method defaults, and there is no path where a missing value falls back to something else.
"""

from __future__ import annotations

# Guard, not a budget. The longest record in the benchmark is 182,304 points
# (`surge_cuxhaven`); this sits far above it so no current case is truncated. It exists so a
# future case of a genuinely different size cannot silently produce a fit that never returns.
TRAIN_POINT_CEILING = 5_000_000

# Windowed methods (DLinear, N-HiTS, PatchTST) build one length-`lookback` row per training
# window. Consecutive windows differ by a single step and are near-duplicates, so a stride is
# taken across the whole span rather than truncating to its tail: coverage stays complete
# while the window matrix stays bounded. At 15,000 windows a fit over the longest record
# still samples it densely, and the matrix is a few hundred MB at the largest lookbacks here.
MAX_TRAIN_WINDOWS = 15_000

# Deep-learning training runs to convergence, not to a fixed step count. The step budget used
# to be the binding constraint on what these models learned — 500 steps in nf_models, 600 in
# torchdl, regardless of how many windows existed — which made a large case see each window
# roughly 12x less often than a small one (research log entry 89). Training now stops when
# held-out loss stops improving; the ceiling below is a safety net for a fit that never
# converges, not a target, and the steps actually used are recorded per cell.
DL_MAX_STEPS = 2_000
# Validation loss is checked this often, and training stops after this many consecutive
# checks without improvement (250 steps of no progress at the defaults below).
DL_VAL_CHECK_STEPS = 50
DL_PATIENCE_CHECKS = 5

# What that held-out loss is measured on is the case's declared val split, not a fraction of
# the training windows (gh #99): the eval loop hands `fit` the train split and the val split
# separately, and no method estimates parameters on the latter. The one place that cannot be
# expressed directly is a multi-segment neuralforecast panel -- see `nf_val_size`.

# Shortest validation cut worth stopping on, in horizons. Four gives a handful of disjoint
# horizon windows per series, enough for the loss curve to have a recognisable minimum. It
# does not bind on any current case: the smallest cut `nf_val_size` chooses is 213 points
# against a floor of 96 (`wwtp_influent_n2o_350k`, h=24).
NF_VAL_MIN_HORIZONS = 4


def nf_val_size(
    other_lens: list[int], target_train_len: int, lookback: int, h: int, val_len: int
) -> int:
    """Panel-wide validation cut for a neuralforecast fit.

    `NeuralForecast.fit(df, val_size=V)` takes one V for the whole panel: it holds out the
    last V steps of *every* series and needs each to retain `lookback + h` points to form a
    training window, dropping those that cannot. There is no per-series val_size and no way
    to pass an explicit validation frame.

    The target series is handed `V` rows of the declared val split appended to its train
    rows, so nf's cut lands exactly on val and the series keeps every training window it has
    — `target_train_len` is there to size those windows, not to constrain V. The other series
    are whole history segments of train data (`other_lens`); each gives up its last V steps
    and is dropped below `lookback + h + V`. So V trades the declared val's coverage against
    how many history segments survive. Sized off the tail segment alone, as it was before
    gh #99, it threw four of `waves_oregon`'s nine segments away.

    Pick the V leaving the most training windows, never above the declared val split and
    never below `NF_VAL_MIN_HORIZONS` horizons. Where V lands below the declared val, the
    remainder of that split is simply left out of the panel: unused, and trained on by
    nobody. A single-series panel needs none of this — one series means the cut is the
    declared val — and gets it anyway, `val_len` being the only candidate.
    """
    floor = NF_VAL_MIN_HORIZONS * h
    if target_train_len < lookback + h + floor and val_len < floor:
        raise ValueError(
            f"target segment holds {target_train_len:,} train points and {val_len} val points, "
            f"too few for a {floor}-step validation cut ({NF_VAL_MIN_HORIZONS} horizons) on top "
            f"of lookback {lookback} + horizon {h}"
        )
    target_windows = max(0, target_train_len - lookback - h + 1)
    best: tuple[int, int] | None = None
    for v in sorted({min(n - lookback - h, val_len) for n in other_lens} | {val_len}):
        if v < floor or v > val_len:
            continue
        kept = [n for n in other_lens if n >= lookback + h + v]
        windows = target_windows + sum(n - v - lookback - h + 1 for n in kept)
        if best is None or windows > best[0]:
            best = (windows, v)
    if best is None:
        raise ValueError(
            f"no validation cut works for this panel: the declared val split holds {val_len} "
            f"points, below the {floor}-step floor ({NF_VAL_MIN_HORIZONS} horizons at h={h})"
        )
    return best[1]
