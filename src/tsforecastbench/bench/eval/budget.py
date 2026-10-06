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
The deep-learning constants say when a training loop has converged, and how many passes over
the data it may take before we call it a fit that never will — a budget expressed in epochs,
so it scales with the record rather than rationing a fixed number of steps between cases of
very different size (gh #101).

Values here are the single source of truth: methods read them, they are not duplicated as
per-method defaults, and there is no path where a missing value falls back to something else.
"""

from __future__ import annotations

# Guard, not a budget. The longest record in the benchmark is 182,304 points
# (`surge_cuxhaven`); this sits far above it so no current case is truncated. It exists so a
# future case of a genuinely different size cannot silently produce a fit that never returns.
TRAIN_POINT_CEILING = 5_000_000

# Windowed methods (DLinear, N-HiTS, PatchTST) build one length-`lookback` row per training
# window, and build every window there is. `MAX_TRAIN_WINDOWS = 15_000` used to stride that
# matrix; it went with gh #101 for the same reason the history caps went with #95, being a
# constant nothing derived. It bound on 5 of 17 (frame, case) pairs, so the neuralforecast
# zoo — which never had it — was already training on pools the other windowed methods were
# not. See `methods.base.build_windows` for what the matrix costs now.

# Deep-learning training runs to convergence, not to a fixed step count. Training stops when
# held-out loss stops improving; the ceiling below is a safety net for a fit that never
# converges, not a target, and the steps actually used are recorded per cell.
#
# It is counted in epochs — passes over the training windows — rather than in steps, which is
# what makes it derived rather than picked. A flat step count was the binding constraint on
# what these models learned (500 steps in nf_models, 600 in torchdl, later 2,000 in both),
# and it made a large case see each window far less often than a small one: research log
# entry 89's finding, and the reason gh #101 could not remove the window cap on its own.
# Counting epochs holds views-per-window equal across cases, so a longer record buys more
# steps rather than thinner ones.
#
# 150 is the long-horizon literature's budget with headroom on top. PatchTST (Nie et al.,
# ICLR 2023) trains for 100 epochs with early stopping, cutting that to 20-50 on its largest
# datasets to save compute. N-HiTS (Challu et al., AAAI 2023, Table A3) instead fixes 1,000
# steps at batch 256 and learning rate 1e-3 — the batch size and learning rate `torchdl`
# already uses, but a flat step count is the shape this benchmark rejected.
DL_MAX_EPOCHS = 150
# Validation loss is checked this often, and training stops after this many consecutive
# checks without improvement (250 steps of no progress at the defaults below).
DL_VAL_CHECK_STEPS = 50
DL_PATIENCE_CHECKS = 5

# MLP's equivalent, kept separate because the two numbers are not comparable. scikit-learn's
# `max_iter` counts epochs too ("the number of epochs ... not the number of gradient steps"),
# but MLPRegressor stops after 10 epochs without improvement while `torchdl` stops after 250
# gradient steps — a fraction of one epoch on a large case. So the same model needs far more
# epochs under the first rule than the second, and a shared constant would bind on one of them.
#
# This is a guard, not a budget: it sits above where every case actually converges. Measured by
# lifting the cap and reading back `n_iter_`, the largest a per-horizon model needed is 586, on
# `wwtp_influent_n2o_350k` — the noisiest target here, and notably not the largest case, since
# a bigger record packs more updates into each epoch and so converges in fewer (87 median on
# `sewerlevel_bellinge` against 322 on `wwtp`). The 300 that stood here was picked rather than
# measured, and it bound on 4 of the 11 cases checked, cutting training short on the noisy ones
# while leaving the rest untouched — a difference between cases that came from the constant
# rather than from the data.
MLP_MAX_EPOCHS = 2_000

# What that held-out loss is measured on is the case's declared val split, not a fraction of
# the training windows (gh #99): the eval loop hands `fit` the train split and the val split
# separately, and no method estimates parameters on the latter. The one place that cannot be
# expressed directly is a multi-segment neuralforecast panel -- see `nf_val_size`.

# Shortest validation cut worth stopping on, in horizons. Four gives a handful of disjoint
# horizon windows per series, enough for the loss curve to have a recognisable minimum. It
# does not bind on any current case: the smallest cut `nf_val_size` chooses is 213 points
# against a floor of 96 (`wwtp_influent_n2o_350k`, h=24).
NF_VAL_MIN_HORIZONS = 4


def dl_step_ceiling(n_windows: int, windows_per_step: int) -> int:
    """Gradient steps a fit may take: `DL_MAX_EPOCHS` passes over `n_windows` windows.

    `windows_per_step` is how many training windows one optimisation step consumes — the
    batch size for the torch models, `windows_batch_size` for a neuralforecast one, which
    that library sets per architecture (32 for iTransformer, 64 for TimesNet, 1,024 for TFT)
    rather than globally. Reading it off the model is what keeps the epoch an epoch.

    Only a fit whose held-out loss never goes stale gets here; early stopping is the normal
    end, and `Forecaster.last_fit_steps` records which of the two a given cell hit.
    """
    return DL_MAX_EPOCHS * max(1, -(-n_windows // windows_per_step))


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
