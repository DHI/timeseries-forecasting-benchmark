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
# Validation loss is checked every `DL_VAL_CHECK_STEPS` steps. Training stops once that many
# steps times the patience from `dl_patience_checks` pass without a new best: two passes over
# the training windows (gh #153), and never fewer than `DL_PATIENCE_CHECKS` checks.
#
# The patience used to be a flat 5 checks (250 steps), which is several passes on a small case
# and a fraction of one on a large one. On the 340k-window `sewerlevel_bellinge` extended
# history it stopped N-HiTS and PatchTST at 0.2-0.7 passes while validation loss kept falling
# to 1.3-4.5 passes, 2.7-7.6% lower. Two passes reached that best on all six runs (two methods,
# three seeds) and one pass on two of them: batches are drawn with replacement, so one pass
# draws only ~63% of distinct windows, two ~86%. On the curated span the same runs stop where
# they did. The flat 5 checks stays as the floor, since two passes of a 4,000-point daily case
# are ~32 steps, shorter than a single check.
DL_VAL_CHECK_STEPS = 50
DL_PATIENCE_CHECKS = 5
DL_PATIENCE_PASSES = 2

# MLP's equivalent, kept separate because the two numbers are not comparable. scikit-learn's
# `max_iter` counts epochs too ("the number of epochs ... not the number of gradient steps"),
# but MLPRegressor stops after 10 epochs without improvement while `torchdl` stops after two
# passes without improvement (`DL_PATIENCE_PASSES`). So the same model needs far more epochs under the first rule than
# the second, and a shared constant would bind on one of them.
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
# separately, then `refit` trains for the step count found on train and val together
# (gh #120). A multi-segment neuralforecast panel holds out the same tail from every segment,
# so segments too short to give up the whole val split are left out (`nf_models._NFBase.fit`).


def dl_step_ceiling(n_windows: int, windows_per_step: int) -> int:
    """Gradient steps a fit may take: `DL_MAX_EPOCHS` passes over `n_windows` windows.

    `windows_per_step` is how many training windows one optimisation step consumes — the
    batch size for the torch models, `windows_batch_size` for a neuralforecast one, which
    that library sets per architecture (32 for iTransformer, 64 for TimesNet) and TFT
    overrides to 256 (gh #109). Reading it off the model is what keeps the epoch an epoch.

    Only a fit whose held-out loss never goes stale gets here; early stopping is the normal
    end, and `Forecaster.last_fit_steps` records which of the two a given cell hit.
    """
    return DL_MAX_EPOCHS * max(1, -(-n_windows // windows_per_step))


def dl_patience_checks(
    n_windows: int, windows_per_step: int, *, check_every: int, min_checks: int, passes: float
) -> int:
    """Validation checks without a new best after which a fit stops: `passes` passes over
    `n_windows` windows, rounded up to whole checks, and at least `min_checks`.

    The caller passes `DL_VAL_CHECK_STEPS`, `DL_PATIENCE_CHECKS` and `DL_PATIENCE_PASSES` as
    its own module imports them, so a test that shrinks them on one module shrinks them here.
    """
    steps_per_pass = max(1, -(-n_windows // windows_per_step))
    return max(min_checks, -(-int(passes * steps_per_pass) // check_every))
