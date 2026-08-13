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
# Fraction of windows held out, chronologically from the end, to decide when to stop.
DL_VAL_FRACTION = 0.2
# Validation loss is checked this often, and training stops after this many consecutive
# checks without improvement (250 steps of no progress at the defaults below).
DL_VAL_CHECK_STEPS = 50
DL_PATIENCE_CHECKS = 5
