"""`build_windows_fut` must agree with `build_windows` on the target, and align its
covariate block to the horizon.

Two separate claims. The first is a regression guard: every univariate DL column keeps its
published numbers only if the target windows are byte-identical to what `build_windows`
built. The second is the one real hazard in the covariate path -- `Xf` holding the covariate
at the wrong timestamp would still train to convergence and just forecast slightly worse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsforecastbench.bench.methods.base import (
    build_windows,
    build_windows_fut,
    segment_arrays,
    segment_exo_arrays,
)

L, H = 12, 4


def _frame(n_per_seg: list[int], n_exo: int) -> pd.DataFrame:
    """Long frame of `len(n_per_seg)` segments; y and each covariate carry distinct ramps."""
    rows = []
    base = 0
    for k, n in enumerate(n_per_seg):
        d = {
            "unique_id": [f"s{k}"] * n,
            "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
            "y": np.arange(base, base + n, dtype=float),
        }
        for j in range(n_exo):
            # covariate j is y shifted by a per-column offset, so a misaligned Xf shows up
            # as an exact integer mismatch rather than a small numeric difference
            d[f"x{j}"] = np.arange(base, base + n, dtype=float) + 1000 * (j + 1)
        rows.append(pd.DataFrame(d))
        base += n
    return pd.concat(rows, ignore_index=True)


@pytest.mark.parametrize("n_per_seg", [[60], [40, 35], [30, 12, 44]])
def test_target_windows_match_build_windows(n_per_seg):
    """The (N, L) and (N, H) matrices are exactly what build_windows produces."""
    df = _frame(n_per_seg, n_exo=0)
    segs, _ = segment_arrays(df)
    x_old, y_old = build_windows(segs, L, H)
    x_new, xf, y_new = build_windows_fut(segs, None, L, H)
    np.testing.assert_array_equal(x_old, x_new)
    np.testing.assert_array_equal(y_old, y_new)
    assert xf.shape == (len(x_old), H, 0)


@pytest.mark.parametrize("n_exo", [1, 3])
def test_covariate_block_is_aligned_to_the_horizon(n_exo):
    """Xf[i, t, j] is covariate j at the same timestamp as Y[i, t]."""
    df = _frame([60], n_exo)
    cols = [f"x{j}" for j in range(n_exo)]
    segs, _ = segment_arrays(df)
    exo = segment_exo_arrays(df, cols)
    _x, xf, y = build_windows_fut(segs, exo, L, H)
    for j in range(n_exo):
        # covariate j is y + 1000*(j+1) at every timestamp, so the horizon block must be too
        np.testing.assert_array_equal(xf[:, :, j], y + 1000 * (j + 1))


def test_covariate_block_never_spans_a_segment_boundary():
    """Multi-segment frames window per segment, covariates included."""
    df = _frame([40, 35], n_exo=2)
    cols = ["x0", "x1"]
    segs, _ = segment_arrays(df)
    exo = segment_exo_arrays(df, cols)
    _x, xf, y = build_windows_fut(segs, exo, L, H)
    expected = sum(max(0, n - L - H + 1) for n in (40, 35))
    assert len(y) == expected
    for j in range(2):
        np.testing.assert_array_equal(xf[:, :, j], y + 1000 * (j + 1))


def test_too_few_windows_returns_none_like_build_windows():
    df = _frame([L + H + 5], n_exo=1)
    segs, _ = segment_arrays(df)
    exo = segment_exo_arrays(df, ["x0"])
    assert build_windows(segs, L, H) == (None, None)
    assert build_windows_fut(segs, exo, L, H) == (None, None, None)


def test_missing_column_becomes_nan_not_a_shifted_column():
    """A column absent from the frame must not shift the remaining covariates left."""
    df = _frame([60], n_exo=1)
    exo = segment_exo_arrays(df, ["x0", "absent"])
    assert exo[0].shape[1] == 2
    assert not np.isnan(exo[0][:, 0]).any()
    assert np.isnan(exo[0][:, 1]).all()
