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


# --- the torch nets' univariate path -------------------------------------------------


@pytest.mark.parametrize(("L", "H"), [(48, 24), (96, 48), (365, 7)])
def test_torch_nets_are_unchanged_without_covariates(L, H):
    """At n_exo=0 both nets must produce exactly what they did before the covariate input.

    The five univariate DL columns keep their numbers only if this holds. `forward` is
    untouched by the covariate work -- the block enters through `exo_term`, applied outside
    the normalised forward pass -- so this should hold trivially, which is the point of
    checking it. Compared in eval mode: PatchTST's encoder carries dropout, so two forward
    passes in train mode differ by RNG draw rather than by architecture.
    """
    torch = pytest.importorskip("torch")
    from tsforecastbench.bench.methods.torchdl import _make_modules

    _, _, PatchTSTNet, NHiTSNet = _make_modules()
    x = torch.randn(4, L)
    for net_cls in (PatchTSTNet, NHiTSNet):
        torch.manual_seed(0)
        with_arg = net_cls(L, H, 0)
        torch.manual_seed(0)
        default = net_cls(L, H)
        with_arg.eval()
        default.eval()
        with torch.no_grad():
            a = default(x)
            b = with_arg(x)
        assert torch.equal(a, b), net_cls.__name__
        # the covariate layer must not exist at all, so it cannot contribute a bias
        assert with_arg.fut_lin is None
        assert with_arg.exo_term(None) == 0.0


@pytest.mark.parametrize(("L", "H", "n_exo"), [(48, 24, 1), (96, 48, 3), (365, 7, 2)])
def test_torch_nets_accept_a_horizon_covariate_block(L, H, n_exo):
    torch = pytest.importorskip("torch")
    from tsforecastbench.bench.methods.torchdl import _make_modules

    _, _, PatchTSTNet, NHiTSNet = _make_modules()
    x, xf = torch.randn(4, L), torch.randn(4, H, n_exo)
    for net_cls in (PatchTSTNet, NHiTSNet):
        net = net_cls(L, H, n_exo, y_scale=3.0)
        net.eval()
        with torch.no_grad():
            # Zero-initialised on purpose, so the +exo net starts identical to its univariate
            # sibling and the covariate path can only earn its contribution.
            assert torch.equal(net.exo_term(xf), torch.zeros(4, H)), net_cls.__name__
            torch.nn.init.ones_(net.fut_lin.weight)
            term = net.exo_term(xf)
            assert (net(x) + term).shape == (4, H)
            # once trained the block must reach the output, or the column is univariate
            assert not torch.equal(term, net.exo_term(xf + 1.0)), net_cls.__name__
            # and it must be scaled by y_scale, which is what makes the weights learnable
            net.y_scale = 6.0
            assert torch.allclose(net.exo_term(xf), 2.0 * term), net_cls.__name__


# --- covariate standardisation -------------------------------------------------------


def test_exo_scaler_treats_a_flatlined_column_as_constant():
    """A covariate that never moves in training must standardise to zero, not to 1e12.

    `waves_oregon`'s `atmp` is 12.7 at every one of its 17,176 training points. The std of
    that comes back as 1.8e-15 from a contiguous 1-D slice and 3.4e-12 from the axis
    reduction over the strided 2-D column, so an absolute epsilon guard cannot separate a
    constant column from a small one -- and dividing a real deviation by 3.4e-12 put 1e12
    into the network.
    """
    from tsforecastbench.bench.methods.base import ExoScaler

    n = 17_176
    block = np.empty((n, 2))
    block[:, 0] = 12.7  # flatlined sensor
    block[:, 1] = np.linspace(1000.0, 1030.0, n)  # real covariate on a large offset
    scaler = ExoScaler.fit([block], 2)
    assert scaler.std[0] == 1.0, "flatlined column was not treated as constant"
    assert scaler.std[1] > 1.0, "a real covariate was flattened"

    # a later value differing from the training constant must stay bounded
    out = scaler.transform(np.array([[15.67, 1018.0]]))
    assert abs(out[0, 0]) < 10, out[0, 0]
    assert np.isfinite(out).all()


def test_exo_scaler_keeps_a_small_relative_variance_column():
    """The guard is relative, so a genuine covariate on a huge offset survives it."""
    from tsforecastbench.bench.methods.base import ExoScaler

    block = np.linspace(1_000_000.0, 1_000_010.0, 500).reshape(-1, 1)
    scaler = ExoScaler.fit([block], 1)
    assert scaler.std[0] > 1.0
    assert abs(scaler.transform(block).std() - 1.0) < 0.01


def test_exo_scaler_sends_non_finite_values_to_the_column_mean():
    from tsforecastbench.bench.methods.base import ExoScaler

    block = np.array([[1.0], [2.0], [3.0], [np.nan]])
    scaler = ExoScaler.fit([block], 1)
    out = scaler.transform(block)
    assert np.isfinite(out).all()
    assert out[3, 0] == 0.0
