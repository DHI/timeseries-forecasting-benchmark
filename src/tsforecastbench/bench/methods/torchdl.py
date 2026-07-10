"""Modern deep-learning forecasters implemented directly in PyTorch.

N-HiTS (Challu et al. 2023) and PatchTST (Nie et al. 2023) are the two headline
long-horizon architectures. The usual route (Nixtla `neuralforecast`) pulls `ray`, which
has no Python-3.13 wheels, so we implement compact, faithful versions on top of plain
`torch` instead — no ray, no lightning. They are univariate (the covariate story is the
windowed-ML `+exo` columns) and use per-window instance normalisation (RevIN-style):
each input window is standardised before the model and the forecast is de-standardised,
which is what makes these models robust across the benchmark's very different scales.

`torch` is the optional `dl` extra; it is imported lazily so the package runs without it.
These are the heaviest CPU methods (each fit trains a small network) — keep rolling-origin
counts modest when including them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.bench.methods.base import Forecaster, lookback_for
from tsforecastbench.data.schema import Case

_MAX_TRAIN = 20000
_MAX_SAMPLES = 6000


def _build_windows(y: np.ndarray, L: int, H: int):
    n = len(y) - L - H + 1
    if n <= 10:
        return None, None
    stride = max(1, n // _MAX_SAMPLES)
    starts = np.arange(0, n, stride)
    X = np.stack([y[s : s + L] for s in starts]).astype(np.float32)
    Y = np.stack([y[s + L : s + L + H] for s in starts]).astype(np.float32)
    return X, Y


def _make_modules():
    """Define the torch modules lazily (only when torch is importable)."""
    import torch
    from torch import nn

    class PatchTSTNet(nn.Module):
        def __init__(
            self, L, H, patch_len=16, stride=8, d_model=64, n_heads=4, n_layers=2, dropout=0.1
        ):
            super().__init__()
            patch_len = min(patch_len, L)
            stride = min(stride, patch_len)
            self.patch_len, self.stride = patch_len, stride
            self.n_patches = (L - patch_len) // stride + 1
            self.embed = nn.Linear(patch_len, d_model)
            self.pos = nn.Parameter(torch.randn(self.n_patches, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model,
                n_heads,
                dim_feedforward=128,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, n_layers)
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(self.n_patches * d_model, H))

        def forward(self, x):  # x: (B, L) already normalised
            patches = x.unfold(1, self.patch_len, self.stride)  # (B, n_patches, patch_len)
            z = self.embed(patches) + self.pos
            z = self.encoder(z)
            return self.head(z)  # (B, H)

    class _NHiTSBlock(nn.Module):
        def __init__(self, L, H, pool_k, hidden, down):
            super().__init__()
            self.pool = nn.AvgPool1d(pool_k, ceil_mode=True)
            lp = -(-L // pool_k)  # ceil
            self.b_basis = max(L // down, 2)
            self.f_basis = max(H // down, 2)
            self.mlp = nn.Sequential(
                nn.Linear(lp, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.b_lin = nn.Linear(hidden, self.b_basis)
            self.f_lin = nn.Linear(hidden, self.f_basis)
            self.L, self.H = L, H

        def forward(self, x):  # x: (B, L)
            h = self.mlp(self.pool(x.unsqueeze(1)).squeeze(1))
            back = self._interp(self.b_lin(h), self.L)
            fore = self._interp(self.f_lin(h), self.H)
            return back, fore

        @staticmethod
        def _interp(theta, size):
            return torch.nn.functional.interpolate(
                theta.unsqueeze(1), size=size, mode="linear", align_corners=False
            ).squeeze(1)

    class NHiTSNet(nn.Module):
        def __init__(self, L, H, pools=(8, 4, 1), downs=(24, 8, 1), hidden=256):
            super().__init__()
            self.blocks = nn.ModuleList(
                _NHiTSBlock(L, H, p, hidden, d) for p, d in zip(pools, downs, strict=False)
            )

        def forward(self, x):  # x: (B, L)
            residual = x
            forecast = 0.0
            for blk in self.blocks:
                back, fore = blk(residual)
                residual = residual - back
                forecast = forecast + fore
            return forecast

    return torch, nn, PatchTSTNet, NHiTSNet


class _TorchDLBase(Forecaster):
    name = "TorchDL"
    family = "deep_learning"
    is_global = True
    max_steps = 600
    batch = 256
    lr = 1e-3

    def _net(self, mods, L, H):
        raise NotImplementedError

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from tsforecastbench.bench.eval.loop import _horizon_to_steps

        mods = _make_modules()
        torch = mods[0]
        torch.manual_seed(0)
        self._mods = mods
        self._H = _horizon_to_steps(case.horizon_primary, case.freq)
        self._L = lookback_for(case.freq, self._H)

        y = pd.to_numeric(train["y"], errors="coerce").to_numpy(dtype=np.float32)
        if len(y) > _MAX_TRAIN:
            y = y[-_MAX_TRAIN:]
        self._last_window = y[-self._L :].copy()

        X, Y = _build_windows(y, self._L, self._H)
        if X is None:
            self._model = None
            return
        Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
        self._model = self._net(mods, self._L, self._H)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        lossf = torch.nn.MSELoss()
        n = len(Xt)
        self._model.train()
        for _step in range(self.max_steps):
            idx = torch.randint(0, n, (min(self.batch, n),))
            xb, yb = Xt[idx], Yt[idx]
            mu = xb.mean(1, keepdim=True)
            sd = xb.std(1, keepdim=True) + 1e-5
            pred = self._model((xb - mu) / sd) * sd + mu  # RevIN-style
            loss = lossf(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        future = self._future_index(case, history["ds"].max(), horizon)
        if getattr(self, "_model", None) is None:
            last = float(pd.to_numeric(history["y"], errors="coerce").ffill().iloc[-1])
            return pd.DataFrame({"ds": future, "y_hat": np.full(horizon, last)})
        torch = self._mods[0]
        self._model.eval()
        with torch.no_grad():
            x = torch.from_numpy(self._last_window.astype(np.float32)).unsqueeze(0)
            mu = x.mean(1, keepdim=True)
            sd = x.std(1, keepdim=True) + 1e-5
            y_hat = (self._model((x - mu) / sd) * sd + mu).squeeze(0).numpy()
        if len(y_hat) < horizon:
            y_hat = np.concatenate([y_hat, np.full(horizon - len(y_hat), y_hat[-1])])
        return pd.DataFrame({"ds": future, "y_hat": y_hat[:horizon].astype(np.float64)})


class NHITSForecaster(_TorchDLBase):
    name = "N-HiTS"

    def _net(self, mods, L, H):
        _, _, _, NHiTSNet = mods
        return NHiTSNet(L, H)


class PatchTSTForecaster(_TorchDLBase):
    name = "PatchTST"

    def _net(self, mods, L, H):
        _, _, PatchTSTNet, _ = mods
        return PatchTSTNet(L, H)
