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

from tsforecastbench.bench.eval.budget import (
    DL_MAX_STEPS,
    DL_PATIENCE_CHECKS,
    DL_VAL_CHECK_STEPS,
    MAX_TRAIN_WINDOWS,
)
from tsforecastbench.bench.methods.base import Forecaster, lookback_for, segment_arrays
from tsforecastbench.bench.methods.scorecard import (
    Capabilities,
    ForecastStrategy,
    Level,
    Scorecard,
)
from tsforecastbench.data.schema import Case


def _build_windows(segs: list[np.ndarray], L: int, H: int):
    """Stack (input, target) windows over every segment, never across a gap.

    No segment is truncated: the whole record is in scope (issue #95). A single shared stride
    across all segments keeps the total window count <= `MAX_TRAIN_WINDOWS`, which bounds the
    window matrix without discarding any part of the span — consecutive windows differ by one
    step and are near-duplicates, so striding costs resolution, not coverage. A local
    (single-segment) frame reduces to the original one-series windowing.
    """
    counts = [max(0, len(y) - L - H + 1) for y in segs]
    total = sum(counts)
    if total <= 10:
        return None, None
    stride = max(1, total // MAX_TRAIN_WINDOWS)
    Xs, Ys = [], []
    for y, n in zip(segs, counts, strict=True):
        if n <= 0:
            continue
        starts = np.arange(0, n, stride)
        Xs.append(np.stack([y[s : s + L] for s in starts]))
        Ys.append(np.stack([y[s + L : s + L + H] for s in starts]))
    X = np.concatenate(Xs).astype(np.float32)
    Y = np.concatenate(Ys).astype(np.float32)
    return X, Y


def _build_val_windows(train_tail: np.ndarray, val_y: np.ndarray, L: int, H: int):
    """Stack the early-stopping windows: every target step lies inside the declared val split.

    Inputs may reach back into train, which is exactly what a forecast made at the first val
    origin conditions on; no target step is ever one the model trained on (gh #99). Targets
    are strided by `H` so they tile the val period without overlapping: consecutive
    single-step-apart windows are near-duplicates that would multiply the cost of every
    validation check for almost no extra information, while one window per horizon spreads
    the check across the whole val period. `surge_cuxhaven`'s 32,171-point val split gives
    670 disjoint 48-step origins spanning 3.7 years.
    """
    series = np.concatenate([train_tail, val_y]).astype(np.float32)
    n_train = len(train_tail)
    first = max(0, n_train - L)  # earliest start whose target begins at or after val
    last = len(series) - L - H  # latest start whose target still fits
    starts = np.arange(first, last + 1, H)
    if len(starts) == 0:
        raise ValueError(
            f"val split holds {len(val_y)} points, too few for one {H}-step validation "
            f"window; early stopping would monitor a metric with no data in it"
        )
    X = np.stack([series[s : s + L] for s in starts])
    Y = np.stack([series[s + L : s + L + H] for s in starts])
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
    family = "dl"
    is_global = True
    pools_segments = True
    max_steps = DL_MAX_STEPS  # ceiling; training normally early-stops well below it
    batch = 256
    lr = 1e-3
    library = "torch (custom implementation, no ray/lightning)"
    compute_profile = (
        "CPU (no GPU dependency); trains a small network from scratch per case over "
        "randomly sampled windows, stopping when held-out loss stops improving rather "
        "than at a fixed step count, so training cost scales with how much the case has "
        "left to learn. Among the heaviest CPU methods in the benchmark; keep "
        "rolling-origin counts modest when including them."
    )
    limitations = (
        "Univariate — ignores exogenous inputs (the covariate story lives in the "
        "windowed-ML +exo columns instead). Each window is instance-normalised "
        "(RevIN-style) before the model and de-normalised after, which is what makes "
        "the same architecture usable across very different series scales."
    )
    # Shared by N-HiTS and PatchTST: both learn a map from the whole lookback window to the
    # whole horizon, so trend and several interacting periods are representable, and neither
    # has a covariate pathway here (that story lives in the windowed-ML +exo columns).
    capabilities = Capabilities(
        trend=True, seasonality=True, multiple_seasonality=True, exogenous=False
    )
    # Among the heaviest CPU fits in the benchmark — a training loop over sampled windows,
    # run to convergence — against a single forward pass at predict. Per-window instance
    # normalisation is what keeps robustness at Medium rather than Low despite the loop.
    scorecard = Scorecard(
        history_requirements=Level.HIGH,
        interpretability=Level.LOW,
        fit_cost=Level.HIGH,
        inference_cost=Level.LOW,
        tuning_effort=Level.HIGH,
        robustness_to_noise=Level.MEDIUM,
    )

    def _net(self, mods, L, H):
        raise NotImplementedError

    def fit(self, case: Case, train: pd.DataFrame, val: pd.DataFrame) -> None:
        from tsforecastbench.bench.eval.loop import _horizon_to_steps

        mods = _make_modules()
        torch = mods[0]
        torch.manual_seed(0)
        self._mods = mods
        self._H = _horizon_to_steps(case.horizon_primary, case.freq)
        self._L = lookback_for(case.freq, self._H)

        # Per-segment windows (unique_id), never across a gap; single-segment local
        # frames behave exactly as before. The tail segment is the one the val split
        # continues from, so it supplies the validation windows' input context.
        segs, tail = segment_arrays(train)
        segs = [s.astype(np.float32) for s in segs]

        X, Y = _build_windows(segs, self._L, self._H)
        if X is None:
            self._model = None
            self.last_fit_steps = None
            return
        # Early stopping runs on the declared val split, not on a fraction carved off the
        # training windows: the boundary the app displays is the boundary the fit uses
        # (gh #99). Every training window's target is inside train, every validation
        # window's target is inside val.
        tail_uid = train["unique_id"].iloc[-1]
        val_y = pd.to_numeric(val.loc[val["unique_id"] == tail_uid, "y"], errors="coerce").to_numpy(
            dtype=np.float32
        )
        Xv_np, Yv_np = _build_val_windows(segs[tail], val_y, self._L, self._H)
        Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
        Xv, Yv = torch.from_numpy(Xv_np), torch.from_numpy(Yv_np)

        self._model = self._net(mods, self._L, self._H)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        lossf = torch.nn.MSELoss()
        n = len(Xt)

        def _revin_pred(xb):
            mu = xb.mean(1, keepdim=True)
            sd = xb.std(1, keepdim=True) + 1e-5
            return self._model((xb - mu) / sd) * sd + mu

        best, best_state, stale = float("inf"), None, 0
        steps = 0
        for step in range(1, self.max_steps + 1):
            self._model.train()
            idx = torch.randint(0, n, (min(self.batch, n),))
            loss = lossf(_revin_pred(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps = step
            if step % DL_VAL_CHECK_STEPS:
                continue
            # Stop when held-out loss stops improving, rather than at a fixed step count: a
            # fixed count made the number of passes over the data vary inversely with case
            # size (research log entry 89). self.max_steps is the safety net, not the target.
            self._model.eval()
            with torch.no_grad():
                val = float(lossf(_revin_pred(Xv), Yv))
            if val < best:
                best, stale = val, 0
                best_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
            else:
                stale += 1
                if stale >= DL_PATIENCE_CHECKS:
                    break
        if best_state is not None:
            self._model.load_state_dict(best_state)
        self.last_fit_steps = steps

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
        # Inference-only: the lookback window is the tail segment of the history handed in
        # at this origin, never state captured at fit — so one fit serves every origin.
        segs, tail = segment_arrays(history)
        last_window = segs[tail][-self._L :].astype(np.float32)
        self._model.eval()
        with torch.no_grad():
            x = torch.from_numpy(last_window).unsqueeze(0)
            mu = x.mean(1, keepdim=True)
            sd = x.std(1, keepdim=True) + 1e-5
            y_hat = (self._model((x - mu) / sd) * sd + mu).squeeze(0).numpy()
        if len(y_hat) < horizon:
            y_hat = np.concatenate([y_hat, np.full(horizon - len(y_hat), y_hat[-1])])
        return pd.DataFrame({"ds": future, "y_hat": y_hat[:horizon].astype(np.float64)})


class NHITSForecaster(_TorchDLBase):
    """Stack of MLP blocks at increasingly coarse pooling rates; each block predicts a
    backcast (subtracted from the residual passed to the next block) and a forecast
    contribution, and the stack's contributions sum to the final forecast."""

    name = "N-HiTS"
    code_snippet = (
        "class NHiTSBlock(nn.Module):        # one block at a given pooling rate\n"
        "    def __init__(self, L, H, pool_k, hidden, down):\n"
        "        self.pool = nn.AvgPool1d(pool_k, ceil_mode=True)     # coarsen the input\n"
        "        self.mlp = nn.Sequential(nn.Linear(L // pool_k, hidden), nn.ReLU(),\n"
        "                                  nn.Linear(hidden, hidden), nn.ReLU())\n"
        "        self.b_lin = nn.Linear(hidden, L // down)            # backcast basis\n"
        "        self.f_lin = nn.Linear(hidden, H // down)            # forecast basis\n\n"
        "    def forward(self, x):\n"
        "        h = self.mlp(self.pool(x))\n"
        "        back = interpolate(self.b_lin(h), size=L)  # upsample basis to input length\n"
        "        fore = interpolate(self.f_lin(h), size=H)  # upsample basis to horizon length\n"
        "        return back, fore\n\n"
        "# stack of blocks, coarse to fine (pool_k=8, 4, 1): each subtracts its backcast\n"
        "# from the residual the next block sees; forecasts from every block sum\n"
        "residual, forecast = window, 0.0\n"
        "for block in blocks:\n"
        "    back, fore = block(residual)\n"
        "    residual = residual - back\n"
        "    forecast = forecast + fore\n\n"
        "# training: per-window instance norm (RevIN-style), plain MSE + Adam\n"
        "mu, sd = window.mean(1), window.std(1)\n"
        "pred = net((window - mu) / sd) * sd + mu\n"
        "loss = mse_loss(pred, target)"
    )
    # Each block's _interp produces its H-length contribution directly; forward() sums
    # them in one pass — no step's output feeds into computing another step.
    forecast_strategy = ForecastStrategy.DIRECT

    def _net(self, mods, L, H):
        _, _, _, NHiTSNet = mods
        return NHiTSNet(L, H)


class PatchTSTForecaster(_TorchDLBase):
    """Splits the lookback window into overlapping patches, embeds and positionally
    encodes them, and runs them through a transformer encoder before a linear head
    projects to the full horizon."""

    name = "PatchTST"
    code_snippet = (
        "class PatchTSTNet(nn.Module):\n"
        "    def __init__(self, L, H, patch_len=16, stride=8, d_model=64, n_heads=4,\n"
        "                 n_layers=2):\n"
        "        self.embed = nn.Linear(patch_len, d_model)\n"
        "        self.pos = nn.Parameter(torch.randn(n_patches, d_model) * 0.02)\n"
        "        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=128)\n"
        "        self.encoder = nn.TransformerEncoder(layer, n_layers)\n"
        "        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n_patches * d_model, H))\n\n"
        "    def forward(self, x):                        # x: (B, L), already normalised\n"
        "        patches = x.unfold(1, self.patch_len, self.stride)  # (B, n_patches, patch_len)\n"
        "        z = self.encoder(self.embed(patches) + self.pos)\n"
        "        return self.head(z)                      # (B, H)\n\n"
        "# training: per-window instance norm (RevIN-style), plain MSE + Adam\n"
        "mu, sd = window.mean(1), window.std(1)\n"
        "pred = net((window - mu) / sd) * sd + mu\n"
        "loss = mse_loss(pred, target)"
    )
    # The head is a single Linear projecting straight to the full H-length output —
    # no step's output feeds into computing another step.
    forecast_strategy = ForecastStrategy.DIRECT

    def _net(self, mods, L, H):
        _, _, PatchTSTNet, _ = mods
        return PatchTSTNet(L, H)
