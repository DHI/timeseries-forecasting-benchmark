"""Modern deep-learning forecasters implemented directly in PyTorch.

N-HiTS (Challu et al. 2023) and PatchTST (Nie et al. 2023) are the two headline
long-horizon architectures. The usual route (Nixtla `neuralforecast`) pulls `ray`, which
has no Python-3.13 wheels, so we implement compact, faithful versions on top of plain
`torch` instead — no ray, no lightning. They are univariate (the covariate story is the
windowed-ML `+exo` columns) and use per-window instance normalisation (RevIN-style):
each input window is standardised before the model and the forecast is de-standardised,
which is what makes these models robust across the benchmark's very different scales.

Each comes in two flavours, like the windowed-ML methods in `ml.py`: univariate, and
exogenous-aware (``use_exogenous=True``) which additionally reads the covariate values over
the forecast horizon. The covariate contract is the ML one -- the value at the step being
predicted, supplied by the eval loop as a perfect-forecast proxy -- so neither architecture
grows a channel dimension over the lookback.

`torch` is the optional `dl` extra; it is imported lazily so the package runs without it.
These are the heaviest CPU methods (each fit trains a small network) — keep rolling-origin
counts modest when including them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.bench.eval.budget import (
    DL_PATIENCE_CHECKS,
    DL_VAL_CHECK_STEPS,
    dl_step_ceiling,
)
from tsforecastbench.bench.methods.base import (
    Forecaster,
    build_windows,
    lookback_for,
    segment_arrays,
)
from tsforecastbench.bench.methods.scorecard import (
    Capabilities,
    ForecastStrategy,
    GrowthShape,
    Level,
    RunningCost,
    Scorecard,
)
from tsforecastbench.data.schema import Case


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
            self,
            L,
            H,
            n_exo=0,
            patch_len=16,
            stride=8,
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
        ):
            super().__init__()
            patch_len = min(patch_len, L)
            stride = min(stride, patch_len)
            self.patch_len, self.stride = patch_len, stride
            self.n_patches = (L - patch_len) // stride + 1
            self.n_exo = n_exo
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
            # The horizon covariates join at the head, not in the patch embedding: patching is
            # defined on the lookback, and a covariate over the horizon has no patch to sit in.
            self.head = nn.Sequential(
                nn.Flatten(), nn.Linear(self.n_patches * d_model + H * n_exo, H)
            )

        def forward(self, x, xf=None):  # x: (B, L) already normalised; xf: (B, H, n_exo)
            patches = x.unfold(1, self.patch_len, self.stride)  # (B, n_patches, patch_len)
            z = self.embed(patches) + self.pos
            z = self.encoder(z)
            if self.n_exo:
                z = torch.cat([z.flatten(1), xf.flatten(1)], dim=1)
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
        def __init__(self, L, H, n_exo=0, pools=(8, 4, 1), downs=(24, 8, 1), hidden=256):
            super().__init__()
            self.blocks = nn.ModuleList(
                _NHiTSBlock(L, H, p, hidden, d) for p, d in zip(pools, downs, strict=False)
            )
            self.n_exo = n_exo
            # A fourth contribution to the same additive stack, so the residual mechanism the
            # blocks share is untouched. The blocks decompose the lookback at three pooling
            # rates; a covariate over the horizon has no lookback to pool, so it cannot enter
            # one of them.
            self.fut_lin = nn.Linear(H * n_exo, H) if n_exo else None

        def forward(self, x, xf=None):  # x: (B, L); xf: (B, H, n_exo)
            residual = x
            forecast = 0.0
            for blk in self.blocks:
                back, fore = blk(residual)
                residual = residual - back
                forecast = forecast + fore
            if self.fut_lin is not None:
                forecast = forecast + self.fut_lin(xf.flatten(1))
            return forecast

    return torch, nn, PatchTSTNet, NHiTSNet


class _TorchDLBase(Forecaster):
    name = "TorchDL"
    family = "dl"
    is_global = True
    pools_segments = True
    # Weight init, the minibatch draw in the training loop, and PatchTST's dropout all come
    # off torch's global RNG, so a different seed is a different network.
    is_stochastic = True
    # Batch size and learning rate are N-HiTS's own (Challu et al., AAAI 2023, Table A3).
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
        "Both architectures take covariates as published, but these compact torch ports "
        "wire none up, so the column runs univariate (the covariate story lives in the "
        "windowed-ML +exo columns instead). Each window is instance-normalised "
        "(RevIN-style) before the model and de-normalised after, which is what makes "
        "the same architecture usable across very different series scales."
    )
    # Shared by N-HiTS and PatchTST: both learn a map from the whole lookback window to the
    # whole horizon, so trend and several interacting periods are representable. `exogenous`
    # describes the architecture rather than this port of it (see the rubric) -- both papers
    # take covariates, neither port passes any, and `limitations` above says so.
    capabilities = Capabilities(
        trend=True, seasonality=True, multiple_seasonality=True, exogenous=True
    )
    # No shared `scorecard`: both fits train to a step ceiling and both predict in one
    # forward pass, but which of the two terms dominates the running cost differs between
    # them, and so does its growth shape. Each subclass declares its own.

    def _net(self, mods, L, H, n_exo):
        raise NotImplementedError

    def fit(self, case: Case, train: pd.DataFrame, val: pd.DataFrame) -> None:
        from tsforecastbench.bench.eval.loop import _horizon_to_steps

        mods = _make_modules()
        torch = mods[0]
        torch.manual_seed(self.seed)
        self._mods = mods
        self._H = _horizon_to_steps(case.horizon_primary, case.freq)
        self._L = lookback_for(case, self._H)

        # Per-segment windows (unique_id), never across a gap; single-segment local
        # frames behave exactly as before. The tail segment is the one the val split
        # continues from, so it supplies the validation windows' input context.
        segs, tail = segment_arrays(train)
        segs = [s.astype(np.float32) for s in segs]

        X, Y = build_windows(segs, self._L, self._H)
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
        # `DL_MAX_EPOCHS` passes over this fit's own window pool, so a longer record buys
        # more steps rather than thinner ones (gh #101). Early stopping below is what
        # normally ends the fit; this is only reached by one that never converges.
        max_steps = dl_step_ceiling(n, self.batch)
        for step in range(1, max_steps + 1):
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
            # size (research log entry 89). `max_steps` is the safety net, not the target.
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
    # A training loop over sampled windows against a single forward pass at predict; a fit
    # plus a thousand forecasts runs to ~28s on a 100,000-point record. Inference is the
    # larger of the two terms and tracks the record, so the growth reads linear.
    scorecard = Scorecard(
        history_requirements=Level.HIGH,
        interpretability=Level.LOW,
        running_cost=RunningCost(level=Level.LOW, growth=GrowthShape.LINEAR),
        tuning_effort=Level.HIGH,
    )

    forecast_strategy = ForecastStrategy.DIRECT

    def _net(self, mods, L, H, n_exo):
        _, _, _, NHiTSNet = mods
        return NHiTSNet(L, H, n_exo)


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
    # The heavier fit of the two (~27s at 100,000 points) against the same one-pass predict,
    # so the fit dominates -- and it trains to a step ceiling, which makes the cost flat: a
    # 3,000-point record costs about what a 150,000-point one does.
    scorecard = Scorecard(
        history_requirements=Level.HIGH,
        interpretability=Level.LOW,
        running_cost=RunningCost(level=Level.LOW, growth=GrowthShape.FLAT),
        tuning_effort=Level.HIGH,
    )

    forecast_strategy = ForecastStrategy.DIRECT

    def _net(self, mods, L, H, n_exo):
        _, _, PatchTSTNet, _ = mods
        return PatchTSTNet(L, H, n_exo)
