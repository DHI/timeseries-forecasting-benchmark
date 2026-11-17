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
    ExoScaler,
    Forecaster,
    _exo_cols,
    build_windows_fut,
    future_exo_block,
    last_observed_exo,
    lookback_for,
    segment_arrays,
    segment_exo_arrays,
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


def _build_val_windows(
    train_tail: np.ndarray,
    val_y: np.ndarray,
    L: int,
    H: int,
    exo_tail: np.ndarray | None = None,
    exo_val: np.ndarray | None = None,
):
    """Stack the early-stopping windows: every target step lies inside the declared val split.

    Inputs may reach back into train, which is exactly what a forecast made at the first val
    origin conditions on; no target step is ever one the model trained on (gh #99). Targets
    are strided by `H` so they tile the val period without overlapping: consecutive
    single-step-apart windows are near-duplicates that would multiply the cost of every
    validation check for almost no extra information, while one window per horizon spreads
    the check across the whole val period. `surge_cuxhaven`'s 32,171-point val split gives
    670 disjoint 48-step origins spanning 3.7 years.

    `exo_tail`/`exo_val` are the matching covariate blocks for the same two spans, windowed
    on the same `starts` so validation sees covariates aligned exactly as training does.
    Returns `Xf` shaped `(n_starts, H, n_exo)`, empty on a case with no covariates.
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
    if exo_tail is None or exo_val is None or exo_tail.shape[1] == 0:
        n_exo = 0 if exo_tail is None else exo_tail.shape[1]
        return X, np.empty((len(starts), H, n_exo), dtype=np.float32), Y
    exo = np.concatenate([exo_tail, exo_val], axis=0).astype(np.float32)
    Xf = np.stack([exo[s + L : s + L + H] for s in starts])
    return X, Xf, Y


def _make_modules():
    """Define the torch modules lazily (only when torch is importable)."""
    import torch
    from torch import nn

    def _zero_init_exo(H, n_exo):
        """The covariate layer, initialised to exactly zero.

        A zero start makes the +exo net identical to its univariate sibling at step 0, so the
        covariate path can only earn its contribution. The default init cannot be used: paired
        with the `y_scale` factor in `_exo_term` its +-0.4 weights put +-68 of noise into every
        forecast on the synthetic check, and Adam's fixed step size could not walk that back
        inside the few hundred steps these fits take -- it measured 2.5x worse than univariate.
        """
        lin = nn.Linear(H * n_exo, H)
        nn.init.zeros_(lin.weight)
        nn.init.zeros_(lin.bias)
        return lin

    def _exo_term(fut_lin, xf, y_scale):
        """The horizon covariates' additive contribution, in the target's own units.

        Two decisions here, both measured rather than assumed.

        It sits *outside* the net's `forward`, which works in RevIN-normalised space. A
        covariate's effect is a relationship in raw units -- a given wind speed raises surge
        by so many metres -- while RevIN divides the target by each window's own standard
        deviation. Inside `forward` the coefficient would have to move with local volatility,
        which fixed weights cannot express, and it measured worse than the univariate net on
        a target that was a pure function of its covariate.

        And it is multiplied by `y_scale`, the training pool's target standard deviation, a
        constant. The covariates arrive standardised, so without it the layer would have to
        learn weights the size of the target's scale -- around 180 on the synthetic check --
        which Adam at lr=1e-3 cannot reach in the few hundred steps these fits take. With it
        the weights stay near 1. A constant, not the per-window `sd`, so the relationship it
        encodes is the same in a calm stretch as in a storm.
        """
        return 0.0 if fut_lin is None else y_scale * fut_lin(xf.flatten(1))

    class PatchTSTNet(nn.Module):
        def __init__(
            self,
            L,
            H,
            n_exo=0,
            y_scale=1.0,
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
            self.y_scale = y_scale
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
            self.fut_lin = _zero_init_exo(H, n_exo) if n_exo else None

        def forward(self, x):  # x: (B, L) already normalised
            patches = x.unfold(1, self.patch_len, self.stride)  # (B, n_patches, patch_len)
            z = self.embed(patches) + self.pos
            z = self.encoder(z)
            return self.head(z)  # (B, H)

        def exo_term(self, xf):
            return _exo_term(self.fut_lin, xf, self.y_scale)

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
        def __init__(
            self, L, H, n_exo=0, y_scale=1.0, pools=(8, 4, 1), downs=(24, 8, 1), hidden=256
        ):
            super().__init__()
            self.blocks = nn.ModuleList(
                _NHiTSBlock(L, H, p, hidden, d) for p, d in zip(pools, downs, strict=False)
            )
            self.n_exo = n_exo
            self.y_scale = y_scale
            self.fut_lin = _zero_init_exo(H, n_exo) if n_exo else None

        def forward(self, x):  # x: (B, L), already normalised
            residual = x
            forecast = 0.0
            for blk in self.blocks:
                back, fore = blk(residual)
                residual = residual - back
                forecast = forecast + fore
            return forecast

        def exo_term(self, xf):
            return _exo_term(self.fut_lin, xf, self.y_scale)

    return torch, nn, PatchTSTNet, NHiTSNet


class _TorchDLBase(Forecaster):
    name = "TorchDL"
    family = "dl"
    is_global = True
    pools_segments = True
    # Weight init, the minibatch draw in the training loop, and PatchTST's dropout all come
    # off torch's global RNG, so a different seed is a different network.
    is_stochastic = True
    use_exogenous: bool = False
    # Batch size and learning rate are N-HiTS's own (Challu et al., AAAI 2023, Table A3).
    batch = 256
    lr = 1e-3
    # The covariate head trains at ten times the backbone's rate. It starts at exactly zero
    # (see `_zero_init_exo`) and has to reach a weight of order one to carry a covariate's
    # full effect, while the backbone starts from a random init that is already the right size.
    # Adam's step is scale-free, so at the backbone's 1e-3 the head needed thousands of steps
    # and early stopping ended the fit after a few hundred with |W| still near 0.05 -- the
    # covariate column came out barely different from its univariate sibling.
    exo_lr_multiplier = 10.0
    library = "torch (custom implementation, no ray/lightning)"
    compute_profile = (
        "CPU (no GPU dependency); trains a small network from scratch per case over "
        "randomly sampled windows, stopping when held-out loss stops improving rather "
        "than at a fixed step count, so training cost scales with how much the case has "
        "left to learn. Among the heaviest CPU methods in the benchmark; keep "
        "rolling-origin counts modest when including them."
    )
    limitations = (
        "The univariate variant ignores exogenous inputs; the +exo variant sees the "
        "covariate values over the forecast horizon and not their history, which is the "
        "same contract the windowed-ML +exo columns use. Each target window is "
        "instance-normalised (RevIN-style) before the model and de-normalised after, which "
        "is what makes the same architecture usable across very different series scales; "
        "covariates are standardised once on the training pool instead, since they carry no "
        "relation to the target's window scale."
    )
    # Shared by N-HiTS and PatchTST: both learn a map from the whole lookback window to the
    # whole horizon, so trend and several interacting periods are representable. `exogenous`
    # is True for the architecture, not for a given column: both read covariates whenever
    # `use_exogenous` is set, which is all the +exo variant switches.
    capabilities = Capabilities(
        trend=True, seasonality=True, multiple_seasonality=True, exogenous=True
    )
    # No shared `scorecard`: both fits train to a step ceiling and both predict in one
    # forward pass, but which of the two terms dominates the running cost differs between
    # them, and so does its growth shape. Each subclass declares its own.

    def __init__(self) -> None:
        self.uses_exogenous = self.use_exogenous
        self._exo: list[str] = []
        self._last_exo: dict[str, float] = {}
        self._scaler = ExoScaler.fit(None, 0)

    def _net(self, mods, L, H, n_exo, y_scale):
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
        self._exo = _exo_cols(case) if self.use_exogenous else []
        self._last_exo = last_observed_exo(train, self._exo)
        exo_segs = segment_exo_arrays(train, self._exo) if self._exo else None
        # Standardised once on the training pool, not per window. RevIN rescales the target by
        # its own window statistics, but a covariate has no relation to the target's window
        # scale -- `waves_oregon` mixes pressure near 1,000 hPa with wind near 5 m/s -- and a
        # small network fed raw values on those scales saturates before it learns anything.
        self._scaler = ExoScaler.fit(exo_segs, len(self._exo))
        if exo_segs is not None:
            exo_segs = [self._scaler.transform(e).astype(np.float32) for e in exo_segs]

        X, Xf, Y = build_windows_fut(segs, exo_segs, self._L, self._H)
        if X is None:
            self._model = None
            self.last_fit_steps = None
            self.last_best_step = None
            return
        # Early stopping runs on the declared val split, not on a fraction carved off the
        # training windows: the boundary the app displays is the boundary the fit uses
        # (gh #99). Every training window's target is inside train, every validation
        # window's target is inside val.
        tail_uid = train["unique_id"].iloc[-1]
        val_y = pd.to_numeric(val.loc[val["unique_id"] == tail_uid, "y"], errors="coerce").to_numpy(
            dtype=np.float32
        )
        exo_val = None
        if self._exo:
            val_tail = val.loc[val["unique_id"] == tail_uid]
            exo_val = self._scaler.transform(segment_exo_arrays(val_tail, self._exo)[0]).astype(
                np.float32
            )
        Xv_np, Xfv_np, Yv_np = _build_val_windows(
            segs[tail],
            val_y,
            self._L,
            self._H,
            exo_segs[tail] if exo_segs is not None else None,
            exo_val,
        )
        Xt, Yt, Xft = torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Xf)
        Xv, Yv, Xfv = (
            torch.from_numpy(Xv_np),
            torch.from_numpy(Yv_np),
            torch.from_numpy(Xfv_np),
        )

        # The target's own scale over the training pool, so the covariate layer's weights
        # stay near 1 -- see `_exo_term`. Guarded against a degenerate constant target.
        y_scale = float(np.nanstd(Y))
        if not np.isfinite(y_scale) or y_scale <= 0:
            y_scale = 1.0
        self._model = self._net(mods, self._L, self._H, len(self._exo), y_scale)
        exo_params = list(self._model.fut_lin.parameters()) if self._exo else []
        exo_ids = {id(p) for p in exo_params}
        backbone = [p for p in self._model.parameters() if id(p) not in exo_ids]
        groups = [{"params": backbone, "lr": self.lr}]
        if exo_params:
            groups.append({"params": exo_params, "lr": self.lr * self.exo_lr_multiplier})
        opt = torch.optim.Adam(groups)
        lossf = torch.nn.MSELoss()
        n = len(Xt)

        def _revin_pred(xb, xfb):
            """Instance-normalise the target window, predict, de-normalise, add covariates.

            The covariate term lands *after* de-normalisation, in target units -- see
            `_exo_term` for why it cannot sit inside the normalised forward pass.
            """
            mu = xb.mean(1, keepdim=True)
            sd = xb.std(1, keepdim=True) + 1e-5
            out = self._model((xb - mu) / sd) * sd + mu
            return out + self._model.exo_term(xfb)

        best, best_state, best_step, stale = float("inf"), None, None, 0
        steps = 0
        # `DL_MAX_EPOCHS` passes over this fit's own window pool, so a longer record buys
        # more steps rather than thinner ones (gh #101). Early stopping below is what
        # normally ends the fit; this is only reached by one that never converges.
        max_steps = dl_step_ceiling(n, self.batch)
        for step in range(1, max_steps + 1):
            self._model.train()
            idx = torch.randint(0, n, (min(self.batch, n),))
            loss = lossf(_revin_pred(Xt[idx], Xft[idx]), Yt[idx])
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
                val = float(lossf(_revin_pred(Xv, Xfv), Yv))
            if val < best:
                best, best_step, stale = val, step, 0
                best_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
            else:
                stale += 1
                if stale >= DL_PATIENCE_CHECKS:
                    break
        if best_state is not None:
            self._model.load_state_dict(best_state)
        self.last_fit_steps = steps
        self.last_best_step = best_step

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
        xf = None
        if self._exo:
            block = future_exo_block(self._exo, history, self._H, future_exog, self._last_exo)
            xf = torch.from_numpy(self._scaler.transform(block).astype(np.float32)).unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            x = torch.from_numpy(last_window).unsqueeze(0)
            mu = x.mean(1, keepdim=True)
            sd = x.std(1, keepdim=True) + 1e-5
            out = self._model((x - mu) / sd) * sd + mu
            y_hat = (out + self._model.exo_term(xf)).squeeze(0).numpy()
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

    def _net(self, mods, L, H, n_exo, y_scale):
        _, _, _, NHiTSNet = mods
        return NHiTSNet(L, H, n_exo, y_scale)


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

    def _net(self, mods, L, H, n_exo, y_scale):
        _, _, PatchTSTNet, _ = mods
        return PatchTSTNet(L, H, n_exo, y_scale)


class NHITSExoForecaster(NHITSForecaster):
    """N-HiTS with the case's declared covariates over the forecast horizon added as a further
    contribution to the block stack's summed forecast."""

    name = "N-HiTS+exo"
    use_exogenous = True


class PatchTSTExoForecaster(PatchTSTForecaster):
    """PatchTST with the case's declared covariates over the forecast horizon concatenated
    into the prediction head alongside the encoded patches."""

    name = "PatchTST+exo"
    use_exogenous = True
