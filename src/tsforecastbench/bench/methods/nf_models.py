"""Neuralforecast-backed deep-learning models — the broader zoo (Linux/WSL only).

N-HiTS and PatchTST are implemented portably in `torchdl.py` (no ray). This module wraps
**Nixtla neuralforecast** to add the models that aren't worth re-implementing by hand —
**TFT**, **iTransformer**, **TimesNet** — and to use GPU automatically when present.

neuralforecast depends on `ray`, which has no Windows / Python-3.13 wheels, so it is the
marker-gated `nf` extra (Linux + py<3.13, i.e. WSL). Everything here imports lazily, so on
a platform without neuralforecast these classes import fine and only fail (cleanly) if you
actually try to fit one. See `scripts/bootstrap_wsl.sh`.
"""

from __future__ import annotations

import inspect
from typing import ClassVar

import numpy as np
import pandas as pd

from tsforecastbench.bench.eval.budget import (
    DL_PATIENCE_CHECKS,
    DL_VAL_CHECK_STEPS,
    dl_step_ceiling,
    nf_val_size,
)
from tsforecastbench.bench.methods.base import Forecaster, lookback_for
from tsforecastbench.bench.methods.scorecard import (
    Capabilities,
    ForecastStrategy,
    GrowthShape,
    Level,
    RunningCost,
    Scorecard,
)
from tsforecastbench.data.schema import Case

_TRAINER_KW = {"accelerator": "auto", "enable_progress_bar": False, "logger": False}


def _naive_ds(ds) -> pd.Series:
    ds = pd.to_datetime(pd.Series(ds) if not isinstance(ds, pd.Series) else ds, utc=True)
    return ds.dt.tz_convert("UTC").dt.tz_localize(None)


class _NFBase(Forecaster):
    """Shared neuralforecast wrapper; subclasses name the model class + extra kwargs."""

    name = "NF"
    family = "dl"
    is_global = True
    pools_segments = True
    # Weight init, the window sampler and dropout are all drawn from the seeded global RNG,
    # so a different seed is a different fit.
    is_stochastic = True
    model_name = "NHITS"  # neuralforecast.models class name
    extra_kwargs: ClassVar[dict] = {}
    library = "neuralforecast + pytorch_lightning"
    compute_profile = (
        "Uses a GPU automatically if one is available (accelerator='auto'), "
        "otherwise falls back to CPU; a heavier training loop (full pytorch_lightning "
        "trainer overhead, run until held-out loss stops improving rather than for a "
        "fixed number of steps) than the custom torch models in torchdl.py. Only "
        "available on Linux with Python <3.13 (the `nf` extra), since neuralforecast "
        "depends on `ray`."
    )
    limitations = (
        "Supports future covariates in neuralforecast, but this wrapper wires none of "
        "them up, so the column runs univariate. Trains a fresh model per case with "
        "no hyperparameter tuning, using the same architecture defaults regardless "
        "of the series' scale or frequency. On an extended-history fit, where the record "
        "is a panel of gap-separated segments, neuralforecast can only hold out an "
        "equal-length tail from every segment, so early stopping there watches a shorter "
        "holdout than the case's full validation split."
    )
    # Shared by all three: window models that learn trend and several interacting periods
    # from the lookback. `exogenous` describes the architecture, not the wiring (see the
    # rubric): neuralforecast takes future covariates for these, this wrapper passes none,
    # and `limitations` above is what says so.
    capabilities = Capabilities(
        trend=True, seasonality=True, multiple_seasonality=True, exogenous=True
    )
    # A full pytorch_lightning training loop per case, run to convergence, against one forward
    # pass at predict. The three land on the same rating from different directions: TFT's 159s
    # fit outweighs its forecasts, while iTransformer's 7s fit is dwarfed by a thousand of
    # them. All three read flat -- they train to a step ceiling, so a 3,000-point record costs
    # about what a 150,000-point one does. Tuning effort is High as the cost of doing it
    # properly: they run on architecture defaults regardless of the series' scale or
    # frequency, which is a limitation rather than an absence of knobs.
    scorecard = Scorecard(
        history_requirements=Level.HIGH,
        interpretability=Level.LOW,
        running_cost=RunningCost(level=Level.MEDIUM, growth=GrowthShape.FLAT),
        tuning_effort=Level.HIGH,
    )

    def fit(self, case: Case, train: pd.DataFrame, val: pd.DataFrame) -> None:
        import neuralforecast.models as nfm
        import pytorch_lightning as pl
        from neuralforecast import NeuralForecast

        from tsforecastbench.bench.eval.loop import _horizon_to_steps

        pl.seed_everything(self.seed)
        self._h = h = _horizon_to_steps(case.horizon_primary, case.freq)
        df = train[["unique_id", "ds", "y"]].copy()
        # The extended-history frame carries one unique_id per contiguous segment;
        # neuralforecast fits them as a panel, so no window spans a gap. The tail segment
        # (latest ds, always the last row) is the series we forecast at predict.
        self._target_uid = df["unique_id"].iloc[-1]
        # No truncation: the whole record is trained on (issue #95).
        df["ds"] = _naive_ds(df["ds"]).to_numpy()
        df["y"] = pd.to_numeric(df["y"], errors="coerce")

        lookback = lookback_for(case.freq, h)
        # Early stopping runs on the declared val split (gh #99). neuralforecast holds out
        # the last `val_size` steps of every series in the panel, so the target series is
        # handed its train rows plus that many val rows: the cut then lands exactly on val
        # and the series keeps every training window it has. The other series are whole
        # history segments of train data, and each gives up its last `val_size` steps —
        # `nf_val_size` picks the cut that leaves the most training windows overall.
        val_target = val[val["unique_id"] == self._target_uid]
        lens = df.groupby("unique_id", sort=False).size()
        target_train = int(lens[self._target_uid])
        other_lens = [int(n) for uid, n in lens.items() if uid != self._target_uid]
        val_size = nf_val_size(other_lens, target_train, lookback, h, len(val_target))
        if val_size < len(val_target):
            print(
                f"  [note] {self.name}: panel validation cut is {val_size:,} of the "
                f"{len(val_target):,}-point val split, so every one of its "
                f"{len(other_lens) + 1} segments trains; the rest of val goes unused."
            )
        keep = [
            uid
            for uid, n in lens.items()
            if uid == self._target_uid or n >= lookback + h + val_size
        ]
        df = df[df["unique_id"].isin(keep)]
        val_head = val_target.iloc[:val_size][["unique_id", "ds", "y"]].copy()
        val_head["ds"] = _naive_ds(val_head["ds"]).to_numpy()
        val_head["y"] = pd.to_numeric(val_head["y"], errors="coerce")
        # Sorted explicitly rather than relying on the frame arriving in order: the val rows
        # are appended after every segment, and `val_size` counts back from each series' last
        # row, so a panel left in append order would put another segment's rows there.
        df = pd.concat([df, val_head], ignore_index=True).sort_values(
            ["unique_id", "ds"], kind="stable", ignore_index=True
        )

        model_cls = getattr(nfm, self.model_name)
        # Windows the panel trains on, after the cut: the target series keeps every window it
        # has, and each surviving history segment gives up its last `val_size` steps.
        train_windows = max(0, target_train - lookback - h + 1) + sum(
            max(0, int(lens[uid]) - val_size - lookback - h + 1)
            for uid in keep
            if uid != self._target_uid
        )
        # One neuralforecast step consumes `windows_batch_size` windows, and that default is
        # per architecture (32 for iTransformer, 64 for TimesNet, 1,024 for TFT), so the
        # epoch has to be measured against the model's own value rather than a shared one.
        windows_per_step = inspect.signature(model_cls).parameters["windows_batch_size"].default
        model = model_cls(
            h=h,
            input_size=lookback,
            # Convergence, not a fixed step count: a fixed count made a large case see each
            # training window far less often than a small one (research log entry 89), which
            # is why the ceiling is `DL_MAX_EPOCHS` passes over this panel rather than a
            # constant (gh #101). It is the safety net for a fit that never converges.
            max_steps=dl_step_ceiling(train_windows, windows_per_step),
            val_check_steps=DL_VAL_CHECK_STEPS,
            early_stop_patience_steps=DL_PATIENCE_CHECKS,
            scaler_type="standard",
            # neuralforecast reseeds torch from its own `random_seed` when the model is
            # built, which would undo the `seed_everything` above if left at its default.
            random_seed=self.seed,
            **self.extra_kwargs,
            **_TRAINER_KW,
        )
        self._nf = NeuralForecast(models=[model], freq=case.freq)
        self._nf.fit(df, val_size=val_size)
        self._col = self.model_name
        # `model` is the pre-fit copy (NeuralForecast deep-copies models_init at construction
        # time and detaches the trainer from the fitted copy once fit() returns), so the
        # step count has to come from the fitted model's own recorded trajectory rather than
        # a live trainer, which reflects where early stopping landed rather than the ceiling
        # above.
        fitted = self._nf.models[0]
        trajectory = getattr(fitted, "train_trajectories", None)
        self.last_fit_steps = int(trajectory[-1][0]) + 1 if trajectory else None

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        # Inference-only: condition the trained network on the tail segment's history as
        # handed in at this origin (neuralforecast's `df=`), never the frame captured at
        # fit — so one fit serves every rolling origin.
        sub = history[history["unique_id"] == self._target_uid]
        new_df = sub[["unique_id", "ds", "y"]].copy()
        new_df["ds"] = _naive_ds(new_df["ds"]).to_numpy()
        new_df["y"] = pd.to_numeric(new_df["y"], errors="coerce")
        preds = self._nf.predict(df=new_df)
        if "unique_id" not in preds.columns:
            preds = preds.reset_index()  # some nf versions return unique_id as the index
        preds = preds[preds["unique_id"] == self._target_uid]
        y_hat = preds[self._col].to_numpy().astype(np.float64)
        if len(y_hat) < horizon:
            y_hat = np.concatenate([y_hat, np.full(horizon - len(y_hat), y_hat[-1])])
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat[:horizon]})


class TFTForecaster(_NFBase):
    """Temporal Fusion Transformer: an LSTM encoder-decoder combined with multi-head
    attention over the lookback window."""

    name = "TFT"
    model_name = "TFT"
    code_snippet = (
        "from neuralforecast import NeuralForecast\n"
        "from neuralforecast.models import TFT\n\n"
        "model = TFT(h=horizon, input_size=lookback, early_stop_patience_steps=5)\n"
        "nf = NeuralForecast(models=[model], freq=case.freq)\n"
        "nf.fit(train_df, val_size=val_size)  # train_df: unique_id, ds, y\n"
        'y_hat = nf.predict()["TFT"]'
    )
    # A single neuralforecast predict() call returns the full horizon at once — TFT's
    # decoder produces multi-horizon predictions in parallel, not step-by-step.
    forecast_strategy = ForecastStrategy.DIRECT


class ITransformerForecaster(_NFBase):
    """Inverted transformer: embeds each variate's whole series as one token and attends
    across variates rather than across time steps."""

    name = "iTransformer"
    model_name = "iTransformer"
    code_snippet = (
        "from neuralforecast import NeuralForecast\n"
        "from neuralforecast.models import iTransformer\n\n"
        "model = iTransformer(h=horizon, input_size=lookback, n_series=1,\n"
        "                     early_stop_patience_steps=5)\n"
        "nf = NeuralForecast(models=[model], freq=case.freq)\n"
        "nf.fit(train_df, val_size=val_size)  # train_df: unique_id, ds, y\n"
        'y_hat = nf.predict()["iTransformer"]'
    )
    # iTransformer is multivariate and fixed to n_series=1 here, so it cannot absorb the
    # extra segments of an extended-history fit -- keep it on the local span only.
    pools_segments = False
    extra_kwargs: ClassVar[dict] = {"n_series": 1}
    limitations = (
        _NFBase.limitations
        + " Architecturally multivariate but fixed to a single series here (n_series=1), "
        "so unlike its siblings it can't pool the extra segments of an extended-history "
        "fit -- and that same pin leaves it without a covariate slot to wire, so it is the "
        "one architecture here whose univariate column is a property of the configuration "
        "rather than of unfinished plumbing."
    )
    # Same one-call, full-horizon-at-once prediction as its siblings — see TFTForecaster.
    forecast_strategy = ForecastStrategy.DIRECT


class TimesNetForecaster(_NFBase):
    """Reshapes the series into 2D tensors by its dominant periods and applies
    convolutional blocks to capture both within-period and across-period patterns."""

    name = "TimesNet"
    model_name = "TimesNet"
    code_snippet = (
        "from neuralforecast import NeuralForecast\n"
        "from neuralforecast.models import TimesNet\n\n"
        "model = TimesNet(h=horizon, input_size=lookback, early_stop_patience_steps=5)\n"
        "nf = NeuralForecast(models=[model], freq=case.freq)\n"
        "nf.fit(train_df, val_size=val_size)  # train_df: unique_id, ds, y\n"
        'y_hat = nf.predict()["TimesNet"]'
    )
    # Same one-call, full-horizon-at-once prediction as its siblings — see TFTForecaster.
    forecast_strategy = ForecastStrategy.DIRECT
