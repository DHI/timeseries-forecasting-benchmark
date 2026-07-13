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

from typing import ClassVar

import numpy as np
import pandas as pd

from tsforecastbench.bench.methods.base import Forecaster, lookback_for
from tsforecastbench.data.schema import Case

_MAX_STEPS = 500
_MAX_TRAIN = 30000
_TRAINER_KW = {"accelerator": "auto", "enable_progress_bar": False, "logger": False}


def _naive_ds(ds) -> pd.Series:
    ds = pd.to_datetime(pd.Series(ds) if not isinstance(ds, pd.Series) else ds, utc=True)
    return ds.dt.tz_convert("UTC").dt.tz_localize(None)


class _NFBase(Forecaster):
    """Shared neuralforecast wrapper; subclasses name the model class + extra kwargs."""

    name = "NF"
    family = "deep_learning"
    is_global = True
    pools_segments = True
    refit_per_origin = False
    model_name = "NHITS"  # neuralforecast.models class name
    extra_kwargs: ClassVar[dict] = {}
    library = "neuralforecast + pytorch_lightning"
    compute_profile = (
        "Uses a GPU automatically if one is available (accelerator='auto'), "
        "otherwise falls back to CPU; a heavier training loop (up to 500 steps, "
        "full pytorch_lightning trainer overhead) than the custom torch models in "
        "torchdl.py. Only available on Linux with Python <3.13 (the `nf` extra), "
        "since neuralforecast depends on `ray`."
    )
    limitations = (
        "Univariate — ignores exogenous inputs. Trains a fresh model per case with "
        "no hyperparameter tuning, using the same architecture defaults regardless "
        "of the series' scale or frequency."
    )
    code_snippet = (
        "import neuralforecast.models as nfm\n"
        "from neuralforecast import NeuralForecast\n\n"
        "model = getattr(nfm, model_name)(h=horizon, input_size=lookback, max_steps=500)\n"
        "nf = NeuralForecast(models=[model], freq=case.freq)\n"
        "nf.fit(train_df)                     # train_df: unique_id, ds, y\n"
        "y_hat = nf.predict()[model_name]"
    )

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        import neuralforecast.models as nfm
        import pytorch_lightning as pl
        from neuralforecast import NeuralForecast

        from tsforecastbench.bench.eval.loop import _horizon_to_steps

        pl.seed_everything(0)
        self._h = h = _horizon_to_steps(case.horizon_primary, case.freq)
        df = train[["unique_id", "ds", "y"]].copy()
        # The extended-history frame carries one unique_id per contiguous segment;
        # neuralforecast fits them as a panel, so no window spans a gap. The tail segment
        # (latest ds, always the last row) is the series we forecast at predict.
        self._target_uid = df["unique_id"].iloc[-1]
        df = df.groupby("unique_id", sort=False).tail(_MAX_TRAIN)
        df["ds"] = _naive_ds(df["ds"]).to_numpy()
        df["y"] = pd.to_numeric(df["y"], errors="coerce")

        model_cls = getattr(nfm, self.model_name)
        model = model_cls(
            h=h,
            input_size=lookback_for(case.freq, h),
            max_steps=_MAX_STEPS,
            scaler_type="standard",
            **self.extra_kwargs,
            **_TRAINER_KW,
        )
        self._nf = NeuralForecast(models=[model], freq=case.freq)
        self._nf.fit(df)
        self._col = self.model_name

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
    name = "TFT"
    model_name = "TFT"


class ITransformerForecaster(_NFBase):
    name = "iTransformer"
    model_name = "iTransformer"
    # iTransformer is multivariate and fixed to n_series=1 here, so it cannot absorb the
    # extra segments of an extended-history fit -- keep it on the local span only.
    pools_segments = False
    extra_kwargs: ClassVar[dict] = {"n_series": 1}
    limitations = (
        _NFBase.limitations
        + " Architecturally multivariate but fixed to a single series here (n_series=1), "
        "so unlike its siblings it can't pool the extra segments of an extended-history fit."
    )


class TimesNetForecaster(_NFBase):
    name = "TimesNet"
    model_name = "TimesNet"
