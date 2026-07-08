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

from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.data.schema import Case

_MAX_STEPS = 500
_MAX_TRAIN = 30000
_TRAINER_KW = {"accelerator": "auto", "enable_progress_bar": False, "logger": False}


def _input_size(freq: str, h: int) -> int:
    f = freq.lower()
    base = {"1d": 64, "1h": 168, "15min": 192, "1w": 52}.get(f, 4 * h)
    return max(base, 2 * h)


def _naive_ds(ds) -> pd.Series:
    ds = pd.to_datetime(pd.Series(ds) if not isinstance(ds, pd.Series) else ds, utc=True)
    return ds.dt.tz_convert("UTC").dt.tz_localize(None)


class _NFBase(Forecaster):
    """Shared neuralforecast wrapper; subclasses name the model class + extra kwargs."""

    name = "NF"
    family = "deep_learning"
    model_name = "NHITS"  # neuralforecast.models class name
    extra_kwargs: ClassVar[dict] = {}

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        import neuralforecast.models as nfm
        from neuralforecast import NeuralForecast

        from tsforecastbench.bench.eval.loop import _horizon_to_steps

        self._h = h = _horizon_to_steps(case.horizon_primary, case.freq)
        df = train[["unique_id", "ds", "y"]].copy()
        if len(df) > _MAX_TRAIN:
            df = df.iloc[-_MAX_TRAIN:]
        df["ds"] = _naive_ds(df["ds"]).to_numpy()
        df["y"] = pd.to_numeric(df["y"], errors="coerce")

        model_cls = getattr(nfm, self.model_name)
        model = model_cls(
            h=h,
            input_size=_input_size(case.freq, h),
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
        preds = self._nf.predict()
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
    # iTransformer needs the number of series (n_series=1 here, univariate)
    extra_kwargs: ClassVar[dict] = {"n_series": 1}


class TimesNetForecaster(_NFBase):
    name = "TimesNet"
    model_name = "TimesNet"
