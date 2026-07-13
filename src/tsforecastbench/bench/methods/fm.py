"""Time-series foundation models (zero-shot).

Wraps pretrained TSFMs behind the common Forecaster interface. These answer the KER5
question directly: does a model pretrained on huge corpora, with *no* case-specific
training, compete with the tuned baselines?

Chronos-Bolt (Amazon) is the laptop-friendly entry: encoder-decoder, CPU-capable, fast.
The dependency is optional — `uv pip install chronos-forecasting` (or the `fm` extra) —
and the wrapper imports it lazily so the rest of the benchmark runs without it. The model
is loaded once and cached across cases/origins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.data.schema import Case

_PIPELINE_CACHE: dict = {}


def _load_pipeline(model_name: str):
    if model_name not in _PIPELINE_CACHE:
        import torch
        from chronos import BaseChronosPipeline

        _PIPELINE_CACHE[model_name] = BaseChronosPipeline.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    return _PIPELINE_CACHE[model_name]


class ChronosForecaster(Forecaster):
    """Zero-shot Chronos-Bolt point forecast (predictive median).

    Univariate: the model conditions only on the target's recent context. Requires the
    optional `chronos-forecasting` dependency; methods are skipped (status=error) by the
    eval loop if it is not installed, so this never breaks a default run.
    """

    name = "Chronos-Bolt"
    family = "foundation_model"
    is_probabilistic = True
    # Zero-shot: there's no fit-time benefit, but a longer history helps up to
    # `context_length` if the shared curated span is shorter than that.
    is_global = True
    library = "chronos-forecasting (Amazon Chronos-Bolt) + torch"
    compute_profile = (
        "CPU-capable; zero-shot, so there's no per-case training — one forward "
        "pass per prediction. The pretrained model is loaded once and cached "
        "across every case and origin."
    )
    limitations = (
        "Zero-shot: never learns anything case-specific, only what it picked up "
        "from pretraining on other series. Univariate — ignores exogenous inputs. "
        "Only conditions on the most recent 2,048 points of history (context_length)."
    )
    code_snippet = (
        "from chronos import BaseChronosPipeline\n\n"
        'pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu")\n'
        "quantiles, _mean = pipe.predict_quantiles(\n"
        "    inputs=context,                  # last 2048 observed values\n"
        "    prediction_length=horizon,\n"
        "    quantile_levels=[0.1, 0.5, 0.9],\n"
        ")\n"
        "y_hat = quantiles[..., 1]            # median"
    )

    def __init__(self, model_name: str = "amazon/chronos-bolt-small", context_length: int = 2048):
        self.model_name = model_name
        self.context_length = context_length

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        # Zero-shot: no training. Keep the recent observed context for predict.
        y = pd.to_numeric(train["y"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
        self._context = y[-self.context_length :]

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        import torch

        pipe = _load_pipeline(self.model_name)
        ctx = torch.tensor(self._context, dtype=torch.float32)
        # predict_quantiles -> (quantiles [B, H, Q], mean [B, H]); take the median quantile
        quantiles, _mean = pipe.predict_quantiles(
            inputs=ctx,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        y_hat = quantiles[0, :, 1].cpu().numpy().astype(np.float64)
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})
