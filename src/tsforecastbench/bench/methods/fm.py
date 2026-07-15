"""Time-series foundation models (zero-shot).

Wraps pretrained TSFMs behind the common Forecaster interface. These answer the KER5
question directly: does a model pretrained on huge corpora, with *no* case-specific
training, compete with the tuned baselines?

Each model's dependency is optional and imported lazily, so the rest of the benchmark
runs without any of them; a method whose library is absent is skipped (status=error) by
the eval loop rather than breaking the run. Pretrained weights are loaded once and cached
across every case and origin. Each wrapper conditions only on the target's recent history
by default. Chronos-2, TimesFM-2.5 and Moirai-2.0 also ship a `+exo` variant
(`use_exogenous=True`) that additionally feeds the case's declared covariates to the
model's native covariate mechanism, mirroring the windowed-ML `+exo` pair in `ml.py` —
future covariate values are a perfect-forecast proxy supplied by the eval loop, not a real
forecast of those covariates. Chronos-Bolt has no covariate mechanism, so it stays
univariate-only.

  - Chronos-Bolt / Chronos-2 (Amazon) — `chronos-forecasting`.
  - TimesFM 2.5 (Google) — `timesfm`.
  - Moirai-2.0 (Salesforce) — `uni2ts`.

All three libraries install together via the single `fm` extra (`uv sync --extra fm`); see
`pyproject.toml` for why that extra caps the project at Python < 3.13. All four models are
CPU-capable; the larger three are slow on CPU, so pair them with a small `--origins` when
running without a GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.bench.methods.ml import _exo_cols, _fit_len
from tsforecastbench.data.schema import Case

_PIPELINE_CACHE: dict = {}
_TIMESFM_CACHE: dict = {}
_MOIRAI_CACHE: dict = {}


def _future_covariate(
    col: str, hist_series: pd.Series, horizon: int, future_exog: pd.DataFrame | None
) -> np.ndarray:
    """Perfect-forecast-proxy future values for one declared covariate.

    Same convention `_WindowedMLBase._future_X` uses: `future_exog` (supplied by the eval
    loop) stands in for a real forecast of the covariate, falling back to the last
    observed value when the column is absent or too short.
    """
    numeric = pd.to_numeric(hist_series, errors="coerce")
    fallback = float(numeric.ffill().iloc[-1]) if numeric.notna().any() else 0.0
    if future_exog is not None and col in future_exog.columns:
        vals = _fit_len(
            pd.to_numeric(future_exog[col], errors="coerce").to_numpy(), horizon, fallback
        )
    else:
        vals = np.full(horizon, fallback)
    return np.where(np.isfinite(vals), vals, fallback)


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


def _load_chronos2_pipeline(model_name: str):
    if model_name not in _PIPELINE_CACHE:
        import torch
        from chronos import Chronos2Pipeline

        _PIPELINE_CACHE[model_name] = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    return _PIPELINE_CACHE[model_name]


def _load_timesfm(
    model_name: str, max_context: int, max_horizon: int, return_backcast: bool = False
):
    # Compilation bakes in the context/horizon caps, so cache per (model, caps) bucket.
    # `return_backcast` is a separate bucket dimension: `forecast_with_covariates` requires
    # it True to fit the xreg regressor, but turning it on changes what plain `forecast()`
    # returns (backcast prepended to the horizon, not just the horizon) — so the
    # covariate-fitting model must never be reused for a univariate call, and vice versa.
    key = (model_name, max_context, max_horizon, return_backcast)
    if key not in _TIMESFM_CACHE:
        import timesfm

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_name)
        model.compile(
            timesfm.ForecastConfig(
                max_context=max_context,
                max_horizon=max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
                return_backcast=return_backcast,
            )
        )
        _TIMESFM_CACHE[key] = model
    return _TIMESFM_CACHE[key]


def _load_moirai_module(model_name: str):
    # Only the pretrained weights are cached; the Moirai2Forecast wrapper bakes in the
    # prediction length, so it is rebuilt per predict call (cheap next to the weights).
    if model_name not in _MOIRAI_CACHE:
        from uni2ts.model.moirai2 import Moirai2Module

        _MOIRAI_CACHE[model_name] = Moirai2Module.from_pretrained(model_name)
    return _MOIRAI_CACHE[model_name]


class ChronosForecaster(Forecaster):
    """Zero-shot Chronos-Bolt point forecast (predictive median).

    Univariate: the model conditions only on the target's recent context. Requires the
    optional `chronos-forecasting` dependency; methods are skipped (status=error) by the
    eval loop if it is not installed, so this never breaks a default run.
    """

    name = "Chronos-Bolt"
    family = "fm"
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
        # Zero-shot: no training and no fit-time state. predict conditions on the recent
        # observed context of the history it is handed, so one fit serves every origin.
        pass

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        import torch

        y = pd.to_numeric(history["y"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
        context = y[-self.context_length :]
        pipe = _load_pipeline(self.model_name)
        ctx = torch.tensor(context, dtype=torch.float32)
        # predict_quantiles -> (quantiles [B, H, Q], mean [B, H]); take the median quantile
        quantiles, _mean = pipe.predict_quantiles(
            inputs=ctx,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        y_hat = quantiles[0, :, 1].cpu().numpy().astype(np.float64)
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})


class Chronos2Forecaster(Forecaster):
    """Zero-shot Chronos-2 point forecast (predictive median).

    The successor to Chronos-Bolt: a single encoder-only model that natively handles
    univariate, multivariate and covariate-informed forecasting. Here it runs univariate,
    conditioning only on the target's recent context. Requires the optional
    `chronos-forecasting` dependency (same `fm` extra as Chronos-Bolt); methods are
    skipped (status=error) by the eval loop if it is not installed.
    """

    name = "Chronos-2"
    family = "fm"
    is_probabilistic = True
    is_global = True
    library = "chronos-forecasting (Amazon Chronos-2) + torch"
    compute_profile = (
        "CPU-capable but slower than Chronos-Bolt (120M params); zero-shot, so there's "
        "no per-case training — one forward pass per prediction via the DataFrame API. "
        "The pretrained model is loaded once and cached across every case and origin."
    )
    limitations = (
        "Zero-shot: never learns anything case-specific, only what it picked up from "
        "pretraining on other series. Univariate by default — Chronos-2 supports "
        "covariates natively, and the +exo variant feeds the case's declared covariates "
        "to `predict_df` as past+future covariate columns, using a perfect-forecast "
        "proxy (not a real forecast of those covariates) for the future values. "
        "Conditions on the most recent `context_length` points of history."
    )
    code_snippet = (
        "from chronos import Chronos2Pipeline\n\n"
        'pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")\n'
        "pred = pipe.predict_df(\n"
        "    context_df,                      # id / timestamp / target (+ covariate) columns\n"
        "    future_df=future_df,             # +exo: future covariate values (id/timestamp/cols)\n"
        "    prediction_length=horizon,\n"
        "    quantile_levels=[0.1, 0.5, 0.9],\n"
        '    id_column="id",\n'
        '    timestamp_column="timestamp",\n'
        '    target="target",\n'
        ")\n"
        'y_hat = pred["predictions"]          # median'
    )

    use_exogenous: bool = False

    def __init__(self, model_name: str = "amazon/chronos-2", context_length: int = 2048):
        self.model_name = model_name
        self.context_length = context_length
        self.uses_exogenous = self.use_exogenous

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        # Zero-shot: no training and no fit-time state (see ChronosForecaster.fit).
        pass

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        y = pd.to_numeric(history["y"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
        ds = pd.DatetimeIndex(history["ds"])
        context_df = pd.DataFrame(
            {
                "id": 0,
                "timestamp": ds[-self.context_length :],
                "target": y[-self.context_length :],
            }
        )
        exo = [c for c in _exo_cols(case) if c in history.columns] if self.use_exogenous else []
        future = self._future_index(case, history["ds"].max(), horizon)
        future_df = None
        if exo:
            tail = history.iloc[-self.context_length :]
            for c in exo:
                context_df[c] = pd.to_numeric(tail[c], errors="coerce").to_numpy()
            future_df = pd.DataFrame({"id": 0, "timestamp": future})
            for c in exo:
                future_df[c] = _future_covariate(c, history[c], horizon, future_exog)
        pipe = _load_chronos2_pipeline(self.model_name)
        pred = pipe.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        y_hat = pred["predictions"].to_numpy(dtype=np.float64)[:horizon]
        return pd.DataFrame({"ds": future, "y_hat": y_hat})


class TimesFMForecaster(Forecaster):
    """Zero-shot TimesFM 2.5 point forecast.

    Google's patch-based decoder foundation model (200M params). Univariate — conditions
    on the target's recent context only. Requires the optional `timesfm` dependency (the
    `timesfm` extra); skipped (status=error) by the eval loop if it is not installed.
    """

    name = "TimesFM-2.5"
    family = "fm"
    is_probabilistic = True
    is_global = True
    library = "timesfm (Google TimesFM 2.5, torch backend)"
    compute_profile = (
        "CPU-capable but slow (200M params); zero-shot, so there's no per-case training — "
        "one forward pass per prediction. The pretrained model is loaded and compiled once "
        "(per context/horizon cap) and cached across every case and origin."
    )
    limitations = (
        "Zero-shot: never learns anything case-specific, only what it picked up from "
        "pretraining. Univariate by default — the +exo variant feeds the case's declared "
        "covariates to `forecast_with_covariates` as an xreg regressor fit alongside the "
        "TimesFM forecast, using a perfect-forecast proxy (not a real forecast of those "
        "covariates) for the future values. Conditions on the most recent `max_context` "
        "points; horizons beyond `max_horizon` are not supported."
    )
    code_snippet = (
        "import timesfm\n\n"
        'model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")\n'
        "model.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=256,\n"
        "    normalize_inputs=True, use_continuous_quantile_head=True))\n"
        "point_forecast, _quantiles = model.forecast(\n"
        "    horizon=horizon,\n"
        "    inputs=[context],                # list of 1D arrays, one per series\n"
        ")\n"
        "y_hat = point_forecast[0]\n"
        "# +exo: model.forecast_with_covariates(inputs=[context],\n"
        "#     dynamic_numerical_covariates={name: [context_and_horizon_values]})"
    )

    use_exogenous: bool = False

    def __init__(
        self,
        model_name: str = "google/timesfm-2.5-200m-pytorch",
        max_context: int = 1024,
    ):
        self.model_name = model_name
        self.max_context = max_context
        self.uses_exogenous = self.use_exogenous

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        # Zero-shot: no training and no fit-time state (see ChronosForecaster.fit).
        pass

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        y = pd.to_numeric(history["y"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
        context = y[-self.max_context :]
        # max_horizon must cover the horizon; bucket to a stable value so the compiled
        # model is reused across origins rather than recompiled each call.
        max_horizon = max(256, horizon)
        exo = [c for c in _exo_cols(case) if c in history.columns] if self.use_exogenous else []
        if exo:
            # forecast_with_covariates needs return_backcast=True to fit the xreg
            # regressor; that changes what plain forecast() returns, so this must be a
            # separately cached compiled model, never the univariate one.
            model = _load_timesfm(
                self.model_name, self.max_context, max_horizon, return_backcast=True
            )
            n = len(context)
            covariates = {
                c: [
                    np.concatenate(
                        [
                            pd.to_numeric(history[c], errors="coerce")
                            .ffill()
                            .bfill()
                            .to_numpy(dtype=float)[-n:],
                            _future_covariate(c, history[c], horizon, future_exog),
                        ]
                    ).tolist()
                ]
                for c in exo
            }
            outputs, _xreg = model.forecast_with_covariates(
                inputs=[context], dynamic_numerical_covariates=covariates
            )
            y_hat = np.asarray(outputs[0], dtype=np.float64)[:horizon]
        else:
            model = _load_timesfm(self.model_name, self.max_context, max_horizon)
            point_forecast, _quantiles = model.forecast(horizon=horizon, inputs=[context])
            y_hat = np.asarray(point_forecast[0], dtype=np.float64)[:horizon]
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})


class Moirai2Forecaster(Forecaster):
    """Zero-shot Moirai-2.0 point forecast (predictive median).

    Salesforce's decoder-only, quantile foundation model, run through its GluonTS
    predictor. Univariate — conditions on the target's recent context only. Requires the
    optional `uni2ts` dependency (the `moirai` extra); skipped (status=error) by the eval
    loop if it is not installed.
    """

    name = "Moirai-2.0"
    family = "fm"
    is_probabilistic = True
    is_global = True
    library = "uni2ts (Salesforce Moirai-2.0) + gluonts + torch"
    compute_profile = (
        "CPU-capable but slow; zero-shot, so there's no per-case training — one forward "
        "pass per prediction through the GluonTS predictor. The pretrained weights are "
        "loaded once and cached across every case and origin."
    )
    limitations = (
        "Zero-shot: never learns anything case-specific, only what it picked up from "
        "pretraining on other series. Univariate by default — Moirai supports covariates "
        "natively, and the +exo variant feeds the case's declared covariates in as "
        "`feat_dynamic_real`, using a perfect-forecast proxy (not a real forecast of those "
        "covariates) for the future values. Conditions on the most recent `context_length` "
        "points of history."
    )
    code_snippet = (
        "from gluonts.dataset.pandas import PandasDataset\n"
        "from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module\n\n"
        "forecaster = Moirai2Forecast(\n"
        '    module=Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"),\n'
        "    prediction_length=horizon, context_length=ctx_len,\n"
        "    target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,\n"
        "    # +exo: feat_dynamic_real_dim=len(exo_cols); dataset built with\n"
        "    # PandasDataset(df, feat_dynamic_real=exo_cols, future_length=horizon)\n"
        ")\n"
        "predictor = forecaster.create_predictor(batch_size=1)\n"
        "forecast = next(iter(predictor.predict(dataset)))\n"
        "y_hat = forecast.median"
    )

    use_exogenous: bool = False

    def __init__(
        self,
        model_name: str = "Salesforce/moirai-2.0-R-small",
        context_length: int = 1680,
    ):
        self.model_name = model_name
        self.context_length = context_length
        self.uses_exogenous = self.use_exogenous

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        # Zero-shot: no training and no fit-time state (see ChronosForecaster.fit).
        pass

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        from gluonts.dataset.pandas import PandasDataset
        from uni2ts.model.moirai2 import Moirai2Forecast

        y = pd.to_numeric(history["y"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
        ctx_len = min(self.context_length, len(y))
        exo = [c for c in _exo_cols(case) if c in history.columns] if self.use_exogenous else []
        forecaster = Moirai2Forecast(
            module=_load_moirai_module(self.model_name),
            prediction_length=horizon,
            context_length=ctx_len,
            target_dim=1,
            feat_dynamic_real_dim=len(exo),
            past_feat_dynamic_real_dim=0,
        )
        predictor = forecaster.create_predictor(batch_size=1)
        hist_ds = pd.DatetimeIndex(history["ds"])
        if exo:
            future = self._future_index(case, history["ds"].max(), horizon)
            data = {"target": np.concatenate([y, np.zeros(horizon)])}
            for c in exo:
                hist_vals = (
                    pd.to_numeric(history[c], errors="coerce").ffill().bfill().to_numpy(dtype=float)
                )
                future_vals = _future_covariate(c, history[c], horizon, future_exog)
                data[c] = np.concatenate([hist_vals, future_vals])
            full_df = pd.DataFrame(data, index=hist_ds.append(future))
            dataset = PandasDataset(
                full_df,
                target="target",
                feat_dynamic_real=exo,
                freq=case.freq,
                future_length=horizon,
            )
        else:
            hist = pd.DataFrame({"target": y}, index=hist_ds)
            dataset = PandasDataset(hist, target="target", freq=case.freq)
        forecast = next(iter(predictor.predict(dataset)))
        y_hat = np.asarray(forecast.median, dtype=np.float64)[:horizon]
        future = self._future_index(case, history["ds"].max(), horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})


class Chronos2ExoForecaster(Chronos2Forecaster):
    name = "Chronos-2+exo"
    use_exogenous = True


class TimesFMExoForecaster(TimesFMForecaster):
    name = "TimesFM-2.5+exo"
    use_exogenous = True


class Moirai2ExoForecaster(Moirai2Forecaster):
    name = "Moirai-2.0+exo"
    use_exogenous = True
