"""The +exo DL methods must actually read the covariate values over the horizon.

The contract these copy is the windowed-ML one: at step t+h the method sees the covariate at
t+h, not the covariate's history. So the decisive test is a series whose target is a pure
function of a covariate the model can only know from `future_exog` -- the univariate variant
has no way to fit it and the +exo variant should recover it almost exactly.

This is also the guard against a misaligned horizon block. A covariate shifted against the
target still trains to convergence and just forecasts worse, so nothing else would catch it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecastbench.data.schema import Case

H_STEPS = 6
NO_VAL = pd.DataFrame({"unique_id": [], "ds": pd.to_datetime([]), "y": []})


def _case() -> Case:
    return Case(
        id="dummy",
        domain="test",
        target="y",
        freq="1h",
        seasonalities=[24],
        horizon_primary=f"{H_STEPS}h",
        splits={"train_end": "2020-01-01", "val_end": "2021-01-01", "test_end": "2022-01-01"},
        source={"name": "test"},
        exogenous=["drive"],
        exogenous_known_future=["drive"],
    )


def _driven_series(n: int = 1400, seed: int = 0) -> pd.DataFrame:
    """y[t] = 10 * drive[t], with drive an independent random walk.

    Nothing about y's own past predicts it beyond what drive carries, so a univariate window
    model can only track the walk's persistence. The covariate at the predicted step
    determines the target outright.
    """
    rng = np.random.default_rng(seed)
    drive = np.cumsum(rng.normal(size=n)) + 50.0
    return pd.DataFrame(
        {
            "unique_id": "a",
            "ds": pd.date_range("2020-01-01", periods=n, freq="h"),
            "y": 10.0 * drive,
            "drive": drive,
        }
    )


def _rmse_over_horizon(cls, df: pd.DataFrame) -> float:
    """Fit on all but the last horizon, forecast it with the true covariates, score."""
    case = _case()
    train, held = df.iloc[:-H_STEPS], df.iloc[-H_STEPS:]
    method = cls()
    method.fit(case, train, NO_VAL)  # DLinear reads no holdout: closed-form fit
    future_exog = held[["ds", "drive"]].reset_index(drop=True)
    pred = method.predict(case, train, H_STEPS, future_exog=future_exog)
    assert np.isfinite(pred["y_hat"]).all()
    return float(np.sqrt(np.mean((pred["y_hat"].to_numpy() - held["y"].to_numpy()) ** 2)))


def test_dlinear_exo_recovers_a_covariate_driven_target():
    from tsforecastbench.bench.methods.dl import DLinearExoForecaster, DLinearForecaster

    df = _driven_series()
    uni = _rmse_over_horizon(DLinearForecaster, df)
    exo = _rmse_over_horizon(DLinearExoForecaster, df)
    # y is 10*drive exactly, so a method reading the horizon's covariates should be near-exact
    # while the univariate one is left extrapolating a random walk.
    assert exo < uni / 10, f"+exo RMSE {exo:.4f} not far below univariate {uni:.4f}"
    assert exo < 0.05 * df["y"].std()


def test_dlinear_exo_is_unchanged_when_the_case_has_no_covariates():
    """On a case declaring nothing, the +exo variant must reduce to its univariate self."""
    from tsforecastbench.bench.methods.dl import DLinearExoForecaster, DLinearForecaster

    case = Case(
        id="dummy",
        domain="test",
        target="y",
        freq="1h",
        seasonalities=[24],
        horizon_primary=f"{H_STEPS}h",
        splits={"train_end": "2020-01-01", "val_end": "2021-01-01", "test_end": "2022-01-01"},
        source={"name": "test"},
    )
    df = _driven_series().drop(columns=["drive"])
    train = df.iloc[:-H_STEPS]
    out = []
    for cls in (DLinearForecaster, DLinearExoForecaster):
        m = cls()
        m.fit(case, train, NO_VAL)
        out.append(m.predict(case, train, H_STEPS)["y_hat"].to_numpy())
    np.testing.assert_allclose(out[0], out[1], rtol=0, atol=0)
