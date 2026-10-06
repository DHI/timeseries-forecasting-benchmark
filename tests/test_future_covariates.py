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
import pytest

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
        dominant_seasonality=24,
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


N_ORIGINS = 40


def _rmse_over_origins(cls, df: pd.DataFrame, needs_val: bool) -> float:
    """One fit, then score `N_ORIGINS` disjoint horizons -- the eval loop's own protocol.

    Averaging over origins is the point. A single horizon is 6 points of a random walk, and
    scoring on one made an architecture that measured 36% better over 40 origins look 70%
    worse; two design decisions were nearly made on that noise.
    """
    case = _case()
    cut = len(df) - N_ORIGINS * H_STEPS
    method = cls()
    if needs_val:
        method.fit(case, df.iloc[: cut - 300], df.iloc[cut - 300 : cut])
    else:
        method.fit(case, df.iloc[:cut], NO_VAL)  # closed-form fit reads no holdout
    errs = []
    for k in range(N_ORIGINS):
        origin = cut + k * H_STEPS
        window = df.iloc[origin : origin + H_STEPS]
        pred = method.predict(
            case,
            df.iloc[:origin],
            H_STEPS,
            future_exog=window[["ds", "drive"]].reset_index(drop=True),
        )
        assert np.isfinite(pred["y_hat"]).all()
        errs.append(np.sqrt(np.mean((pred["y_hat"].to_numpy() - window["y"].to_numpy()) ** 2)))
    return float(np.mean(errs))


def test_dlinear_exo_recovers_a_covariate_driven_target():
    from tsforecastbench.bench.methods.dl import DLinearExoForecaster, DLinearForecaster

    df = _driven_series(n=2000)
    uni = _rmse_over_origins(DLinearForecaster, df, needs_val=False)
    exo = _rmse_over_origins(DLinearExoForecaster, df, needs_val=False)
    # y is 10*drive exactly and the ridge solves in closed form, so it recovers the map
    # almost outright -- measured 17.97 to 0.010.
    assert exo < uni / 100, f"+exo RMSE {exo:.4f} vs univariate {uni:.4f}"


def test_dlinear_exo_is_unchanged_when_the_case_has_no_covariates():
    """On a case declaring nothing, the +exo variant must reduce to its univariate self."""
    from tsforecastbench.bench.methods.dl import DLinearExoForecaster, DLinearForecaster

    case = Case(
        id="dummy",
        domain="test",
        target="y",
        freq="1h",
        seasonalities=[24],
        dominant_seasonality=24,
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


@pytest.mark.parametrize(
    ("base_name", "exo_name"),
    [("NHITSForecaster", "NHITSExoForecaster"), ("PatchTSTForecaster", "PatchTSTExoForecaster")],
)
def test_torch_exo_beats_its_univariate_base_on_a_driven_target(base_name, exo_name):
    """Same claim as for DLinear: the covariate at the predicted step carries the target."""
    pytest.importorskip("torch")
    import tsforecastbench.bench.methods.torchdl as td

    df = _driven_series(n=2000)
    scores = {
        name: _rmse_over_origins(getattr(td, name), df, needs_val=True)
        for name in (base_name, exo_name)
    }
    # Gradient-trained with early stopping, so nothing like DLinear's closed-form recovery:
    # measured -31% for N-HiTS and -36% for PatchTST over 40 origins. The threshold is loose
    # enough to survive a seed draw and tight enough to fail if the covariate path is inert,
    # which it was under every rejected variant in `_exo_term`'s notes.
    assert scores[exo_name] < 0.85 * scores[base_name], scores
