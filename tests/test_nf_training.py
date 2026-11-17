"""How the neuralforecast zoo trains: which weights it keeps when early stopping fires.

Runs only where the `nf` extra is installed (Linux/WSL); the wrappers import it lazily.
"""

from __future__ import annotations

import pytest

from tsforecastbench.data.registry import get_case


@pytest.fixture(scope="module")
def fitted_itransformer():
    """An iTransformer fit on synthetic data with an impatient early stop, so the test is
    fast and the run is sure to train past its best check before stopping."""
    pytest.importorskip("neuralforecast")
    from tsforecastbench.bench.methods import nf_models

    mp = pytest.MonkeyPatch()
    mp.setattr(nf_models, "DL_VAL_CHECK_STEPS", 5)
    mp.setattr(nf_models, "DL_PATIENCE_CHECKS", 2)
    case = get_case("discharge_flashy_catskills")
    splits = case.split_frame(case.load(prefer_synth=True, curated=True))
    method = nf_models.ITransformerForecaster()
    method.fit(case, splits["train"], splits["val"])
    mp.undo()
    return method


@pytest.fixture(scope="module")
def fitted(fitted_itransformer):
    """The fitted neuralforecast model itself (NeuralForecast fits a deep copy)."""
    return fitted_itransformer._nf.models[0]


def _restore_callback(model):
    return next(
        cb for cb in model.trainer_kwargs["callbacks"] if type(cb).__name__ == "RestoreBestWeights"
    )


def _checks(model) -> list[tuple[int, float]]:
    """`(step, val_loss)` of the real checks, leaving out Lightning's step-0 sanity pass,
    which scores an untrained model on two batches and is not a check."""
    return [(step, loss) for step, loss in model.valid_trajectories if step > 0]


def test_the_fitted_model_keeps_the_best_checks_weights(fitted):
    import torch

    cb = _restore_callback(fitted)
    assert cb.best_loss == min(loss for _, loss in _checks(fitted))
    state = fitted.state_dict()
    assert state.keys() == cb.best_state.keys()
    assert all(torch.equal(state[k], cb.best_state[k]) for k in state)


def test_training_ran_past_the_best_check(fitted):
    """Otherwise the test above would pass with or without the restore."""
    losses = [loss for _, loss in _checks(fitted)]
    assert losses.index(min(losses)) < len(losses) - 1


def test_the_kept_step_is_the_best_checks_step(fitted_itransformer, fitted):
    best_step, _ = min(_checks(fitted), key=lambda c: c[1])
    assert fitted_itransformer.last_best_step == best_step
    assert fitted_itransformer.last_best_step < fitted_itransformer.last_fit_steps


def test_training_and_early_stopping_use_mse_like_torchdl(fitted):
    from neuralforecast.losses.pytorch import MSE

    assert isinstance(fitted.loss, MSE)
    assert isinstance(fitted.valid_loss, MSE)
