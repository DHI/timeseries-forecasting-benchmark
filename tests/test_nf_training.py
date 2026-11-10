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
    return method._nf.models[0]


def _restore_callback(fitted):
    return next(
        cb for cb in fitted.trainer_kwargs["callbacks"] if type(cb).__name__ == "RestoreBestWeights"
    )


def _checks(fitted) -> list[float]:
    """Validation losses of the real checks, leaving out Lightning's step-0 sanity pass,
    which scores an untrained model on two batches and is not a check."""
    return [loss for step, loss in fitted.valid_trajectories if step > 0]


def test_the_fitted_model_keeps_the_best_checks_weights(fitted_itransformer):
    import torch

    cb = _restore_callback(fitted_itransformer)
    assert cb.best_loss == min(_checks(fitted_itransformer))
    state = fitted_itransformer.state_dict()
    assert state.keys() == cb.best_state.keys()
    assert all(torch.equal(state[k], cb.best_state[k]) for k in state)


def test_training_ran_past_the_best_check(fitted_itransformer):
    """Otherwise the test above would pass with or without the restore."""
    losses = _checks(fitted_itransformer)
    assert losses.index(min(losses)) < len(losses) - 1


def test_training_and_early_stopping_use_mse_like_torchdl(fitted_itransformer):
    from neuralforecast.losses.pytorch import MSE

    assert isinstance(fitted_itransformer.loss, MSE)
    assert isinstance(fitted_itransformer.valid_loss, MSE)
