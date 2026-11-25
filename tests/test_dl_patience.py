"""How long a deep-learning fit waits for a new best validation loss (gh #153)."""

from __future__ import annotations

from tsforecastbench.bench.eval.budget import dl_patience_checks


def _checks(n_windows: int, windows_per_step: int = 256) -> int:
    return dl_patience_checks(n_windows, windows_per_step, check_every=50, min_checks=5, passes=2)


def test_a_small_case_keeps_the_floor():
    # 4,000 windows at batch 256 is 16 steps a pass: two passes are under one check.
    assert _checks(4_000) == 5


def test_a_large_case_waits_two_passes():
    # The sewer extended history: 1,329 steps a pass, so 2,658 steps, rounded up to checks.
    assert _checks(340_164) == 54


def test_patience_scales_with_the_steps_a_pass_takes():
    # Same windows at TimesNet's batch of 64: 5,316 steps a pass, so 10,632 steps.
    assert _checks(340_164, windows_per_step=64) == 213


def test_zero_passes_falls_back_to_the_floor():
    assert dl_patience_checks(340_164, 256, check_every=50, min_checks=5, passes=0) == 5
