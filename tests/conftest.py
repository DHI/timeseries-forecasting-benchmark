"""Pytest configuration.

Force the non-interactive `Agg` matplotlib backend before any test imports pyplot,
so headless test runs (CI, uv-managed Python without Tcl/Tk on Windows) don't try
to open a GUI window.

Also pin the seed sweep off. `DEFAULT_SEEDS` is 5 for real runs, which would make every
test that puts a stochastic method through the eval loop cost five fits for a result the
test does not read. Tests that mean to exercise the sweep pass `n_seeds=` explicitly.
Both are `setdefault` and read at import time, so an env var set outside still wins.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TSFB_SEEDS", "1")
