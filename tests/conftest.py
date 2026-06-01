"""Pytest configuration.

Force the non-interactive `Agg` matplotlib backend before any test imports pyplot,
so headless test runs (CI, uv-managed Python without Tcl/Tk on Windows) don't try
to open a GUI window.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
