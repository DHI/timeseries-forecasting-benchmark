"""Shared evaluation defaults.

A single source of truth for knobs that several entry points share, so the same
value isn't hard-coded (and drifting) across the CLI, reports, and notebooks.

`DEFAULT_ORIGINS` is the number of rolling-origin cutoffs used per cases x methods
cell. Lower it for fast iteration during development; raise it for publication-grade
scoring with tighter error bars. Override per-run without code edits via the
`TSFB_ORIGINS` environment variable, e.g.::

    TSFB_ORIGINS=3 uv run python scripts/run_pilot.py   # quick smoke test
    TSFB_ORIGINS=24 uv run python scripts/run_pilot.py  # reliable scoring

Rule of thumb: 8 is the dev default (legible lead-time curves, enough window
diversity to surface edge-case failures, cheap to run); 24 for final numbers.
"""

from __future__ import annotations

import os

# Dev default: fast to iterate on, still gives readable lead-time curves and
# enough window coverage for review. Bump to ~24 for reliable published scores.
DEFAULT_ORIGINS = int(os.environ.get("TSFB_ORIGINS", "8"))
