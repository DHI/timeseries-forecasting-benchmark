"""Local pre-push check: runs the same ruff + pytest commands as CI, plus shows
any uncommitted/unstaged changes so files don't accidentally get left out of a
push.

    uv run python scripts/precheck.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def _run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd)


def _git_status() -> None:
    """Print uncommitted/unstaged changes (warning, not failure)."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    if out.strip():
        print("\nuncommitted changes (won't be in the next push):")
        for line in out.strip().splitlines():
            print(f"  {line}")
        print("if any of these are meant to land, run: git add -A")


def main() -> int:
    if shutil.which("uv") is None:
        print("uv not found; install from https://docs.astral.sh/uv/")
        return 2

    rc = 0
    rc |= _run(["uvx", "ruff", "check", "."])
    rc |= _run(["uv", "run", "--extra", "dev", "pytest", "-q"])
    _git_status()

    if rc == 0:
        print("\nlint + tests passed - check the git status warning above before pushing")
    else:
        print("\nchecks failed - fix and re-run before pushing")
    return rc


if __name__ == "__main__":
    sys.exit(main())
