"""Download (or rebuild) every case's processed Parquet from raw sources.

Convenience over `download_pilot.py`: walks every case in the registry and calls its
download.main(). Useful for fresh clones that want to regenerate processed/ from
scratch, or after data updates.

Skips gracefully when a case needs a manual archive (GESLA, WWTP) and prints how to
satisfy it.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsforecastbench.data.registry import discover_cases


def main() -> int:
    rc = 0
    cases = discover_cases()
    for cid in cases:
        print(f"\n=== {cid} ===")
        try:
            mod = importlib.import_module(f"tsforecastbench.data.cases.{cid}.download")
            rc |= mod.main([])
        except SystemExit as e:
            rc |= int(e.code or 0)
        except Exception as exc:
            print(f"[{cid}] failed: {exc}")
            rc |= 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
