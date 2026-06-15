"""Download real data for the pilot cases (delegates to each case's download.py).

Run all:
    uv run python scripts/download_pilot.py

Run one, or discovery mode:
    uv run python -m tsforecastbench.data.cases.discharge_lowland_denmark.download --list
    uv run python -m tsforecastbench.data.cases.sewerlevel_bellinge.download --list
    uv run python -m tsforecastbench.data.cases.surge_cuxhaven.download --station-file <name>

Needs the `data` extra for some cases:
    uv sync --extra data       # xarray, netcdf4, utide, cdsapi, copernicusmarine
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PILOT_CASE_IDS = ("discharge_lowland_denmark", "sewerlevel_bellinge", "surge_cuxhaven")


def main() -> int:
    rc = 0
    for cid in PILOT_CASE_IDS:
        print(f"\n=== {cid} ===")
        mod = importlib.import_module(f"tsforecastbench.data.cases.{cid}.download")
        try:
            rc |= mod.main([])
        except SystemExit as e:
            rc |= int(e.code or 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
