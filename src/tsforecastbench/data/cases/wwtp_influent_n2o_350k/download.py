"""Build the WWTP influent-flow case from a manually downloaded Mendeley archive.

Mendeley's API surface is fragile across versions, so we use a manual one-time download
of the dataset zip (single "Download All" button on the dataset page):

    https://data.mendeley.com/datasets/xmbxhscgpr/1

Save the resulting zip (or its extracted contents) under
    data/raw/wwtp_influent_n2o_350k/
Then run:
    uv run python -m tsforecastbench.data.cases.wwtp_influent_n2o_350k.download --list   # see members
    uv run python -m tsforecastbench.data.cases.wwtp_influent_n2o_350k.download
After --list, edit CONFIG below if the influent-flow column is not auto-detected.
"""

from __future__ import annotations

import argparse
import sys
import zipfile

import pandas as pd

from tsforecastbench.data.cases import _io
from tsforecastbench.data.registry import get_case

DATASET_URL = "https://data.mendeley.com/datasets/xmbxhscgpr/1"

CONFIG = {
    "tabular_match": "",  # empty -> first CSV/XLSX found
    # We want INFLUENT wastewater flow. Many WWTP SCADA exports also contain AIR FLOW,
    # GAS FLOW, effluent flow etc. - so use a positive-include list ranked by preference
    # AND an exclude list, rather than a substring match on "flow" alone.
    "flow_col_include": [
        "INFLUENT.FLOW",
        "INF.FLOW",
        "Q_INFLUENT",
        "WW.FLOW",
        "WASTEWATER",
        "RAW.FLOW",
        "INLET.FLOW",
        "influent_flow",
        "flow_in",
        "Qin",
    ],
    "flow_col_exclude": ["AIR", "GAS", "BLOWER", "EFFLUENT", "EFF.", "SLUDGE", "RECIRC"],
    "flow_col_override": "INLET.Q value",  # pinned after inspecting aved_raw.csv (49 cols)
    "date_col_candidates": ["datetime", "timestamp", "time", "date", "ds"],
}


def _gather_tabular(raw_dir):
    """Return [(path, source_label)] for all CSV/XLSX in raw_dir, extracting any zips."""
    out = []
    # extract any zip into raw_dir/extracted (idempotent)
    for z in raw_dir.glob("*.zip"):
        extract_dir = raw_dir / "extracted"
        with zipfile.ZipFile(z) as zf:
            zf.extractall(extract_dir)
    for p in (
        list(raw_dir.rglob("*.csv")) + list(raw_dir.rglob("*.xlsx")) + list(raw_dir.rglob("*.xls"))
    ):
        out.append(p)
    return sorted(out)


def _pick_flow(cols) -> str | None:
    """Pick the influent-wastewater-flow column from CONFIG include/exclude rules.

    1. If override is set, return it (must exist).
    2. Otherwise score each column: + matches in include list, - matches in exclude list.
       Prefer earlier include entries (ranked). Reject any column with an exclude hit.
    3. Return highest-scored remaining column or None.
    """
    if CONFIG["flow_col_override"]:
        return CONFIG["flow_col_override"] if CONFIG["flow_col_override"] in cols else None
    excluded = {c for c in cols if any(x.lower() in c.lower() for x in CONFIG["flow_col_exclude"])}
    candidates = [c for c in cols if c not in excluded]
    best, best_rank = None, len(CONFIG["flow_col_include"]) + 1
    for c in candidates:
        for i, inc in enumerate(CONFIG["flow_col_include"]):
            if inc.lower() in c.lower() and i < best_rank:
                best, best_rank = c, i
    return best


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true")
    p.add_argument(
        "--columns", action="store_true", help="Print column names of the chosen file and exit"
    )
    args = p.parse_args(argv)

    case = get_case("wwtp_influent_n2o_350k")
    raw_dir = _io.DATA_RAW / case.id
    raw_dir.mkdir(parents=True, exist_ok=True)

    files = _gather_tabular(raw_dir)
    if not files:
        print(f"[wwtp_influent_n2o_350k] No CSV/XLSX (or zip of) under {raw_dir}/.")
        print(f"Download the dataset zip from {DATASET_URL} (Download All button),")
        print(f"save it under {raw_dir}/, then re-run.")
        return 1

    if args.list:
        for p_ in files:
            print(f"  {p_.relative_to(raw_dir)}  ({p_.stat().st_size} bytes)")
        return 0

    candidates = files
    if CONFIG["tabular_match"]:
        candidates = [p_ for p_ in files if CONFIG["tabular_match"].lower() in p_.name.lower()]
    if not candidates:
        print(f"[wwtp_influent_n2o_350k] no file matched {CONFIG['tabular_match']!r}. Run --list.")
        return 1
    src_path = candidates[0]
    print(f"[wwtp_influent_n2o_350k] reading {src_path.name}")
    raw = _io.read_table(src_path)

    if args.columns:
        for c in raw.columns:
            print(f"  {c}")
        print(f"\n{len(raw.columns)} columns in {src_path.name}")
        return 0

    dcol = next(
        (
            c
            for c in raw.columns
            for cand in CONFIG["date_col_candidates"]
            if cand.lower() == c.lower()
        ),
        None,
    )
    fcol = _pick_flow(raw.columns)
    if dcol is None or fcol is None:
        print(
            f"[wwtp_influent_n2o_350k] could not auto-detect date/flow. Columns: {list(raw.columns)}"
        )
        print("Run with --columns to see them all, then set CONFIG['flow_col_override'].")
        return 1
    print(f"[wwtp_influent_n2o_350k] using flow column: {fcol}")

    out = pd.DataFrame(
        {
            "unique_id": case.id,
            "ds": pd.to_datetime(raw[dcol], utc=True, errors="coerce"),
            "y": pd.to_numeric(raw[fcol], errors="coerce"),
        }
    )
    out["influent_flow"] = out["y"]
    out = out.dropna(subset=["ds"]).sort_values("ds")
    out = _io.resample_regular(out, case.freq)
    path = _io.write_canonical(case, out)
    print(
        f"[wwtp_influent_n2o_350k] wrote {path} ({len(out)} rows) from {src_path.name} col {fcol}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
