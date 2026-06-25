"""Download CAMELS-DK discharge for one gauged catchment from the GEUS Dataverse.

CAMELS-DK (Koch et al. 2025, ESSD), DOI 10.22008/FK2/AZXSYP. The per-catchment daily
dynamics live inside a single zip on the Dataverse: 'Dynamics/Gauged_catchments.zip'
(file id 83022). Each catchment is one file inside that zip.

Pipeline:
  1. Download Gauged_catchments.zip (cached under data/raw/).
  2. List members; pick the one for CONFIG['catchment_id'] (or the first if unset).
  3. Parse it (CSV or NetCDF), map columns, write canonical Parquet.

Run:
    uv run python -m tsforecastbench.data.cases.discharge_lowland_denmark.download
    uv run python -m tsforecastbench.data.cases.discharge_lowland_denmark.download --list-zip   # show members
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from tsforecastbench.data.cases import _io
from tsforecastbench.data.registry import get_case

DATAVERSE_BASE = "https://dataverse.geus.dk"
GAUGED_DYNAMICS_FILE_ID = 83022  # Dynamics/Gauged_catchments.zip
SIGNATURE_OBS_FILE_ID = 84628  # Attibutes/CAMELS_DK_signature_obs_based.csv
TOPOGRAPHY_FILE_ID = 84631  # Attibutes/CAMELS_DK_topography.csv

CONFIG = {
    # Set to a station id substring to pick a specific catchment; None -> first member.
    # Choose a lowland, baseflow-dominated catchment from Attibutes/CAMELS_DK_topography.csv
    # + signature_obs_based.csv once data is in hand.
    "catchment_id": "37470707",  # BFI 0.97, low slope 0.88: groundwater-dominated lowland
    "col_map_candidates": {
        "q_obs": ["Qobs", "q_obs", "streamflow", "discharge", "Q", "obs_streamflow"],
        "precip": ["precipitation", "precip", "P", "Psum"],
        "temp": ["temperature", "temp", "Tmean", "T"],
        "pet": ["pet", "PET", "Ep", "potential_evapotranspiration", "ref_et"],
    },
}


def _download_zip(case):
    dest = _io.DATA_RAW / case.id / "Gauged_catchments.zip"
    if dest.exists():
        print(f"[discharge_lowland_denmark] using cached {dest.name}")
        return dest
    url = f"{DATAVERSE_BASE}/api/access/datafile/{GAUGED_DYNAMICS_FILE_ID}"
    print("[discharge_lowland_denmark] downloading Gauged_catchments.zip ...")
    return _io.http_download(url, dest)


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _download_attr(case, file_id: int, name: str):
    dest = _io.DATA_RAW / case.id / "attributes" / name
    if dest.exists():
        return dest
    return _io.http_download(f"{DATAVERSE_BASE}/api/access/datafile/{file_id}", dest)


def rank_catchments(case, top: int = 20) -> pd.DataFrame:
    """Rank gauged catchments by baseflow index (groundwater dominance) and low slope.

    Pulls signature_obs_based.csv (+ topography.csv) and prints the most
    groundwater-dominated lowland candidates. Column names are detected flexibly.
    """
    sig_path = _download_attr(case, SIGNATURE_OBS_FILE_ID, "signature_obs_based.csv")
    topo_path = _download_attr(case, TOPOGRAPHY_FILE_ID, "topography.csv")
    sig = _io.read_table(sig_path)
    topo = _io.read_table(topo_path)

    def find(df, cands):
        low = {c.lower(): c for c in df.columns}
        for cc in cands:
            if cc.lower() in low:
                return low[cc.lower()]
        return None

    id_sig = find(sig, ["catch_id", "catchment_id", "gauge_id", "id", "station_id"])
    bfi = find(sig, ["baseflow_index", "bfi", "sig_bfi", "baseflowindex"])
    if id_sig is None or bfi is None:
        print(f"[rank] could not find id/BFI columns. signature columns: {list(sig.columns)}")
        return sig
    id_topo = find(topo, ["catch_id", "catchment_id", "gauge_id", "id", "station_id"])
    slope = find(topo, ["slope_mean", "mean_slope", "slope", "gauge_slope"])
    area = find(topo, ["area", "catchment_area", "area_km2"])

    df = sig[[id_sig, bfi]].copy()
    df.columns = ["catch_id", "baseflow_index"]
    if id_topo is not None:
        keep = [id_topo] + [c for c in (slope, area) if c]
        t = topo[keep].copy()
        t.columns = ["catch_id"] + [n for n, c in (("slope", slope), ("area", area)) if c]
        df = df.merge(t, on="catch_id", how="left")

    df = df.sort_values("baseflow_index", ascending=False)
    print(f"[rank] top {top} catchments by baseflow index (groundwater-dominated first):")
    print(df.head(top).to_string(index=False))
    print("\nSet CONFIG['catchment_id'] to one of the catch_id values above.")
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list-zip", action="store_true", help="List zip members and exit")
    p.add_argument("--rank", action="store_true", help="Rank catchments by baseflow index")
    args = p.parse_args(argv)

    case = get_case("discharge_lowland_denmark")

    if args.rank:
        try:
            rank_catchments(case)
            return 0
        except Exception as exc:
            print(f"[discharge_lowland_denmark] rank failed: {exc}")
            return 1
    try:
        zip_path = _download_zip(case)
    except Exception as exc:
        print(f"[discharge_lowland_denmark] download failed: {exc}")
        return 1

    members = [m for m in _io.list_zip(zip_path) if not m.endswith("/")]
    data_members = [m for m in members if m.lower().endswith((".csv", ".txt", ".nc", ".nc4"))]

    if args.list_zip:
        for m in data_members[:50]:
            print(f"  {m}")
        print(f"\n{len(data_members)} data members (of {len(members)} total).")
        return 0

    if CONFIG["catchment_id"]:
        sel = next((m for m in data_members if CONFIG["catchment_id"] in m), None)
        if sel is None:
            print(
                f"[discharge_lowland_denmark] catchment {CONFIG['catchment_id']!r} not found. "
                "Run --list-zip."
            )
            return 1
    else:
        sel = data_members[0]
        print(f"[discharge_lowland_denmark] no catchment_id set; using first member: {sel}")

    member_path = _io.extract_member(zip_path, sel, _io.DATA_RAW / case.id / "extracted")
    src = _io.read_table(member_path)

    date_col = next(
        (c for c in src.columns if c.lower() in {"date", "time", "datetime", "ds"}), None
    )
    if date_col is None:
        print(f"[discharge_lowland_denmark] no date column in {list(src.columns)}")
        return 1

    out = pd.DataFrame({"unique_id": case.id, "ds": pd.to_datetime(src[date_col], utc=True)})
    for canon, cands in CONFIG["col_map_candidates"].items():
        col = _pick_column(src, cands)
        if col is None:
            print(
                f"[discharge_lowland_denmark] WARNING no source col for {canon} "
                f"(tried {cands}); columns are {list(src.columns)}"
            )
            continue
        out[canon] = pd.to_numeric(src[col], errors="coerce").to_numpy()
    if "q_obs" not in out.columns:
        print("[discharge_lowland_denmark] could not find a discharge column; aborting.")
        return 1
    out["y"] = out["q_obs"]

    path = _io.write_canonical(case, out)
    print(f"[discharge_lowland_denmark] wrote {path} ({len(out)} rows) from {sel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
