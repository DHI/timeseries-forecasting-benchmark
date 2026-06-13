"""Shared download + canonical-write helpers used by every case download.py.

Network access goes through `requests`, which respects HTTP(S)_PROXY env vars.
The canonical writer enforces the long-format schema and records a SHA-256 in a
committed manifest so re-downloads are byte-verifiable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from tsforecastbench.data.schema import DATA_PROCESSED, REPO_ROOT, REQUIRED_COLS, Case

DATA_RAW = REPO_ROOT / "data" / "raw"
MANIFEST = REPO_ROOT / "data" / "manifest.json"


def http_download(url: str, dest: Path, chunk: int = 1 << 20, timeout: int = 120) -> Path:
    """Stream a URL to disk. Respects proxy env vars via requests."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for block in r.iter_content(chunk_size=chunk):
                if block:
                    f.write(block)
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _update_manifest(case_id: str, parquet: Path) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if MANIFEST.exists():
        data = json.loads(MANIFEST.read_text())
    data[case_id] = {
        "parquet": str(parquet.relative_to(REPO_ROOT)),
        "sha256": sha256(parquet),
        "rows": int(pd.read_parquet(parquet).shape[0]),
    }
    MANIFEST.write_text(json.dumps(data, indent=2, sort_keys=True))


def write_canonical(case: Case, df: pd.DataFrame) -> Path:
    """Validate schema, sort, write data/processed/<id>.parquet, update manifest."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{case.id}: canonical frame missing {missing}; got {list(df.columns)}")
    for c in case.exogenous:
        if c not in df.columns:
            raise ValueError(f"{case.id}: canonical frame missing exogenous {c}")

    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"], utc=True)
    df = df.sort_values("ds").reset_index(drop=True)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / f"{case.id}.parquet"
    df.to_parquet(out, index=False)
    _update_manifest(case.id, out)
    return out


def resample_regular(df: pd.DataFrame, freq: str, agg: dict | str = "mean") -> pd.DataFrame:
    """Resample an irregular/high-rate frame onto a regular `freq` grid.

    Keeps unique_id, sets ds as index, resamples, leaves gaps as NaN (explicit rows).
    """
    uid = df["unique_id"].iloc[0]
    s = df.drop(columns=["unique_id"]).set_index("ds").sort_index()
    out = s.resample(freq).agg(agg)
    out = out.reset_index()
    out.insert(0, "unique_id", uid)
    return out


def list_zip(zip_path: Path) -> list[str]:
    """Return the member names inside a zip."""
    import zipfile

    with zipfile.ZipFile(zip_path) as z:
        return z.namelist()


def extract_member(zip_path: Path, member: str, dest_dir: Path) -> Path:
    """Extract a single member from a zip to dest_dir, return its path."""
    import zipfile

    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extract(member, path=dest_dir)
    return dest_dir / member


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV/TSV (sniffing separator + encoding) or a NetCDF via xarray.

    Danish data files are frequently Latin-1 / cp1252 (e.g. the Bellinge rain gauge
    has chars like 'Aa'), so we try several encodings. Latin-1 decodes any byte, so
    it is the guaranteed last resort.
    """
    if str(path).endswith((".nc", ".nc4")):
        import xarray as xr

        return xr.open_dataset(path).to_dataframe().reset_index()
    if str(path).endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    return pd.read_csv(path, encoding="latin-1")
