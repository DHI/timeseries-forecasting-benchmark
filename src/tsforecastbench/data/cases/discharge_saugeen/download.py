"""Build the Saugeen River discharge case from the Monash TSF archive (Zenodo).

Downloads saugeenday_dataset.zip from Zenodo record 4656058, extracts the
single-series .tsf file, and converts it to the canonical long-format Parquet.

    uv run python -m tsforecastbench.data.cases.discharge_saugeen.download
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd

from tsforecastbench.data.cases import _io
from tsforecastbench.data.registry import get_case

ZIP_URL = "https://zenodo.org/records/4656058/files/saugeenday_dataset.zip?download=1"
ZIP_NAME = "saugeenday_dataset.zip"
TSF_NAME = "saugeenday_dataset.tsf"


def _parse_tsf(path: Path) -> pd.Series:
    """Parse a Monash .tsf file, return a Series of daily discharge indexed by UTC date."""
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    in_data = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() == "@data":
            in_data = True
            continue
        if in_data:
            # Format: series_name:start_timestamp:v1,v2,...
            # Timestamps use hyphens in the time part (e.g. "1915-01-01 00-00-00")
            parts = stripped.split(":", 2)
            if len(parts) != 3:
                raise ValueError(f"Malformed TSF data row in {path}: {stripped[:120]}")
            start_str = parts[1].strip().split(" ")[0]  # keep only the date part
            start = pd.Timestamp(start_str, tz="UTC")
            values = [
                float("nan") if v.strip() in ("?", "NaN", "") else float(v)
                for v in parts[2].split(",")
            ]
            idx = pd.date_range(start=start, periods=len(values), freq="1D")
            return pd.Series(values, index=idx, name="discharge")

    raise ValueError(f"No data found in {path}")


def main() -> int:
    case = get_case("discharge_saugeen")
    raw_dir = _io.DATA_RAW / case.id
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / ZIP_NAME
    if not zip_path.exists():
        print(f"[discharge_saugeen] downloading {ZIP_URL} ...")
        _io.http_download(ZIP_URL, zip_path)
    else:
        print(f"[discharge_saugeen] found cached {zip_path}")

    print("[discharge_saugeen] extracting .tsf ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extract(TSF_NAME, raw_dir)

    series = _parse_tsf(raw_dir / TSF_NAME)
    print(
        f"[discharge_saugeen] parsed {len(series)} daily observations "
        f"({series.index[0].date()} to {series.index[-1].date()}), "
        f"{series.notna().mean():.0%} finite"
    )

    df = pd.DataFrame(
        {
            "unique_id": case.id,
            "ds": series.index,
            "y": series.to_numpy(),
            "discharge": series.to_numpy(),
        }
    )

    path = _io.write_canonical(case, df)
    print(f"[discharge_saugeen] wrote {path} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
