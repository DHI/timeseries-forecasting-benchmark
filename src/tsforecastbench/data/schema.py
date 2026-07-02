"""Pydantic schema for a benchmark Case loaded from per-case TOML."""

from __future__ import annotations

import importlib
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

REQUIRED_COLS = ("unique_id", "ds", "y")


class Splits(BaseModel):
    """Calendar train/val/test split (inclusive end dates). Used when split_mode=calendar."""

    model_config = ConfigDict(extra="forbid")
    train_end: date
    val_end: date
    test_end: date

    @field_validator("val_end")
    @classmethod
    def _val_after_train(cls, v, info):
        if "train_end" in info.data and v <= info.data["train_end"]:
            raise ValueError("val_end must be after train_end")
        return v

    @field_validator("test_end")
    @classmethod
    def _test_after_val(cls, v, info):
        if "val_end" in info.data and v <= info.data["val_end"]:
            raise ValueError("test_end must be after val_end")
        return v


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    doi: str | None = None
    url: str | None = None
    license: str | None = None
    citation: str | None = None


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    domain: str
    notes: str = ""
    target: str
    target_unit: str = ""
    freq: str
    horizon_primary: str
    horizons_eval: list[str] = Field(default_factory=list)
    input_window: str
    exogenous: list[str] = Field(default_factory=list)
    exogenous_known_future: list[str] = Field(default_factory=list)
    # Domain-expert judgment, not derivable from measured skill: which exogenous columns
    # are necessary for a meaningful forecast (e.g. rain for a flood case). Everything
    # else in `exogenous`/`exogenous_known_future` is implicitly optional — see
    # `optional_exogenous` below and RESEARCH_LOG.md 2026-07-02.
    necessary_exogenous: list[str] = Field(default_factory=list)
    static_attrs: list[str] = Field(default_factory=list)
    splits: Splits
    source: Source
    data_type: Literal["observed", "simulated", "mixed"] = "observed"
    synth_steps: int = 4 * 365

    # Split strategy. "calendar" uses the fixed dates in `splits`. "fraction" derives
    # split dates from quantiles of the OBSERVED (non-NaN y) timestamps, which is robust
    # to series whose record doesn't cover the calendar window.
    split_mode: Literal["calendar", "fraction"] = "calendar"
    split_fraction: list[float] = Field(default_factory=lambda: [0.7, 0.85])

    @model_validator(mode="after")
    def _necessary_exogenous_is_available(self) -> "Case":
        available = set(self.exogenous) | set(self.exogenous_known_future)
        unavailable = set(self.necessary_exogenous) - available
        if unavailable:
            raise ValueError(
                f"Case {self.id}: necessary_exogenous references {sorted(unavailable)}, "
                "not present in exogenous/exogenous_known_future"
            )
        return self

    @property
    def optional_exogenous(self) -> list[str]:
        """Exogenous columns not marked necessary — i.e. enrichment, not prerequisite."""
        available = set(self.exogenous) | set(self.exogenous_known_future)
        return sorted(available - set(self.necessary_exogenous))

    @property
    def exo_tier(self) -> Literal["none", "optional", "necessary"]:
        """Domain-expert judgment of how essential exogenous input is to this case.

        "necessary": at least one exogenous column is required for a meaningful forecast.
        "optional": exogenous columns exist but are enrichments, not prerequisites.
        "none": the case has no exogenous columns at all.
        """
        if self.necessary_exogenous:
            return "necessary"
        if self.optional_exogenous:
            return "optional"
        return "none"

    @property
    def parquet_path(self) -> Path:
        return DATA_PROCESSED / f"{self.id}.parquet"

    @property
    def raw_dir(self) -> Path:
        return DATA_RAW / self.id

    def load(self, prefer_synth: bool = False) -> pd.DataFrame:
        if prefer_synth or not self.parquet_path.exists():
            return self.synth()
        df = pd.read_parquet(self.parquet_path)
        _validate_frame(self, df)
        return df

    def synth(self) -> pd.DataFrame:
        mod = importlib.import_module(f"tsforecastbench.data.cases.{self.id}.synth")
        df = mod.generate(self)
        _validate_frame(self, df)
        return df

    def _fraction_split_dates(self, df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
        obs = pd.to_datetime(df.loc[df["y"].notna(), "ds"])
        if obs.empty:
            raise ValueError(f"Case {self.id}: no observed y values to split on")
        ns = obs.astype("int64").to_numpy()
        f0, f1 = self.split_fraction
        t0 = pd.Timestamp(int(np.quantile(ns, f0)))
        t1 = pd.Timestamp(int(np.quantile(ns, f1)))
        return t0, t1

    def split_frame(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        ds_cmp = pd.to_datetime(df["ds"])
        if ds_cmp.dt.tz is not None:
            ds_cmp = ds_cmp.dt.tz_convert("UTC").dt.tz_localize(None)

        if self.split_mode == "fraction":
            train_end, val_end = self._fraction_split_dates(df.assign(ds=ds_cmp))
            obs_ds = ds_cmp[df["y"].notna()]
            test_end = obs_ds.max() if not obs_ds.empty else ds_cmp.max()
        else:
            train_end = pd.Timestamp(self.splits.train_end)
            val_end = pd.Timestamp(self.splits.val_end)
            test_end = pd.Timestamp(self.splits.test_end)

        return {
            "train": df[ds_cmp <= train_end].reset_index(drop=True),
            "val": df[(ds_cmp > train_end) & (ds_cmp <= val_end)].reset_index(drop=True),
            "test": df[(ds_cmp > val_end) & (ds_cmp <= test_end)].reset_index(drop=True),
        }


def _validate_frame(case: Case, df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Case {case.id}: missing required columns {missing}. Got {list(df.columns)}"
        )
    for c in case.exogenous:
        if c not in df.columns:
            raise ValueError(f"Case {case.id}: missing exogenous column {c}")


def load_case_from_toml(path: Path) -> Case:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return Case(**data)
