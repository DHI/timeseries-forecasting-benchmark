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
DATA_CURATED = REPO_ROOT / "data" / "curated"
# Extended-history frames for `Forecaster.is_global` methods (schema.py:Case.load_extended).
DATA_CURATED_EXTENDED = REPO_ROOT / "data" / "curated_extended"

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
    # Seasonal periods in steps of `freq`, dominant first (ranked by detection prominence,
    # not by period length — the longest period is not necessarily first); [] = non-seasonal.
    # The single source of truth for seasonality in the train/predict cycle (methods and the
    # MASE season all resolve from here — there is no frequency-based fallback). Values are
    # fixed once at curation time from train-split evidence (notebooks/seasonality.py);
    # deviations from the detection proposal need a justification in `notes`.
    seasonalities: list[int]
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

    @field_validator("seasonalities")
    @classmethod
    def _seasonalities_valid(cls, v: list[int]) -> list[int]:
        if any(p < 2 for p in v):
            raise ValueError(f"seasonalities must all be >= 2 steps, got {v}")
        if len(set(v)) != len(v):
            raise ValueError(f"seasonalities must be unique, got {v}")
        return v

    @model_validator(mode="after")
    def _necessary_exogenous_is_available(self) -> Case:
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
    def curated_path(self) -> Path:
        return DATA_CURATED / f"{self.id}.parquet"

    @property
    def curated_extended_path(self) -> Path:
        return DATA_CURATED_EXTENDED / f"{self.id}.parquet"

    @property
    def raw_dir(self) -> Path:
        return DATA_RAW / self.id

    def load(self, prefer_synth: bool = False, curated: bool = False) -> pd.DataFrame:
        """Load the case frame.

        `curated=True` returns the benchmark-ready series (tail contiguous span,
        small gaps filled, `y_observed` mask) that the eval loops consume: it reads
        `data/curated/<id>.parquet` when present, else applies the gap policy on the
        fly. The default (`curated=False`) returns the raw series in
        `data/processed/`, gaps and all, for EDA and case review.
        """
        if curated:
            return self._load_curated(prefer_synth=prefer_synth)
        if prefer_synth or not self.parquet_path.exists():
            return self.synth()
        df = pd.read_parquet(self.parquet_path)
        _validate_frame(self, df)
        return df

    def _load_curated(self, prefer_synth: bool = False) -> pd.DataFrame:
        from tsforecastbench.data.curate import curate_frame

        if not prefer_synth and self.curated_path.exists():
            return pd.read_parquet(self.curated_path)
        raw = self.load(prefer_synth=prefer_synth)
        exog = list(dict.fromkeys([*self.exogenous, *self.exogenous_known_future]))
        curated, _ = curate_frame(
            raw,
            self.freq,
            seasonalities=self.seasonalities,
            horizon_primary=self.horizon_primary,
            split_fraction=self.split_fraction,
            exog=exog,
        )
        return curated

    def load_extended(self, prefer_synth: bool = False) -> pd.DataFrame:
        """Full single-series history for `Forecaster.is_global` methods.

        Gap-filled the same way as `load(curated=True)` (small gaps <= the per-case
        ACF-derived cap), but without trimming to the tail segment: every observation
        at or before the tail segment's `val_end` is kept, gaps larger than the cap
        included and marked unobserved rather than used as a trim boundary. Cutting off
        at `val_end` — not the raw series' end — keeps this frame strictly before the
        test window every method (local or global) is scored on, so it adds history,
        never leaks it.
        """
        from tsforecastbench.data.curate import curate_frame, extend_frame

        if not prefer_synth and self.curated_extended_path.exists():
            return pd.read_parquet(self.curated_extended_path)
        raw = self.load(prefer_synth=prefer_synth)
        exog = list(dict.fromkeys([*self.exogenous, *self.exogenous_known_future]))
        curated, _ = curate_frame(
            raw,
            self.freq,
            seasonalities=self.seasonalities,
            horizon_primary=self.horizon_primary,
            split_fraction=self.split_fraction,
            exog=exog,
        )
        _, val_end, _ = self.split_bounds(curated)
        extended, _ = extend_frame(raw, self.freq, cutoff=val_end, exog=exog)
        return extended

    def synth(self) -> pd.DataFrame:
        mod = importlib.import_module(f"tsforecastbench.data.cases.{self.id}.synth")
        df = mod.generate(self)
        _validate_frame(self, df)
        return df

    def _fraction_split_dates(self, df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
        obs = pd.to_datetime(df.loc[_observed_mask(df), "ds"])
        if obs.empty:
            raise ValueError(f"Case {self.id}: no observed y values to split on")
        ns = obs.astype("int64").to_numpy()
        f0, f1 = self.split_fraction
        t0 = pd.Timestamp(int(np.quantile(ns, f0)))
        t1 = pd.Timestamp(int(np.quantile(ns, f1)))
        return t0, t1

    def split_bounds(self, df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
        """(train_end, val_end, test_end), tz-naive UTC, from `split_mode`."""
        ds_cmp = pd.to_datetime(df["ds"])
        if ds_cmp.dt.tz is not None:
            ds_cmp = ds_cmp.dt.tz_convert("UTC").dt.tz_localize(None)

        if self.split_mode == "fraction":
            train_end, val_end = self._fraction_split_dates(df.assign(ds=ds_cmp))
            obs_ds = ds_cmp[_observed_mask(df)]
            test_end = obs_ds.max() if not obs_ds.empty else ds_cmp.max()
        else:
            train_end = pd.Timestamp(self.splits.train_end)
            val_end = pd.Timestamp(self.splits.val_end)
            test_end = pd.Timestamp(self.splits.test_end)
        return train_end, val_end, test_end

    def split_frame(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        ds_cmp = pd.to_datetime(df["ds"])
        if ds_cmp.dt.tz is not None:
            ds_cmp = ds_cmp.dt.tz_convert("UTC").dt.tz_localize(None)
        train_end, val_end, test_end = self.split_bounds(df)

        return {
            "train": df[ds_cmp <= train_end].reset_index(drop=True),
            "val": df[(ds_cmp > train_end) & (ds_cmp <= val_end)].reset_index(drop=True),
            "test": df[(ds_cmp > val_end) & (ds_cmp <= test_end)].reset_index(drop=True),
        }


def _observed_mask(df: pd.DataFrame) -> pd.Series:
    """Genuine-observation mask. On a curated frame `y` is gap-filled, so the truth of
    what was measured lives in `y_observed`; on a raw frame it is simply `y` non-NaN."""
    if "y_observed" in df.columns:
        return df["y_observed"].astype(bool)
    return df["y"].notna()


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
