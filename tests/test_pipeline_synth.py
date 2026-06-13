"""End-to-end smoke + unit tests. Everything runs on synthetic data (no network, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsforecastbench.bench.eval import run_case  # noqa: E402
from tsforecastbench.bench.eval.metrics import mae, mase, rmse, skill_score  # noqa: E402
from tsforecastbench.bench.eval.rolling import run_leadtime_curve  # noqa: E402
from tsforecastbench.bench.methods import (  # noqa: E402
    Climatology,
    LightGBMExoForecaster,
    LightGBMForecaster,
    LinearLagsExoForecaster,
    MeanBaseline,
    Persistence,
    SeasonalNaive,
)
from tsforecastbench.bench.viz import plot_leadtime_curve, render_matrix  # noqa: E402
from tsforecastbench.data.registry import discover_cases, get_case  # noqa: E402

ALL_CASES = list(discover_cases())


def test_expected_cases_present():
    for cid in ("camels_dk_discharge", "bellinge_sewer_level", "gesla_storm_surge",
                "ndbc_waves", "wwtp_influent"):
        assert cid in ALL_CASES


@pytest.mark.parametrize("cid", ALL_CASES)
def test_synth_schema(cid):
    case = get_case(cid)
    df = case.synth()
    assert {"unique_id", "ds", "y"}.issubset(df.columns), cid
    for c in case.exogenous:
        assert c in df.columns, f"{cid} missing exogenous {c}"
    assert len(df) > 100
    assert pd.api.types.is_datetime64_any_dtype(df["ds"])


@pytest.mark.parametrize("cid", ALL_CASES)
def test_splits_non_empty(cid):
    case = get_case(cid)
    sp = case.split_frame(case.synth())
    for part in ("train", "val", "test"):
        assert len(sp[part]) > 0, f"{cid} {part} empty"


@pytest.mark.parametrize("cid", ALL_CASES)
def test_baselines_run(cid):
    case = get_case(cid)
    methods = [MeanBaseline(), Persistence(), SeasonalNaive(), Climatology()]
    res = run_case(case, methods, prefer_synth=True)
    assert len(res) == len(methods) * 5  # MAE, RMSE, MASE, Skill_vs_Persistence, Skill_vs_Mean
    assert (res["status"] == "ok").all(), res[res["status"] != "ok"]["status"].tolist()


def test_lightgbm_runs_on_one_case():
    res = run_case(get_case("ndbc_waves"), [LightGBMForecaster()], prefer_synth=True)
    assert (res["status"] == "ok").all()


def test_rolling_matrix_runs():
    from tsforecastbench.bench.eval import run_case_rolling

    case = get_case("usgs_river_discharge")
    methods = [Persistence(), LightGBMExoForecaster()]
    res = run_case_rolling(case, methods, n_origins=6, prefer_synth=True)
    assert len(res) == len(methods) * 5
    assert (res["status"] == "ok").all(), res[res["status"] != "ok"]["status"].tolist()
    # pooled rolling score is finite and persistence skill-vs-itself is ~0
    pers = res[(res.method == "Persistence") & (res.metric == "Skill_vs_Persistence")]["value"]
    assert abs(float(pers.iloc[0])) < 1e-6


@pytest.mark.parametrize("cid", ["gesla_storm_surge", "ndbc_waves", "camels_dk_discharge"])
def test_exogenous_methods_run(cid):
    """Exogenous-aware windowed ML runs on every case that declares covariates."""
    case = get_case(cid)
    methods = [LinearLagsExoForecaster(), LightGBMExoForecaster()]
    res = run_case(case, methods, prefer_synth=True)
    assert (res["status"] == "ok").all(), res[res["status"] != "ok"]["status"].tolist()
    assert all(m.uses_exogenous for m in methods)


def test_exogenous_method_on_univariate_case():
    """+exo variants reduce to univariate (no crash) on a case with no covariates."""
    res = run_case(get_case("wd_battledim"), [LightGBMExoForecaster()], prefer_synth=True)
    assert (res["status"] == "ok").all()


def test_fraction_split_handles_truncated_obs():
    """CAMELS uses fraction splits; even if obs end early the test window is scorable."""
    case = get_case("camels_dk_discharge")
    df = case.synth()
    df.loc[int(len(df) * 0.6):, "y"] = np.nan  # observations stop at 60%
    sp = case.split_frame(df)
    assert sp["test"]["y"].notna().sum() > 0


def test_metrics_math():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    yhat = y + 1.0
    assert mae(y, yhat) == pytest.approx(1.0)
    assert rmse(y, yhat) == pytest.approx(1.0)
    # skill vs a worse reference is positive
    ref = y + 2.0
    assert skill_score(y, yhat, ref) > 0
    # MASE of a perfect forecast is 0
    assert mase(y, y, np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0]), season=1) == 0.0


def test_leadtime_curve_shape():
    case = get_case("ndbc_waves")
    curve = run_leadtime_curve(
        case, [MeanBaseline(), Persistence(), Climatology()], n_origins=6, prefer_synth=True
    )
    assert {"method", "lead", "MAE", "RMSE", "bias", "n_origins"}.issubset(curve.columns)
    assert curve["lead"].max() >= 1


def test_matrix_renders(tmp_path):
    case = get_case("ndbc_waves")
    res = run_case(case, [MeanBaseline(), Persistence(), Climatology()], prefer_synth=True)
    render_matrix(res, metric="MASE", out_path=tmp_path / "m.png")
    assert (tmp_path / "m.png").exists()


def test_leadtime_plot_renders(tmp_path):
    case = get_case("ndbc_waves")
    curve = run_leadtime_curve(case, [MeanBaseline(), Persistence()], n_origins=5, prefer_synth=True)
    plot_leadtime_curve(curve, out_path=tmp_path / "c.png")
    assert (tmp_path / "c.png").exists()
