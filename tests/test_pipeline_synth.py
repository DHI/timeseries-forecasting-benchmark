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
    CatBoostExoForecaster,
    CatBoostForecaster,
    Climatology,
    LightGBMExoForecaster,
    LightGBMForecaster,
    LinearLagsExoForecaster,
    MeanBaseline,
    Persistence,
    SeasonalNaive,
    ensemble_baselines,
    ensemble_baselines_theta,
)
from tsforecastbench.bench.viz import plot_leadtime_curve, render_matrix  # noqa: E402
from tsforecastbench.data.registry import discover_cases, get_case  # noqa: E402
from tsforecastbench.data.seasonality import dominant_period  # noqa: E402

ALL_CASES = list(discover_cases())


def test_expected_cases_present():
    for cid in (
        "discharge_lowland_denmark",
        "sewerlevel_bellinge",
        "surge_cuxhaven",
        "waves_oregon",
        "wwtp_influent_n2o_350k",
    ):
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
    if case.seasonalities:
        assert (res["status"] == "ok").all(), res[res["status"] != "ok"]["status"].tolist()
    else:
        # A declared-non-seasonal case makes SeasonalNaive honestly inapplicable (it
        # would duplicate Persistence); every other baseline still runs.
        sn = res["method"] == "SeasonalNaive"
        assert (res.loc[~sn, "status"] == "ok").all()
        assert res.loc[sn, "status"].str.contains("non-seasonal").all()


def test_climatology_forecast_is_periodic_in_declared_dominant_period():
    # Climatology keys on the case's dominant period, so its forecast repeats with that
    # period regardless of frequency (issue #54). demand_ltown is hourly; its strong
    # hour-of-day swing makes 24 the period that explains the most variance, so
    # dominant_period selects 24 and the forecast cycles cleanly every 24 steps.
    case = get_case("demand_ltown")
    train = case.split_frame(case.synth())["train"]
    period = dominant_period(case.seasonalities, train["y"].to_numpy())

    clim = Climatology()
    clim.fit(case, train)
    horizon = 2 * period + 5
    preds = clim.predict(case, train, horizon)["y_hat"].to_numpy()
    np.testing.assert_allclose(preds[:period], preds[period : 2 * period])


def test_climatology_on_nonseasonal_case_is_global_mean():
    # A declared-non-seasonal case has no cycle to key on, so every position collapses to
    # one bucket and the forecast is the global training mean (decision B on issue #54).
    from tsforecastbench.data.schema import Case

    case = Case(
        id="dummy",
        domain="test",
        target="y",
        freq="1h",
        seasonalities=[],
        horizon_primary="6h",
        input_window="24h",
        splits={"train_end": "2020-01-01", "val_end": "2021-01-01", "test_end": "2022-01-01"},
        source={"name": "test"},
    )
    ds = pd.date_range("2019-01-01", periods=240, freq="1h")
    train = pd.DataFrame({"ds": ds, "y": np.arange(240, dtype=float)})

    clim = Climatology()
    clim.fit(case, train)
    preds = clim.predict(case, train, 6)["y_hat"].to_numpy()
    np.testing.assert_allclose(preds, train["y"].mean())


def test_ensemble_baselines_run():
    case = get_case("discharge_lowland_denmark")
    methods = [ensemble_baselines(), ensemble_baselines_theta()]
    res = run_case(case, methods, prefer_synth=True)
    assert len(res) == len(methods) * 5  # MAE, RMSE, MASE, Skill_vs_Persistence, Skill_vs_Mean
    assert (res["status"] == "ok").all(), res[res["status"] != "ok"]["status"].tolist()


def test_ensemble_is_median_of_members():
    case = get_case("discharge_lowland_denmark")
    train = case.split_frame(case.synth())["train"]
    horizon = 24

    ens = ensemble_baselines()
    ens.fit(case, train)
    combined = ens.predict(case, train, horizon)["y_hat"].to_numpy()

    member_preds = []
    for member in ens.members:
        member.fit(case, train)
        member_preds.append(member.predict(case, train, horizon)["y_hat"].to_numpy())
    expected = np.nanmedian(np.vstack(member_preds), axis=0)

    np.testing.assert_allclose(combined, expected)


def test_lightgbm_runs_on_one_case():
    res = run_case(get_case("waves_oregon"), [LightGBMForecaster()], prefer_synth=True)
    assert (res["status"] == "ok").all()


def test_catboost_runs_on_one_case():
    res = run_case(get_case("waves_oregon"), [CatBoostForecaster()], prefer_synth=True)
    assert (res["status"] == "ok").all()


def test_rolling_matrix_runs():
    from tsforecastbench.bench.eval import run_case_rolling

    case = get_case("discharge_flashy_catskills")
    methods = [Persistence(), LightGBMExoForecaster()]
    res, snaps = run_case_rolling(case, methods, n_origins=6, prefer_synth=True)
    assert len(res) == len(methods) * 5
    assert (res["status"] == "ok").all(), res[res["status"] != "ok"]["status"].tolist()
    # pooled rolling score is finite and persistence skill-vs-itself is ~0
    pers = res[(res.method == "Persistence") & (res.metric == "Skill_vs_Persistence")]["value"]
    assert abs(float(pers.iloc[0])) < 1e-6
    # snapshots: full predicted series per method x origin, aligned on shared origins
    assert set(snaps.columns) == {"case_id", "method", "origin_id", "origin_ds", "ds", "y_hat"}
    assert set(snaps["method"].unique()) == {m.name for m in methods}
    per_method_origins = snaps.groupby("method")["origin_id"].apply(lambda s: sorted(s.unique()))
    assert per_method_origins.iloc[0] == per_method_origins.iloc[1]


@pytest.mark.parametrize("cid", ["surge_cuxhaven", "waves_oregon", "discharge_lowland_denmark"])
def test_exogenous_methods_run(cid):
    """Exogenous-aware windowed ML runs on every case that declares covariates."""
    case = get_case(cid)
    methods = [LinearLagsExoForecaster(), LightGBMExoForecaster(), CatBoostExoForecaster()]
    res = run_case(case, methods, prefer_synth=True)
    assert (res["status"] == "ok").all(), res[res["status"] != "ok"]["status"].tolist()
    assert all(m.uses_exogenous for m in methods)


def test_exogenous_method_on_univariate_case():
    """+exo variants reduce to univariate (no crash) on a case with no covariates."""
    res = run_case(get_case("demand_ltown"), [LightGBMExoForecaster()], prefer_synth=True)
    assert (res["status"] == "ok").all()


def test_dlinear_and_theta_run():
    from tsforecastbench.bench.methods import DLinearForecaster, ThetaForecaster

    res = run_case(
        get_case("discharge_flashy_catskills"),
        [DLinearForecaster(), ThetaForecaster()],
        prefer_synth=True,
    )
    assert (res["status"] == "ok").all(), res[res["status"] != "ok"]["status"].tolist()


def test_mstl_classicals_run():
    """MSTL-ETS / MSTL-ARIMA fit a seasonal classical model and produce finite MASE."""
    from tsforecastbench.bench.methods import (
        MSTLAutoARIMAForecaster,
        MSTLAutoETSForecaster,
    )

    res = run_case(
        get_case("discharge_lowland_denmark"),
        [MSTLAutoETSForecaster(), MSTLAutoARIMAForecaster()],
        prefer_synth=True,
    )
    assert (res["status"] == "ok").all(), res[res["status"] != "ok"]["status"].tolist()
    mase = res[res["metric"] == "MASE"]["value"]
    assert mase.notna().all() and (mase > 0).all()


def test_mstl_is_honestly_inapplicable_on_declared_nonseasonal_case():
    # Every committed case now declares seasonalities, so we build a non-seasonal variant
    # explicitly: same synthetic data, seasonalities overridden to []. MSTL must then report
    # itself inapplicable rather than degenerating into the plain non-seasonal trend model.
    from tsforecastbench.bench.methods import MSTLAutoETSForecaster

    case = get_case("discharge_flashy_catskills").model_copy(update={"seasonalities": []})
    res = run_case(case, [MSTLAutoETSForecaster()], prefer_synth=True)
    assert res["status"].str.contains("MSTL inapplicable").all()


def test_methods_package_imports_without_fm_deps():
    """The methods package (incl. the FM wrapper) must import without torch/chronos."""
    import tsforecastbench.bench.methods as m

    assert hasattr(m, "ChronosForecaster")


def test_chronos_runs_if_installed():
    pytest.importorskip("chronos")
    from tsforecastbench.bench.methods import ChronosForecaster

    res = run_case(get_case("waves_oregon"), [ChronosForecaster()], prefer_synth=True)
    assert (res["status"] == "ok").all(), res[res["status"] != "ok"]["status"].tolist()


def test_torch_dl_classes_importable():
    import tsforecastbench.bench.methods as m

    assert hasattr(m, "NHITSForecaster") and hasattr(m, "PatchTSTForecaster")


def test_neuralforecast_zoo_classes_importable():
    # The nf zoo (TFT/iTransformer/TimesNet) is Linux/WSL-only at runtime, but the wrapper
    # classes must always import (neuralforecast is imported lazily inside fit).
    import tsforecastbench.bench.methods as m

    assert all(
        hasattr(m, c) for c in ("TFTForecaster", "ITransformerForecaster", "TimesNetForecaster")
    )


def test_torch_dl_runs_if_installed():
    pytest.importorskip("torch")
    from tsforecastbench.bench.methods import NHITSForecaster, PatchTSTForecaster

    # tiny training budget keeps the test fast
    methods = [NHITSForecaster(), PatchTSTForecaster()]
    for mdl in methods:
        mdl.max_steps = 5
    res = run_case(get_case("discharge_flashy_catskills"), methods, prefer_synth=True)
    assert (res["status"] == "ok").all(), res[res["status"] != "ok"]["status"].tolist()


def test_fraction_split_handles_truncated_obs():
    """CAMELS uses fraction splits; even if obs end early the test window is scorable."""
    case = get_case("discharge_lowland_denmark")
    df = case.synth()
    df.loc[int(len(df) * 0.6) :, "y"] = np.nan  # observations stop at 60%
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
    case = get_case("waves_oregon")
    curve = run_leadtime_curve(
        case, [MeanBaseline(), Persistence(), Climatology()], n_origins=6, prefer_synth=True
    )
    assert {"method", "lead", "MAE", "RMSE", "bias", "n_origins"}.issubset(curve.columns)
    assert curve["lead"].max() >= 1


def test_matrix_renders(tmp_path):
    case = get_case("waves_oregon")
    res = run_case(case, [MeanBaseline(), Persistence(), Climatology()], prefer_synth=True)
    render_matrix(res, metric="MASE", out_path=tmp_path / "m.png")
    assert (tmp_path / "m.png").exists()


def test_matrix_colour_scale_clamped_to_outlier(tmp_path):
    """A catastrophic outlier must not skew the colour scale: the norm clamps to [-1, 1]
    for skill while the cell still renders. (User-reported: a -16 cell washed out the map.)"""
    rows = []
    for method, val in [("Good", 0.5), ("Bad", -16.65)]:
        rows.append(
            {
                "case_id": "c1",
                "domain": "d",
                "method": method,
                "horizon": "1h",
                "metric": "Skill_vs_Persistence",
                "value": val,
            }
        )
    fig = render_matrix(
        pd.DataFrame(rows), metric="Skill_vs_Persistence", out_path=tmp_path / "skill.png"
    )
    norm = fig.axes[0].images[0].norm
    assert norm.vmin == -1.0 and norm.vmax == 1.0  # clamped, not -16.65
    assert (tmp_path / "skill.png").exists()


def test_leadtime_plot_renders(tmp_path):
    case = get_case("waves_oregon")
    curve = run_leadtime_curve(
        case, [MeanBaseline(), Persistence()], n_origins=5, prefer_synth=True
    )
    plot_leadtime_curve(curve, out_path=tmp_path / "c.png")
    assert (tmp_path / "c.png").exists()
