"""Consistency checks for scripts/export_web_data.py's JSON output.

Runs against the real committed docs/results/scores.parquet (small, always present — see
tsforecastbench.webdata's module docstring) rather than a synthetic fixture, so these
checks catch a real mismatch between the exported JSON and the data it's derived from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from export_web_data import (  # noqa: E402
    SNAPSHOTS_PATH,
    export_cases,
    export_methods,
    export_scores,
    export_snapshots,
)


def _export(tmp_path: Path) -> tuple[list[dict], dict, dict]:
    export_scores(tmp_path)
    export_methods(tmp_path)
    export_cases(tmp_path)
    scores = json.loads(tmp_path.joinpath("scores.json").read_text())
    methods = json.loads(tmp_path.joinpath("methods.json").read_text())
    cases = json.loads(tmp_path.joinpath("cases.json").read_text())
    return scores, methods, cases


def test_export_produces_nonempty_valid_json(tmp_path):
    scores, methods, cases = _export(tmp_path)
    assert len(scores) > 0
    assert len(methods) > 0
    assert len(cases) > 0


def test_every_score_row_case_id_is_in_cases(tmp_path):
    scores, _methods, cases = _export(tmp_path)
    missing = {row["case_id"] for row in scores} - set(cases)
    assert not missing, f"scores.json references case_id(s) missing from cases.json: {missing}"


def test_every_score_row_method_is_in_methods_registry(tmp_path):
    scores, methods, _cases = _export(tmp_path)
    missing = {row["method"] for row in scores} - set(methods)
    assert not missing, f"scores.json references method(s) missing from methods.json: {missing}"


def test_methods_carry_capabilities_and_scorecard(tmp_path):
    """The method card renders these as markup rather than through the markdown path, so a
    missing key is a crash in the app rather than a blank line. Field names and the
    cross-field rules are checked in test_method_scorecards.py; this only pins that the
    export actually carries them."""
    _scores, methods, _cases = _export(tmp_path)
    for name, meta in methods.items():
        assert isinstance(meta["capabilities"], dict), name
        assert isinstance(meta["scorecard"], dict), name
        assert all(isinstance(v, bool) for v in meta["capabilities"].values()), name
        graded = {k: v for k, v in meta["scorecard"].items() if k != "running_cost"}
        assert all(v in {"Low", "Medium", "High"} for v in graded.values()), name
        # `running_cost` is a level plus the measured growth shape, not a bare level.
        cost = meta["scorecard"]["running_cost"]
        assert cost["level"] in {"Low", "Medium", "High"}, name
        assert cost["growth"] in {"flat", "linear", "superlinear"}, name


def test_scores_have_no_nan_json_literal(tmp_path):
    export_scores(tmp_path)
    raw = tmp_path.joinpath("scores.json").read_text()
    # Python's json module accepts the bare "NaN"/"Infinity" tokens it itself writes for
    # float('nan')/inf, but browsers' JSON.parse does not — a stray one here would silently
    # break every fetch of this file in the deployed app.
    assert "NaN" not in raw
    assert "Infinity" not in raw


def test_cases_have_domain_metrics_field(tmp_path):
    """`domain_metrics` isn't reliably derivable from which metrics happen to appear in
    scores.parquet for a case (the eval pipeline can compute NSE/KGE for a case even when
    its case.toml doesn't declare that domain_metrics eligibility) — this is the field the
    client-side `cases_eligible_for_metric` port must read instead."""
    _scores, _methods, cases = _export(tmp_path)
    for case_id, meta in cases.items():
        assert "domain_metrics" in meta, case_id
        assert isinstance(meta["domain_metrics"], list), case_id


def test_cases_have_no_nan_json_literal(tmp_path):
    export_cases(tmp_path)
    raw = tmp_path.joinpath("cases.json").read_text()
    assert "NaN" not in raw
    assert "Infinity" not in raw


def test_case_series_have_matching_ds_y_lengths(tmp_path):
    """train_tail/test_split are exported as parallel ds/y arrays (not one array of
    records) to keep the JSON smaller — a length mismatch would silently misalign every
    point once the client zips them back together."""
    _scores, _methods, cases = _export(tmp_path)
    for case_id, meta in cases.items():
        for field in ("train_tail", "train_tail_extended", "test_split"):
            series = meta[field]
            assert len(series["ds"]) == len(series["y"]), f"{case_id}.{field}"


def test_train_tail_is_bounded(tmp_path):
    """The training quick-look plot is a bounded tail, not a case's full (possibly
    multi-hundred-thousand-point) history — this is the field that keeps a case like
    wwtp_influent_n2o_350k from shipping its raw series."""
    _scores, _methods, cases = _export(tmp_path)
    for case_id, meta in cases.items():
        assert len(meta["train_tail"]["ds"]) <= 1000, case_id
        assert len(meta["train_tail_extended"]["ds"]) <= 1000, case_id


def test_split_counts_extended_only_differs_in_train(tmp_path):
    """`split_counts`/`split_counts_extended` should only ever differ in the train count
    — extended history never reaches into val/test (see webdata.load_case_split_counts)."""
    _scores, _methods, cases = _export(tmp_path)
    for case_id, meta in cases.items():
        assert meta["split_counts"]["val"] == meta["split_counts_extended"]["val"], case_id
        assert meta["split_counts"]["test"] == meta["split_counts_extended"]["test"], case_id


def test_distribution_x_grid_matches_density_length(tmp_path):
    _scores, _methods, cases = _export(tmp_path)
    for case_id, meta in cases.items():
        for field in ("distribution", "distribution_extended"):
            dist = meta[field]
            n = len(dist["x_grid"])
            if dist["train_density"] is not None:
                assert len(dist["train_density"]) == n, f"{case_id}.{field}"
            if dist["test_density"] is not None:
                assert len(dist["test_density"]) == n, f"{case_id}.{field}"


# export_snapshots takes ~1-2 minutes (curation runs 9 metrics x every method x every
# case) — a module-scoped fixture runs it once and every snapshot test below reads that
# same output, rather than each test re-running the full export.
@pytest.fixture(scope="module")
def snapshot_paths(tmp_path_factory) -> list[Path]:
    if not SNAPSHOTS_PATH.exists():
        pytest.skip(f"{SNAPSHOTS_PATH} not present — run a rolling pilot to produce it")
    out_dir = tmp_path_factory.mktemp("snapshots")
    export_snapshots(out_dir)
    return sorted((out_dir / "snapshots").glob("*.json"))


@pytest.fixture(scope="module")
def snapshot_bundles(snapshot_paths) -> dict[str, dict]:
    return {path.stem: json.loads(path.read_text()) for path in snapshot_paths}


def test_snapshots_produce_one_bundle_per_case(snapshot_bundles):
    assert len(snapshot_bundles) > 0


def test_snapshot_actual_window_length_matches_history_plus_horizon(snapshot_bundles):
    for case_id, bundle in snapshot_bundles.items():
        expected_len = bundle["n_history"] + bundle["horizon_steps"]
        for origin_id, values in bundle["actual"].items():
            assert len(values) == expected_len, f"{case_id} origin {origin_id}"


def test_snapshot_curation_origin_ids_exist_in_origins(snapshot_bundles):
    """Every curated origin_id must be one this case bundle actually has data for — a
    mismatch here would mean the client looks up a prediction/actual window that was
    never exported."""
    for case_id, bundle in snapshot_bundles.items():
        known_origins = {o["origin_id"] for o in bundle["origins"]}
        for metric, by_method in bundle["curation"].items():
            for method, by_variant in by_method.items():
                for variant, curated in by_variant.items():
                    for entry in curated:
                        assert entry["origin_id"] in known_origins, (
                            f"{case_id}/{metric}/{method}/{variant}: "
                            f"unknown origin_id {entry['origin_id']}"
                        )


def test_snapshot_predictions_keyed_by_known_origins(snapshot_bundles):
    for case_id, bundle in snapshot_bundles.items():
        known_origins = {o["origin_id"] for o in bundle["origins"]}
        for method, by_variant in bundle["predictions"].items():
            for variant, by_origin in by_variant.items():
                for origin_id in by_origin:
                    assert int(origin_id) in known_origins, f"{case_id}/{method}/{variant}"


def test_snapshots_have_no_nan_json_literal(snapshot_paths):
    for path in snapshot_paths:
        raw = path.read_text()
        assert "NaN" not in raw, path
        assert "Infinity" not in raw, path
