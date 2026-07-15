"""Consistency checks for scripts/export_web_data.py's JSON output.

Runs against the real committed docs/results/scores.parquet (small, always present — see
app/data.py's module docstring) rather than a synthetic fixture, so these checks catch a
real mismatch between the exported JSON and the data it's derived from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "app"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from export_web_data import export_cases, export_methods, export_scores  # noqa: E402


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
    — extended history never reaches into val/test (see data.load_case_split_counts)."""
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
