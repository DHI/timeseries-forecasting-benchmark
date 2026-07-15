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
