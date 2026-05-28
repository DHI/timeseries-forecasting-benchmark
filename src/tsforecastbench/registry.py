"""Discover and load all cases under src/tsforecastbench/cases/<id>/case.toml."""

from __future__ import annotations

from pathlib import Path

from tsforecastbench.schema import Case, load_case_from_toml

CASES_DIR = Path(__file__).resolve().parent / "cases"


def discover_cases() -> dict[str, Case]:
    """Walk cases/ and return {case_id: Case} for every case.toml found."""
    cases: dict[str, Case] = {}
    for toml in sorted(CASES_DIR.glob("*/case.toml")):
        case = load_case_from_toml(toml)
        if case.id in cases:
            raise ValueError(f"Duplicate case id {case.id}")
        cases[case.id] = case
    return cases


def get_case(case_id: str) -> Case:
    cases = discover_cases()
    if case_id not in cases:
        raise KeyError(f"Unknown case {case_id!r}. Known: {list(cases)}")
    return cases[case_id]
