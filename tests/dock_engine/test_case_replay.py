"""The C-CASES corpus replayed against the Python layout rules.

Every ``validate`` case gives its ``(reason, where)`` — or no refusal for ``valid: true`` —
through :func:`layout_refusal`, and every ``normalize`` case gives its ``output`` through
:func:`normalize_layout`, sizes within ±0.005. The on-disk case counts are pinned to
``CUTOVER.json`` first, so a case that goes missing cannot shrink the replay unnoticed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dock_engine import cases_dir, layout_refusal, normalize_layout

PKG = Path(__file__).resolve().parents[2] / "packages" / "dock-engine"
CUTOVER = json.loads((PKG / "CUTOVER.json").read_text(encoding="utf-8"))
SIZE_ABS = 0.005

VALIDATE = sorted((cases_dir() / "validate").glob("*.json"))
NORMALIZE = sorted((cases_dir() / "normalize").glob("*.json"))


def _case(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_form(case: dict[str, Any]) -> str:
    return case.get("options", {}).get("form", "dock")


def _normalize_form(case: dict[str, Any]) -> str:
    return case.get("form", "dock")


def _assert_tree(got: Any, want: Any, at: str) -> None:
    """Compare two layout trees: exact everywhere except ``size``, which is within SIZE_ABS."""
    if isinstance(want, dict):
        assert isinstance(got, dict), at
        assert set(got) == set(want), at
        for key, value in want.items():
            if key == "size":
                assert isinstance(got[key], (int, float)) and not isinstance(got[key], bool), at
                assert got[key] == pytest.approx(value, abs=SIZE_ABS), f"{at}.size"
            else:
                _assert_tree(got[key], value, f"{at}.{key}")
    elif isinstance(want, list):
        assert isinstance(got, list), at
        assert len(got) == len(want), at
        for index, (g, w) in enumerate(zip(got, want, strict=True)):
            _assert_tree(g, w, f"{at}[{index}]")
    else:
        assert got == want and type(got) is type(want), at


def test_case_counts_match_cutover() -> None:
    """Every case kind on disk holds exactly the count CUTOVER.json records."""
    on_disk = {
        kind.name: len(list(kind.glob("*.json")))
        for kind in sorted(cases_dir().iterdir())
        if kind.is_dir()
    }
    assert on_disk == CUTOVER["case_counts"]
    assert len(VALIDATE) == CUTOVER["case_counts"]["validate"]
    assert len(NORMALIZE) == CUTOVER["case_counts"]["normalize"]


def test_both_forms_are_replayed() -> None:
    """The validate and normalize corpora each carry dock and collection cases."""
    assert {_validate_form(_case(p)) for p in VALIDATE} == {"dock", "collection"}
    assert {_normalize_form(_case(p)) for p in NORMALIZE} == {"dock", "collection"}


@pytest.mark.parametrize("path", VALIDATE, ids=[p.stem for p in VALIDATE])
def test_validate_case(path: Path) -> None:
    """A validate case refuses with its (reason, where), or not at all when valid."""
    case = _case(path)
    options = case.get("options", {})
    expected = None if case.get("valid") else (case["reason"], case["where"])
    got = layout_refusal(case["input"], options.get("form", "dock"), options.get("members"))
    assert got == expected


@pytest.mark.parametrize("path", NORMALIZE, ids=[p.stem for p in NORMALIZE])
def test_normalize_case(path: Path) -> None:
    """A normalize case yields its output tree, sizes within ±0.005."""
    case = _case(path)
    before = json.dumps(case["input"], sort_keys=True)
    got = normalize_layout(case["input"], _normalize_form(case))
    _assert_tree(got, case["output"], "layout")
    assert json.dumps(case["input"], sort_keys=True) == before
