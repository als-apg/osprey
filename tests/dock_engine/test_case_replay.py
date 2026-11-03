"""The dock engine's validate and normalize cases replayed against the server's layout rules.

Every ``validate`` case gives its ``(reason, where)`` — or no refusal for ``valid: true`` —
through :func:`layout_refusal`, and every ``normalize`` case gives its ``output`` through
:func:`normalize_layout`, sizes within ±0.005. The cases are byte copies of the engine's
own (``cases-pin.json``), so the server refuses a layout exactly as the browser's engine does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey.interfaces.web_terminal.dock_layout import (
    LAYOUT_REASONS,
    layout_refusal,
    normalize_layout,
)

CASES = Path(__file__).resolve().parent / "cases"
SIZE_ABS = 0.005

VALIDATE = sorted((CASES / "validate").glob("*.json"))
NORMALIZE = sorted((CASES / "normalize").glob("*.json"))


def _case(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def test_both_forms_are_replayed() -> None:
    """The validate and normalize cases each carry dock and collection cases."""
    assert {_case(p).get("options", {}).get("form", "dock") for p in VALIDATE} == {
        "dock",
        "collection",
    }
    assert {_case(p).get("form", "dock") for p in NORMALIZE} == {"dock", "collection"}


def test_every_reason_is_one_a_case_names() -> None:
    """The server's reason vocabulary is exactly the one the cases refuse with."""
    named = {_case(p)["reason"] for p in VALIDATE if not _case(p).get("valid")}
    assert named == set(LAYOUT_REASONS)


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
    got = normalize_layout(case["input"], case.get("form", "dock"))
    _assert_tree(got, case["output"], "layout")
    assert json.dumps(case["input"], sort_keys=True) == before
