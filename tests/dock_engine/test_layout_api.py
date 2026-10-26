"""The C-PY surface of ``dock_engine``: every name exists, and calls the way CANOPY makes them.

The full rule corpus is replayed elsewhere (``test_case_replay.py``); these tests pin
the API shape — names, signatures, the refusal carrier and the recursion rule.
"""

from __future__ import annotations

import sys
import typing

import pytest

import dock_engine
from dock_engine import (
    LAYOUT_REASONS,
    REASON_TEXT,
    SIZE_TOLERANCE,
    SPLITS,
    LayoutForm,
    LayoutRefused,
    layout_keys,
    layout_refusal,
    normalize_layout,
    old_shape_refusal,
    validate_layout,
)

C_PY_NAMES = (
    "LAYOUT_REASONS",
    "REASON_TEXT",
    "LayoutForm",
    "SIZE_TOLERANCE",
    "SPLITS",
    "LayoutRefused",
    "layout_refusal",
    "validate_layout",
    "normalize_layout",
    "layout_keys",
    "old_shape_refusal",
    "cases_dir",
    "rules_digest",
    "__version__",
)

DOCK = {
    "split": "row",
    "items": [
        {"tile": "a", "tabs": ["terminal", "logs"], "showing": "logs", "size": 0.5},
        {"tile": "b", "tabs": ["files"], "showing": "files", "size": 0.5},
    ],
}
COLLECTION = {"split": "column", "items": [{"tabs": [0]}, {"tabs": [1], "size": 2}]}


def _deep(depth: int) -> dict[str, typing.Any]:
    node: dict[str, typing.Any] = {"tile": "leaf", "tabs": ["x"], "showing": "x"}
    for _ in range(depth):
        node = {"split": "row", "items": [node, {"tile": "t", "tabs": ["y"], "showing": "y"}]}
    return node


def test_every_c_py_name_is_exported() -> None:
    """The package root exports the whole C-PY surface, and ``__all__`` names it."""
    for name in C_PY_NAMES:
        assert hasattr(dock_engine, name), name
        assert name in dock_engine.__all__, name


def test_vocabulary_constants() -> None:
    """Fifteen reasons, each with its text; the two splits; the dock-form tolerance."""
    assert isinstance(LAYOUT_REASONS, tuple)
    assert len(LAYOUT_REASONS) == 15
    assert len(set(LAYOUT_REASONS)) == 15
    assert set(REASON_TEXT) == set(LAYOUT_REASONS)
    assert all(isinstance(text, str) and text for text in REASON_TEXT.values())
    assert SPLITS == ("row", "column")
    assert SIZE_TOLERANCE == 0.001
    assert typing.get_args(LayoutForm) == ("dock", "collection")


def test_layout_refused_carries_reason_and_where() -> None:
    """``LayoutRefused`` is a ``ValueError`` naming the reason and the path."""
    refused = LayoutRefused("bad_split", "layout.split")
    assert isinstance(refused, ValueError)
    assert refused.reason == "bad_split"
    assert refused.where == "layout.split"
    assert "layout.split" in str(refused) and "bad_split" in str(refused)


def test_validate_layout_raises_the_first_refusal() -> None:
    """A broken layout raises with the same ``(reason, where)`` ``layout_refusal`` returns."""
    broken = {"split": "diagonal", "items": DOCK["items"]}
    with pytest.raises(LayoutRefused) as caught:
        validate_layout(broken)
    assert (caught.value.reason, caught.value.where) == layout_refusal(broken)
    assert (caught.value.reason, caught.value.where) == ("bad_split", "layout.split")
    assert validate_layout(DOCK) is None


def test_form_and_members_are_positional_or_keyword() -> None:
    """CANOPY passes ``form`` and ``members`` positionally; OSPREY by keyword. Both mean the same."""
    assert layout_refusal(COLLECTION, "collection", 2) is None
    assert layout_refusal(COLLECTION, form="collection", members=2) is None
    assert layout_refusal(COLLECTION, "collection", 3) == ("member_missing", "layout")
    assert layout_refusal(COLLECTION, form="collection", members=3) == ("member_missing", "layout")
    with pytest.raises(LayoutRefused):
        validate_layout(COLLECTION, "collection", 3)
    validate_layout(COLLECTION, form="collection", members=2)
    assert normalize_layout(COLLECTION, "collection") == normalize_layout(
        COLLECTION, form="collection"
    )


def test_members_count_only_in_collection_form() -> None:
    """``members`` is ignored for the dock form, as the engine ignores it."""
    assert layout_refusal(DOCK, "dock", 99) is None


def test_rename_table_right_hand_names_are_callable_positionally() -> None:
    """``check_layout(l, f, m)`` becomes ``layout_refusal(l, f, m)`` + ``layout_keys(l)``;
    ``find_old_shape(v, w)`` becomes ``old_shape_refusal(v, w)``."""
    assert layout_refusal(DOCK, "dock", None) is None
    assert layout_keys(DOCK) == ["terminal", "logs", "files"]
    assert layout_keys(COLLECTION) == ["0", "1"]
    assert layout_keys(None) == []
    assert old_shape_refusal(DOCK, "layout") is None
    assert old_shape_refusal({"split": "row", "items": [{"slot": 0}]}, "layout") == (
        "old_slot_shape",
        "layout.items[0]",
    )
    assert old_shape_refusal({"key": "x"}, "here") == ("old_key_shape", "here")
    assert old_shape_refusal({"key": "x"}) == ("old_key_shape", "layout")


def test_normalize_layout_returns_a_fresh_tree() -> None:
    """Weights become fractions, ``showing`` is filled in, and the input is untouched."""
    out = normalize_layout(COLLECTION, "collection")
    assert [item["size"] for item in out["items"]] == pytest.approx([1 / 3, 2 / 3])
    assert [item["showing"] for item in out["items"]] == [0, 1]
    assert "showing" not in COLLECTION["items"][0]


def test_too_deep_a_tree_is_not_a_layout() -> None:
    """A tree deeper than the interpreter can walk is refused, never raised."""
    deep = _deep(sys.getrecursionlimit() + 100)
    assert layout_refusal(deep) == ("not_layout", "layout")
    assert old_shape_refusal(deep) == ("not_layout", "layout")
    with pytest.raises(LayoutRefused) as caught:
        validate_layout(deep)
    assert caught.value.reason == "not_layout"
