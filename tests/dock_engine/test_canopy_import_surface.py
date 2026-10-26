"""CANOPY's import surface on ``dock_engine``, called the way CANOPY calls it.

Every right-hand name of the C-PY rename table (``check_layout`` → ``layout_refusal`` +
``layout_keys``, ``find_old_shape`` → ``old_shape_refusal``) and every name CANOPY's
``collections.py`` and ``workspace.py`` import unchanged (``SPLITS``, ``LAYOUT_REASONS``,
``LayoutForm``, ``LayoutRefused``, ``normalize_layout``) is imported and called positionally.
"""

from __future__ import annotations

import typing

import pytest

from dock_engine import (
    LAYOUT_REASONS,
    SPLITS,
    LayoutForm,
    LayoutRefused,
    layout_keys,
    layout_refusal,
    normalize_layout,
    old_shape_refusal,
)


def _tabs(keys: list[typing.Any]) -> dict[str, typing.Any]:
    return {"tabs": keys, "showing": keys[0]}


def _dock() -> dict[str, typing.Any]:
    return {
        "split": "row",
        "items": [{"tile": "t1", **_tabs(["a"])}, {"tile": "t2", **_tabs(["b", "c"])}],
    }


def _collection() -> dict[str, typing.Any]:
    return {"split": "column", "items": [_tabs([0]), _tabs([1])]}


def test_unchanged_names_import() -> None:
    """SPLITS, LAYOUT_REASONS and LayoutForm keep the shapes CANOPY relies on."""
    assert set(SPLITS) == {"row", "column"}
    assert "row" in SPLITS
    assert "member_missing" in LAYOUT_REASONS
    assert set(typing.get_args(LayoutForm)) == {"dock", "collection"}


def test_layout_refusal_positional_with_members() -> None:
    """layout_refusal(layout, form, members) replaces check_layout(layout, form, members)."""
    assert layout_refusal(_collection(), "collection", 2) is None
    refused = layout_refusal(_collection(), "collection", 1)
    assert refused is not None
    reason, where = refused
    assert reason in LAYOUT_REASONS
    assert isinstance(where, str)


def test_placed_keys_check() -> None:
    """layout_refusal(layout) + layout_keys(layout) replace `placed = check_layout(layout)`."""
    layout = _dock()
    assert layout_refusal(layout) is None
    assert set(layout_keys(layout)) == {"a", "b", "c"}
    assert layout_keys(_collection()) == ["0", "1"]
    assert layout_keys(None) == []


def test_refusal_carrier() -> None:
    """A refusal raises as LayoutRefused carrying .reason and .where, caught as ValueError."""
    refused = layout_refusal(42)
    assert refused is not None
    error = LayoutRefused(*refused)
    assert (error.reason, error.where) == refused
    with pytest.raises(ValueError):
        raise error


def test_old_shape_refusal_positional() -> None:
    """old_shape_refusal(value, "layout") replaces find_old_shape(value, "layout")."""
    assert old_shape_refusal(_dock(), "layout") is None
    old = {"split": "row", "items": [{"slot": 0}, {"tabs": [1]}]}
    assert old_shape_refusal(old, "layout") == ("old_slot_shape", "layout.items[0]")


def test_normalize_layout_positional() -> None:
    """normalize_layout(layout, form) returns a fresh, valid tree for both forms."""
    for layout, form in ((_dock(), "dock"), (_collection(), "collection")):
        before = repr(layout)
        normalized = normalize_layout(layout, form)
        assert normalized is not layout
        assert repr(layout) == before
        members = 2 if form == "collection" else None
        assert layout_refusal(normalized, form, members) is None
