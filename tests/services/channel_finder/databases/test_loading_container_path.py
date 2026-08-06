"""The container path ``_hierarchical_loading`` reports for a missing ``_expansion``.

An instance level whose container exists but carries no ``_expansion`` is the
only caller of ``_get_container_path``: ``_validate_instance_level`` builds the
path purely to name the offending node in its error message. The path is
assembled from the *declared* hierarchy rather than from the tree walk, so each
preceding level contributes a different token -- ``[CATEGORY]`` for a ``tree``
level, ``['NAME']`` for an ``instances`` level -- and a level at index 0
contributes nothing at all. Those arms are mutually exclusive, so one document
cannot reach them all; the three below split them.

Which container gets validated is not free choice: ``_find_level_container``
walks only the *first* non-underscore dict child at each tree level. For
:func:`range_instances_doc` that makes ``tree['SR']['BPM']['DEVICE']`` the
validated node -- the ``QUAD`` branch's ``DEVICE`` is never reached, so mutating
it would leave the document loading cleanly and prove nothing.

Every case asserts the exact path string, both where the message reports the
container and where it reports the missing key. A test that only asserted "some
ValueError was raised" would pass against a path builder that emitted the wrong
tokens, which is the failure this file exists to catch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from tests.services.channel_finder.conftest import _document, range_instances_doc

if TYPE_CHECKING:
    from collections.abc import Callable

_RANGE_EXPANSION = {"_type": "range", "_pattern": "D{}", "_range": [1, 2]}


def tree_preceded_doc(*, with_expansion: bool) -> dict:
    """Instance level at index 2, preceded by two ``tree`` levels.

    The shared range document already has that shape; only the ``_expansion`` on
    the validated container is at issue, so it is dropped rather than rebuilt.
    Two preceding tree levels mean two ``[CATEGORY]`` tokens.
    """
    doc = range_instances_doc()
    if not with_expansion:
        del doc["tree"]["SR"]["BPM"]["DEVICE"]["_expansion"]
    return doc


def instances_preceded_doc(*, with_expansion: bool) -> dict:
    """Instance level at index 2, preceded by a ``tree`` then an ``instances`` level.

    Consecutive instance levels must be nested, so ``DEVICE`` lives inside
    ``UNIT``; the outer ``UNIT`` keeps its expansion so validation reaches the
    inner container instead of failing earlier. The preceding instances level
    contributes its own name, not ``[CATEGORY]``.
    """
    device: dict[str, Any] = {"Power": {"_description": "Forward power"}}
    if with_expansion:
        device["_expansion"] = dict(_RANGE_EXPANSION)
    return _document(
        levels=[
            {"name": "system", "type": "tree"},
            {"name": "unit", "type": "instances"},
            {"name": "device", "type": "instances"},
            {"name": "signal", "type": "tree"},
        ],
        naming_pattern="{system}:{unit}:{device}:{signal}",
        tree={
            "RF": {
                "_description": "RF system",
                "UNIT": {
                    "_expansion": {"_type": "list", "_instances": ["MAIN", "BACKUP"]},
                    "DEVICE": device,
                },
            },
        },
    )


def root_instances_doc(*, with_expansion: bool) -> dict:
    """Instance level at index 0, with nothing in front of it.

    No preceding level means the loop body never runs, so the path is the root
    token and the container name alone.
    """
    device: dict[str, Any] = {"X": {"_description": "Horizontal position"}}
    if with_expansion:
        device["_expansion"] = dict(_RANGE_EXPANSION)
    return _document(
        levels=[
            {"name": "device", "type": "instances"},
            {"name": "signal", "type": "tree"},
        ],
        naming_pattern="{device}:{signal}",
        tree={"DEVICE": device},
    )


# Each case: (document builder, level named in the message, expected path).
CONTAINER_PATHS = [
    pytest.param(
        tree_preceded_doc,
        "device",
        "tree[CATEGORY][CATEGORY]['DEVICE']",
        id="preceded-by-tree-levels",
    ),
    pytest.param(
        instances_preceded_doc,
        "device",
        "tree[CATEGORY]['UNIT']['DEVICE']",
        id="preceded-by-instances-level",
    ),
    pytest.param(
        root_instances_doc,
        "device",
        "tree['DEVICE']",
        id="first-level",
    ),
]


@pytest.mark.parametrize(("make_doc", "level_name", "expected_path"), CONTAINER_PATHS)
def test_missing_expansion_reports_container_path(
    hier_db_factory: Callable[..., Any],
    make_doc: Callable[..., dict],
    level_name: str,
    expected_path: str,
) -> None:
    """A container without ``_expansion`` is rejected, named by its declared path."""
    with pytest.raises(ValueError, match="missing '_expansion' definition") as excinfo:
        hier_db_factory(make_doc(with_expansion=False))

    message = str(excinfo.value)
    assert f"Instance level '{level_name}'" in message
    assert f"Found container at: {expected_path}\n" in message
    assert f"Missing: {expected_path}['_expansion']\n" in message


@pytest.mark.parametrize(
    ("make_doc", "expected_channels"),
    [
        pytest.param(
            tree_preceded_doc,
            {"SR:BPM:B01:X", "SR:QUAD:Q1:Setpoint"},
            id="preceded-by-tree-levels",
        ),
        pytest.param(
            instances_preceded_doc,
            {"RF:MAIN:D1:Power", "RF:BACKUP:D2:Power"},
            id="preceded-by-instances-level",
        ),
        pytest.param(root_instances_doc, {"D1:X", "D2:X"}, id="first-level"),
    ],
)
def test_same_document_loads_once_expansion_is_present(
    hier_db_factory: Callable[..., Any],
    make_doc: Callable[..., dict],
    expected_channels: set[str],
) -> None:
    """Control case: the dropped ``_expansion`` is the only thing wrong.

    Without this, a case whose document tripped some *earlier* guard -- a
    misdeclared level, an unreachable container -- would still raise and still
    look green while never touching the path builder.
    """
    db = hier_db_factory(make_doc(with_expansion=True))

    assert expected_channels <= set(db.channel_map)
