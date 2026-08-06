"""Rejection arms of ``_hierarchical_loading._validate_expansion_definition``.

Every case here writes a document whose ``_expansion`` block breaks exactly one
rule and asserts that loading it raises. The base documents come from the shared
``*_doc()`` builders in ``tests/services/channel_finder/conftest.py`` -- each
call returns a fresh dict, so replacing one ``_expansion`` cannot leak into
another case, and the surrounding document stays valid so the failure is
attributable to the mutation and nothing else.

Which container gets validated is not free choice: ``_find_level_container``
walks the *first* tree branch it finds at each tree level, so for
:func:`range_instances_doc` the validated container is ``tree['SR']['BPM']
['DEVICE']`` and for :func:`list_instances_doc` it is ``tree['RF']['UNIT']``.
Those are the two nodes the helpers below mutate.

The validator's messages are long and multi-line; each assertion matches a short
distinguishing substring rather than the whole text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from tests.services.channel_finder.conftest import (
    list_instances_doc,
    range_instances_doc,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _range_doc_with(expansion: dict[str, Any]) -> dict:
    """A valid range document with the validated ``_expansion`` replaced."""
    doc = range_instances_doc()
    doc["tree"]["SR"]["BPM"]["DEVICE"]["_expansion"] = expansion
    return doc


def _list_doc_with(expansion: dict[str, Any]) -> dict:
    """A valid list document with the validated ``_expansion`` replaced."""
    doc = list_instances_doc()
    doc["tree"]["RF"]["UNIT"]["_expansion"] = expansion
    return doc


# Each case: (document builder, level named in the message, message substring).
# One case per rejection arm, in source order.
EXPANSION_REJECTIONS = [
    pytest.param(
        lambda: _range_doc_with({"_pattern": "B{:02d}", "_range": [1, 3]}),
        "device",
        "missing '_type' field",
        id="missing-type",
    ),
    pytest.param(
        lambda: _range_doc_with({"_type": "range", "_range": [1, 3]}),
        "device",
        "requires '_pattern' field",
        id="range-missing-pattern",
    ),
    pytest.param(
        lambda: _range_doc_with({"_type": "range", "_pattern": "B{:02d}"}),
        "device",
        "requires '_range' field",
        id="range-missing-range",
    ),
    pytest.param(
        lambda: _range_doc_with({"_type": "range", "_pattern": "B{:02d}", "_range": [1, 2, 3]}),
        "device",
        "'_range' must be",
        id="range-not-two-elements",
    ),
    pytest.param(
        lambda: _range_doc_with({"_type": "range", "_pattern": "B{:02d}", "_range": ["1", 3]}),
        "device",
        "start and end must be integers",
        id="range-non-integer-bounds",
    ),
    pytest.param(
        lambda: _range_doc_with({"_type": "range", "_pattern": "B{:02d}", "_range": [5, 1]}),
        "device",
        "start must be <= end",
        id="range-start-after-end",
    ),
    pytest.param(
        lambda: _list_doc_with({"_type": "list"}),
        "unit",
        "requires '_instances' field",
        id="list-missing-instances",
    ),
    pytest.param(
        lambda: _list_doc_with({"_type": "list", "_instances": "MAIN"}),
        "unit",
        "'_instances' must be a list",
        id="list-instances-not-a-list",
    ),
    pytest.param(
        lambda: _list_doc_with({"_type": "list", "_instances": []}),
        "unit",
        "'_instances' cannot be empty",
        id="list-instances-empty",
    ),
    pytest.param(
        lambda: _range_doc_with({"_type": "sequence", "_range": [1, 3]}),
        "device",
        "invalid '_type'",
        id="unknown-type",
    ),
]


@pytest.mark.parametrize(("make_doc", "level_name", "match"), EXPANSION_REJECTIONS)
def test_invalid_expansion_is_rejected(
    hier_db_factory: Callable[..., Any],
    make_doc: Callable[[], dict],
    level_name: str,
    match: str,
) -> None:
    """Each malformed ``_expansion`` raises ValueError naming its level."""
    with pytest.raises(ValueError, match=match) as excinfo:
        hier_db_factory(make_doc())

    assert f"'{level_name}'" in str(excinfo.value)


@pytest.mark.parametrize("make_doc", [range_instances_doc, list_instances_doc])
def test_base_document_loads_unmutated(
    hier_db_factory: Callable[..., Any],
    make_doc: Callable[[], dict],
) -> None:
    """Control case: the documents the rejections mutate are themselves valid.

    Without this, a suite whose cases all tripped some earlier guard would still
    look green.
    """
    db = hier_db_factory(make_doc())

    assert db.channel_map
