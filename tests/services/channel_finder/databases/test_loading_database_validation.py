"""Document-level rejection arms of ``_hierarchical_loading.load_database``.

These are the malformed-*document* failures -- a missing ``hierarchy`` section, a
naming pattern out of sync with the declared levels, a bad level ``type``, an
instance level whose container is missing or misplaced. The malformed
``_expansion`` arms live next door in ``test_loading_expansion_validation.py``.

Each case starts from a shared ``*_doc()`` builder in
``tests/services/channel_finder/conftest.py`` (or, where no shared shape fits, a
minimal document built here) and breaks exactly one rule, so the failure is
attributable to the mutation and to nothing that came before it. Validation runs
in a fixed order -- ``_validate_naming_pattern`` before
``_validate_hierarchy_config`` before ``_validate_tree_structure`` -- and every
document below is valid up to the arm it is meant to reach.

One ordering trap is worth stating outright: ``load_database`` reads ``tree``
*before* it checks for ``hierarchy``, so a document missing both raises
``KeyError`` from the ``tree`` lookup and never reaches the ``hierarchy`` guard.
The missing-``hierarchy`` case below therefore keeps its ``tree``.

The messages are long and multi-line; each assertion matches a short
distinguishing substring rather than the whole text.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pytest
from tests.services.channel_finder.conftest import (
    optional_instance_level_doc,
    optional_tree_level_doc,
    tree_only_doc,
)

if TYPE_CHECKING:
    from collections.abc import Callable

LOADING_LOGGER = "osprey.services.channel_finder.databases._hierarchical_loading"


def _consecutive_instances_doc(nested: bool) -> dict:
    """Two adjacent ``instances`` levels, declared nested or as tree siblings.

    ``system(tree) -> cell(instances) -> device(instances)``. With ``nested``
    the ``DEVICE`` container sits inside ``CELL``, which is the layout
    ``_validate_instance_level`` demands; without it the two containers are
    siblings under ``SR``, which is the rejected layout. No shared builder has
    two consecutive instance levels, so this shape is built here.
    """
    device = {"_expansion": {"_type": "list", "_instances": ["A", "B"]}, "Value": {}}
    cell = {"_expansion": {"_type": "range", "_pattern": "C{:02d}", "_range": [1, 2]}}

    if nested:
        cell["DEVICE"] = device
        system: dict[str, Any] = {"CELL": cell}
    else:
        system = {"CELL": cell, "DEVICE": device}

    return {
        "hierarchy": {
            "levels": [
                {"name": "system", "type": "tree"},
                {"name": "cell", "type": "instances"},
                {"name": "device", "type": "instances"},
            ],
            "naming_pattern": "{system}:{cell}:{device}",
        },
        "tree": {"SR": {"_description": "Storage Ring", **system}},
    }


def _no_hierarchy_doc() -> dict:
    """A document with a valid ``tree`` and no ``hierarchy`` section at all."""
    return {"tree": tree_only_doc()["tree"]}


def _no_naming_pattern_doc() -> dict:
    """A ``hierarchy`` section with its ``levels``, but no ``naming_pattern``."""
    doc = tree_only_doc()
    del doc["hierarchy"]["naming_pattern"]
    return doc


def _no_levels_doc() -> dict:
    """A ``hierarchy`` section with its ``naming_pattern``, but no ``levels``."""
    doc = tree_only_doc()
    del doc["hierarchy"]["levels"]
    return doc


def _undefined_pattern_level_doc() -> dict:
    """Valid levels, but the pattern names a level that was never declared."""
    doc = tree_only_doc()
    doc["hierarchy"]["naming_pattern"] = "{system}:{device}:{signal}:{bogus}"
    return doc


def _optional_level_missing_placeholder_doc() -> dict:
    """An ``optional: true`` level dropped from the naming pattern."""
    doc = optional_tree_level_doc()
    doc["hierarchy"]["naming_pattern"] = "{system}:{device}:{signal}"
    return doc


def _invalid_level_type_doc() -> dict:
    """A level whose ``type`` is neither ``"tree"`` nor ``"instances"``."""
    doc = tree_only_doc()
    doc["hierarchy"]["levels"][2]["type"] = "category"
    return doc


def _required_instance_without_container_doc() -> dict:
    """A *required* instance level whose first tree branch has no container.

    ``optional_instance_level_doc`` lists ``BR`` -- which has no ``DEVICE``
    container -- before ``SR``, and ``_find_level_container`` only ever walks
    the first branch it finds. As shipped the level is optional, so the missing
    container is tolerated; dropping ``optional`` turns the same document into
    the required-level rejection.
    """
    doc = optional_instance_level_doc()
    del doc["hierarchy"]["levels"][1]["optional"]
    return doc


# Each case: (document builder, message substring). One case per rejection arm,
# in the order load_database evaluates them.
DOCUMENT_REJECTIONS = [
    pytest.param(
        _no_hierarchy_doc,
        "must contain 'hierarchy' section",
        id="missing-hierarchy-section",
    ),
    pytest.param(
        _no_naming_pattern_doc,
        "has no 'naming_pattern'",
        id="missing-naming-pattern",
    ),
    pytest.param(
        _no_levels_doc,
        "has no 'levels'",
        id="missing-levels",
    ),
    pytest.param(
        _undefined_pattern_level_doc,
        "references undefined hierarchy levels",
        id="pattern-references-undefined-level",
    ),
    pytest.param(
        _optional_level_missing_placeholder_doc,
        "must appear in naming_pattern",
        id="optional-level-missing-placeholder",
    ),
    pytest.param(
        _invalid_level_type_doc,
        "has invalid type",
        id="invalid-level-type",
    ),
    pytest.param(
        _required_instance_without_container_doc,
        "requires container named",
        id="required-instance-level-without-container",
    ),
    pytest.param(
        lambda: _consecutive_instances_doc(nested=False),
        "Consecutive instance levels",
        id="consecutive-instance-levels-as-siblings",
    ),
]


@pytest.mark.parametrize(("make_doc", "match"), DOCUMENT_REJECTIONS)
def test_malformed_document_is_rejected(
    hier_db_factory: Callable[..., Any],
    make_doc: Callable[[], dict],
    match: str,
) -> None:
    """Each malformed document raises ValueError from the constructor."""
    with pytest.raises(ValueError, match=match):
        hier_db_factory(make_doc())


def test_missing_hierarchy_raises_value_error_not_key_error(
    hier_db_factory: Callable[..., Any],
) -> None:
    """The missing-``hierarchy`` guard fires, rather than a ``tree`` KeyError.

    ``load_database`` reads ``data["tree"]`` first, so a document missing both
    keys dies with ``KeyError('tree')`` before the guard runs -- that failure
    would still look like a rejection to a bare ``pytest.raises(Exception)``
    while leaving the guard uncovered. Pinning the type here is what keeps the
    case honest.
    """
    doc = _no_hierarchy_doc()
    assert "tree" in doc
    assert "hierarchy" not in doc

    with pytest.raises(ValueError) as excinfo:
        hier_db_factory(doc)

    assert not isinstance(excinfo.value, KeyError)
    assert "hierarchy" in str(excinfo.value)


def test_level_absent_from_pattern_is_navigation_only(
    hier_db_factory: Callable[..., Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-optional level left out of the pattern loads and logs a note.

    Pattern placeholders must be a *subset* of the declared levels, not equal to
    them: a level with no placeholder is usable for navigation and simply
    contributes nothing to channel names. This is the accepting counterpart of
    the ``pattern-references-undefined-level`` rejection above.
    """
    doc = tree_only_doc()
    doc["hierarchy"]["naming_pattern"] = "{system}:{device}"

    with caplog.at_level(logging.INFO, logger=LOADING_LOGGER):
        db = hier_db_factory(doc)

    assert "navigation only" in caplog.text
    assert "'signal'" in caplog.text
    assert "signal" in db.hierarchy_levels
    assert db.channel_map


def test_optional_instance_level_without_container_loads(
    hier_db_factory: Callable[..., Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The accepting counterpart of the required-instance-level rejection.

    ``optional_instance_level_doc`` is the *same* document as
    :func:`_required_instance_without_container_doc` with ``optional`` still in
    place: ``_find_level_container`` returns None for the first branch either
    way, and only the ``optional`` flag decides whether that is tolerated with a
    log line or raised on. Loading it unmutated is what shows the rejection
    above is attributable to dropping the flag and not to the missing container
    on its own.
    """
    with caplog.at_level(logging.INFO, logger=LOADING_LOGGER):
        db = hier_db_factory(optional_instance_level_doc())

    assert "no container in some branches" in caplog.text
    assert "'device'" in caplog.text
    assert db.channel_map


def test_nested_consecutive_instance_levels_load(
    hier_db_factory: Callable[..., Any],
) -> None:
    """Control case for the consecutive-instances rejection.

    The same two instance levels load once ``DEVICE`` is nested inside ``CELL``,
    which is what shows the rejection above comes from the sibling layout and
    not from anything else in that document.
    """
    db = hier_db_factory(_consecutive_instances_doc(nested=True))

    assert db.channel_map
