"""Shared hierarchical-database fixtures for the channel-finder tests.

Every database here is written to disk as real JSON under ``tmp_path`` and then
loaded. There is no in-memory constructor: ``BaseDatabase.__init__`` calls
``load_database()`` eagerly, so a :class:`HierarchicalChannelDatabase` only
exists once its document exists as a file.

The family is organised in two layers:

* ``*_doc()`` -- module-level builders returning the raw JSON document as a
  plain dict. Each call builds a fresh dict, so a test may mutate the result to
  produce a near-miss variant (a broken ``_expansion``, a container removed, an
  extra level) without leaking into any other test.
* ``*_db`` -- fixtures that write the matching document and return the loaded
  database.

Every builder docstring states the shape it produces *and* the production branch
it exists to reach, so a consumer can pick a fixture without re-reading
``osprey.services.channel_finder.databases``. Module references below are
relative to that package.

Nothing here touches the network, a container, or the repository tree; the only
filesystem writes go to the per-test ``tmp_path``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from osprey.services.channel_finder.databases.hierarchical import (
    HierarchicalChannelDatabase,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


# ---------------------------------------------------------------------------
# Document builders
#
# A hierarchical document is ``{"hierarchy": {"levels": [...],
# "naming_pattern": "..."}, "tree": {...}}``. ``levels`` entries carry
# ``name``/``type`` (``"tree"`` or ``"instances"``) and an optional
# ``optional: true``. Optional levels MUST appear in ``naming_pattern`` --
# ``_hierarchical_loading._validate_naming_pattern`` rejects the document
# otherwise.
# ---------------------------------------------------------------------------


def _document(levels: list[dict[str, Any]], naming_pattern: str, tree: dict) -> dict:
    """Assemble a hierarchical document from its three parts."""
    return {
        "hierarchy": {"levels": levels, "naming_pattern": naming_pattern},
        "tree": tree,
    }


def tree_only_doc() -> dict:
    """Three tree levels, no instance expansion anywhere.

    ``system:device:signal`` over ``SR``/``BR``. The baseline shape: exercises
    plain tree recursion in ``_hierarchical_naming._build_channel_map`` and the
    tree branch of ``_hierarchical_query._navigate_to_node`` /
    ``_extract_tree_options`` with nothing else in play. ``BR`` has no children,
    so it is an automatic leaf (no ``_is_leaf`` marker needed) -- use it to
    reach the "node has no children" early return in ``_is_leaf_node``.

    Channels: ``SR:BPM:X``, ``SR:BPM:Y``, ``SR:QUAD:Setpoint``, ``BR``.
    """
    return _document(
        levels=[
            {"name": "system", "type": "tree"},
            {"name": "device", "type": "tree"},
            {"name": "signal", "type": "tree"},
        ],
        naming_pattern="{system}:{device}:{signal}",
        tree={
            "SR": {
                "_description": "Storage Ring",
                "BPM": {
                    "_description": "Beam Position Monitor",
                    "X": {"_description": "Horizontal position"},
                    "Y": {"_description": "Vertical position"},
                },
                "QUAD": {"_description": "Quadrupole", "Setpoint": {}},
            },
            "BR": {"_description": "Booster Ring"},
        },
    )


def range_instances_doc() -> dict:
    """An ``instances`` level expanded by ``_type: "range"``.

    ``system(tree) -> device(instances) -> signal(tree)``, with the required
    ``DEVICE`` container carrying ``{"_type": "range", "_pattern": "B{:02d}",
    "_range": [1, 3]}`` under ``BPM`` and a second, differently-patterned range
    under ``QUAD``. Reaches the range arm of ``_hierarchical_base
    ._get_instance_names`` and ``_hierarchical_query._expand_instances``, the
    instances arm of ``_build_channel_map``, and the full happy path of
    ``_hierarchical_loading._validate_expansion_definition`` for ranges
    (``_pattern`` and ``_range`` both present and well formed).

    Channels: ``SR:BPM:B01:X`` ... ``SR:BPM:B03:Y``, ``SR:QUAD:Q1:Setpoint`` ...
    """
    return _document(
        levels=[
            {"name": "system", "type": "tree"},
            {"name": "family", "type": "tree"},
            {"name": "device", "type": "instances"},
            {"name": "signal", "type": "tree"},
        ],
        naming_pattern="{system}:{family}:{device}:{signal}",
        tree={
            "SR": {
                "_description": "Storage Ring",
                "BPM": {
                    "_description": "Beam Position Monitors",
                    "DEVICE": {
                        "_expansion": {
                            "_type": "range",
                            "_pattern": "B{:02d}",
                            "_range": [1, 3],
                        },
                        "X": {"_description": "Horizontal position"},
                        "Y": {"_description": "Vertical position"},
                    },
                },
                "QUAD": {
                    "_description": "Quadrupoles",
                    "DEVICE": {
                        "_expansion": {
                            "_type": "range",
                            "_pattern": "Q{}",
                            "_range": [1, 2],
                        },
                        "Setpoint": {},
                        "Readback": {},
                    },
                },
            },
        },
    )


def list_instances_doc() -> dict:
    """An ``instances`` level expanded by ``_type: "list"``.

    ``system(tree) -> unit(instances) -> signal(tree)``, with the required
    ``UNIT`` container carrying ``{"_type": "list", "_instances": ["MAIN",
    "BACKUP", "TEST"]}``. The list arm is the counterpart of
    :func:`range_instances_doc`: it reaches the ``list`` branches of
    ``_get_instance_names``, ``_expand_instances`` and
    ``_validate_expansion_definition`` (which checks ``_instances`` is present,
    is a list, and is non-empty).

    Channels: ``RF:MAIN:Power``, ``RF:MAIN:Phase``, ``RF:BACKUP:Power``, ...
    """
    return _document(
        levels=[
            {"name": "system", "type": "tree"},
            {"name": "unit", "type": "instances"},
            {"name": "signal", "type": "tree"},
        ],
        naming_pattern="{system}:{unit}:{signal}",
        tree={
            "RF": {
                "_description": "RF system",
                "UNIT": {
                    "_expansion": {
                        "_type": "list",
                        "_instances": ["MAIN", "BACKUP", "TEST"],
                    },
                    "Power": {"_description": "Forward power"},
                    "Phase": {"_description": "Cavity phase"},
                },
            },
        },
    )


def optional_tree_level_doc() -> dict:
    """A tree level declared ``optional: true`` that some branches skip.

    ``system -> device -> subdevice(tree, optional) -> signal``. Under ``PS``,
    ``PSU`` is a real subdevice container while ``Heartbeat`` carries
    ``_is_leaf: true`` and no children, so ``_build_channel_map`` takes the
    "OPTIONAL LEVEL SKIP" branch and re-assigns it to ``signal``. The resulting
    empty ``subdevice`` component leaves a ``::`` artifact that
    ``_hierarchical_naming._clean_optional_separators`` collapses.

    Channels: ``SR:PS:PSU:Voltage``, ``SR:PS:PSU:Current``, ``SR:PS:Heartbeat``.
    """
    return _document(
        levels=[
            {"name": "system", "type": "tree"},
            {"name": "device", "type": "tree"},
            {"name": "subdevice", "type": "tree", "optional": True},
            {"name": "signal", "type": "tree"},
        ],
        naming_pattern="{system}:{device}:{subdevice}:{signal}",
        tree={
            "SR": {
                "_description": "Storage Ring",
                "PS": {
                    "_description": "Power supplies",
                    "PSU": {
                        "_description": "Supply unit",
                        "Voltage": {},
                        "Current": {},
                    },
                    "Heartbeat": {"_is_leaf": True, "_description": "Liveness"},
                },
            },
        },
    )


def optional_instance_level_doc() -> dict:
    """An ``optional`` instance level whose *first* branch has no container.

    ``system(tree) -> device(instances, optional) -> signal(tree)``. Validation
    walks only the first tree branch it finds, so listing ``BR`` (no ``DEVICE``
    container) before ``SR`` (which has one) is what reaches the
    "optional instance level has no container in some branches" early return in
    ``_hierarchical_loading._validate_instance_level``. A *required* instance
    level in the same position raises instead -- that error path needs a
    mutated copy of this document with ``optional`` dropped.

    ``BR`` also reaches the ``expansion_def is None`` fall-through in the
    instances arm of ``_build_channel_map``: the branch simply yields no
    channels.

    Channels: ``SR:D1:X``, ``SR:D2:X`` (nothing from ``BR``).
    """
    return _document(
        levels=[
            {"name": "system", "type": "tree"},
            {"name": "device", "type": "instances", "optional": True},
            {"name": "signal", "type": "tree"},
        ],
        naming_pattern="{system}:{device}:{signal}",
        tree={
            "BR": {"_description": "Booster - no DEVICE container", "Status": {}},
            "SR": {
                "_description": "Storage Ring",
                "DEVICE": {
                    "_expansion": {"_type": "range", "_pattern": "D{}", "_range": [1, 2]},
                    "X": {"_description": "Horizontal position"},
                },
            },
        },
    )


def hybrid_expansion_doc() -> dict:
    """``_expansion`` on a *tree* child rather than on an instance container.

    All three levels are ``tree``-typed, but the ``CH`` node under ``SR`` owns
    an ``_expansion``. That is the hybrid pattern: ``_build_channel_map``
    expands ``CH`` into ``CH-1``..``CH-3`` and assigns each instance to the
    surrounding *tree* level, and ``_hierarchical_query._extract_tree_options``
    returns the expanded instances instead of the container name ``CH``.
    ``_navigate_to_node`` and ``_collect_separator_overrides`` reach their
    "selection is an expanded instance, find the container that generates it"
    fallbacks with a selection such as ``device="CH-2"``.

    No level is ``instances``-typed, so ``_validate_instance_level`` never runs
    against this document.

    Channels: ``SR:CH-1:Voltage`` ... ``SR:CH-3:Current``, ``SR:PS:Status``.
    """
    return _document(
        levels=[
            {"name": "system", "type": "tree"},
            {"name": "device", "type": "tree"},
            {"name": "signal", "type": "tree"},
        ],
        naming_pattern="{system}:{device}:{signal}",
        tree={
            "SR": {
                "_description": "Storage Ring",
                "CH": {
                    "_description": "Corrector channels",
                    "_expansion": {"_type": "range", "_pattern": "CH-{}", "_range": [1, 3]},
                    "Voltage": {},
                    "Current": {},
                },
                "PS": {"_description": "Power supply", "Status": {}},
            },
        },
    )


def scalar_leaves_doc() -> dict:
    """Nodes whose children are scalars rather than dicts.

    ``BPM`` has only scalar children (``X``/``Y`` are ints) and ``QUAD`` mixes a
    scalar (``units``) into a dict child. Scalars are invisible to every
    ``isinstance(value, dict)`` guard, so a node with only scalar children is an
    automatic leaf in ``_hierarchical_naming._is_leaf_node`` and contributes no
    recursion in ``_build_channel_map`` -- but
    ``_hierarchical_base._count_channels`` *does* count them (its ``else``
    branch), which is the arithmetic the tree preview reports.

    Channels: ``SR:BPM``, ``SR:QUAD:Setpoint``.
    """
    return _document(
        levels=[
            {"name": "system", "type": "tree"},
            {"name": "device", "type": "tree"},
            {"name": "signal", "type": "tree"},
        ],
        naming_pattern="{system}:{device}:{signal}",
        tree={
            "SR": {
                "_description": "Storage Ring",
                "BPM": {"X": 1, "Y": 2},
                "QUAD": {"Setpoint": {"units": "A"}},
            },
        },
    )


def separator_override_doc() -> dict:
    """``_separator`` overrides at the root child *and* at depth > 0.

    ``SR`` carries ``"_separator": "-"``, which ``_build_channel_map`` binds to
    the ``(system, device)`` pair; ``BPM``, one level deeper, carries
    ``"_separator": "_"``, bound to ``(device, signal)``. Both are collected
    again -- by a different code path -- in
    ``_hierarchical_naming._collect_separator_overrides`` when
    ``build_channels_from_selections`` is called with matching selections.
    ``QUAD`` deliberately carries no override, which is what proves the
    per-subtree copy of the override dict does not leak into siblings.

    Channels: ``SR-BPM_X:RB``, ``SR-BPM_X:SP``, ``SR-QUAD:Setpoint:RB``.
    """
    return _document(
        levels=[
            {"name": "system", "type": "tree"},
            {"name": "device", "type": "tree"},
            {"name": "signal", "type": "tree"},
            {"name": "suffix", "type": "tree"},
        ],
        naming_pattern="{system}:{device}:{signal}:{suffix}",
        tree={
            "SR": {
                "_description": "Storage Ring",
                "_separator": "-",
                "BPM": {
                    "_description": "Beam Position Monitor",
                    "_separator": "_",
                    "X": {"RB": {}, "SP": {}},
                },
                "QUAD": {"_description": "Quadrupole", "Setpoint": {"RB": {}}},
            },
        },
    )


def channel_part_doc() -> dict:
    """``_channel_part`` overrides, including the navigation-only empty string.

    ``Magnets`` renames itself to ``MAG`` and its child ``Power Supplies`` to
    ``PS``; ``Building-1`` sets ``"_channel_part": ""``, which
    ``_hierarchical_naming._get_channel_part`` returns verbatim so the level
    contributes nothing to the name. The resulting empty first component leaves
    a leading ``:`` that ``_clean_optional_separators`` strips, and the
    "value is falsy, skip it" arm of ``_build_channel_with_separators`` is what
    drops the component in the first place.

    Note the tree keys stay human-readable: navigation still uses ``Magnets``
    and ``Building-1``, only the emitted names change. One consequence to expect
    before asserting on it: ``_hierarchical_query.get_statistics`` buckets
    channels by comparing the stored path against the raw *tree key*, so both
    systems report ``channel_count: 0`` here even though two channels exist.

    Channels: ``MAG:PS:Current``, ``RACK:Temp``.
    """
    return _document(
        levels=[
            {"name": "system", "type": "tree"},
            {"name": "device", "type": "tree"},
            {"name": "signal", "type": "tree"},
        ],
        naming_pattern="{system}:{device}:{signal}",
        tree={
            "Magnets": {
                "_channel_part": "MAG",
                "_description": "Magnet systems",
                "Power Supplies": {"_channel_part": "PS", "Current": {}},
            },
            "Building-1": {
                "_channel_part": "",
                "_description": "Navigation-only grouping",
                "RACK": {"Temp": {}},
            },
        },
    )


def leaf_with_children_doc() -> dict:
    """``_is_leaf: true`` on a node that *also* has dict children (required level).

    ``SIGNAL-Y`` sits at the required ``signal`` level and is both a complete
    channel and the parent of the ``RB``/``SP`` suffix variants. This is the
    only shape that reaches the "process children of leaf nodes" block in
    ``_build_channel_map``: the node emits its own channel, then re-enters
    recursion with its children assigned to the next *tree* level.

    Contrast with :func:`optional_leaf_with_children_doc`, where the identical
    node under an optional level takes a different branch and its children are
    dropped.

    Channels: ``SR:PS:SIGNAL-Y``, ``SR:PS:SIGNAL-Y:RB``, ``SR:PS:SIGNAL-Y:SP``.
    """
    return _document(
        levels=[
            {"name": "system", "type": "tree"},
            {"name": "device", "type": "tree"},
            {"name": "signal", "type": "tree"},
            {"name": "suffix", "type": "tree"},
        ],
        naming_pattern="{system}:{device}:{signal}:{suffix}",
        tree={
            "SR": {
                "_description": "Storage Ring",
                "PS": {
                    "_description": "Power supply",
                    "SIGNAL-Y": {
                        "_is_leaf": True,
                        "_description": "Complete channel with suffix variants",
                        "RB": {"_description": "Readback"},
                        "SP": {"_description": "Setpoint"},
                    },
                },
            },
        },
    )


def optional_leaf_with_children_doc() -> dict:
    """``_is_leaf: true`` plus dict children, this time under an *optional* level.

    Same ``SIGNAL-Y`` node as :func:`leaf_with_children_doc`, but the level it
    sits at (``subdevice``) is declared ``optional: true``. The optional-level
    check in ``_build_channel_map`` fires *first*, so the node skips its level
    and is assigned to ``suffix`` instead -- by which point every level has been
    consumed and its ``RB``/``SP`` children are never visited. Use this pair to
    characterise that asymmetry rather than assuming it.

    ``PSU`` is the ordinary sibling that does occupy the optional level.

    Channels: ``SR:PS:SIGNAL-Y``, ``SR:PS:PSU:Voltage`` (no ``RB``/``SP``).
    """
    return _document(
        levels=[
            {"name": "system", "type": "tree"},
            {"name": "device", "type": "tree"},
            {"name": "subdevice", "type": "tree", "optional": True},
            {"name": "suffix", "type": "tree"},
        ],
        naming_pattern="{system}:{device}:{subdevice}:{suffix}",
        tree={
            "SR": {
                "_description": "Storage Ring",
                "PS": {
                    "_description": "Power supply",
                    "SIGNAL-Y": {
                        "_is_leaf": True,
                        "_description": "Complete channel with suffix variants",
                        "RB": {"_description": "Readback"},
                        "SP": {"_description": "Setpoint"},
                    },
                    "PSU": {"_description": "Supply unit", "Voltage": {}},
                },
            },
        },
    )


def all_optional_next_instances_doc() -> dict:
    """Every remaining level optional, with the next level ``instances``-typed.

    ``system(tree) -> device(instances, optional) -> signal(tree, optional)``.
    At the ``SR`` node both remaining levels are optional, so ``_is_leaf_node``
    evaluates ``all_remaining_optional`` as True -- the only shape in this file
    that does -- and then takes its ``instances`` arm: ``SR`` has no ``DEVICE``
    container, so it *is* a leaf. ``BR`` has one, so the same arm returns False
    and the branch expands normally.

    ``SR`` is listed first for the same reason as in
    :func:`optional_instance_level_doc`: validation walks only the first branch.

    Channels: ``SR``, ``SR:X``, ``SR:Y``, ``BR:D1:Z``, ``BR:D2:Z``.
    """
    return _document(
        levels=[
            {"name": "system", "type": "tree"},
            {"name": "device", "type": "instances", "optional": True},
            {"name": "signal", "type": "tree", "optional": True},
        ],
        naming_pattern="{system}:{device}:{signal}",
        tree={
            "SR": {
                "_description": "No DEVICE container - all remaining levels optional",
                "X": {"_description": "Horizontal position"},
                "Y": {"_description": "Vertical position"},
            },
            "BR": {
                "_description": "Has a DEVICE container",
                "DEVICE": {
                    "_expansion": {"_type": "range", "_pattern": "D{}", "_range": [1, 2]},
                    "Z": {},
                },
            },
        },
    )


def all_optional_next_tree_doc() -> dict:
    """Every remaining level optional, with the next level ``tree``-typed.

    ``system(tree) -> subdevice(tree, optional) -> signal(tree, optional)``.
    The counterpart of :func:`all_optional_next_instances_doc` for the ``tree``
    arm of the ``all_remaining_optional`` block in ``_is_leaf_node``.

    Worth knowing before writing assertions: that arm can only ever return
    False. It re-tests ``has_tree_children`` with the same predicate the
    ``has_children`` guard above it already applied, and a node that failed that
    guard returned True several lines earlier. The branch is reachable -- this
    document reaches it -- but no document can make it return True.

    Channels: ``SR:PSU:Voltage``, ``SR:PSU:Current``.
    """
    return _document(
        levels=[
            {"name": "system", "type": "tree"},
            {"name": "subdevice", "type": "tree", "optional": True},
            {"name": "signal", "type": "tree", "optional": True},
        ],
        naming_pattern="{system}:{subdevice}:{signal}",
        tree={
            "SR": {
                "_description": "Storage Ring",
                "PSU": {"_description": "Supply unit", "Voltage": {}, "Current": {}},
            },
        },
    )


# ---------------------------------------------------------------------------
# Writer and loader
# ---------------------------------------------------------------------------


@pytest.fixture
def write_hier_json(tmp_path: Path) -> Callable[..., Path]:
    """Factory writing a hierarchical document to ``tmp_path`` as JSON.

    Returns:
        Callable ``(doc, name="hier.json") -> Path``. Pass distinct names when
        one test needs several databases side by side; the document is written
        verbatim and never validated, so this is also the seam for producing
        deliberately malformed documents that ``load_database`` must reject.
    """

    def _write(doc: dict, name: str = "hier.json") -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(doc, indent=2))
        return path

    return _write


@pytest.fixture
def hier_db_factory(
    write_hier_json: Callable[..., Path],
) -> Callable[..., HierarchicalChannelDatabase]:
    """Factory writing a document and returning the loaded database.

    Returns:
        Callable ``(doc, name="hier.json") -> HierarchicalChannelDatabase``.
        Loading happens inside :class:`BaseDatabase`'s constructor, so a
        document that fails validation raises from this call.
    """

    def _load(doc: dict, name: str = "hier.json") -> HierarchicalChannelDatabase:
        return HierarchicalChannelDatabase(str(write_hier_json(doc, name)))

    return _load


# ---------------------------------------------------------------------------
# Loaded databases, one per shape
#
# Each fixture is the loaded counterpart of the like-named ``*_doc()`` builder;
# see that builder's docstring for the shape and the branch it reaches.
# ---------------------------------------------------------------------------


@pytest.fixture
def tree_only_db(hier_db_factory) -> HierarchicalChannelDatabase:
    """Loaded :func:`tree_only_doc` -- three tree levels, no expansion."""
    return hier_db_factory(tree_only_doc(), "tree_only.json")


@pytest.fixture
def range_instances_db(hier_db_factory) -> HierarchicalChannelDatabase:
    """Loaded :func:`range_instances_doc` -- ``_type: "range"`` instance level."""
    return hier_db_factory(range_instances_doc(), "range_instances.json")


@pytest.fixture
def list_instances_db(hier_db_factory) -> HierarchicalChannelDatabase:
    """Loaded :func:`list_instances_doc` -- ``_type: "list"`` instance level."""
    return hier_db_factory(list_instances_doc(), "list_instances.json")


@pytest.fixture
def optional_tree_level_db(hier_db_factory) -> HierarchicalChannelDatabase:
    """Loaded :func:`optional_tree_level_doc` -- optional tree level, skipped by a leaf."""
    return hier_db_factory(optional_tree_level_doc(), "optional_tree_level.json")


@pytest.fixture
def optional_instance_level_db(hier_db_factory) -> HierarchicalChannelDatabase:
    """Loaded :func:`optional_instance_level_doc` -- optional instance level, no container."""
    return hier_db_factory(optional_instance_level_doc(), "optional_instance_level.json")


@pytest.fixture
def hybrid_expansion_db(hier_db_factory) -> HierarchicalChannelDatabase:
    """Loaded :func:`hybrid_expansion_doc` -- ``_expansion`` on a tree child."""
    return hier_db_factory(hybrid_expansion_doc(), "hybrid_expansion.json")


@pytest.fixture
def scalar_leaves_db(hier_db_factory) -> HierarchicalChannelDatabase:
    """Loaded :func:`scalar_leaves_doc` -- non-dict leaf values."""
    return hier_db_factory(scalar_leaves_doc(), "scalar_leaves.json")


@pytest.fixture
def separator_override_db(hier_db_factory) -> HierarchicalChannelDatabase:
    """Loaded :func:`separator_override_doc` -- ``_separator`` at two depths."""
    return hier_db_factory(separator_override_doc(), "separator_override.json")


@pytest.fixture
def channel_part_db(hier_db_factory) -> HierarchicalChannelDatabase:
    """Loaded :func:`channel_part_doc` -- ``_channel_part`` overrides, one empty."""
    return hier_db_factory(channel_part_doc(), "channel_part.json")


@pytest.fixture
def leaf_with_children_db(hier_db_factory) -> HierarchicalChannelDatabase:
    """Loaded :func:`leaf_with_children_doc` -- ``_is_leaf`` node with children."""
    return hier_db_factory(leaf_with_children_doc(), "leaf_with_children.json")


@pytest.fixture
def optional_leaf_with_children_db(hier_db_factory) -> HierarchicalChannelDatabase:
    """Loaded :func:`optional_leaf_with_children_doc` -- same node, optional level."""
    return hier_db_factory(optional_leaf_with_children_doc(), "optional_leaf_with_children.json")


@pytest.fixture
def all_optional_next_instances_db(hier_db_factory) -> HierarchicalChannelDatabase:
    """Loaded :func:`all_optional_next_instances_doc` -- all-optional tail, instances next."""
    return hier_db_factory(all_optional_next_instances_doc(), "all_optional_instances.json")


@pytest.fixture
def all_optional_next_tree_db(hier_db_factory) -> HierarchicalChannelDatabase:
    """Loaded :func:`all_optional_next_tree_doc` -- all-optional tail, tree next."""
    return hier_db_factory(all_optional_next_tree_doc(), "all_optional_tree.json")
