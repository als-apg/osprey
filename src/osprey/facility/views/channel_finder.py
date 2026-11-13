"""The channel-finder index views: each selected pipeline's database.

The in_context index is written to ``<render>/data/channel_finder/in_context.json``::

    {
      "schema": "osprey.facility.channel_finder/1",
      "count": <rows>,
      "channels": [
        {"channel": <name>, "address": <address>, "description": <text>},
        ...
      ]
    }

One row per channel, sorted by address; a facility that tags channels
``in_context`` narrows the index to those channels. A row's ``channel`` is the
channel's ``label``, else its address; its ``description`` is the channel's
own. ``count`` is the number of rows. A render carries the index when its
``channel_finder.pipeline_mode`` is ``in_context``; a facility with no channel then stops the build with
``view-unsupported``.

The hierarchical index is written to
``<render>/data/channel_finder/hierarchical.json``::

    {
      "schema": "osprey.facility.channel_finder/1",
      "count": <rows>,
      "hierarchy": {
        "levels": [{"name": <level>, "type": "tree"}, ...],
        "naming_pattern": "{<level>}{<level>}..."
      },
      "tree": {<place node>: ... {<class>: {<device>: {<leaf>: {...}}}}}
    }

The levels are the level words of the places that hold a channel or are an
ancestor of one (:func:`channel_places`), ordered by the depth at which each
first appears, then ``class``, ``device`` and ``leaf``; a place tree that holds
no channel, such as a rack tree beside the ring, adds no level. Every level is
present for every channel: an absent place, class or device is the node ``-``
(``_description: "no <level>"``), so every leaf sits at the same depth. A place
node is keyed by its id relative to the place above it in the tree, a class node
by the class name, a device node by the device id. A non-leaf node's
``_channel_part`` is empty and a leaf's is the channel's address, so the pattern
of bare placeholders spells each address exactly.

A leaf is keyed by its channel's signal when no other leaf of its device node
shares it, else by ``<signal>:<role>`` when that is unique, else by its address;
a channel with no signal is keyed by its address. A leaf's ``_description`` is
its channel's description. A place node is described by its place, a class
node by its class (the facility's class row, else the vocabulary's), and a
device node by its device. ``count`` is the number of leaves. A render carries
the index when its ``channel_finder.pipeline_mode`` is ``hierarchical``; a facility
with no channel writes an empty tree. The build stops with
``view-unsupported`` when two level words of those places first appear at one
depth or one branch orders two of them against their depths, and
when a tree key (a place or device id, a class, a signal, or an address used as
a leaf key) begins with ``_``, which the loader reads as a meta key, and when
two channels of one device node take one leaf key, which happens only when one
channel's address is another's signal.

The middle-layer index is written to
``<render>/data/channel_finder/middle_layer.json``, with the DuckDB database
``run_sql`` queries beside it as ``middle_layer.duckdb``::

    {
      "schema": "osprey.facility.channel_finder/1",
      "count": <rows>,
      <System>: {
        "_description": <text>,
        <Family>: {
          "_description": <text>,
          "_setup": {"CommonNames": [...], "DeviceList": [...], "PlaceList": [...],
                     "ElementList": [...]},
          <Field>: {"_description": <text>, "ChannelNames": [<address>, ...]}
        }
      }
    }

A Family is a group: every group is filed under the System of each member,
the top place of the member's place (``-`` for a member with no place), and
named by its id less a leading ``<System>/``. A classed device that no
single-class group holds under its System files into a Family named by its
class under that System, described by the class; a group whose short name is
such a class name keeps its id. A channel belongs to its ``on`` device and to
every device it is an ``endpoint_of``, and so to every Family holding one of
those devices. A channel's field is its signal. A Field lists, for each member
in ``CommonNames`` order (members by ``s``, then id), its one channel with that
signal, so ``ChannelNames`` aligns with ``DeviceList``, and is described by the
vocabulary's sentence for the signal, else by the description its channels
share; a channel whose signal some member lacks or holds twice, or that has no
signal, is its own Field keyed by its address and described by the channel.
``_setup`` is derived from the members, never read from them: ``CommonNames``
is each member's ``label``, else its id; ``DeviceList`` is ``[index, k]`` — ``index``
the member's place numbered among its siblings (places of one parent and level,
ordered by the lowest ``s`` below each, then in natural id order), else 0 for a
member whose place has no sibling or that has no place; ``k`` its ordinal among
the family's members with that place, the index-0 members counted together;
``PlaceList`` is each member's place id, else ``null``; ``ElementList`` is each
member's position in member order. No record carries a ``DeviceList``, a
``PlaceList``, an ``ElementList`` or a positional name. A System is
described by its place's description, a group's Family by the group's
description, else its label. A channel of no family is left out; the build
names how many channels it left out and how many it keyed by address in one
note. ``count`` is the number of distinct addresses the Fields list; a channel
of no family is not counted. A render carries the index when its
``channel_finder.pipeline_mode`` is ``middle_layer``; a facility with no device in a group or with a class stops
the build with ``view-unsupported``, as does a host where DuckDB cannot load
its full-text-search extension, which the database's search index needs, and a
System or Family key the loader would not read back: one beginning with
``_``, or a System ``schema`` or ``count``, and a channel whose address is the
signal another channel's Field in the same Family is keyed by.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.facility import DEFAULT_ROLE
from osprey.facility.records import FacilityRecords
from osprey.facility.views import PIPELINE_MODE_KEY, ViewInputs

__all__ = [
    "ABSENT",
    "CHANNEL_FINDER_SCHEMA",
    "HIERARCHICAL_FILE",
    "HIERARCHICAL_MODE",
    "IN_CONTEXT_FILE",
    "IN_CONTEXT_MODE",
    "IN_CONTEXT_TAG",
    "MIDDLE_LAYER_DUCKDB_FILE",
    "MIDDLE_LAYER_FILE",
    "MIDDLE_LAYER_MODE",
    "TAIL_LEVELS",
    "channel_places",
    "hierarchical_document",
    "hierarchy_levels",
    "hierarchical_selected",
    "in_context_document",
    "in_context_selected",
    "middle_layer_document",
    "middle_layer_families",
    "middle_layer_selected",
    "write_hierarchical",
    "write_in_context",
    "write_middle_layer",
]

CHANNEL_FINDER_SCHEMA = "osprey.facility.channel_finder/1"
IN_CONTEXT_FILE = "in_context.json"
IN_CONTEXT_MODE = "in_context"
IN_CONTEXT_TAG = "in_context"
HIERARCHICAL_FILE = "hierarchical.json"
HIERARCHICAL_MODE = "hierarchical"
MIDDLE_LAYER_FILE = "middle_layer.json"
MIDDLE_LAYER_DUCKDB_FILE = "middle_layer.duckdb"
MIDDLE_LAYER_MODE = "middle_layer"

#: The node standing for an absent place, class or device.
ABSENT = "-"

#: The levels below the place levels, in order.
TAIL_LEVELS: tuple[str, ...] = ("class", "device", "leaf")

#: A level word the naming pattern can carry as a placeholder.
_LEVEL_WORD = re.compile(r"\w+")


def _row(channel: Mapping[str, Any]) -> dict[str, Any]:
    address = str(channel["id"])
    return {
        "channel": channel.get("label") or address,
        "address": address,
        "description": channel.get("description"),
    }


def in_context_document(records: FacilityRecords) -> dict[str, Any]:
    """The in_context index of one facility file.

    Args:
        records: The facility file's records.

    Returns:
        ``{schema, count, channels}``, one row per channel tagged
        ``in_context`` when any is, else one per channel, sorted by address;
        ``count`` is the number of rows.
    """
    channels = records.of_kind("channel")
    tagged = [channel for channel in channels if IN_CONTEXT_TAG in (channel.get("tags") or [])]
    rows = [_row(channel) for channel in tagged or channels]
    rows.sort(key=lambda row: row["address"])
    return {"schema": CHANNEL_FINDER_SCHEMA, "count": len(rows), "channels": rows}


def in_context_selected(inputs: ViewInputs) -> bool:
    """Whether the render selects the in_context pipeline.

    Args:
        inputs: The render's view inputs.

    Returns:
        True when the rendered ``channel_finder.pipeline_mode`` is ``in_context``.
    """
    return _pipeline_mode(inputs) == IN_CONTEXT_MODE


def _pipeline_mode(inputs: ViewInputs) -> str | None:
    channel_finder = inputs.rendered_config.get("channel_finder")
    if not isinstance(channel_finder, Mapping):
        return None
    mode = channel_finder.get("pipeline_mode")
    return mode if isinstance(mode, str) else None


def _mode_unsupported(detail: str, remedy: str) -> Exception:
    """The ``view-unsupported`` stop on the rendered ``channel_finder.pipeline_mode``."""
    from osprey.facility.errors import FacilityBuildError
    from osprey.facility.served import CONFIG_SOURCE

    return FacilityBuildError(
        "view-unsupported",
        PIPELINE_MODE_KEY,
        [CONFIG_SOURCE],
        remedy,
        record_kind="path",
        detail=detail,
    )


def write_in_context(root: Path, inputs: ViewInputs) -> list[Path]:
    """Write the in_context index into ``root``.

    Args:
        root: The render's ``data/channel_finder`` directory.
        inputs: The render's view inputs.

    Returns:
        The file written.

    Raises:
        FacilityBuildError: ``view-unsupported`` when the facility has no
            channel.
    """
    from osprey.facility.views import view_bytes

    document = in_context_document(inputs.records)
    if not document["channels"]:
        raise _mode_unsupported(
            f"selects {IN_CONTEXT_MODE} and the facility has no channel",
            "add a channel, or select another channel_finder_mode",
        )
    root.mkdir(parents=True, exist_ok=True)
    target = root / IN_CONTEXT_FILE
    target.write_bytes(view_bytes(document))
    return [target]


# --- the hierarchical index --------------------------------------------------------


def hierarchical_selected(inputs: ViewInputs) -> bool:
    """Whether the render selects the hierarchical pipeline.

    Args:
        inputs: The render's view inputs.

    Returns:
        True when the rendered ``channel_finder.pipeline_mode`` is ``hierarchical``.
    """
    return _pipeline_mode(inputs) == HIERARCHICAL_MODE


def _unsupported(
    record: Mapping[str, Any], record_kind: str, detail: str, remedy: str
) -> Exception:
    """The ``view-unsupported`` stop naming one facility record."""
    from osprey.facility.errors import FacilityBuildError
    from osprey.facility.validate import stating_files

    return FacilityBuildError(
        "view-unsupported",
        str(record["id"]),
        stating_files(record, None),
        remedy,
        record_kind=record_kind,
        detail=detail,
    )


#: What a hierarchical tree key may not begin with: the loader reads such a key as meta.
_META_PREFIX = "_"


def _checked_key(
    key: str, record: Mapping[str, Any], record_kind: str, index: str = HIERARCHICAL_MODE
) -> str:
    """``key``, unless it begins with ``_`` and so would be read as a meta key.

    Args:
        key: The key the index would write.
        record: The facility record the key comes from.
        record_kind: The kind of ``record``, as a stop names it.
        index: The index whose loader reads the key, as a stop names it.

    Raises:
        FacilityBuildError: ``view-unsupported`` naming ``record`` and the key.
    """
    if not key.startswith(_META_PREFIX):
        return key
    raise _unsupported(
        record,
        record_kind,
        f"its tree key `{key}` begins with `_`, "
        f"and a key beginning with `_` is a meta key of the {index} index",
        f"give the {record_kind} a key that does not begin with `_`, "
        "or select another channel_finder_mode",
    )


#: The remedy of a stop on two channels that take one index key.
_SHARED_KEY_REMEDY = (
    "give one of them a signal or address that differs, or select another channel_finder_mode"
)


def channel_places(records: FacilityRecords) -> list[Mapping[str, Any]]:
    """The places that hold a channel or are an ancestor of one, in file order.

    A channel is held by its device's place, else by the place it is on; a
    device's place may be one a relation projected.

    Args:
        records: The facility file's records.

    Returns:
        Those place records.
    """
    held: set[str] = set()
    for channel in records.of_kind("channel"):
        on = channel.get("on") or {}
        device = records.record("device", str(on["device"])) if on.get("device") else None
        place_id = device.get("place") if device is not None else on.get("place")
        if place_id is None:
            continue
        segments = str(place_id).split("/")
        held.update("/".join(segments[:depth]) for depth in range(1, len(segments) + 1))
    return [place for place in records.of_kind("place") if str(place["id"]) in held]


def _shallowest(places: Iterable[Mapping[str, Any]]) -> dict[str, tuple[int, Mapping[str, Any]]]:
    """Each level word's shallowest depth and the first place there to carry it."""
    shallowest: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for place in places:
        word = place.get("level")
        if word is None:
            continue
        depth = str(place["id"]).count("/")
        if str(word) not in shallowest or depth < shallowest[str(word)][0]:
            shallowest[str(word)] = (depth, place)
    return shallowest


def hierarchy_levels(records: FacilityRecords) -> list[str]:
    """The place levels of the hierarchical index, without checking them.

    The level words of :func:`channel_places`, ordered by the depth at which
    each first appears, then by word. Where the index builds, these are its
    place levels; this never stops the build, so a view written in any
    channel_finder mode can name them.

    Args:
        records: The facility file's records.

    Returns:
        The level words, shallowest first.
    """
    shallowest = _shallowest(channel_places(records))
    return sorted(shallowest, key=lambda word: (shallowest[word][0], word))


def _tree(place: Mapping[str, Any]) -> str:
    """The id of the top place of ``place``'s tree."""
    return str(place["id"]).split("/")[0]


def _place_levels(records: FacilityRecords) -> list[str]:
    """The checked place levels of the hierarchical index.

    Raises:
        FacilityBuildError: ``view-unsupported`` when a level word cannot name
            a level, or when two level words first appear at one depth.
    """
    shallowest = _shallowest(channel_places(records))
    for word, (_depth, place) in sorted(shallowest.items()):
        if not _LEVEL_WORD.fullmatch(word) or word in TAIL_LEVELS:
            raise _unsupported(
                place,
                "place",
                f"level `{word}` cannot name a hierarchical level",
                "use a level word of letters, digits and `_` other than "
                f"{', '.join(TAIL_LEVELS)}, or select another channel_finder_mode",
            )
    at_depth: dict[int, list[str]] = defaultdict(list)
    for word, (depth, _place) in shallowest.items():
        at_depth[depth].append(word)
    for depth, words in sorted(at_depth.items()):
        if len(words) > 1:
            first, second = sorted(words)[:2]
            trees = sorted({_tree(shallowest[first][1]), _tree(shallowest[second][1])})
            named = " and ".join(f"`{tree}`" for tree in trees)
            where = f"the trees {named}" if len(trees) > 1 else f"the tree {named}"
            raise _unsupported(
                shallowest[second][1],
                "place",
                f"levels `{first}` and `{second}` both first appear at depth {depth} "
                f"among the places that hold channels, in {where}",
                f"use one level word per depth across the places of {where} that hold "
                "channels, keep channels out of one of them, "
                "or select another channel_finder_mode",
            )
    return sorted(shallowest, key=lambda word: (shallowest[word][0], word))


def _place_keys(
    place_id: str | None,
    records: FacilityRecords,
    levels: Sequence[str],
) -> tuple[list[str], list[str | None]]:
    """The node key and the place id at each place level, for one place.

    Returns:
        The keys, ``-`` where the place has no ancestor of that level, and the
        place ids, ``None`` there.
    """
    keys = [ABSENT] * len(levels)
    ids: list[str | None] = [None] * len(levels)
    if place_id is None:
        return keys, ids
    index = {word: position for position, word in enumerate(levels)}
    segments = place_id.split("/")
    parent = ""
    last = -1
    for depth in range(1, len(segments) + 1):
        ancestor_id = "/".join(segments[:depth])
        ancestor = records.record("place", ancestor_id)
        if ancestor is None or ancestor.get("level") is None:
            continue
        position = index[str(ancestor["level"])]
        if position <= last:
            raise _unsupported(
                ancestor,
                "place",
                f"level `{ancestor['level']}` sits below level `{levels[last]}` "
                "against the order of their shallowest depths",
                "order the level words the same way on every branch, "
                "or select another channel_finder_mode",
            )
        keys[position] = _checked_key(
            ancestor_id[len(parent) + 1 :] if parent else ancestor_id, ancestor, "place"
        )
        ids[position] = ancestor_id
        parent = ancestor_id
        last = position
    return keys, ids


def _class_descriptions(records: FacilityRecords) -> dict[str, str]:
    from osprey.facility.validate import vocabulary

    descriptions = {
        str(row["name"]): str(row["description"])
        for row in vocabulary()["classes"]
        if row.get("description")
    }
    for row in records.document.get("classes") or []:
        if row.get("description"):
            descriptions[str(row["class"])] = str(row["description"])
    return descriptions


def _leaf_keys(channels: Sequence[Mapping[str, Any]]) -> list[str]:
    """Each leaf's key among the leaves of one device node, in the order given."""
    signals = Counter(channel.get("signal") for channel in channels)
    pairs = Counter(
        (channel.get("signal"), channel.get("role", DEFAULT_ROLE)) for channel in channels
    )
    keys: list[str] = []
    for channel in channels:
        signal = channel.get("signal")
        role = channel.get("role", DEFAULT_ROLE)
        if signal is None:
            keys.append(str(channel["id"]))
        elif signals[signal] == 1:
            keys.append(str(signal))
        elif pairs[(signal, role)] == 1:
            keys.append(f"{signal}:{role}")
        else:
            keys.append(str(channel["id"]))
    taken = Counter(keys)
    return [
        str(channel["id"]) if taken[key] > 1 else key
        for channel, key in zip(channels, keys, strict=True)
    ]


def _shared_leaf_key(
    first: Mapping[str, Any], second: Mapping[str, Any], key: str, device: Mapping[str, Any] | None
) -> Exception:
    """The ``view-unsupported`` stop on two channels of one device node that take one leaf key.

    The two channels are named in address order, so the line does not depend on
    the order the facility file lists them in.
    """
    first, second = sorted((first, second), key=lambda channel: str(channel["id"]))
    index = f"of the {HIERARCHICAL_MODE} index"
    if device is None:
        return _unsupported(
            second,
            "channel",
            f"it and channel `{first['id']}` take the one leaf key `{key}` under no device {index}",
            _SHARED_KEY_REMEDY,
        )
    return _unsupported(
        device,
        "device",
        f"its channels `{first['id']}` and `{second['id']}` take the one leaf key `{key}` {index}",
        _SHARED_KEY_REMEDY,
    )


def hierarchical_document(records: FacilityRecords) -> dict[str, Any]:
    """The hierarchical index of one facility file.

    Args:
        records: The facility file's records.

    Returns:
        ``{schema, count, hierarchy, tree}``; ``count`` is the number of
        leaves the tree holds.

    Raises:
        FacilityBuildError: ``view-unsupported`` when two level words of the
            places that hold channels first appear at one depth, when such a
            level word cannot be a level name,
            when one branch orders two level words against their depths, when
            a tree key begins with ``_``, or when two channels of one device
            node take one leaf key.
    """
    place_levels = _place_levels(records)
    levels = [*place_levels, *TAIL_LEVELS]

    descriptions: dict[tuple[str, ...], str | None] = {}
    leaves: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)

    for channel in records.of_kind("channel"):
        on = channel.get("on") or {}
        device = records.record("device", str(on["device"])) if on.get("device") else None
        place_id = device.get("place") if device is not None else on.get("place")
        keys, ids = _place_keys(
            str(place_id) if place_id is not None else None, records, place_levels
        )
        for position, word in enumerate(place_levels):
            place_node = tuple(keys[: position + 1])
            if place_node not in descriptions:
                place_ref = ids[position]
                descriptions[place_node] = (
                    f"no {word}"
                    if place_ref is None
                    else records.require("place", place_ref).get("description")
                )
        class_key = device_key = ABSENT
        if device is not None:
            if device.get("class"):
                class_key = _checked_key(str(device["class"]), device, "device")
            device_key = _checked_key(str(device["id"]), device, "device")
        class_node = (*keys, class_key)
        device_node = (*class_node, device_key)
        if device is not None:
            descriptions[device_node] = device.get("description")
        else:
            descriptions[device_node] = "no device"
        leaves[device_node].append(channel)

    class_texts = _class_descriptions(records)
    for class_node in {node[: len(place_levels) + 1] for node in leaves}:
        if class_node[-1] == ABSENT:
            descriptions[class_node] = "no class"
        else:
            descriptions[class_node] = class_texts.get(class_node[-1])

    tree: dict[str, Any] = {}
    count = 0
    for device_node, channels in leaves.items():
        node = tree
        for depth in range(len(device_node)):
            key = device_node[depth]
            if key not in node:
                child: dict[str, Any] = {"_channel_part": ""}
                description = descriptions.get(device_node[: depth + 1])
                if description:
                    child["_description"] = description
                node[key] = child
            node = node[key]
        holders: dict[str, Mapping[str, Any]] = {}
        for channel, key in zip(channels, _leaf_keys(channels), strict=True):
            _checked_key(key, channel, "channel")
            if key in holders:
                raise _shared_leaf_key(
                    holders[key], channel, key, records.record("device", device_node[-1])
                )
            holders[key] = channel
            leaf: dict[str, Any] = {"_channel_part": str(channel["id"])}
            if channel.get("description"):
                leaf["_description"] = channel["description"]
            count += key not in node
            node[key] = leaf

    return {
        "schema": CHANNEL_FINDER_SCHEMA,
        "count": count,
        "hierarchy": {
            "levels": [{"name": level, "type": "tree"} for level in levels],
            "naming_pattern": "".join(f"{{{level}}}" for level in levels),
        },
        "tree": tree,
    }


def write_hierarchical(root: Path, inputs: ViewInputs) -> list[Path]:
    """Write the hierarchical index into ``root``.

    Args:
        root: The render's ``data/channel_finder`` directory.
        inputs: The render's view inputs.

    Returns:
        The file written.

    Raises:
        FacilityBuildError: ``view-unsupported`` when the place levels cannot
            be ordered into one hierarchy, when a tree key begins with ``_``, or
            when two channels of one device node take one leaf key.
    """
    from osprey.facility.views import view_bytes

    document = hierarchical_document(inputs.records)
    root.mkdir(parents=True, exist_ok=True)
    target = root / HIERARCHICAL_FILE
    target.write_bytes(view_bytes(document))
    return [target]


# --- the middle-layer index --------------------------------------------------------


def middle_layer_selected(inputs: ViewInputs) -> bool:
    """Whether the render selects the middle-layer pipeline.

    Args:
        inputs: The render's view inputs.

    Returns:
        True when the rendered ``channel_finder.pipeline_mode`` is ``middle_layer``.
    """
    return _pipeline_mode(inputs) == MIDDLE_LAYER_MODE


def _member_order(device: Mapping[str, Any]) -> tuple[bool, float, str]:
    position = device.get("s")
    return (position is None, float(position) if position is not None else 0.0, str(device["id"]))


def _short_name(group_id: str, system: str) -> str:
    prefix = f"{system}/"
    return group_id.removeprefix(prefix)


def _family_names(system: str, group_ids: Sequence[str], taken: Iterable[str] = ()) -> list[str]:
    """The Family names of the groups filed under one System, in their order.

    A group is named by its id less a leading ``<System>/``, unless another
    group under that System, or a name in ``taken``, has the same short name:
    then the group keeps its id.
    """
    short = Counter(_short_name(group_id, system) for group_id in group_ids)
    reserved = set(taken)
    return [
        group_id
        if short[_short_name(group_id, system)] > 1 or _short_name(group_id, system) in reserved
        else _short_name(group_id, system)
        for group_id in group_ids
    ]


@dataclass(frozen=True)
class _Family:
    """One Family of the middle-layer index under one System."""

    name: str
    description: str | None
    members: list[Mapping[str, Any]]
    record: Mapping[str, Any]
    record_kind: str


def _system_of(device: Mapping[str, Any]) -> str:
    place = device.get("place")
    return str(place).split("/", 1)[0] if place else ABSENT


def _families_by_system(records: FacilityRecords) -> dict[str, list[_Family]]:
    """Each System's Families: its groups, then its derived (System, class) Families.

    A group is filed under the System of each member, the top place of the
    member's place (``-`` for a member with no place). A classed device that no
    single-class group holds under its System (a group whose members filed
    there all carry one class) files into the derived Family of its System and
    class, named by the class. Members are ordered by ``s``, then id.
    """
    split: dict[str, list[tuple[Mapping[str, Any], list[Mapping[str, Any]]]]] = defaultdict(list)
    covered: dict[str, set[str]] = defaultdict(set)
    for group in records.of_kind("group"):
        by_system: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for member_id in group.get("members") or []:
            device = records.record("device", str(member_id))
            if device is not None:
                by_system[_system_of(device)].append(device)
        for system, members in by_system.items():
            split[system].append((group, sorted(members, key=_member_order)))
            member_classes = {member.get("class") for member in members}
            if len(member_classes) == 1 and None not in member_classes:
                covered[system].update(str(member["id"]) for member in members)
    derived: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for device in records.of_kind("device"):
        system = _system_of(device)
        if device.get("class") and str(device["id"]) not in covered[system]:
            derived[system][str(device["class"])].append(device)

    class_texts = _class_descriptions(records)
    out: dict[str, list[_Family]] = {}
    for system in dict.fromkeys([*split, *derived]):
        entries = split.get(system, [])
        classes = sorted(derived.get(system, {}))
        names = _family_names(system, [str(group["id"]) for group, _ in entries], classes)
        families = [
            _Family(
                name,
                group.get("description") or group.get("label"),
                members,
                group,
                "group",
            )
            for (group, members), name in zip(entries, names, strict=True)
        ]
        for class_name in classes:
            record = {"id": class_name}
            _checked_key(class_name, record, "class", _MIDDLE_LAYER_INDEX)
            members = sorted(derived[system][class_name], key=_member_order)
            families.append(
                _Family(class_name, class_texts.get(class_name), members, record, "class")
            )
        out[system] = families
    return out


def middle_layer_families(records: FacilityRecords) -> dict[str, list[tuple[str, str]]]:
    """Each device class's families as the middle-layer index files and names them.

    Args:
        records: The facility file's records.

    Returns:
        Class name -> the ``(System, Family)`` pairs under which the index lists
        a member of that class, sorted; a class no family holds is absent.
    """
    out: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for system, families in _families_by_system(records).items():
        for family in families:
            for member in family.members:
                if member.get("class"):
                    out[str(member["class"])].add((system, family.name))
    return {name: sorted(pairs) for name, pairs in sorted(out.items())}


#: A run of digits, which natural order compares as a number.
_DIGITS = re.compile(r"(\d+)")

#: The place index of a device whose place has no sibling places; sibling
#: indices start at 1, so these rows meet no indexed row.
NO_PLACE_INDEX = 0


def _natural(text: str) -> tuple[tuple[int, int | str], ...]:
    """``text`` as natural-order parts: digit runs compare as numbers."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part) for part in _DIGITS.split(text) if part
    )


def _place_indices(records: FacilityRecords) -> dict[str, int]:
    """Each place's 1-based index among its siblings, for a place that has siblings.

    The siblings of a place are the places with the same parent and the same
    level; a top place is a System and is never indexed. Siblings are ordered
    by the lowest ``s`` of any device placed in the place or below it; those
    with no positioned device follow in natural id order.
    """
    lowest: dict[str, float] = {}
    for device in records.of_kind("device"):
        place, position = device.get("place"), device.get("s")
        if not place or position is None:
            continue
        segments = str(place).split("/")
        for depth in range(1, len(segments) + 1):
            ancestor = "/".join(segments[:depth])
            if ancestor not in lowest or float(position) < lowest[ancestor]:
                lowest[ancestor] = float(position)
    siblings: dict[tuple[str, str | None], list[str]] = defaultdict(list)
    for place in records.of_kind("place"):
        place_id = str(place["id"])
        if "/" not in place_id:
            continue
        level = place.get("level")
        siblings[(place_id.rsplit("/", 1)[0], None if level is None else str(level))].append(
            place_id
        )
    indices: dict[str, int] = {}
    for ids in siblings.values():
        if len(ids) < 2:
            continue
        ordered = sorted(
            ids,
            key=lambda place_id: (
                place_id not in lowest,
                lowest.get(place_id, 0.0),
                _natural(place_id),
            ),
        )
        indices.update((place_id, index) for index, place_id in enumerate(ordered, start=1))
    return indices


def _setup(members: Sequence[Mapping[str, Any]], indices: Mapping[str, int]) -> dict[str, Any]:
    """A family's ``_setup``, derived from its members in member order.

    ``CommonNames`` holds each member's ``label``, else its id. ``DeviceList``
    holds ``[index, k]``: ``index`` the member's place numbered among its
    sibling places, else 0 for a member whose place has none (one at its
    System, in an only-child place, or with no place); ``k`` its 1-based
    ordinal among the family's members with that place, all index-0 members
    counted together. ``PlaceList`` holds each member's place id, else
    ``None``. ``ElementList`` holds each member's 1-based position.
    """
    names = [str(member.get("label") or member["id"]) for member in members]
    seen: Counter[str | None] = Counter()
    device_list: list[list[int]] = []
    place_list: list[str | None] = []
    for member in members:
        place = str(member["place"]) if member.get("place") else None
        index = indices.get(place) if place else None
        counted = place if index is not None else None
        seen[counted] += 1
        device_list.append([NO_PLACE_INDEX if index is None else index, seen[counted]])
        place_list.append(place)
    return {
        "CommonNames": names,
        "DeviceList": device_list,
        "PlaceList": place_list,
        "ElementList": list(range(1, len(members) + 1)),
    }


def _signal_descriptions(records: FacilityRecords) -> dict[str, str]:
    """Each signal role's description: the vocabulary's and the facility-added ones."""
    from osprey.facility.validate import vocabulary

    rows = [*vocabulary()["signal_roles"], *(records.document.get("signals") or [])]
    return {str(row["name"]): str(row["description"]) for row in rows if row.get("description")}


def _common_description(channels: Sequence[Mapping[str, Any]]) -> str | None:
    """The description every one of ``channels`` states, else ``None``."""
    texts = {channel.get("description") for channel in channels}
    if len(texts) != 1:
        return None
    (text,) = texts
    return str(text) if text else None


def _family_fields(
    system: str,
    family: _Family,
    records: FacilityRecords,
    signal_texts: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], int]:
    """A family's Fields under one System, and how many channels are keyed by address.

    A channel's field key is its signal. A signal every member holds exactly
    once is one Field listing the members' channels in member order, described
    by the vocabulary's sentence for the signal, else by the description its
    channels share. Every other channel, and one with no signal, is its own
    Field keyed by its address and described by the channel.

    Raises:
        FacilityBuildError: ``view-unsupported`` naming the family's record,
            when one channel's address is the signal another channel's Field
            is keyed by.
    """
    holders: dict[str, str] = {}

    def hold(key: str, channel: Mapping[str, Any]) -> None:
        address = str(channel["id"])
        held = holders.setdefault(key, address)
        if held == address:
            return
        first, second = sorted((held, address))
        raise _unsupported(
            family.record,
            family.record_kind,
            f"its Family `{family.name}` under System `{system}` gives channels `{first}` and "
            f"`{second}` the one Field key `{key}` of the {_MIDDLE_LAYER_INDEX} index",
            _SHARED_KEY_REMEDY,
        )

    per_member: list[dict[str | None, list[Mapping[str, Any]]]] = []
    for member in family.members:
        fields: dict[str | None, list[Mapping[str, Any]]] = defaultdict(list)
        for channel in records.channels_on(str(member["id"])):
            signal = channel.get("signal")
            fields[str(signal) if signal else None].append(channel)
        per_member.append(fields)

    keys = {key for fields in per_member for key in fields}
    out: dict[str, dict[str, Any]] = {}
    by_address = 0
    for key in sorted(keys, key=lambda key: (key is None, key or "")):
        if key is not None and all(len(fields.get(key, ())) == 1 for fields in per_member):
            channels = [fields[key][0] for fields in per_member]
            hold(key, channels[0])
            field: dict[str, Any] = {"ChannelNames": [str(c["id"]) for c in channels]}
            description = signal_texts.get(key) or _common_description(channels)
            if description:
                field["_description"] = description
            out[key] = field
            continue
        for fields in per_member:
            for channel in fields.get(key, ()):
                address = str(channel["id"])
                hold(address, channel)
                field = {"ChannelNames": [address]}
                if channel.get("description"):
                    field["_description"] = channel["description"]
                out[address] = field
                by_address += 1
    return out, by_address


#: The middle-layer index as a stop names it.
_MIDDLE_LAYER_INDEX = "middle-layer"


def _checked_system(system: str, place: Mapping[str, Any]) -> None:
    """Stop on a System key the middle-layer loader would not read as a System.

    Raises:
        FacilityBuildError: ``view-unsupported`` naming the place, when the key
            is a document key of the index or begins with ``_``.
    """
    from osprey.services.channel_finder.databases.middle_layer import DOCUMENT_KEYS

    if system in DOCUMENT_KEYS:
        raise _unsupported(
            place,
            "place",
            f"its System key `{system}` is a document key of the middle-layer index",
            f"give the place an id other than `{system}`, or select another channel_finder_mode",
        )
    _checked_key(system, place, "place", _MIDDLE_LAYER_INDEX)


def middle_layer_document(records: FacilityRecords) -> tuple[dict[str, Any], int, int]:
    """The middle-layer index of one facility file.

    Args:
        records: The facility file's records.

    Returns:
        ``{schema, count, <System>: {<Family>: {...}}}``; the number of
        channels in no family, which the index leaves out; and the number of
        channels the index keys by their address. ``count`` is the number of
        distinct addresses its Fields list.

    Raises:
        FacilityBuildError: ``view-unsupported`` when a System or Family key
            begins with ``_``, a System key is a document key of the index
            (``schema``, ``count``), or one channel's address is the signal
            another channel's Field in the same Family is keyed by.
    """
    indices = _place_indices(records)

    signal_texts = _signal_descriptions(records)
    in_family: set[str] = set()
    listed: set[str] = set()
    document: dict[str, Any] = {"schema": CHANNEL_FINDER_SCHEMA}
    by_address = 0
    for system, families in _families_by_system(records).items():
        place = records.record("place", system)
        if system != ABSENT:
            _checked_system(system, place or {"id": system})
        node: dict[str, Any] = {}
        if system == ABSENT:
            node["_description"] = "no place"
        elif place is not None and place.get("description"):
            node["_description"] = place["description"]
        for entry in families:
            _checked_key(entry.name, entry.record, entry.record_kind, _MIDDLE_LAYER_INDEX)
            for member in entry.members:
                in_family.update(str(c["id"]) for c in records.channels_on(str(member["id"])))
            fields, keyed = _family_fields(system, entry, records, signal_texts)
            listed.update(address for field in fields.values() for address in field["ChannelNames"])
            by_address += keyed
            family: dict[str, Any] = {"_setup": _setup(entry.members, indices), **fields}
            if entry.description:
                family["_description"] = entry.description
            node[entry.name] = family
        document[system] = node
    document["count"] = len(listed)

    left_out = sum(
        1 for channel in records.of_kind("channel") if str(channel["id"]) not in in_family
    )
    return document, left_out, by_address


def _imported_from(database: Path, digest: str) -> bool:
    """Whether ``database`` still sits beside the index whose bytes hash to ``digest``.

    A database is copied only while the index it was imported from is
    unchanged beside it: a render directory rewritten or removed since leaves
    nothing to copy.
    """
    index = database.with_name(MIDDLE_LAYER_FILE)
    if not (database.is_file() and index.is_file()):
        return False
    return hashlib.sha256(index.read_bytes()).hexdigest() == digest


def _import_database(index: Path, database: Path) -> None:
    """Import the middle-layer index into a DuckDB database with its search index.

    Raises:
        FacilityBuildError: ``view-unsupported`` when DuckDB cannot write the
            database.
    """
    import duckdb

    from osprey.services.channel_finder.databases.duckdb_import import import_to_duckdb

    try:
        import_to_duckdb(str(index), str(database))
    except duckdb.Error as exc:
        database.unlink(missing_ok=True)
        raise _mode_unsupported(
            f"selects {MIDDLE_LAYER_MODE} and its DuckDB database cannot be written ({exc})",
            "install DuckDB's `fts` extension on this host, or select another channel_finder_mode",
        ) from exc


def write_middle_layer(root: Path, inputs: ViewInputs) -> list[Path]:
    """Write the middle-layer index and its DuckDB database into ``root``.

    The channels the index leaves out or keys by address are counted in one
    note on stderr. A render whose index bytes match a database this build has
    already imported (``inputs.built_databases``) copies that database rather
    than importing the index again.

    Args:
        root: The render's ``data/channel_finder`` directory.
        inputs: The render's view inputs.

    Returns:
        The files written: the index and its database.

    Raises:
        FacilityBuildError: ``view-unsupported`` when no device of the facility
            is in a group or has a class, or when DuckDB cannot write the database (its
            full-text-search extension is neither installed nor reachable).
    """
    from osprey.facility.views import report_note, view_bytes
    from osprey.services.channel_finder.databases.middle_layer import DOCUMENT_KEYS

    document, left_out, by_address = middle_layer_document(inputs.records)
    if not document.keys() - DOCUMENT_KEYS:
        raise _mode_unsupported(
            f"selects {MIDDLE_LAYER_MODE} and no device of the facility is in a group "
            "or has a class",
            "add a group or give the devices a class, or select another channel_finder_mode",
        )
    root.mkdir(parents=True, exist_ok=True)
    target = root / MIDDLE_LAYER_FILE
    payload = view_bytes(document)
    target.write_bytes(payload)
    database = root / MIDDLE_LAYER_DUCKDB_FILE
    database.unlink(missing_ok=True)

    digest = hashlib.sha256(payload).hexdigest()
    built = inputs.built_databases
    source = None if built is None else built.get(digest)
    if source is not None and _imported_from(source, digest):
        shutil.copyfile(source, database)
    else:
        _import_database(target, database)
        if built is not None:
            built[digest] = database
    if left_out or by_address:
        report_note(
            inputs,
            f"view {MIDDLE_LAYER_MODE}: {left_out} channels in no family left out, "
            f"{by_address} keyed by address",
        )
    return [target, database]
