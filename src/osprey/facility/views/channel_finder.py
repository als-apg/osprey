"""The channel-finder index views: each selected pipeline's database.

The in_context index is written to ``<render>/data/channel_finder/in_context.json``::

    {
      "schema": "osprey.facility.channel_finder/1",
      "channels": [
        {"channel": <name>, "address": <address>, "description": <text>},
        ...
      ]
    }

One row per channel tagged ``in_context``, sorted by address. A row's
``channel`` is the channel's first ``names`` entry, else its address; its
``description`` is the channel's own. A render carries the index when its
``channel_finder.pipeline_mode`` is ``in_context``; a facility with no tagged
channel then stops the build with ``view-unsupported``.

The hierarchical index is written to
``<render>/data/channel_finder/hierarchical.json``::

    {
      "schema": "osprey.facility.channel_finder/1",
      "hierarchy": {
        "levels": [{"name": <level>, "type": "tree"}, ...],
        "naming_pattern": "{<level>}{<level>}..."
      },
      "tree": {<place node>: ... {<class>: {<device>: {<leaf>: {...}}}}}
    }

The levels are the facility's place level words, ordered by the depth at which
each first appears, then ``class``, ``device`` and ``leaf``. Every level is
present for every channel: an absent place, class or device is the node ``-``
(``_description: "no <level>"``), so every leaf sits at the same depth. A place
node is keyed by its id relative to the place above it in the tree, a class node
by the class name, a device node by the device id. A non-leaf node's
``_channel_part`` is empty and a leaf's is the channel's address, so the pattern
of bare placeholders spells each address exactly.

A leaf is keyed by its channel's signal when no other leaf of its device node
shares it, else by ``<signal>:<role>`` when that is unique, else by its address;
a channel with no signal is keyed by its address. A leaf's ``_description`` is
the ``signals`` sentence of a group naming its device for the channel's kind of
signal (the group's ``signals`` key, longest first, whose ``/``-separated parts
are the last alphanumeric runs of the address), else the channel's own. A place
node is described by its place, a class node by the smallest described group
holding every device under it (else by its class), and a device node by its device
(else by the smallest described group naming it). A render carries the index
when its ``channel_finder.pipeline_mode`` is ``hierarchical``; a facility
with no channel writes an empty tree. The build stops with
``view-unsupported`` when two place level words first appear at one depth, and
when a tree key (a place or device id, a class, a signal, or an address used as
a leaf key) begins with ``_``, which the loader reads as a meta key.

The middle-layer index is written to
``<render>/data/channel_finder/middle_layer.json``, with the DuckDB database
``run_sql`` queries beside it as ``middle_layer.duckdb``::

    {
      "schema": "osprey.facility.channel_finder/1",
      <System>: {
        "_description": <text>,
        <Family>: {
          "_description": <text>,
          "_setup": {"CommonNames": [...], "DeviceList": [...], "ElementList": [...]},
          <Field>: {"_description": <text>, "ChannelNames": [<address>, ...]}
        }
      }
    }

A Family is a group: every group is filed under the System of each member,
the top place of the member's place (``-`` for a member with no place), and
named by its id less a leading ``<System>/``. A channel belongs to its ``on``
device and to every device it is an ``endpoint_of``, and so to every Family
whose group holds one of those devices. A channel's field is the family's
``signals`` key its address ends with (the longest), else its signal. A Field
lists, for each member in ``CommonNames`` order (members by ``s``, then id), its
one channel with that field, so ``ChannelNames`` aligns with ``DeviceList``; a
channel whose field some member lacks or holds twice, or that has no field, is
its own Field keyed by its address. A Field's ``_description`` is the family's
``signals`` sentence under the longest key every one of its addresses ends
with; a group without ``signals`` writes no Field sentence. ``_setup`` holds
each member's last name (its common name) and its ``DeviceList`` and
``ElementList`` attributes, each list only when every member states it: a
device's common name is its last names entry; its first is its source name,
which the graph view reads; a device with no names is named by its id. A
channel of no family is left out; the build names how many channels it left
out and how many it keyed by address in one note. A render
carries the index when its ``channel_finder.pipeline_mode`` is
``middle_layer``; a facility with no group stops the build with
``view-unsupported``, as does a host where DuckDB cannot load its
full-text-search extension, which the database's search index needs, and a
System or Family key the loader would not read back: one beginning with
``_``, or a System ``schema``.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

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
    "hierarchical_document",
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

#: One alphanumeric run of an address or of a ``signals`` key.
_ALNUM_RUN = re.compile(r"[A-Za-z0-9]+")


def _row(channel: Mapping[str, Any]) -> dict[str, Any]:
    address = str(channel["id"])
    names = channel.get("names")
    name = names[0] if isinstance(names, list) and names else address
    return {"channel": name, "address": address, "description": channel.get("description")}


def in_context_document(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The in_context index of one facility file.

    Args:
        doc: The facility file.

    Returns:
        ``{schema, channels}``, one row per tagged channel, sorted by address.
    """
    rows = [
        _row(channel)
        for channel in doc.get("channels", [])
        if IN_CONTEXT_TAG in (channel.get("tags") or [])
    ]
    rows.sort(key=lambda row: row["address"])
    return {"schema": CHANNEL_FINDER_SCHEMA, "channels": rows}


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
        FacilityBuildError: ``view-unsupported`` when no channel is tagged
            ``in_context``.
    """
    from osprey.facility.views import view_bytes

    document = in_context_document(inputs.doc)
    if not document["channels"]:
        raise _mode_unsupported(
            f"selects {IN_CONTEXT_MODE} and no channel is tagged `{IN_CONTEXT_TAG}`",
            f"tag at least one channel `{IN_CONTEXT_TAG}`, or select another channel_finder_mode",
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


def _place_levels(places: Iterable[Mapping[str, Any]]) -> list[str]:
    """The place level words, ordered by the depth at which each first appears."""
    shallowest: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for place in places:
        word = place.get("level")
        if word is None:
            continue
        depth = str(place["id"]).count("/")
        if str(word) not in shallowest or depth < shallowest[str(word)][0]:
            shallowest[str(word)] = (depth, place)
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
            raise _unsupported(
                shallowest[second][1],
                "place",
                f"levels `{first}` and `{second}` both first appear at depth {depth}",
                "give the places at one depth one level word, "
                "or select another channel_finder_mode",
            )
    return sorted(shallowest, key=lambda word: shallowest[word][0])


def _place_keys(
    place_id: str | None,
    places: Mapping[str, Mapping[str, Any]],
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
        ancestor = places.get(ancestor_id)
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


def _smallest_group_description(
    groups: Sequence[Mapping[str, Any]], members: set[str]
) -> str | None:
    """The description of the smallest described group holding every one of ``members``.

    Ties between groups of one size go to the earlier group in the facility file.
    """
    best: tuple[int, int, str] | None = None
    for position, group in enumerate(groups):
        description = group.get("description")
        held = set(group.get("members") or [])
        if not description or not members <= held:
            continue
        rank = (len(held), position, str(description))
        if best is None or rank < best:
            best = rank
    return best[2] if best is not None else None


def _alnum_runs(text: str) -> list[str]:
    return _ALNUM_RUN.findall(text)


def _key_runs(key: Any) -> list[str]:
    """A ``signals`` key's runs: each ``/``-separated part read as alphanumeric runs."""
    return [run for part in str(key).split("/") for run in _alnum_runs(part)]


def _ends_with(runs: Sequence[str], key_runs: Sequence[str]) -> bool:
    """Whether a non-empty ``key_runs`` ends ``runs``."""
    return bool(key_runs) and list(runs[-len(key_runs) :]) == list(key_runs)


def _signal_sentence(address: str, family_signals: Sequence[Mapping[str, Any]]) -> str | None:
    """The family sentence for one address: the longest ``signals`` key its runs end with.

    A key's ``/``-separated parts are read as alphanumeric runs too, so a part
    spelled with ``_`` matches the runs the address spells it with. Ties between
    keys of one length go to the earlier group, then to the key sorted first.
    """
    runs = _alnum_runs(address)
    best: tuple[int, int, str, str] | None = None
    for position, signals in enumerate(family_signals):
        for key, sentence in signals.items():
            key_runs = _key_runs(key)
            if not _ends_with(runs, key_runs):
                continue
            rank = (-len(key_runs), position, str(key), str(sentence))
            if best is None or rank < best:
                best = rank
    return best[3] if best is not None else None


def _class_descriptions(doc: Mapping[str, Any]) -> dict[str, str]:
    from osprey.facility.validate import vocabulary

    descriptions = {
        str(row["name"]): str(row["description"])
        for row in vocabulary()["classes"]
        if row.get("description")
    }
    for row in doc.get("classes") or []:
        if row.get("description"):
            descriptions[str(row["class"])] = str(row["description"])
    return descriptions


def _leaf_keys(channels: Sequence[Mapping[str, Any]]) -> list[str]:
    """Each leaf's key among the leaves of one device node, in the order given."""
    signals = Counter(channel.get("signal") for channel in channels)
    pairs = Counter(
        (channel.get("signal"), channel.get("role", "readback")) for channel in channels
    )
    keys: list[str] = []
    for channel in channels:
        signal = channel.get("signal")
        role = channel.get("role", "readback")
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


def hierarchical_document(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The hierarchical index of one facility file.

    Args:
        doc: The facility file.

    Returns:
        ``{schema, hierarchy, tree}``.

    Raises:
        FacilityBuildError: ``view-unsupported`` when two place level words
            first appear at one depth, when a level word cannot be a level name,
            when one branch orders two level words against their depths, or
            when a tree key begins with ``_``.
    """
    places = {str(place["id"]): place for place in doc.get("places", [])}
    devices = {str(device["id"]): device for device in doc.get("devices", [])}
    groups: list[Mapping[str, Any]] = list(doc.get("groups", []))
    place_levels = _place_levels(places.values())
    levels = [*place_levels, *TAIL_LEVELS]

    family_signals: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for group in groups:
        if group.get("signals"):
            for member in group.get("members") or []:
                family_signals[str(member)].append(group["signals"])

    descriptions: dict[tuple[str, ...], str | None] = {}
    class_members: dict[tuple[str, ...], set[str]] = defaultdict(set)
    leaves: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)

    for channel in doc.get("channels", []):
        on = channel.get("on") or {}
        device = devices.get(str(on["device"])) if on.get("device") else None
        place_id = device.get("place") if device is not None else on.get("place")
        keys, ids = _place_keys(
            str(place_id) if place_id is not None else None, places, place_levels
        )
        for position, word in enumerate(place_levels):
            place_node = tuple(keys[: position + 1])
            if place_node not in descriptions:
                place_ref = ids[position]
                descriptions[place_node] = (
                    f"no {word}" if place_ref is None else places[place_ref].get("description")
                )
        class_key = device_key = ABSENT
        if device is not None:
            if device.get("class"):
                class_key = _checked_key(str(device["class"]), device, "device")
            device_key = _checked_key(str(device["id"]), device, "device")
        class_node = (*keys, class_key)
        device_node = (*class_node, device_key)
        if device is not None:
            class_members[class_node].add(str(device["id"]))
            descriptions[device_node] = device.get("description") or _smallest_group_description(
                groups, {str(device["id"])}
            )
        else:
            descriptions[device_node] = "no device"
        leaves[device_node].append(channel)

    class_texts = _class_descriptions(doc)
    for class_node in {node[: len(place_levels) + 1] for node in leaves}:
        if class_node[-1] == ABSENT:
            descriptions[class_node] = "no class"
        else:
            descriptions[class_node] = _smallest_group_description(
                groups, class_members[class_node]
            ) or class_texts.get(class_node[-1])

    tree: dict[str, Any] = {}
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
        family = family_signals.get(device_node[-1], []) if device_node[-1] != ABSENT else []
        for channel, key in zip(channels, _leaf_keys(channels), strict=True):
            _checked_key(key, channel, "channel")
            address = str(channel["id"])
            leaf: dict[str, Any] = {"_channel_part": address}
            sentence = _signal_sentence(address, family)
            description = sentence or channel.get("description")
            if description:
                leaf["_description"] = description
            node[key] = leaf

    return {
        "schema": CHANNEL_FINDER_SCHEMA,
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
            be ordered into one hierarchy, or when a tree key begins with ``_``.
    """
    from osprey.facility.views import view_bytes

    document = hierarchical_document(inputs.doc)
    root.mkdir(parents=True, exist_ok=True)
    target = root / HIERARCHICAL_FILE
    target.write_bytes(view_bytes(document))
    return [target]


# --- the middle-layer index --------------------------------------------------------


#: The device attributes ``_setup`` carries, one entry per member, in this order.
SETUP_ATTRIBUTES: tuple[str, ...] = ("DeviceList", "ElementList")


def middle_layer_selected(inputs: ViewInputs) -> bool:
    """Whether the render selects the middle-layer pipeline.

    Args:
        inputs: The render's view inputs.

    Returns:
        True when the rendered ``channel_finder.pipeline_mode`` is ``middle_layer``.
    """
    return _pipeline_mode(inputs) == MIDDLE_LAYER_MODE


def _signal_key(address: str, signals: Mapping[str, Any]) -> str | None:
    """The longest ``signals`` key whose runs end the address's runs, else ``None``.

    Ties between keys of one length go to the key sorted first.
    """
    runs = _alnum_runs(address)
    best: tuple[int, str] | None = None
    for key in signals:
        key_runs = _key_runs(key)
        if not _ends_with(runs, key_runs):
            continue
        rank = (-len(key_runs), str(key))
        if best is None or rank < best:
            best = rank
    return best[1] if best is not None else None


def _cell_sentence(
    addresses: Sequence[str], signals: Mapping[str, Any], role: str | None = None
) -> str | None:
    """The sentence a Field's cell carries.

    A Field keyed by a signal role takes the sentence under its own key, where
    ``signals`` states one; every other Field takes the sentence under the
    longest ``signals`` key every address ends with.
    """
    if role is not None and signals.get(role):
        return str(signals[role])
    best: tuple[int, str, str] | None = None
    for key, sentence in signals.items():
        key_runs = _key_runs(key)
        if key_runs and all(_ends_with(_alnum_runs(address), key_runs) for address in addresses):
            rank = (-len(key_runs), str(key), str(sentence))
            if best is None or rank < best:
                best = rank
    return best[2] if best is not None else None


def _member_order(device: Mapping[str, Any]) -> tuple[bool, float, str]:
    position = device.get("s")
    return (position is None, float(position) if position is not None else 0.0, str(device["id"]))


def _short_name(group_id: str, system: str) -> str:
    prefix = f"{system}/"
    return group_id[len(prefix) :] if group_id.startswith(prefix) else group_id


def _family_names(system: str, group_ids: Sequence[str]) -> list[str]:
    """The Family names of the groups filed under one System, in their order.

    A group is named by its id less a leading ``<System>/``, unless another
    group under that System has the same short name: then both keep their ids.
    """
    taken = Counter(_short_name(group_id, system) for group_id in group_ids)
    return [
        group_id if taken[_short_name(group_id, system)] > 1 else _short_name(group_id, system)
        for group_id in group_ids
    ]


def _families_by_system(
    doc: Mapping[str, Any],
) -> dict[str, list[tuple[Mapping[str, Any], str, list[Mapping[str, Any]]]]]:
    """Each System's families: the group, its Family name and its members there.

    A group is filed under the System of each member, the top place of
    the member's place (``-`` for a member with no place); its members under
    one System are ordered by ``s``, then id.
    """
    devices = {str(device["id"]): device for device in doc.get("devices", [])}
    split: dict[str, list[tuple[Mapping[str, Any], list[Mapping[str, Any]]]]] = defaultdict(list)
    for group in doc.get("groups", []):
        by_system: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for member_id in group.get("members") or []:
            device = devices.get(str(member_id))
            if device is None:
                continue
            place = device.get("place")
            by_system[str(place).split("/", 1)[0] if place else ABSENT].append(device)
        for system, members in by_system.items():
            split[system].append((group, sorted(members, key=_member_order)))
    return {
        system: [
            (group, name, members)
            for (group, members), name in zip(
                entries,
                _family_names(system, [str(group["id"]) for group, _ in entries]),
                strict=True,
            )
        ]
        for system, entries in split.items()
    }


def middle_layer_families(doc: Mapping[str, Any]) -> dict[str, list[tuple[str, str]]]:
    """Each device class's families as the middle-layer index files and names them.

    Args:
        doc: The facility file.

    Returns:
        Class name -> the ``(System, Family)`` pairs under which the index lists
        a member of that class, sorted; a class no family holds is absent.
    """
    out: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for system, entries in _families_by_system(doc).items():
        for _group, name, members in entries:
            for member in members:
                if member.get("class"):
                    out[str(member["class"])].add((system, name))
    return {name: sorted(pairs) for name, pairs in sorted(out.items())}


def _setup(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A family's ``_setup``: each member's common name and the attributes every member states.

    A device's common name is its last names entry; its first is its source
    name, which the graph view reads; a device with no names is named by its id.
    """
    names = [str((member.get("names") or [member["id"]])[-1]) for member in members]
    setup: dict[str, Any] = {"CommonNames": names}
    for attribute in SETUP_ATTRIBUTES:
        values = [(member.get("attributes") or {}).get(attribute) for member in members]
        if all(value is not None for value in values):
            setup[attribute] = values
    return setup


def _family_fields(
    members: Sequence[Mapping[str, Any]],
    channels_of: Mapping[str, Sequence[Mapping[str, Any]]],
    signals: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], int]:
    """A family's Fields under one System, and how many channels are keyed by address."""
    per_member: list[dict[str | None, list[str]]] = []
    roles: set[str] = set()
    for member in members:
        fields: dict[str | None, list[str]] = defaultdict(list)
        for channel in channels_of.get(str(member["id"]), ()):
            address = str(channel["id"])
            signal = channel.get("signal")
            key = _signal_key(address, signals)
            if key is None and signal:
                key = str(signal)
                roles.add(key)
            fields[key].append(address)
        per_member.append(fields)

    keys = {key for fields in per_member for key in fields}
    cells: dict[str, list[str]] = {}
    by_address = 0
    for key in sorted(keys, key=lambda key: (key is None, key or "")):
        if key is not None and all(len(fields.get(key, ())) == 1 for fields in per_member):
            cells[key] = [fields[key][0] for fields in per_member]
            continue
        for fields in per_member:
            for address in fields.get(key, ()):
                cells[address] = [address]
                by_address += 1

    out: dict[str, dict[str, Any]] = {}
    for key, addresses in cells.items():
        field: dict[str, Any] = {"ChannelNames": addresses}
        sentence = _cell_sentence(addresses, signals, key if key in roles else None)
        if sentence:
            field["_description"] = sentence
        out[key] = field
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
            f"its System key `{system}` is the document key of the middle-layer index",
            f"give the place an id other than `{system}`, or select another channel_finder_mode",
        )
    _checked_key(system, place, "place", _MIDDLE_LAYER_INDEX)


def middle_layer_document(doc: Mapping[str, Any]) -> tuple[dict[str, Any], int, int]:
    """The middle-layer index of one facility file.

    Args:
        doc: The facility file.

    Returns:
        ``{schema, <System>: {<Family>: {...}}}``; the number of channels in no
        family, which the index leaves out; and the number of channels the
        index keys by their address.

    Raises:
        FacilityBuildError: ``view-unsupported`` when a System or Family key
            begins with ``_``, or a System key is ``schema``.
    """
    places = {str(place["id"]): place for place in doc.get("places", [])}

    channels_of: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for channel in doc.get("channels", []):
        owners = [(channel.get("on") or {}).get("device"), *(channel.get("endpoint_of") or [])]
        for owner in dict.fromkeys(str(owner) for owner in owners if owner):
            channels_of[owner].append(channel)

    in_family: set[str] = set()
    document: dict[str, Any] = {"schema": CHANNEL_FINDER_SCHEMA}
    by_address = 0
    for system, entries in _families_by_system(doc).items():
        place = places.get(system)
        if system != ABSENT:
            _checked_system(system, place or {"id": system})
        node: dict[str, Any] = {}
        if system == ABSENT:
            node["_description"] = "no place"
        elif place is not None:
            description = place.get("description") or next(iter(place.get("names") or []), None)
            if description:
                node["_description"] = description
        for group, name, members in entries:
            _checked_key(name, group, "group", _MIDDLE_LAYER_INDEX)
            for member in members:
                in_family.update(str(c["id"]) for c in channels_of.get(str(member["id"]), ()))
            fields, keyed = _family_fields(members, channels_of, group.get("signals") or {})
            by_address += keyed
            family: dict[str, Any] = {"_setup": _setup(members), **fields}
            description = group.get("description") or next(iter(group.get("names") or []), None)
            if description:
                family["_description"] = description
            node[name] = family
        document[system] = node

    left_out = sum(1 for channel in doc.get("channels", []) if str(channel["id"]) not in in_family)
    return document, left_out, by_address


def write_middle_layer(root: Path, inputs: ViewInputs) -> list[Path]:
    """Write the middle-layer index and its DuckDB database into ``root``.

    The channels the index leaves out or keys by address are counted in one
    note on stderr.

    Args:
        root: The render's ``data/channel_finder`` directory.
        inputs: The render's view inputs.

    Returns:
        The files written: the index and its database.

    Raises:
        FacilityBuildError: ``view-unsupported`` when the facility has no
            group, or when DuckDB cannot write the database (its
            full-text-search extension is neither installed nor reachable).
    """
    from osprey.facility.views import report_note, view_bytes

    document, left_out, by_address = middle_layer_document(inputs.doc)
    if len(document) == 1:
        raise _mode_unsupported(
            f"selects {MIDDLE_LAYER_MODE} and the facility has no group",
            "add at least one group, or select another channel_finder_mode",
        )
    root.mkdir(parents=True, exist_ok=True)
    target = root / MIDDLE_LAYER_FILE
    target.write_bytes(view_bytes(document))
    database = root / MIDDLE_LAYER_DUCKDB_FILE
    database.unlink(missing_ok=True)

    import duckdb

    from osprey.services.channel_finder.databases.duckdb_import import import_to_duckdb

    try:
        import_to_duckdb(str(target), str(database))
    except duckdb.Error as exc:
        database.unlink(missing_ok=True)
        raise _mode_unsupported(
            f"selects {MIDDLE_LAYER_MODE} and its DuckDB database cannot be written ({exc})",
            "install DuckDB's `fts` extension on this host, or select another channel_finder_mode",
        ) from exc
    if left_out or by_address:
        report_note(
            inputs,
            f"view {MIDDLE_LAYER_MODE}: {left_out} channels in no family left out, "
            f"{by_address} keyed by address",
        )
    return [target, database]
