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
the sentence its device's family group keeps for the channel's kind of signal
(the group's ``signals`` key, longest first, whose ``/``-separated parts are the
last alphanumeric runs of the address), else the channel's own. A place node is
described by its place, a class node by the smallest described group holding
every device under it (else by its class), and a device node by its device
(else by the smallest described group naming it). A render carries the index
when its ``channel_finder.pipeline_mode`` is ``hierarchical``; two place level
words that first appear at one depth stop the build with ``view-unsupported``.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from osprey.facility.views import ViewInputs

__all__ = [
    "ABSENT",
    "CHANNEL_FINDER_SCHEMA",
    "HIERARCHICAL_FILE",
    "HIERARCHICAL_MODE",
    "IN_CONTEXT_FILE",
    "IN_CONTEXT_TAG",
    "TAIL_LEVELS",
    "hierarchical_document",
    "hierarchical_selected",
    "in_context_document",
    "in_context_selected",
    "write_hierarchical",
    "write_in_context",
]

CHANNEL_FINDER_SCHEMA = "osprey.facility.channel_finder/1"
IN_CONTEXT_FILE = "in_context.json"
IN_CONTEXT_TAG = "in_context"
HIERARCHICAL_FILE = "hierarchical.json"
HIERARCHICAL_MODE = "hierarchical"

#: The node standing for an absent place, class or device.
ABSENT = "-"

#: The levels below the place levels, in order.
TAIL_LEVELS: tuple[str, ...] = ("class", "device", "leaf")

#: A level word the naming pattern can carry as a placeholder.
_LEVEL_WORD = re.compile(r"\w+")

#: One alphanumeric run of an address or of a ``signals`` key.
_ALNUM_RUN = re.compile(r"[A-Za-z0-9]+")

#: The rendered config key that selects a pipeline.
PIPELINE_MODE_KEY = "channel_finder.pipeline_mode"


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
    return _pipeline_mode(inputs) == IN_CONTEXT_TAG


def _pipeline_mode(inputs: ViewInputs) -> str | None:
    channel_finder = inputs.rendered_config.get("channel_finder")
    if not isinstance(channel_finder, Mapping):
        return None
    mode = channel_finder.get("pipeline_mode")
    return mode if isinstance(mode, str) else None


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
    from osprey.facility.errors import FacilityBuildError
    from osprey.facility.served import CONFIG_SOURCE
    from osprey.facility.views import view_bytes

    document = in_context_document(inputs.doc)
    if not document["channels"]:
        raise FacilityBuildError(
            "view-unsupported",
            PIPELINE_MODE_KEY,
            [CONFIG_SOURCE],
            f"tag at least one channel `{IN_CONTEXT_TAG}`, or select another channel_finder_mode",
            record_kind="path",
            detail=f"selects {IN_CONTEXT_TAG} and no channel is tagged `{IN_CONTEXT_TAG}`",
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


def _record_sources(record: Mapping[str, Any]) -> list[str]:
    provenance = record.get("provenance") or {}
    return sorted({str(source["file"]) for source in provenance.get("sources", [])})


def _unsupported(place: Mapping[str, Any], detail: str, remedy: str) -> Exception:
    from osprey.facility.errors import FacilityBuildError

    return FacilityBuildError(
        "view-unsupported",
        str(place["id"]),
        _record_sources(place),
        remedy,
        record_kind="place",
        detail=detail,
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
                f"level `{ancestor['level']}` sits below level `{levels[last]}` "
                "against the order of their shallowest depths",
                "order the level words the same way on every branch, "
                "or select another channel_finder_mode",
            )
        keys[position] = ancestor_id[len(parent) + 1 :] if parent else ancestor_id
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
            key_runs = [run for part in str(key).split("/") for run in _alnum_runs(part)]
            if not key_runs or runs[-len(key_runs) :] != key_runs:
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
            or when one branch orders two level words against their depths.
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
        class_name = device.get("class") if device is not None else None
        class_node = (*keys, str(class_name) if class_name else ABSENT)
        device_node = (*class_node, str(device["id"]) if device is not None else ABSENT)
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
            be ordered into one hierarchy.
    """
    from osprey.facility.views import view_bytes

    document = hierarchical_document(inputs.doc)
    root.mkdir(parents=True, exist_ok=True)
    target = root / HIERARCHICAL_FILE
    target.write_bytes(view_bytes(document))
    return [target]
