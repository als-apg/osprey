"""Benchmark channel helpers over hierarchical and middle-layer databases.

Expands a hierarchical channel database into flat channel lists and collects
the addresses a middle-layer database lists.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _expand_instances(expansion_def: dict) -> list[str]:
    """Expand an ``_expansion`` directive into a list of instance names.

    Supports the two expansion types a hierarchical database carries:

    * **range** -- ``_pattern`` + ``_range`` (inclusive on both ends)
    * **list** -- ``_instances`` explicit list
    """
    expansion_type = expansion_def.get("_type")

    if expansion_type == "range":
        pattern = expansion_def.get("_pattern", "{}")
        start, end = expansion_def.get("_range", [1, 1])
        return [pattern.format(i) for i in range(start, end + 1)]

    if expansion_type == "list":
        return list(expansion_def.get("_instances", []))

    return []


def _is_metadata_key(key: str) -> bool:
    """Return True if *key* is a metadata key (starts with ``_``)."""
    return key.startswith("_")


def expand_hierarchy(tree_data: dict) -> list[dict]:
    """Expand a hierarchical channel tree into flat channel entries.

    Traverses the 6-level hierarchy
    (ring -> system -> family -> DEVICE -> field -> subfield)
    and expands all ``_expansion`` directives into concrete channel records.

    Args:
        tree_data: The full JSON object loaded from the hierarchical
            channel database (must contain a ``"tree"`` key).

    Returns:
        Sorted list of dicts, each with keys:
        ``pv``, ``ring``, ``system``, ``family``, ``device``,
        ``field``, ``subfield``.
    """
    tree = tree_data.get("tree", tree_data)
    channels: list[dict] = []

    for ring_name, ring_node in tree.items():
        if _is_metadata_key(ring_name):
            continue

        for system_name, system_node in ring_node.items():
            if _is_metadata_key(system_name):
                continue

            for family_name, family_node in system_node.items():
                if _is_metadata_key(family_name):
                    continue

                # The DEVICE key holds _expansion + field/subfield siblings
                device_node = family_node.get("DEVICE", {})
                expansion = device_node.get("_expansion")
                if expansion is None:
                    continue

                device_names = _expand_instances(expansion)

                # Collect field -> [subfield, ...] from siblings of
                # _expansion inside the DEVICE node
                for field_name, field_node in device_node.items():
                    if _is_metadata_key(field_name):
                        continue
                    if not isinstance(field_node, dict):
                        continue

                    for subfield_name, subfield_node in field_node.items():
                        if _is_metadata_key(subfield_name):
                            continue
                        if not isinstance(subfield_node, dict):
                            continue

                        for device in device_names:
                            pv = ":".join(
                                [
                                    ring_name,
                                    system_name,
                                    family_name,
                                    device,
                                    field_name,
                                    subfield_name,
                                ]
                            )
                            channels.append(
                                {
                                    "pv": pv,
                                    "ring": ring_name,
                                    "system": system_name,
                                    "family": family_name,
                                    "device": device,
                                    "field": field_name,
                                    "subfield": subfield_name,
                                }
                            )

    channels.sort(key=lambda c: c["pv"])
    return channels


def collect_middle_layer_pvs(data: dict) -> set[str]:
    """Recursively collect all PVs from ``ChannelNames`` arrays in a middle-layer DB."""
    pvs: set[str] = set()

    for key, value in data.items():
        if key == "ChannelNames" and isinstance(value, list):
            pvs.update(value)
        elif isinstance(value, dict):
            pvs.update(collect_middle_layer_pvs(value))

    return pvs
