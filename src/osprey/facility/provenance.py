"""The one builder of a record's ``provenance`` block.

Every record of the facility file carries::

    provenance:
      sources:  [{layer, file, fields}]   one per (layer, file), sorted by both
      fixes:    [{op, why}]               the fixes applied, in fixes.yaml order
      defaults: [field]                   the fields the build filled, sorted
      place_from: span | authored | fix   a device's place, when it has one

Views read only this block, so every stage that adds to a record's history goes
through the functions here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = ["PLACE_FROM", "add_defaults", "build_provenance", "set_place_from"]

#: Where a device's ``place`` may come from.
PLACE_FROM: tuple[str, ...] = ("span", "authored", "fix")


def build_provenance(
    *,
    sources: Iterable[tuple[str, str, Iterable[str]]] = (),
    fixes: Iterable[tuple[str, str]] = (),
    defaults: Iterable[str] = (),
    place_from: str | None = None,
) -> dict[str, Any]:
    """Build a provenance block.

    Args:
        sources: ``(layer, file, fields)`` for each source that stated a field;
            entries naming the same layer and file are joined.
        fixes: ``(op, why)`` for each fix applied, in fixes.yaml order.
        defaults: The fields the build filled.
        place_from: Where a device's place came from, one of ``PLACE_FROM``.

    Returns:
        The block, with ``place_from`` present only when given.
    """
    joined: dict[tuple[str, str], set[str]] = {}
    for layer, file, fields in sources:
        joined.setdefault((layer, file), set()).update(fields)
    block: dict[str, Any] = {
        "sources": [
            {"layer": layer, "file": file, "fields": sorted(fields)}
            for (layer, file), fields in sorted(joined.items())
        ],
        "fixes": [{"op": op, "why": why} for op, why in fixes],
        "defaults": sorted(set(defaults)),
    }
    if place_from is not None:
        return set_place_from(block, place_from)
    return block


def add_defaults(provenance: dict[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    """Return the block with more filled fields recorded.

    Args:
        provenance: A block from ``build_provenance``.
        fields: The fields the build filled.

    Returns:
        A new block; ``provenance`` is left as it was.
    """
    return {**provenance, "defaults": sorted(set(provenance.get("defaults", ())) | set(fields))}


def set_place_from(provenance: dict[str, Any], place_from: str) -> dict[str, Any]:
    """Return the block with a device's place origin recorded.

    Args:
        provenance: A block from ``build_provenance``.
        place_from: One of ``PLACE_FROM``.

    Returns:
        A new block; ``provenance`` is left as it was.

    Raises:
        ValueError: ``place_from`` is not one of ``PLACE_FROM``.
    """
    if place_from not in PLACE_FROM:
        raise ValueError(f"place_from must be one of {PLACE_FROM}, not {place_from!r}")
    return {**provenance, "place_from": place_from}
