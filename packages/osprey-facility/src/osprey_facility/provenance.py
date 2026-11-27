"""The one builder of a record's ``provenance`` block.

Every record of the facility file carries::

    provenance:
      sources:  [{layer, file, fields, basis?, rule?}]   one per (layer, file, basis,
                                          rule), sorted; ``basis: inferred`` and its
                                          ``rule`` when the record said its fields
                                          rest on a named rule, absent when stated
      fixes:    [{op, why}]               the fixes applied, in fixes.yaml order
      defaults: [field]                   the fields the build filled, sorted
      rules:    [{fields, rule}]          of those, the ones the build derived by a
                                          named rule (absent when none)
      place_from: span | authored | fix | imported | relation   a device's place, when it has one

Views read only this block, so every stage that adds to a record's history goes
through the functions here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, NamedTuple

__all__ = ["PLACE_FROM", "Stating", "add_defaults", "build_provenance", "set_place_from"]


class Stating(NamedTuple):
    """One layer file's statement of some of a record's fields.

    Attributes:
        layer: The layer.
        file: The file, relative to ``facility/``.
        fields: The fields it states.
        basis: ``inferred`` when the record said a named rule derived them, else ``None``.
        rule: That rule.
    """

    layer: str
    file: str
    fields: Iterable[str]
    basis: str | None = None
    rule: str | None = None


#: Where a device's ``place`` may come from.
PLACE_FROM: tuple[str, ...] = ("span", "authored", "fix", "imported", "relation")


def build_provenance(
    *,
    sources: Iterable[tuple[str, str, Iterable[str]] | Stating] = (),
    fixes: Iterable[tuple[str, str]] = (),
    defaults: Iterable[str] = (),
    place_from: str | None = None,
    rules: Iterable[tuple[str, Iterable[str]]] = (),
) -> dict[str, Any]:
    """Build a provenance block.

    Args:
        sources: ``(layer, file, fields)``, or a :class:`Stating` with its basis
            and rule, for each source that stated a field; entries naming the
            same layer, file, basis and rule are joined.
        fixes: ``(op, why)`` for each fix applied, in fixes.yaml order.
        defaults: The fields the build filled.
        place_from: Where a device's place came from, one of ``PLACE_FROM``.
        rules: ``(rule, fields)`` for the fields the build derived by a named rule.

    Returns:
        The block, with ``place_from`` and ``rules`` present only when given.
    """
    joined: dict[tuple[str, str, str, str], set[str]] = {}
    for source in sources:
        stating = Stating(*source)
        key = (stating.layer, stating.file, stating.basis or "", stating.rule or "")
        joined.setdefault(key, set()).update(stating.fields)
    block: dict[str, Any] = {
        "sources": [
            {
                "layer": layer,
                "file": file,
                "fields": sorted(fields),
                **({"basis": basis, "rule": rule} if basis else {}),
            }
            for (layer, file, basis, rule), fields in sorted(joined.items())
        ],
        "fixes": [{"op": op, "why": why} for op, why in fixes],
        "defaults": sorted(set(defaults)),
    }
    derived: dict[str, set[str]] = {}
    for rule, fields in rules:
        derived.setdefault(rule, set()).update(fields)
    if derived:
        block["rules"] = [
            {"fields": sorted(fields), "rule": rule} for rule, fields in sorted(derived.items())
        ]
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
