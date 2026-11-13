"""Claim ids for imported relation claims.

A claim id is ``<layer>/<type>/<hash>``. The hash covers what makes a claim the
claim it is: its type, both ends (the record an end names, its ``port`` and its
``unit``), the key attributes the type's ``single_per`` names, and the citation's
``source.document``. It never covers the citation's ``version``, ``locator``,
``state`` or ``imported`` date, nor any attribute that is not a key attribute, so
a new document version, a re-pagination or a re-import keeps every id.

The ends of a symmetric type are sorted before hashing, so a row written in
either direction is one claim. Two rows that differ only in the object are two
claims; one fact cited from two documents is two claims.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

__all__ = ["HASH_LENGTH", "claim_id"]

#: Hex characters of the sha256 digest kept in a claim id.
HASH_LENGTH = 12

#: The kinds one end may name, and the end fields hashed with it.
_END_FIELDS: tuple[str, ...] = ("device", "place", "group", "port", "unit")

_ATTRIBUTE_PREFIX = "attributes."


def _end(claim: Mapping[str, Any], side: str) -> dict[str, Any]:
    end = claim.get(side)
    if not isinstance(end, Mapping):
        return {}
    return {name: _text(end[name]) for name in _END_FIELDS if end.get(name) is not None}


def _text(value: Any) -> Any:
    # A port read as 3 from YAML and "3" from CSV is one port.
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _key_attributes(claim: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    attributes = claim.get("attributes")
    if not isinstance(attributes, Mapping):
        return {}
    found: dict[str, Any] = {}
    for field in row.get("single_per") or []:
        if isinstance(field, str) and field.startswith(_ATTRIBUTE_PREFIX):
            name = field.removeprefix(_ATTRIBUTE_PREFIX)
            if attributes.get(name) is not None:
                found[name] = attributes[name]
    return found


def _document(claim: Mapping[str, Any]) -> Any:
    source = claim.get("source")
    return source.get("document") if isinstance(source, Mapping) else None


def claim_id(layer: str, claim: Mapping[str, Any], types: Mapping[str, Mapping[str, Any]]) -> str:
    """The id of *claim* in *layer*.

    Args:
        layer: The imported layer's name.
        claim: One relation claim: ``type``, ``subject``, ``object``,
            ``attributes`` and ``source``.
        types: The relation type rows keyed by name, as
            :func:`osprey.facility.validate.relation_types` returns them. A type
            missing from it is read as directed with no key attributes.

    Returns:
        ``<layer>/<type>/<hash>``, the hash the first :data:`HASH_LENGTH` hex
        characters of a sha256 over the claim's identifying fields.
    """
    type_ = str(claim.get("type"))
    row = types.get(type_) or {}
    ends = [_end(claim, "subject"), _end(claim, "object")]
    if row.get("symmetric"):
        ends.sort(key=_canonical)
    basis = {
        "type": type_,
        "ends": ends,
        "attributes": _key_attributes(claim, row),
        "document": _document(claim),
    }
    digest = hashlib.sha256(_canonical(basis).encode("utf-8")).hexdigest()[:HASH_LENGTH]
    return f"{layer}/{type_}/{digest}"
