"""The ontology branches a mapping may name, read from the packaged vocabulary.

The class set is the facility vocabulary's generated table
(``osprey/facility/schema/_generated/vocabulary.json``), so the mapping checker,
the ontology emitter and the facility build share one spelling of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources

from osprey.facility import PN_LOCAL

__all__ = ["ROOT_CLASS", "PackagedClass", "is_pn_local", "packaged_classes"]


@dataclass(frozen=True)
class PackagedClass:
    """One class of the packaged vocabulary.

    Attributes:
        name: The class name.
        parent: The parent class, ``None`` only for the root.
        alt_labels: The vocabulary's aliases for the class, in table order.
        iri: The class IRI.
    """

    name: str
    parent: str | None
    alt_labels: tuple[str, ...]
    iri: str


@cache
def _table() -> tuple[PackagedClass, ...]:
    path = resources.files("osprey.facility.schema._generated") / "vocabulary.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["classes"]
    return tuple(
        PackagedClass(
            name=row["name"],
            parent=row["parent"],
            alt_labels=tuple(row.get("aliases") or ()),
            iri=row["iri"],
        )
        for row in rows
    )


def _root_class() -> str:
    # The generated vocabulary holds exactly one parentless class.
    (root,) = (klass.name for klass in _table() if klass.parent is None)
    return root


ROOT_CLASS: str = _root_class()
"""Name of the single parentless class in the packaged table."""


def packaged_classes() -> dict[str, PackagedClass]:
    """Return every packaged class except the root, keyed by class name.

    The returned dict is a fresh copy; mutating it does not affect the table.
    """
    return {klass.name: klass for klass in _table() if klass.parent is not None}


def is_pn_local(token: str) -> bool:
    """Return whether ``token`` is a valid Turtle local name for generated IRIs."""
    return PN_LOCAL.fullmatch(token) is not None
