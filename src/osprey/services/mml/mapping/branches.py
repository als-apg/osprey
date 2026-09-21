"""The ontology branches a mapping may name, read from the packaged class table.

The class set is the same validated table ``build-ttl`` reads
(:func:`~osprey.services.facility_knowledge.ttl_generator.ontology_map.load_demo_ontology`),
so the mapping checker and the ontology emitter share one spelling of it.
"""

from __future__ import annotations

from osprey.services.facility_knowledge.ttl_generator.model import PN_LOCAL
from osprey.services.facility_knowledge.ttl_generator.ontology_map import (
    ClassDef,
    load_demo_ontology,
)

__all__ = ["ROOT_CLASS", "is_pn_local", "packaged_classes"]


def _root_class() -> str:
    # The table's own validation guarantees exactly one parentless class.
    (root,) = (k.name for k in load_demo_ontology().classes.values() if k.parent is None)
    return root


ROOT_CLASS: str = _root_class()
"""Name of the single parentless class in the packaged table."""


def packaged_classes() -> dict[str, ClassDef]:
    """Return every packaged class except the root, keyed by class name.

    The returned dict is a fresh copy; mutating it does not affect the table.
    """
    return {
        name: klass
        for name, klass in load_demo_ontology().classes.items()
        if klass.parent is not None
    }


def is_pn_local(token: str) -> bool:
    """Return whether ``token`` is a valid Turtle local name for generated IRIs."""
    return PN_LOCAL.fullmatch(token) is not None
