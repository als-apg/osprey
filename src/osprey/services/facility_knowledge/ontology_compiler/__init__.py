"""Compiles an authored LinkML schema into the ontology table the runtime reads.

``osprey knowledge compile-ontology SOURCE OUTPUT`` runs the four stages this
package holds: :mod:`.loader` reads the schema, :mod:`.payload` turns it into
the table shape
:func:`~osprey.services.facility_knowledge.ttl_generator.ontology_map.parse_ontology`
already accepts, :mod:`.compile` validates it through that same function, and
:mod:`.render` writes deterministic JSON bytes.  :mod:`.check` re-renders in
memory and diffs against the committed artifact, so CI can prove the checked-in
table still matches the schema it claims to come from.

The compiler is an **authoring-time** tool.  The runtime read path stays where
it is, in :mod:`~osprey.services.facility_knowledge.ttl_generator`, reading the
compiled JSON with the standard library alone.  This package is a *sibling* of
that one, not a module inside it, because ``linkml_runtime`` imports ``rdflib``
and ``tests/services/facility_knowledge/test_import_isolation.py`` discovers
every module under ``ttl_generator/`` by :meth:`~pathlib.Path.rglob` and proves
that importing it pulls in neither ``rdflib`` nor ``neo4j``.

Importing *this* package is free.  The re-exports below are resolved lazily
through :func:`__getattr__`, and every ``linkml_runtime`` import in the package
is function-local, so ``import
osprey.services.facility_knowledge.ontology_compiler`` never loads the LinkML
toolchain — and never loads ``rdflib`` with it.  That also lets the package be
imported while a submodule that a caller does not touch is absent or failing.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from .check import check_artifact
    from .compile import CompiledOntology, compile_schema
    from .errors import OntologyCompileError
    from .loader import load_schema
    from .payload import schema_to_payload
    from .render import GENERATED_HEADER, render_json

#: The package's public surface.
__all__ = [
    "GENERATED_HEADER",
    "CompiledOntology",
    "OntologyCompileError",
    "check_artifact",
    "compile_schema",
    "render_json",
]

#: Attribute name -> submodule it lives in.  Covers the public names in
#: :data:`__all__` plus the two per-stage entry points a caller may want
#: directly, so every submodule of the package is reachable by name.
_EXPORTS = {
    "OntologyCompileError": "errors",
    "load_schema": "loader",
    "schema_to_payload": "payload",
    "CompiledOntology": "compile",
    "compile_schema": "compile",
    "render_json": "render",
    "GENERATED_HEADER": "render",
    "check_artifact": "check",
}


def __getattr__(name: str) -> Any:
    """Resolve a re-export by importing its submodule on first access.

    Args:
        name: Attribute being looked up on the package.

    Returns:
        The attribute, read from the submodule that defines it.

    Raises:
        AttributeError: *name* is not one of the package's exports.
    """
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    """List the package's exports alongside its real attributes."""
    return sorted(set(globals()) | set(_EXPORTS))
