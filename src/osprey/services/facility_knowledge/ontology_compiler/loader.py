"""Reading an authored LinkML schema into a :class:`~linkml_runtime.SchemaView`.

This is the compiler's first stage: it turns a path into a loaded schema, and
turns every way that can fail into one :class:`.OntologyCompileError` naming the
file.  ``linkml_runtime`` reports a misspelled slot as a :class:`TypeError` from
a generated ``__init__``, malformed YAML as a :class:`yaml.YAMLError`, and a
missing file as an :class:`OSError`; none of those read like an authoring
mistake on their own, and none of them name the schema in a way the CLI can
print.  Wrapping them here means the later stages, and the CLI, only ever have
to know about one exception type.

**Why this package sits beside** :mod:`~osprey.services.facility_knowledge.ttl_generator`
**rather than inside it.**  ``linkml_runtime`` imports ``rdflib``, and
``tests/services/facility_knowledge/test_import_isolation.py`` walks every
module under ``ttl_generator/`` with :meth:`~pathlib.Path.rglob` and spawns a
subprocess per module asserting that importing it leaves ``rdflib`` (and
``neo4j``) out of ``sys.modules``.  A compiler module dropped into that package
would be discovered automatically and would turn that suite red.  The runtime
read path — ``ontology_map`` loading a compiled JSON table — must stay free of
the authoring toolchain, so the authoring toolchain lives in its own package.

The same discipline applies inside *this* package: every ``linkml_runtime``
import is function-local, so importing
:mod:`osprey.services.facility_knowledge.ontology_compiler` costs nothing and
pulls in no ``rdflib``.  Only calling :func:`load_schema` does.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from .errors import OntologyCompileError

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from linkml_runtime import SchemaView

#: Exceptions ``linkml_runtime`` and its YAML loader raise for a schema an
#: author got wrong.  ``TypeError`` is the interesting one: an unrecognised slot
#: (``aliasees:`` for ``aliases:``) reaches a generated dataclass ``__init__``
#: as an unexpected keyword argument, and the message even suggests the
#: correction — worth carrying through verbatim.
_LOAD_FAILURES = (OSError, ValueError, TypeError, yaml.YAMLError)


def load_schema(source: Path) -> SchemaView:
    """Load a LinkML schema, wrapping every failure as an authoring error.

    The schema is loaded eagerly: ``SchemaView`` construction parses the file,
    and :meth:`~linkml_runtime.SchemaView.imports_closure` resolves the
    ``imports`` block.  Both happen inside this call so a broken schema fails
    here, with a message naming the file, rather than several stages later.

    ``imports: [linkml:types]`` resolves offline — ``linkml_runtime`` ships the
    LinkML metamodel schemas as package data and maps the ``linkml:`` prefix to
    them locally, so no network access is involved.

    Args:
        source: Path to the ``.yaml`` schema.

    Returns:
        The loaded :class:`~linkml_runtime.SchemaView`.

    Raises:
        OntologyCompileError: The file is missing or unreadable, its YAML does
            not parse, or ``linkml_runtime`` rejects its contents.  The
            underlying message is carried through unchanged.
    """
    from linkml_runtime import SchemaView  # noqa: PLC0415 - keeps rdflib off the import path

    try:
        view = SchemaView(str(source))
        _ = view.schema
        view.imports_closure()
    except _LOAD_FAILURES as exc:
        raise OntologyCompileError(source, str(exc)) from exc
    return view
