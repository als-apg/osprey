"""The one exception the ontology compiler raises for schemas it rejects.

Every compiler-owned rejection — an unreadable file, YAML the parser cannot
follow, a slot LinkML does not recognise, a class that breaks the authoring
rules — surfaces as :class:`OntologyCompileError`, so a caller (the CLI, a
test) has a single type to catch.  Structural complaints about the *table* the
schema compiles down to keep their existing type,
:class:`~osprey.services.facility_knowledge.ttl_generator.ontology_map.OntologyMapError`;
the split is deliberate, because the two name different things an operator has
to fix.

The error carries the source file rather than folding it into the message, so
callers can report the path their own way, and ``str()`` still opens with the
file name an operator has to edit.
"""

from __future__ import annotations

from pathlib import Path


class OntologyCompileError(ValueError):
    """Raised when a LinkML schema cannot be compiled into an ontology table.

    Subclasses :class:`ValueError` to match
    :class:`~osprey.services.facility_knowledge.ttl_generator.ontology_map.OntologyMapError`,
    so a caller that already treats a malformed table as a value error keeps
    working.

    Args:
        source: The schema file that was being compiled.  Kept whole on the
            exception; only its name reaches :meth:`__str__`.
        message: What is wrong with the schema, phrased for the person who has
            to edit it.
    """

    def __init__(self, source: Path, message: str) -> None:
        super().__init__(message)
        self.source = source
        self.message = message

    def __str__(self) -> str:
        """Return ``"<file name>: <message>"``.

        The file name — never the full path — leads, because compiler errors
        are read next to a schema an operator opened by name, and an absolute
        path differs per checkout.
        """
        return f"{self.source.name}: {self.message}"
