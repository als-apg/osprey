"""Driving the compiler's stages and validating the table they produce.

This module is the compiler's spine: it reads the schema (:mod:`.loader`),
translates it (:mod:`.payload`), and then hands the result to
:func:`~osprey.services.facility_knowledge.ttl_generator.ontology_map.parse_ontology`
— the *same* function the runtime uses to read the committed JSON artifact.
That reuse is the point.  Every structural rule the runtime enforces on a table
— exactly one parentless class and it must be the root, every parent declared,
every family mapped to a class that exists, no cycles — is enforced on the
compiler's output by the very code that will later read it, so an artifact this
package writes cannot fail to load. Re-implementing those checks here would mean
two sets of rules to keep in step, and the day they drifted the compiler would
emit a table the runtime rejects.

The cost of that reuse is a message written for the wrong reader.
``parse_ontology`` speaks in the vocabulary of the compiled *table* — ``parent``,
``classes``, ``family_to_class`` — because that is what its usual caller is
holding.  Someone compiling a schema is holding YAML, where those things are
spelled ``is_a`` and ``DeviceFamily``.  So the error is re-raised, as the same
type, with :data:`LINKML_PREAMBLE` prepended and the original chained onto it:
one sentence of translation in front of a message that is otherwise carried
through word for word.  Rewriting the message instead would put the compiler in
the business of paraphrasing rules it does not own.

The two error types stay distinct on purpose, and the delegation order decides
which one a caller sees.  A schema the loader or the translator rejects raises
:class:`.OntologyCompileError` and never reaches ``parse_ontology`` — those are
authoring faults visible in the YAML alone.  Only a schema that is *valid
LinkML* yet describes a table that will not stand up gets
:class:`~osprey.services.facility_knowledge.ttl_generator.ontology_map.OntologyMapError`.

Like the rest of the package, this module imports no ``linkml_runtime`` at
module scope — the toolchain loads only when :func:`compile_schema` calls into
the loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..ttl_generator.ontology_map import OntologyMap, OntologyMapError, parse_ontology
from .loader import load_schema
from .payload import schema_to_payload

#: Prepended to every :class:`OntologyMapError` raised while validating a
#: compiled payload.  ``parse_ontology`` names the table's own field spellings;
#: this sentence maps them back onto the schema the reader has open, so the
#: complaint about a "parentless class" lands on the ``is_a`` line that caused
#: it.  The trailing space is part of the constant: the original message follows
#: it directly.
LINKML_PREAMBLE = "In LinkML, `is_a` is the parent and the `DeviceFamily` enum is the family map. "


@dataclass(frozen=True)
class CompiledOntology:
    """A schema that compiled, together with both forms of what it compiled to.

    Both forms are kept because the callers want different ones and neither
    should have to redo the work: :func:`~.render.render_json` serialises
    *payload*, while a caller that wants to ask questions of the result — which
    class does this family map to, what are its ancestors — wants *table*.
    Recomputing either from the other would mean re-running a stage.

    Frozen so a compiled result cannot be edited between the validation that
    approved it and the artifact written from it.  Note that freezing the
    dataclass does not freeze *payload*'s contents; the guarantee is that this
    object keeps pointing at the payload that was validated.

    Args:
        source: Schema file this was compiled from.  Carried so a caller can
            name the file — in an error, in a generated header — without having
            to thread the path alongside the result.
        payload: The ontology table as decoded-JSON data, ready for
            :func:`~.render.render_json`.
        table: The same table parsed and validated, as the runtime would read
            it back.
    """

    source: Path
    payload: dict[str, object]
    table: OntologyMap


def compile_schema(source: Path) -> CompiledOntology:
    """Compile a LinkML schema into a validated ontology table.

    Runs the three stages in order — load, translate, validate — and stops at
    the first one that objects.  The order is what gives the two error types
    their meaning: anything wrong with the *schema* surfaces as
    :class:`.OntologyCompileError` before validation is ever reached, so an
    :class:`~osprey.services.facility_knowledge.ttl_generator.ontology_map.OntologyMapError`
    from this function always means the schema was well-formed and the ontology
    it describes was not.

    Args:
        source: Path to the ``.yaml`` schema to compile.

    Returns:
        The validated result, carrying the payload to render and the table it
        parses into.

    Raises:
        OntologyCompileError: The schema could not be read, or uses something
            the ontology table cannot represent.
        OntologyMapError: The schema is valid LinkML but describes a table that
            does not validate — a dangling parent, a family mapping to a class
            that is not declared, more or fewer than one parentless class.  The
            message is the original prefixed with :data:`LINKML_PREAMBLE`, and
            the original is chained as ``__cause__``.
    """
    payload = schema_to_payload(load_schema(source), source)
    try:
        table = parse_ontology(payload, source_path=source)
    except OntologyMapError as exc:
        raise OntologyMapError(f"{LINKML_PREAMBLE}{exc}") from exc
    return CompiledOntology(source=source, payload=payload, table=table)
