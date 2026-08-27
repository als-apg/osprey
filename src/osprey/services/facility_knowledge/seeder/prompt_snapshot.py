"""Seed-time schema snapshot, baked into the graph agent's rendered prompt.

The facility-knowledge-graph agent — and the channel finder in its graph
paradigm, which reads the same store — needs the store's vocabulary: labels,
relationship types, property spellings — before it can write a Cypher query
that returns rows instead of a confident zero. At run time that vocabulary
comes from the ``get_schema`` and ``example_queries`` tools; this module moves
the common case earlier, so a fresh subagent starts already knowing it. Each
agent file is baked with the example catalogue its own server serves.

**The seeder owns the baked block, not the build.** ``osprey build`` renders
the agent prompt before any store exists, so it ships a placeholder that tells
the agent to call the tools. Whichever verb then touches the store — the
deploy-time staging step on every ``osprey up``, or ``osprey knowledge
seed-graph`` — captures the schema *from the live store it just verified* and
rewrites the placeholder in every rendered agent file. Sync between prompt and
store is therefore by construction: the writer of one is the writer of the
other, stamped with the same seed-marker checksum, and a rebuild that resets
the block to the placeholder self-heals on the next ``up``.

The capture is **complete where the tool samples**: ``get_schema`` bounds its
per-label property scan because it answers live queries on request, while this
capture runs once per seed and can afford the full walk. Both go through
:func:`collect_schema`, so the bookkeeping exclusions cannot drift apart.

**Three things are captured, and the block says which is which.** The schema
(:func:`collect_schema`); the facility's own class synonyms
(:func:`collect_vocabulary`), which are authored in its ontology rather than
hard-coded here, so a corpus spelling its classes differently gets its own
spellings; and one specimen device (:func:`resolve_example_values`) whose real
name, section and address are substituted into the curated examples. Each of
the latter two is queried under its own guard: a store that answers the schema
but not them still deploys, the examples keep their shipped literals, and the
block labels those *framework defaults* rather than passing them off as facts
about this corpus.

The tools stay registered regardless. They are the recovery path when a query
returns zero rows for a name the snapshot lists (a store re-seeded out of
band), and the only path on a render whose store was never seeded.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from osprey.services.facility_knowledge.seeder.graph_seeder import NARAD_PREFIXES
from osprey.services.facility_knowledge.seeder.ttl_seeder import local_name
from osprey.utils.workspace import BUILD_DIR_NAME, IMAGE_DIR_NAME

logger = logging.getLogger(__name__)

#: One read-only Cypher execution: ``(cypher, params) -> rows``. The tool binds
#: this to ``GraphContext.run_read`` (timeout- and read-enforced); the seeder
#: binds it to its open driver session. Everything in this module that dials
#: the store does so through this seam, so both callers issue byte-identical
#: queries.
RunCypher = Callable[[str, dict[str, Any] | None], list[dict[str, Any]]]

# ---------------------------------------------------------------------------
# Schema collection — shared with the get_schema MCP tool
# ---------------------------------------------------------------------------

#: Labels holding store state rather than facility knowledge: neosemantics'
#: graph config and namespace-prefix nodes, and the seeder's own marker.
BOOKKEEPING_LABELS = frozenset({"_GraphConfig", "_NsPrefDef", "_OspreySeed"})

#: Property names that belong to the bookkeeping nodes above. Filtered on top
#: of the label exclusion as belt-and-braces: were one of these ever to land on
#: a knowledge node, listing it would invite the agent to query the seed marker.
BOOKKEEPING_PROPERTIES = frozenset({"sha256", "seededAt", "kind", "directionSource"})

LABELS_CYPHER = "CALL db.labels() YIELD label RETURN label ORDER BY label"

RELATIONSHIP_TYPES_CYPHER = (
    "CALL db.relationshipTypes() YIELD relationshipType "
    "RETURN relationshipType ORDER BY relationshipType"
)

NAMING_NOTE = (
    "n10s applyNeo4jNaming: relationship types are uppercased (HASBINDING, "
    "READSSIGNAL, WRITESSIGNAL, SUBCLASSOF, TYPE); rdf:type is both a label and "
    "a TYPE edge to a Class node"
)


def is_reportable_label(label: str) -> bool:
    """Whether *label* names facility knowledge rather than store bookkeeping.

    Underscore-prefixed labels are excluded as a class, which already covers
    :data:`BOOKKEEPING_LABELS`; the named set is kept so the exclusion stays
    legible and survives a bookkeeping label that does not follow the
    convention.
    """
    return not label.startswith("_") and label not in BOOKKEEPING_LABELS


def sampled_keys_cypher(label: str) -> str:
    """Build the bounded property-name sample for one label.

    The label is interpolated because Cypher takes no parameter in label
    position. Callers must have rejected any label containing a backtick first —
    that is the only character that could break out of the quoting.
    """
    return (
        f"MATCH (n:`{label}`) WITH n LIMIT $k UNWIND keys(n) AS key "
        "RETURN DISTINCT key ORDER BY key"
    )


def _full_keys_cypher(label: str) -> str:
    """The unbounded variant: every property key any node of *label* carries.

    Same quoting contract as :func:`sampled_keys_cypher`. Used only by the
    seed-time capture, which runs once per seed and can afford the full walk
    the live tool deliberately bounds.
    """
    return f"MATCH (n:`{label}`) UNWIND keys(n) AS key RETURN DISTINCT key ORDER BY key"


def _column(rows: list[dict[str, Any]], key: str) -> list[str]:
    """Pull one column out of query rows, dropping nulls."""
    return [str(row[key]) for row in rows if row.get(key) is not None]


def collect_schema(run: RunCypher, *, sample_size: int | None = None) -> dict[str, Any]:
    """Read the store's queryable vocabulary through *run*.

    Args:
        run: The Cypher seam — see :data:`RunCypher`.
        sample_size: Per-label bound on the property-name scan. ``None`` walks
            every node of every label, which is what the seed-time capture
            wants; the ``get_schema`` tool passes its documented bound.

    Returns:
        The same shape ``get_schema`` serves: ``labels``,
        ``relationship_types``, ``properties_by_label``, ``prefixes`` and
        ``naming`` (whose ``sample_size`` is ``None`` for a complete capture).
    """
    labels = [
        label for label in _column(run(LABELS_CYPHER, None), "label") if is_reportable_label(label)
    ]
    relationship_types = _column(run(RELATIONSHIP_TYPES_CYPHER, None), "relationshipType")

    properties_by_label: dict[str, list[str]] = {}
    for label in labels:
        if "`" in label:
            # Unquotable in label position, so it cannot be scanned safely.
            # Left out of properties_by_label entirely rather than mapped to an
            # empty list, which would claim the label carries no properties;
            # the label itself still appears in ``labels``.
            logger.warning("Skipping property scan for label with a backtick: %r", label)
            continue
        if sample_size is None:
            rows = run(_full_keys_cypher(label), None)
        else:
            rows = run(sampled_keys_cypher(label), {"k": sample_size})
        properties_by_label[label] = [
            key for key in _column(rows, "key") if key not in BOOKKEEPING_PROPERTIES
        ]

    return {
        "labels": labels,
        "relationship_types": relationship_types,
        "properties_by_label": properties_by_label,
        "prefixes": dict(NARAD_PREFIXES),
        "naming": {"note": NAMING_NOTE, "sample_size": sample_size},
    }


# ---------------------------------------------------------------------------
# Vocabulary collection — the facility's own class synonyms
# ---------------------------------------------------------------------------

#: Every ontology class that declares synonyms, with them. The facility authors
#: these as ``aliases`` in its LinkML ontology; ``compile-ontology`` emits them
#: as ``skos:altLabel`` and n10s lands them on ``(c:Class).altLabel``. Capturing
#: them here is what keeps the prompt's vocabulary *this* facility's rather than
#: a table of names hard-coded in the framework, which a corpus spelling its
#: classes differently would silently fail to match.
VOCABULARY_CYPHER = (
    "MATCH (c:Class) WHERE c.altLabel IS NOT NULL "
    "RETURN c.uri AS uri, c.altLabel AS synonyms ORDER BY c.uri"
)


def _class_display_name(uri: str) -> str:
    """The display name of a class IRI: its local name, never empty.

    Classes carry no ``rdfs:label`` in this corpus shape, so the local name is
    what the agent will actually type in a Cypher label position. The split
    itself is :func:`~osprey.services.facility_knowledge.seeder.ttl_seeder.local_name`'s
    — the package's one rule for IRI → local-name derivation, shared rather
    than restated so this table and the stub bundles cannot come to disagree
    about where a name ends. Only the empty case is decided here: a URI ending
    on its separator has no local name, and the whole URI is a better row label
    than a blank one.
    """
    return local_name(uri) or uri


def _as_list(value: Any) -> list[Any]:
    """Coerce a possibly-multivalued store property to a list.

    ``skos:altLabel`` is configured ``handleMultival: ARRAY``, so the store
    normally answers with a list. A store whose n10s config was set up by hand
    can answer with the bare scalar instead; wrapping it costs one branch and
    keeps a single-synonym class from rendering one row per character.
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def collect_vocabulary(run: RunCypher) -> list[dict[str, Any]]:
    """Read the facility's class synonyms through *run*.

    Args:
        run: The Cypher seam — see :data:`RunCypher`.

    Returns:
        One entry per class carrying synonyms, ordered by URI:
        ``{"name": <IRI local part>, "uri": ..., "synonyms": [...]}`` with the
        synonyms sorted and deduplicated so two bakes of one store render
        byte-identically.

        An empty list when the store holds no synonyms **or** when the query
        fails. The guard is the point: :func:`bake_snapshot` runs inside
        ``osprey up``'s staging step, and a store that answers
        :func:`collect_schema` but not this query must still deploy — the block
        then tells the agent to read ``altLabel`` itself.
    """
    try:
        rows = run(VOCABULARY_CYPHER, None)
    except Exception:  # noqa: BLE001 - any store failure degrades to no vocabulary
        logger.warning("Could not capture class synonyms from the store", exc_info=True)
        return []

    vocabulary: list[dict[str, Any]] = []
    for row in rows:
        uri = row.get("uri")
        if uri is None:
            continue
        uri = str(uri)
        synonyms = sorted({str(item) for item in _as_list(row.get("synonyms")) if item is not None})
        vocabulary.append({"name": _class_display_name(uri), "uri": uri, "synonyms": synonyms})
    return vocabulary


# ---------------------------------------------------------------------------
# Example parameter values — one specimen device, resolved live
# ---------------------------------------------------------------------------

#: One real device that actually has a binding, picked deterministically. A
#: single specimen rather than a query per parameter, because it buys internal
#: consistency the per-parameter shape could not: the ``pv`` example's address
#: belongs to the same device the ``name``/``section`` examples name.
SPECIMEN_VALUES_CYPHER = (
    "MATCH (d:Resource)-[:HASBINDING]->(b:ChannelBinding) "
    "WHERE d.sourceName IS NOT NULL AND d.sectionCode IS NOT NULL AND b.fullPv IS NOT NULL "
    "RETURN d.sourceName AS name, d.sectionCode AS section, d.system AS system, "
    "b.fullPv AS pv "
    "ORDER BY d.sectionCode, d.sourceName, b.fullPv LIMIT 1"
)

#: The parameter names the specimen answers. Substitution is keyed on the
#: parameter *name*, which already means the same thing in both catalogues, so
#: the frozen ``ExampleQuery`` never has to change.
SPECIMEN_PARAMETERS = ("name", "section", "system", "pv")


def resolve_example_values(run: RunCypher) -> dict[str, Any]:
    """Resolve the curated examples' corpus-valued parameters through *run*.

    Only parameters that are *facts about the store* are resolved: ``name``,
    ``section``, ``system`` and ``pv``. Search terms (``phrase``,
    ``field_meaning``, ``role``, ``synonym``, …) are deliberately English and
    exist to demonstrate a prose search — "resolving" them would be meaningless.
    Absence from this result is therefore the declaration that a parameter is
    not resolvable; a future example taking a new search term is left alone.

    ``root_uri`` and ``class_uri`` are deliberately **not** resolved either.
    The ontology has more than one parentless class, so no query can pick a
    "root" honestly, and ``class_uri``'s "all magnets" is a domain concept no
    query can pick out of an arbitrary ontology. Both stay framework defaults
    and are labelled as such, which is exactly what the labelling is for.

    Args:
        run: The Cypher seam — see :data:`RunCypher`.

    Returns:
        The resolved values as strings, keyed by parameter name, with any column
        the specimen left null dropped. Coerced rather than passed through: the
        block is rendered by ``json.dumps``, and a driver type that does not
        serialise would fail there — downstream of both guards, taking the whole
        snapshot with it. An empty mapping when the store holds no such device
        or the query fails — guarded for the same reason
        :func:`collect_vocabulary` is: every example then keeps its shipped
        literal and the block labels it a framework default.
    """
    try:
        rows = run(SPECIMEN_VALUES_CYPHER, None)
    except Exception:  # noqa: BLE001 - any store failure degrades to shipped defaults
        logger.warning("Could not resolve example parameter values from the store", exc_info=True)
        return {}

    if not rows:
        return {}
    row = rows[0]
    return {key: str(row[key]) for key in SPECIMEN_PARAMETERS if row.get(key) is not None}


# ---------------------------------------------------------------------------
# Rendering and applying the block
# ---------------------------------------------------------------------------

#: The managed region in the rendered agent prompt. The agent template ships
#: these markers around a placeholder; every bake replaces marker-to-marker,
#: markers included, so applying twice is applying once.
SNAPSHOT_BEGIN = "<!-- osprey:graph-snapshot begin -->"
SNAPSHOT_END = "<!-- osprey:graph-snapshot end -->"

#: The heading the shared template partial puts above the markers. A file that
#: carries it without the marker pair has been hand-edited.
SNAPSHOT_HEADING = "## The Graph at Hand"

#: The rendered agent files this module patches, each paired with the module
#: holding the curated example catalogue its own MCP server serves. One
#: registry, so a file cannot be found without a catalogue or rendered without
#: being found. The catalogues are named rather than imported: both are pure
#: stdlib data, but reaching them through the tools packages must not become
#: an import-time dependency of the seeder.
AGENT_FILENAME = "facility-knowledge-graph.md"
CHANNEL_FINDER_FILENAME = "channel-finder.md"
_CATALOGUE_MODULES: dict[str, str] = {
    AGENT_FILENAME: "osprey.mcp_server.graph.tools.examples_data",
    CHANNEL_FINDER_FILENAME: "osprey.mcp_server.channel_finder_graph.tools.examples_data",
}
TARGET_FILENAMES = tuple(_CATALOGUE_MODULES)


def _example_catalogues() -> dict[str, Sequence[Any]]:
    """The curated catalogue each target file is baked with, keyed by filename."""
    return {
        filename: import_module(module).EXAMPLE_QUERIES
        for filename, module in _CATALOGUE_MODULES.items()
    }


#: Rendered when a store answered the schema but carried no class synonyms —
#: an empty table would read as "this corpus has no synonyms", which is a claim
#: the capture cannot make when the query simply failed.
NO_VOCABULARY_NOTE = (
    "No class synonyms were captured from this corpus; call `get_schema()` and "
    "query `(c:Class).altLabel` directly."
)

#: How to match against a multivalued property. Spelled once, next to the table
#: it applies to: ``altLabel`` is a list, and ``= $term`` against a list is the
#: confident-zero this whole block exists to prevent.
VOCABULARY_MATCH_RULE = (
    "Match with `ANY(l IN c.altLabel WHERE toLower(l) CONTAINS $term)` — "
    "`altLabel` is a list, never compare with `=`."
)

#: The two labels a rendered parameter set can carry. Never unlabelled: an
#: unlabelled value is one the agent has no way to tell apart from a fact about
#: its own corpus, which is precisely how the shipped demo's names ended up
#: being typed at other facilities' stores.
CAPTURED_PARAMETERS_NOTE = "values captured from this corpus"
DEFAULT_PARAMETERS_NOTE = "framework defaults; substitute values from this corpus"

#: How each ``READSSIGNAL``/``WRITESSIGNAL`` edge in *this* corpus came to point
#: the way it does. ``build-ttl`` knows; the seeder carries it in the marker;
#: the agent needs it because the two derivations differ in what they can get
#: wrong — a grammar-derived corpus mislabels any writable channel whose address
#: does not end in ``:SP``. Deliberately worded without the build host's path:
#: the whole rendered prompt is guarded against leaking one.
DIRECTION_PROVENANCE_LINES = {
    "limits": (
        "Direction provenance: read/write edges derived from this facility's channel limits."
    ),
    "grammar": (
        "Direction provenance: read/write edges derived from the address grammar "
        "(`:SP` writes), because no channel limits were available."
    ),
}


def _render_direction(direction_source: str | None) -> list[str]:
    """The provenance line's lines, or none at all.

    A corpus built by an older ``osprey`` recorded no source, and there is no
    honest default to fall back on — the two derivations are not
    interchangeable — so an absent source renders nothing rather than a guess.
    An unrecognised value is printed verbatim: a newer builder's spelling is
    still more informative than silence.
    """
    if not direction_source:
        return []
    known = DIRECTION_PROVENANCE_LINES.get(direction_source)
    return ["", known or f"Direction provenance: `{direction_source}`."]


def _render_vocabulary(vocabulary: Sequence[Mapping[str, Any]]) -> list[str]:
    """The Vocabulary section's lines, table or note."""
    lines = ["### Vocabulary", ""]
    if not vocabulary:
        return [*lines, NO_VOCABULARY_NOTE]

    lines += ["| Class | Synonyms (altLabel) |", "| --- | --- |"]
    for entry in vocabulary:
        synonyms = ", ".join(f"`{synonym}`" for synonym in entry["synonyms"])
        lines.append(f"| `{entry['name']}` | {synonyms or '(none)'} |")
    return [*lines, "", VOCABULARY_MATCH_RULE]


def _render_parameters(parameters: Mapping[str, Any], values: Mapping[str, Any]) -> str:
    """The one Parameters line for an example, substituted and labelled.

    Substitution is by parameter **name**: a key *values* resolved is replaced,
    a key it did not keeps the catalogue's shipped literal. ``ExampleQuery`` is
    frozen and shared by both catalogues, so nothing is mutated — the rendered
    dict is built here and thrown away with the block.
    """
    if not parameters:
        return "Parameters: none"

    rendered = {key: values.get(key, value) for key, value in parameters.items()}
    every_key_captured = all(key in values for key in parameters)
    note = CAPTURED_PARAMETERS_NOTE if every_key_captured else DEFAULT_PARAMETERS_NOTE
    return f"Parameters: `{json.dumps(rendered)}` — {note}"


def render_block(
    schema: dict[str, Any],
    examples: Sequence[Any],
    *,
    digest: str | None,
    resource_count: int,
    vocabulary: Sequence[Mapping[str, Any]] = (),
    values: Mapping[str, Any] | None = None,
    direction_source: str | None = None,
) -> str:
    """Render the snapshot block, markers included.

    Args:
        schema: A :func:`collect_schema` result.
        examples: The curated ``ExampleQuery`` catalogue.
        digest: The seed marker's sha256, or ``None`` on an unmanaged store.
        resource_count: ``(:Resource)`` nodes in the store, for the provenance
            line.
        vocabulary: A :func:`collect_vocabulary` result. Empty renders the note
            rather than an empty table.
        values: A :func:`resolve_example_values` result. Each example's
            parameters are substituted by name from it; whatever it does not
            carry stays the shipped literal and the line says so.
        direction_source: How the corpus's read/write edges were derived, as the
            seed marker recorded it. ``None`` renders no provenance line.
    """
    values = values or {}
    lines: list[str] = [SNAPSHOT_BEGIN, ""]

    stamp = f"corpus checksum `{digest[:12]}`" if digest else "corpus unmanaged (no seed marker)"
    lines += [
        f"Captured from the live store at seed time — {resource_count} Resource "
        f"nodes, {stamp}. Schema, vocabulary and the example parameters marked "
        "*captured* are this corpus's own; parameters marked *framework "
        "defaults* are the shipped catalogue's and must be swapped for values "
        "from this corpus. It is rewritten whenever the store is seeded or "
        "re-verified (`osprey up`, `osprey knowledge seed-graph`). If a name "
        "listed here returns zero rows, or you need vocabulary beyond it, call "
        "`get_schema()` / `example_queries()` — the live store always wins over "
        "this text.",
        *_render_direction(direction_source),
        "",
        "### Schema",
        "",
        f"- **Node labels:** {', '.join(schema['labels'])}",
        f"- **Relationship types:** {', '.join(schema['relationship_types'])}",
        "- **Properties by label** (complete at capture time):",
    ]
    for label, keys in schema["properties_by_label"].items():
        lines.append(f"  - `{label}`: {', '.join(keys) if keys else '(none)'}")
    lines += [
        "- **Prefixes** (for reading and building `uri` values):",
    ]
    for prefix, namespace in schema["prefixes"].items():
        lines.append(f"  - `{prefix}:` → `{namespace}`")
    lines += [
        f"- **Naming:** {schema['naming']['note']}",
        "",
        *_render_vocabulary(vocabulary),
        "",
        "### Curated examples",
        "",
        "Adapt the closest example — swap a parameter value, add a WHERE "
        "clause, widen or narrow the LIMIT — rather than composing a new query "
        "shape. Pass every value through `params`; never paste values into the "
        "query text. Every example ends in a LIMIT, so a truncated result was "
        "truncated by the server's row cap, not by the query.",
    ]

    for example in examples:
        lines += [
            "",
            f"#### {example.key} — {example.title}",
            "",
            example.description,
            "",
            "```cypher",
            example.cypher,
            "```",
        ]
        lines.append(_render_parameters(example.parameters, values))

    lines += ["", SNAPSHOT_END]
    return "\n".join(lines)


def snapshot_targets(render_dir: Path) -> list[Path]:
    """Every rendered graph-querying agent file under *render_dir*.

    Three places, all of them renders of this one deployment: the render's own
    ``.claude/agents/``; one directory level down, where attached persona
    renders (the operator terminals sharing this deployment's store) keep
    theirs; and the container-path copies ``osprey build`` stages as each
    image's build context (:func:`osprey.utils.workspace.container_image_context`),
    which the images are built from.

    Deliberately not a recursive walk: a render carries a ``.venv`` that makes
    ``rglob`` pay for tens of thousands of directories to find a handful of
    files.
    """
    agent_dirs = [
        render_dir / ".claude" / "agents",
        *sorted(render_dir.glob("*/.claude/agents")),
        *sorted(render_dir.glob(f"{IMAGE_DIR_NAME}/*/{BUILD_DIR_NAME}/.claude/agents")),
    ]
    candidates = [agents / filename for agents in agent_dirs for filename in TARGET_FILENAMES]
    return [path for path in candidates if path.is_file()]


def apply_snapshot(render_dir: Path, blocks: Mapping[str, str]) -> list[Path]:
    """Replace the managed region of every target file with its block.

    Args:
        render_dir: The directory holding the rendered ``config.yml``.
        blocks: The rendered block per target filename (:data:`TARGET_FILENAMES`).

    A file without the marker pair is never appended to. One that still shows
    a trace of the section — the heading, or a lone marker — has been
    hand-edited, and growing an unmarked section in someone's edited prompt is
    worse than leaving the tools to answer at run time, so it is reported and
    left alone. One with no trace at all never shipped the section (the
    channel finder outside its graph paradigm) and is skipped quietly.

    Returns:
        The files now carrying their block (rewritten or already current).
    """
    patched: list[Path] = []
    for path in snapshot_targets(render_dir):
        text = path.read_text(encoding="utf-8")
        begin = text.find(SNAPSHOT_BEGIN)
        end = text.find(SNAPSHOT_END)
        if begin == -1 or end < begin:
            if begin != -1 or end != -1 or SNAPSHOT_HEADING in text:
                logger.warning("No snapshot marker pair in %s; leaving it alone", path)
            continue
        end += len(SNAPSHOT_END)
        updated = text[:begin] + blocks[path.name] + text[end:]
        if updated != text:
            path.write_text(updated, encoding="utf-8")
        patched.append(path)
    return patched


def bake_snapshot(session: Any, render_dir: Path) -> list[Path]:
    """Capture the live store's schema and bake it into *render_dir*'s prompts.

    The one entry point both writers share — the deploy-time staging step and
    the ``seed-graph`` verb — so anything that seeds or re-verifies the store
    refreshes the prompt with it.

    Args:
        session: An open driver session on the store just seeded or verified.
        render_dir: The directory holding the rendered ``config.yml``.

    Returns:
        The rendered agent files now carrying the snapshot; empty when the
        render has none (the agents are disabled, the channel finder runs
        another paradigm, or this is a store-only project).
    """
    from osprey.services.facility_knowledge.seeder import graph_seeder

    def run(cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return [record.data() for record in session.run(cypher, params or {})]

    catalogues = _example_catalogues()
    # Three captures, each issued once and rendered once per catalogue: schema,
    # class synonyms and the example specimen are all facts about the *store*
    # and therefore the same for both agents; only the curated examples differ.
    schema = collect_schema(run)
    vocabulary = collect_vocabulary(run)
    values = resolve_example_values(run)
    digest = graph_seeder.read_marker(session)
    resource_count = graph_seeder.resource_count(session)
    direction_source = graph_seeder.read_direction_source(session)
    blocks = {
        filename: render_block(
            schema,
            examples,
            digest=digest,
            resource_count=resource_count,
            vocabulary=vocabulary,
            values=values,
            direction_source=direction_source,
        )
        for filename, examples in catalogues.items()
    }
    return apply_snapshot(render_dir, blocks)
