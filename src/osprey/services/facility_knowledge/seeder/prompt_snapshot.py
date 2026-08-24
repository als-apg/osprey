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
BOOKKEEPING_PROPERTIES = frozenset({"sha256", "seededAt", "kind"})

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
# Corpus detection — which example parameter set matches this store
# ---------------------------------------------------------------------------

#: Every device node carries ``narad_p:facility`` (``"demo"``, ``"ALS"``, …),
#: so the store itself says which curated parameter set fits it. Lowercased in
#: Cypher because the example keys are lowercase by convention.
_FACILITY_CYPHER = (
    "MATCH (d:Resource) WHERE d.facility IS NOT NULL "
    "RETURN DISTINCT toLower(d.facility) AS facility LIMIT 10"
)


def detect_corpus(run: RunCypher, known: frozenset[str] | set[str]) -> str | None:
    """The single example-parameter corpus this store matches, or ``None``.

    ``None`` — a store naming no facility, several, or one the curated examples
    carry no parameters for — keeps every parameter set in the snapshot, with
    the pick-the-matching-one note the ``example_queries`` tool ships.
    """
    facilities = set(_column(run(_FACILITY_CYPHER, None), "facility"))
    matches = facilities & set(known)
    if len(facilities) == 1 and len(matches) == 1:
        return next(iter(matches))
    return None


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


def render_block(
    schema: dict[str, Any],
    examples: Sequence[Any],
    *,
    corpus: str | None,
    digest: str | None,
    resource_count: int,
) -> str:
    """Render the snapshot block, markers included.

    Args:
        schema: A :func:`collect_schema` result.
        examples: The curated ``ExampleQuery`` catalogue.
        corpus: The detected example-parameter corpus, or ``None`` to keep
            every parameter set.
        digest: The seed marker's sha256, or ``None`` on an unmanaged store.
        resource_count: ``(:Resource)`` nodes in the store, for the provenance
            line.
    """
    lines: list[str] = [SNAPSHOT_BEGIN, ""]

    stamp = f"corpus checksum `{digest[:12]}`" if digest else "corpus unmanaged (no seed marker)"
    lines += [
        f"Captured from the live store at seed time — {resource_count} Resource "
        f"nodes, {stamp}. It is rewritten whenever the store is seeded or "
        "re-verified (`osprey up`, `osprey knowledge seed-graph`). If a name "
        "listed here returns zero rows, or you need vocabulary beyond it, call "
        "`get_schema()` / `example_queries()` — the live store always wins over "
        "this text.",
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
        if corpus is not None:
            values = example.parameters.get(corpus, {})
            rendered = f"`{json.dumps(values)}`" if values else "none"
            lines.append(f"Parameters: {rendered}")
        else:
            sets = "; ".join(
                f"`{key}`: `{json.dumps(values)}`" for key, values in example.parameters.items()
            )
            lines.append(f"Parameters (pick the set matching the seeded corpus): {sets or 'none'}")

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
    known = frozenset(
        corpus
        for examples in catalogues.values()
        for example in examples
        for corpus in example.parameters
    )
    # One capture, rendered once per catalogue: the schema is the store's and
    # the same for both agents; only the curated examples differ.
    schema = collect_schema(run)
    corpus = detect_corpus(run, known)
    digest = graph_seeder.read_marker(session)
    resource_count = graph_seeder.resource_count(session)
    blocks = {
        filename: render_block(
            schema, examples, corpus=corpus, digest=digest, resource_count=resource_count
        )
        for filename, examples in catalogues.items()
    }
    return apply_snapshot(render_dir, blocks)
