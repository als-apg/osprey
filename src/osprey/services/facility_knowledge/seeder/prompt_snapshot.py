"""Seed-time schema snapshot, baked into the graph agent's rendered prompt.

The facility-knowledge-graph agent needs the store's vocabulary — labels,
relationship types, property spellings — before it can write a Cypher query
that returns rows instead of a confident zero. At run time that vocabulary
comes from the ``get_schema`` and ``example_queries`` tools; this module moves
the common case earlier, so a fresh subagent starts already knowing it.

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
from collections.abc import Callable
from pathlib import Path
from typing import Any

from osprey.services.facility_knowledge.seeder.graph_seeder import NARAD_PREFIXES

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

#: The rendered agent file this module patches, wherever a render holds one.
AGENT_FILENAME = "facility-knowledge-graph.md"


def render_block(
    schema: dict[str, Any],
    examples: list[Any],
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
    """Every rendered graph-agent file under *render_dir*.

    The render's own ``.claude/agents/`` plus one directory level down, which
    is where attached persona renders (the operator terminals sharing this
    deployment's store) keep theirs. Deliberately not a recursive walk: a
    render carries a ``.venv`` that makes ``rglob`` pay for tens of thousands
    of directories to find at most a handful of files.
    """
    candidates = [render_dir / ".claude" / "agents" / AGENT_FILENAME]
    candidates += sorted(render_dir.glob(f"*/.claude/agents/{AGENT_FILENAME}"))
    return [path for path in candidates if path.is_file()]


def apply_snapshot(render_dir: Path, block: str) -> list[Path]:
    """Replace the managed region of every target file with *block*.

    A file without both markers is skipped rather than appended to: the marker
    pair ships with the agent template, so its absence means a hand-edited or
    pre-snapshot render, and growing an unmarked section in someone's edited
    prompt is worse than leaving the tools to answer at run time.

    Returns:
        The files now carrying *block* (rewritten or already current).
    """
    patched: list[Path] = []
    for path in snapshot_targets(render_dir):
        text = path.read_text(encoding="utf-8")
        begin = text.find(SNAPSHOT_BEGIN)
        end = text.find(SNAPSHOT_END)
        if begin == -1 or end < begin:
            logger.warning("No snapshot markers in %s; leaving it alone", path)
            continue
        end += len(SNAPSHOT_END)
        updated = text[:begin] + block + text[end:]
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
        render has none (the agent is disabled, or this is a store-only
        project).
    """
    # Function-local: examples_data is pure stdlib data, but importing it
    # through the tools package must not become an import-time dependency of
    # the seeder.
    from osprey.mcp_server.graph.tools.examples_data import EXAMPLE_QUERIES
    from osprey.services.facility_knowledge.seeder import graph_seeder

    def run(cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return [record.data() for record in session.run(cypher, params or {})]

    known = frozenset(corpus for example in EXAMPLE_QUERIES for corpus in example.parameters)
    block = render_block(
        collect_schema(run),
        list(EXAMPLE_QUERIES),
        corpus=detect_corpus(run, known),
        digest=graph_seeder.read_marker(session),
        resource_count=graph_seeder.resource_count(session),
    )
    return apply_snapshot(render_dir, block)
