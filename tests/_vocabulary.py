"""The words the vocabulary guards hold, in one place.

``RATCHET_WORD`` is the word the shipped code loses file by file; the file
scan in :mod:`tests.facility.test_word_ratchet` and the agent-facing guard in
:mod:`tests.mcp_server.test_channel_finder_tool_vocabulary` both match it.
``PROTOCOL_WORDS`` are the words the text an agent reads never names.
``ENGINE_WORDS`` are the simulation engine's own words, which only the engine
plug-ins, the importers and the facility definitions speak; the second scan in
:mod:`tests.facility.test_word_ratchet` matches them.

``HARDCODED_VOCABULARY_TOKENS`` hold the facility-vocabulary rule: facility
terminology has exactly one source of truth. For the graph paradigm that source
is the store's own ``(c:Class).altLabel``, captured into the *Graph at Hand*
block when the corpus is seeded — so no framework template may ship a
facility's device kinds, class labels or operator synonyms of its own.

Two rendered agents carry the graph terminology partial — the
facility-knowledge-graph subagent and the channel finder in its graph paradigm
— and each is asserted by the test module that owns its render. The token list
and the matching rule live here so the two guards cannot drift into disagreeing
about what the rule forbids.

Lives in a ``_``-prefixed module rather than ``conftest.py`` because it holds
constants and plain functions, not fixtures. Same convention as
:mod:`tests.cli._lifecycle_build`.
"""

from __future__ import annotations

import re

#: One agent-facing text: ``(source, name, field)``. The source is an MCP
#: server, ``ariel_search`` for the ARIEL search descriptors, or ``rendered``
#: for a file a build renders for the agent, named by its path under the
#: render root.
TextKey = tuple[str, str, str]

#: The word, case-insensitive on its own and as a snake_case part; the
#: capitalised camelCase part (``PyATRingModel``) is matched case-sensitively.
#: A string, not a compiled pattern: ``git grep -P`` takes it as written.
RATCHET_WORD = r"(?i:\bring\b|_ring\b|\bring_)|Ring(?=[A-Z_])"

#: The pyAT attributes a corrector or a strength writes and the ``axis`` key a
#: monitor reads. Code outside the engine's zones reads a binding's ``role``
#: and ``plane`` instead. A string for ``git grep -P``, as ``RATCHET_WORD``.
ENGINE_WORDS = r"\bKickAngle\b|\bPolynom[AB]\b|[\"']axis[\"']"

# Case-sensitive on purpose: lowercase ``epics`` names a connector type in config
# text, and the bare demo prefix ``SR`` without a colon is not a match.
PROTOCOL_WORDS = re.compile(r"\bPVs?\b|\bEPICS\b|SR:")

# Texts that still name a protocol word; an entry leaves when its wording changes.
PENDING_REWORDING: frozenset[TextKey] = frozenset(
    {
        (
            "channel_finder_in_context",
            "ask_channels",
            "inputSchema/properties/question/description",
        ),
        ("channel_finder_graph", "read_cypher", "description"),
        ("channel_finder_graph", "search_channels", "description"),
        ("ariel", "keyword_search", "description"),
        ("ariel_search", "keyword_search", "description"),
        ("control_system", "archiver_read", "inputSchema/properties/channels/description"),
        ("control_system", "channel_read", "inputSchema/properties/channels/description"),
        ("graph", "example_queries", "description"),
        ("graph", "read_cypher", "description"),
        ("phoebus", "phoebus_open_databrowser", "description"),
        ("phoebus", "phoebus_open_databrowser", "inputSchema/properties/channels/description"),
        ("phoebus", "phoebus_perceive", "description"),
        ("rendered", ".claude/agents/channel-finder.md", "text"),
        ("rendered", ".claude/agents/data-visualizer.md", "text"),
        ("rendered", ".claude/agents/facility-knowledge-graph.md", "text"),
        ("rendered", ".claude/agents/pyat-specialist.md", "text"),
        ("rendered", ".claude/output-styles/control-operator.md", "text"),
        ("rendered", ".claude/rules/control-system-safety.md", "text"),
        ("rendered", ".claude/rules/safety.md", "text"),
        ("rendered", ".claude/rules/test-ioc-safety.md", "text"),
        ("rendered", ".claude/skills/session-report/reference.md", "text"),
        ("rendered", ".claude/skills/writing-bluesky-plans/SKILL.md", "text"),
        ("rendered", "CLAUDE.md", "text"),
    }
)

#: Agent-facing texts that still carry ``RATCHET_WORD``, tagged like the file
#: ratchet's allowlist with the stage that rewords them.
RATCHET_PENDING: dict[TextKey, str] = {
    (
        "channel_finder_hierarchical",
        "build_channels",
        "inputSchema/properties/selections/description",
    ): "rename:12",
    (
        "channel_finder_hierarchical",
        "get_options",
        "inputSchema/properties/level/description",
    ): "rename:12",
    (
        "channel_finder_hierarchical",
        "get_options",
        "inputSchema/properties/selections/description",
    ): "rename:12",
    (
        "channel_finder_middle_layer",
        "list_families",
        "inputSchema/properties/system/description",
    ): "rename:12",
    ("rendered", ".claude/agents/channel-finder.md", "text"): "rename:12",
    ("rendered", ".claude/agents/facility-knowledge-graph.md", "text"): "rename:12",
    ("rendered", ".claude/agents/pyat-specialist.md", "text"): "rename:12",
    ("rendered", ".claude/output-styles/control-operator.md", "text"): "rename:12",
}


#: Facility vocabulary that a framework prompt may not spell for itself: the
#: concrete class labels and the operator synonyms the partial used to ship.
HARDCODED_VOCABULARY_TOKENS: tuple[str, ...] = (
    "dcct",
    "bcm",
    "bpm",
    "quad",
    "Quadrupole",
    "BeamPositionMonitor",
    "HCorrector",
    "VCorrector",
    "Corrector",
)


def hardcoded_vocabulary_hits(text: str) -> list[str]:
    """Tokens from the facility-vocabulary rule that survive in *text*.

    The CamelCase class names are matched case-sensitively, so English prose
    about "a quadrupole" is not read as the class label ``Quadrupole``. The
    lowercase operator tokens are matched case-insensitively but word-bounded,
    so "quadrupole" does not false-positive on the abbreviation "quad".
    """
    hits: list[str] = []
    for token in HARDCODED_VOCABULARY_TOKENS:
        if token[0].isupper():
            if token in text:
                hits.append(token)
        elif re.search(rf"\b{re.escape(token)}\b", text, re.IGNORECASE):
            hits.append(token)
    return hits
