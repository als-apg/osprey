"""The words the vocabulary guards hold, in one place.

``RATCHET_WORD`` is the word no shipped file or agent-facing text carries; the file
scan in :mod:`tests.facility.test_word_ratchet` and the agent-facing guard in
:mod:`tests.mcp_server.test_channel_finder_tool_vocabulary` both match it.
``PROTOCOL_WORDS`` are the words the text an agent reads never names.
``ENGINE_WORDS`` are the simulation engine's own words, which only the engine
plug-ins, the importers and the facility definitions speak; the second scan in
:mod:`tests.facility.test_word_ratchet` matches them.
``RETIRED_ROSTER_WORD`` is the roster's retired word, which no tracked file
names, by content or by path, outside the code that reads the old spelling;
the third scan in :mod:`tests.facility.test_word_ratchet` matches it.
``SITE_EXAMPLE_WORD`` is the example deployment name that named a site, which no
core text carries; the fourth scan in :mod:`tests.facility.test_word_ratchet`
matches it.

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

Rendered text is judged after the strings it received from the facility
definition are masked by origin: ``FACILITY_CONTEXT_KEYS`` name the render
context the build fills from the facility, and every string leaf and key under
them is replaced by ``FACILITY_MASK`` before the words are matched. Masks match
whole words. Every facility-sourced string is masked whatever it is, a bare
guarded word included, because OSPREY's own wording is guarded at source — the
templates and ``src/`` are scanned by :mod:`tests.facility.test_word_ratchet` —
so a guarded word that survives masking is OSPREY's. OSPREY's vocabulary table
(:func:`osprey.facility.validate.vocabulary`) is never masked and stays judged.

Lives in a ``_``-prefixed module rather than ``conftest.py`` because it holds
constants and plain functions, not fixtures. Same convention as
:mod:`tests.cli._lifecycle_build`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

# Framework-free, so it lives in the connectors suite's harness; re-exported here.
from osprey_connectors_tests._harness._vocabulary import REFUSAL_BAN_LIST  # noqa: F401

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

#: The roster's retired word, singular or plural: on its own, as a snake_case
#: or kebab-case part, as the capitalised part of a camelCase name, or all
#: capitals. Letter lookarounds rather than ``\b``, because ``_`` is a word
#: character and ``\b`` would miss the word in the middle of a snake_case name.
#: Other words that merely contain it (the adjective for one's own, an
#: impersonation) are not matches. A string for ``git grep -P``, as
#: ``RATCHET_WORD``; its own source line carries the scan's line marker.
RETIRED_ROSTER_WORD = r"(?<![A-Za-z])[Pp]ersonas?(?![a-z])|(?<=[a-z0-9])Personas?(?![a-z])|(?<![A-Za-z])PERSONAS?(?![A-Za-z])"  # retired-spelling

#: The example deployment name that named a site; examples use ``my-assistant``.
SITE_EXAMPLE_WORD = r"\bals-assistant\b"

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

#: The render-context keys the build fills from the facility definition.
FACILITY_CONTEXT_KEYS: tuple[str, ...] = (
    "facility_facts",
    "middle_layer_families",
    "served_decks",
    "measurement",
)

#: What a facility-sourced string reads as once masked; it matches neither
#: ``RATCHET_WORD`` nor ``PROTOCOL_WORDS``, so a failing guard stays readable.
FACILITY_MASK = "[facility]"


def _string_leaves(value: Any) -> Iterator[str]:
    """Every stripped, non-empty string leaf and mapping key under *value*."""
    if isinstance(value, str):
        if stripped := value.strip():
            yield stripped
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _string_leaves(key)
            yield from _string_leaves(item)
    elif isinstance(value, Iterable) and not isinstance(value, bytes | bytearray):
        for item in value:
            yield from _string_leaves(item)


def facility_strings(*values: Any) -> frozenset[str]:
    """The strings *values* carry, minus the leaves of OSPREY's vocabulary table."""
    from osprey.facility.validate import vocabulary

    return frozenset(_string_leaves(values)) - frozenset(_string_leaves(vocabulary()))


def render_root_facility_strings(root: Path) -> frozenset[str]:
    """The facility-sourced strings the build handed the templates of one render root."""
    from osprey.cli.templates.claude_code import config_derived_context

    config_file = root / "config.yml"
    config = (
        yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        if config_file.is_file()
        else {}
    )
    context = config_derived_context(config, root)
    return facility_strings(*(context[key] for key in FACILITY_CONTEXT_KEYS))


def mask_facility_text(text: str, strings: Iterable[str]) -> str:
    """*text* with every whole-word occurrence of a guarded facility string masked.

    Only the strings that carry a ``RATCHET_WORD`` or ``PROTOCOL_WORDS`` match
    are candidates, a string that is nothing but a guarded word included; the
    longest is tried first so a short string never splits a longer one.
    """
    candidates = sorted(
        {s for s in strings if re.search(RATCHET_WORD, s) or PROTOCOL_WORDS.search(s)},
        key=lambda s: (-len(s), s),
    )
    if not candidates:
        return text
    pattern = r"(?<!\w)(?:" + "|".join(re.escape(s) for s in candidates) + r")(?!\w)"
    return re.sub(pattern, FACILITY_MASK, text)


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
