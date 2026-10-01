"""The words the vocabulary guards hold, in one place.

``RATCHET_WORD`` is the word the shipped code loses file by file; the file
scan in :mod:`tests.facility.test_word_ratchet` and the agent-facing guard in
:mod:`tests.mcp_server.test_channel_finder_tool_vocabulary` both match it.
``PROTOCOL_WORDS`` are the words the text an agent reads never names.

Lives in a ``_``-prefixed module rather than ``conftest.py`` because it holds
constants, not fixtures. Same convention as :mod:`tests.cli._vocabulary_guard`.
"""

from __future__ import annotations

import re

#: One agent-facing text: ``(source, name, field)``. The source is an MCP
#: server, or ``ariel_search`` for the ARIEL search descriptors.
TextKey = tuple[str, str, str]

#: The word, case-insensitive on its own and as a snake_case part; the
#: capitalised camelCase part (``PyATRingModel``) is matched case-sensitively.
#: A string, not a compiled pattern: ``git grep -P`` takes it as written.
RATCHET_WORD = r"(?i:\bring\b|_ring\b|\bring_)|Ring(?=[A-Z_])"

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
}
