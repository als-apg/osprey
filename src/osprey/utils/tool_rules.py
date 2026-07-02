"""Shared tool-name denylist matching.

Single implementation of the denylist semantics used by the dispatch worker
(``dispatch_api.DENIED_TOOLS``) and the web-terminal permission callback
(``sdk_context.make_tool_allowlist``): entries ending in ``*`` match by prefix
(e.g. ``mcp__plugin_playwright_playwright__*`` blocks every playwright tool);
all other entries match exactly.

Lives in ``osprey.utils`` (leaf package) so both the dispatch worker and the
web terminal can import it without pulling in each other's package trees.
"""

from __future__ import annotations

from collections.abc import Iterable


def matches_denylist(tool_name: str, entries: Iterable[str]) -> bool:
    """Return True if ``tool_name`` matches any denylist entry.

    Args:
        tool_name: Tool name as reported by the SDK (e.g. ``Bash``,
            ``mcp__controls__channel_read``).
        entries: Denylist entries. An entry ending in ``*`` matches any tool
            whose name starts with the part before the ``*``; every other
            entry matches exactly.

    Returns:
        True on the first matching entry, False if none match.
    """
    for entry in entries:
        if entry.endswith("*"):
            if tool_name.startswith(entry[:-1]):
                return True
        elif tool_name == entry:
            return True
    return False
