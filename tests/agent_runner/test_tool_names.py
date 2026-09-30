"""Unit tests for :mod:`osprey.agent_runner.tool_names`.

The relations between the lists (subset, superset) are pinned beside their
consumers; these tests pin the properties every list shares.
"""

import pytest

from osprey.agent_runner import tool_names


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("DENY_DEFAULTS", tuple),
        ("READ_ONLY_DENIED_BUILTINS", tuple),
        ("OPEN_MODE_EGRESS_TOOLS", tuple),
        ("WRITE_CAPABLE_BUILTINS", tuple),
        ("DISPATCH_DENIED_TOOLS", frozenset),
    ],
)
def test_each_list_is_immutable(name, kind):
    """No consumer can widen or reorder a shared floor at run time."""
    assert type(getattr(tool_names, name)) is kind


def test_the_interactive_floor_renders_in_this_order():
    """The order is the rendered ``permissions.deny`` order of every built project.

    Changing it is a deliberate edit of this test.
    """
    assert tool_names.DENY_DEFAULTS == (
        "Bash",
        "Edit",
        "WebFetch",
        "WebSearch",
        "mcp__plugin_*",
        "mcp__claude_ai_*",
    )


@pytest.mark.parametrize(
    ("server", "namespace"),
    [
        ("plugin", "mcp__plugin_*"),
        ("plugin_x", "mcp__plugin_*"),
        ("plugin.x", "mcp__plugin_*"),
        ("claude_ai", "mcp__claude_ai_*"),
        ("claude_ai_Gmail", "mcp__claude_ai_*"),
        ("Plugin_x", None),
        ("pluginx", None),
        ("claude_aix", None),
        ("claude-ai_x", None),
        ("controls", None),
        ("channel-finder", None),
    ],
)
def test_foreign_mcp_namespace(server, namespace):
    """The agent CLI's own outcomes: it maps every character outside
    ``[A-Za-z0-9_-]`` to ``_`` and matches deny globs case-sensitively by prefix."""
    assert tool_names.foreign_mcp_namespace(server) == namespace
