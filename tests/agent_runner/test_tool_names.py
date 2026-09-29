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
        "mcp__plugin_playwright_playwright__*",
        "mcp__plugin_context7_context7__*",
    )
