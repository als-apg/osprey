"""Unit tests for the shared tool denylist matcher (osprey.utils.tool_rules)."""

import pytest

from osprey.utils.tool_rules import matches_denylist


@pytest.mark.parametrize(
    ("tool", "entries", "expected"),
    [
        pytest.param("Bash", ["Bash", "WebFetch"], True, id="exact-entry-matches-exact-tool"),
        pytest.param("BashOutput", ["Bash"], False, id="exact-entry-does-not-match-other-tool"),
        pytest.param(
            "mcp__plugin_playwright_playwright__click",
            ["mcp__plugin_playwright_playwright__*"],
            True,
            id="prefix-entry-matches-by-prefix",
        ),
        pytest.param(
            "mcp__osprey_workspace__artifact_list",
            ["mcp__plugin_playwright_playwright__*"],
            False,
            id="prefix-entry-does-not-match-unrelated-tool",
        ),
        pytest.param("anything", ["*"], True, id="bare-star-matches-anything"),
        pytest.param("", ["*"], True, id="bare-star-matches-empty-tool"),
        pytest.param("Bash", [], False, id="empty-entries-match-nothing"),
        pytest.param("", ["Bash"], False, id="empty-tool-does-not-match-named-entry"),
        pytest.param("", [""], True, id="empty-tool-matches-empty-entry"),
        # "foo*" prefix-matches "foo" itself (empty remainder)
        pytest.param("foo", ["foo*"], True, id="prefix-entry-matches-its-own-prefix-exactly"),
    ],
)
def test_matches_denylist(tool, entries, expected):
    """matches_denylist semantics shared by dispatch + web terminal."""
    assert matches_denylist(tool, entries) is expected
