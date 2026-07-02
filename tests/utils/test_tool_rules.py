"""Unit tests for the shared tool denylist matcher (osprey.utils.tool_rules)."""

from osprey.utils.tool_rules import matches_denylist


class TestMatchesDenylist:
    """AAA tests for matches_denylist semantics shared by dispatch + web terminal."""

    def test_exact_entry_matches_exact_tool(self):
        # Arrange
        entries = ["Bash", "WebFetch"]

        # Act
        result = matches_denylist("Bash", entries)

        # Assert
        assert result is True

    def test_exact_entry_does_not_match_other_tool(self):
        # Arrange
        entries = ["Bash"]

        # Act
        result = matches_denylist("BashOutput", entries)

        # Assert
        assert result is False

    def test_prefix_entry_matches_by_prefix(self):
        # Arrange
        entries = ["mcp__plugin_playwright_playwright__*"]

        # Act
        result = matches_denylist("mcp__plugin_playwright_playwright__click", entries)

        # Assert
        assert result is True

    def test_prefix_entry_does_not_match_unrelated_tool(self):
        # Arrange
        entries = ["mcp__plugin_playwright_playwright__*"]

        # Act
        result = matches_denylist("mcp__osprey_workspace__data_list", entries)

        # Assert
        assert result is False

    def test_bare_star_matches_everything(self):
        # Arrange
        entries = ["*"]

        # Act / Assert
        assert matches_denylist("anything", entries) is True
        assert matches_denylist("", entries) is True

    def test_empty_entries_matches_nothing(self):
        # Arrange
        entries: list[str] = []

        # Act
        result = matches_denylist("Bash", entries)

        # Assert
        assert result is False

    def test_empty_tool_name_only_matches_bare_star_or_empty_entry(self):
        # Arrange / Act / Assert
        assert matches_denylist("", ["Bash"]) is False
        assert matches_denylist("", [""]) is True

    def test_prefix_entry_matches_its_own_prefix_exactly(self):
        # Arrange: "foo*" prefix-matches "foo" itself (empty remainder)
        entries = ["foo*"]

        # Act / Assert
        assert matches_denylist("foo", entries) is True
