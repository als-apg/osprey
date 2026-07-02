"""Unit tests for parse_project_agents (dispatch_worker.agent_surfaces).

The parser reads the provisioned ``<project>/.claude/agents/*.md`` files and
returns ``{agent_name: frozenset(tools) | None}`` — ``None`` marks a declared
agent without an explicit ``tools:`` list (inherits-all semantics), which the
dispatch policy treats as non-delegable.
"""

import logging
from pathlib import Path

from osprey.mcp_server.dispatch_worker.agent_surfaces import parse_project_agents


def _write_agent(agents_dir: Path, filename: str, content: str) -> Path:
    agents_dir.mkdir(parents=True, exist_ok=True)
    path = agents_dir / filename
    path.write_text(content)
    return path


def _project(tmp_path: Path) -> Path:
    return tmp_path / "project"


def _agents_dir(tmp_path: Path) -> Path:
    return _project(tmp_path) / ".claude" / "agents"


class TestParseProjectAgents:
    def test_comma_string_tools_parsed(self, tmp_path):
        # Arrange — templates render tools: as a comma-separated scalar
        _write_agent(
            _agents_dir(tmp_path),
            "channel-finder.md",
            "---\n"
            "name: channel-finder\n"
            "tools: mcp__channel-finder__search, mcp__osprey_workspace__submit_response\n"
            "---\n\n# Channel Finder Agent\n",
        )

        # Act
        surfaces = parse_project_agents(_project(tmp_path))

        # Assert
        assert surfaces == {
            "channel-finder": frozenset(
                {"mcp__channel-finder__search", "mcp__osprey_workspace__submit_response"}
            )
        }

    def test_yaml_list_tools_parsed(self, tmp_path):
        # Arrange — customized files may legally use YAML list form
        _write_agent(
            _agents_dir(tmp_path),
            "custom.md",
            "---\nname: custom\ntools:\n  - mcp__ariel__keyword_search\n  - Read\n---\nbody\n",
        )

        # Act
        surfaces = parse_project_agents(_project(tmp_path))

        # Assert
        assert surfaces == {"custom": frozenset({"mcp__ariel__keyword_search", "Read"})}

    def test_missing_tools_key_yields_none(self, tmp_path, caplog):
        # Arrange — inherits-all agent: declared but non-delegable
        _write_agent(
            _agents_dir(tmp_path),
            "wild.md",
            "---\nname: wild\ndescription: no tools declared\n---\nbody\n",
        )

        # Act
        with caplog.at_level(logging.WARNING):
            surfaces = parse_project_agents(_project(tmp_path))

        # Assert
        assert surfaces == {"wild": None}
        assert any("wild" in r.message for r in caplog.records)

    def test_malformed_yaml_skips_file(self, tmp_path, caplog):
        # Arrange
        _write_agent(
            _agents_dir(tmp_path),
            "broken.md",
            "---\nname: [unclosed\ntools: {{{{\n---\nbody\n",
        )

        # Act
        with caplog.at_level(logging.WARNING):
            surfaces = parse_project_agents(_project(tmp_path))

        # Assert
        assert surfaces == {}
        assert any("broken.md" in r.message for r in caplog.records)

    def test_file_without_frontmatter_skipped(self, tmp_path):
        # Arrange
        _write_agent(_agents_dir(tmp_path), "readme.md", "# Not an agent file\n")

        # Act / Assert
        assert parse_project_agents(_project(tmp_path)) == {}

    def test_empty_file_skipped(self, tmp_path):
        # Arrange
        _write_agent(_agents_dir(tmp_path), "empty.md", "")

        # Act / Assert
        assert parse_project_agents(_project(tmp_path)) == {}

    def test_duplicate_name_last_wins_with_warning(self, tmp_path, caplog):
        # Arrange — two files declaring the same frontmatter name
        _write_agent(_agents_dir(tmp_path), "a-first.md", "---\nname: dup\ntools: ToolA\n---\n")
        _write_agent(_agents_dir(tmp_path), "b-second.md", "---\nname: dup\ntools: ToolB\n---\n")

        # Act
        with caplog.at_level(logging.WARNING):
            surfaces = parse_project_agents(_project(tmp_path))

        # Assert — sorted glob order: b-second.md parsed last
        assert surfaces == {"dup": frozenset({"ToolB"})}
        assert any("dup" in r.message for r in caplog.records)

    def test_frontmatter_name_differs_from_filename(self, tmp_path):
        # Arrange — the CLI dispatches on frontmatter name, not filename
        _write_agent(_agents_dir(tmp_path), "some-file.md", "---\nname: real-name\ntools: T\n---\n")

        # Act
        surfaces = parse_project_agents(_project(tmp_path))

        # Assert
        assert "real-name" in surfaces
        assert "some-file" not in surfaces

    def test_missing_name_skips_file(self, tmp_path, caplog):
        # Arrange
        _write_agent(_agents_dir(tmp_path), "anon.md", "---\ntools: T\n---\n")

        # Act
        with caplog.at_level(logging.WARNING):
            surfaces = parse_project_agents(_project(tmp_path))

        # Assert
        assert surfaces == {}

    def test_missing_agents_dir_returns_empty(self, tmp_path):
        # Arrange — project exists, .claude/agents does not
        _project(tmp_path).mkdir(parents=True)

        # Act / Assert
        assert parse_project_agents(_project(tmp_path)) == {}

    def test_subdirectories_not_descended(self, tmp_path):
        # Arrange — templates ship a _terminology sibling dir
        sub = _agents_dir(tmp_path) / "_terminology"
        _write_agent(sub, "nested.md", "---\nname: nested\ntools: T\n---\n")

        # Act / Assert
        assert parse_project_agents(_project(tmp_path)) == {}

    def test_non_dict_frontmatter_skipped(self, tmp_path):
        # Arrange — YAML that parses to a scalar, not a mapping
        _write_agent(_agents_dir(tmp_path), "scalar.md", "---\njust a string\n---\n")

        # Act / Assert
        assert parse_project_agents(_project(tmp_path)) == {}

    def test_tools_odd_type_yields_none(self, tmp_path):
        # Arrange — tools: as a mapping is not a valid surface declaration
        _write_agent(_agents_dir(tmp_path), "odd.md", "---\nname: odd\ntools:\n  key: val\n---\n")

        # Act
        surfaces = parse_project_agents(_project(tmp_path))

        # Assert
        assert surfaces == {"odd": None}

    def test_comma_string_strips_and_drops_empties(self, tmp_path):
        # Arrange — trailing comma as rendered by the Jinja loop
        _write_agent(
            _agents_dir(tmp_path),
            "trail.md",
            "---\nname: trail\ntools: ToolA, ToolB, \n---\n",
        )

        # Act
        surfaces = parse_project_agents(_project(tmp_path))

        # Assert
        assert surfaces == {"trail": frozenset({"ToolA", "ToolB"})}
