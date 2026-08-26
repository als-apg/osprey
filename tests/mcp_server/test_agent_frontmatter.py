"""Tests for the shared ``.claude/agents/*.md`` frontmatter reader.

The dispatch worker reads what an agent declares about itself — its name and
its tools — from the rendered agent file.
"""

import logging

from osprey.mcp_server.agent_frontmatter import parse_agent_frontmatter


def _write_agent(agents_dir, filename, body):
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / filename).write_text(body, encoding="utf-8")


class TestParseAgentFrontmatter:
    def test_keyed_by_frontmatter_name_not_filename(self, tmp_path):
        agents = tmp_path / ".claude" / "agents"
        _write_agent(agents, "file.md", "---\nname: declared\ntools: a, b\n---\nbody\n")

        parsed = parse_agent_frontmatter(tmp_path)

        assert set(parsed) == {"declared"}
        assert parsed["declared"]["tools"] == "a, b"

    def test_missing_dir_and_subdirs(self, tmp_path):
        assert parse_agent_frontmatter(tmp_path) == {}

        agents = tmp_path / ".claude" / "agents"
        _write_agent(agents / "_terminology", "x.md", "---\nname: nested\n---\n")
        assert parse_agent_frontmatter(tmp_path) == {}

    def test_malformed_and_nameless_files_skipped_with_warning(self, tmp_path, caplog):
        agents = tmp_path / ".claude" / "agents"
        _write_agent(agents, "broken.md", "---\nname: [unclosed\n---\n")
        _write_agent(agents, "nameless.md", "---\ntools: a\n---\n")
        _write_agent(agents, "plain.md", "no frontmatter here\n")
        _write_agent(agents, "scalar.md", "---\njust a string\n---\n")

        with caplog.at_level(logging.WARNING):
            parsed = parse_agent_frontmatter(tmp_path)

        assert parsed == {}
        assert any("broken.md" in r.message for r in caplog.records)
        assert any("nameless.md" in r.message for r in caplog.records)

    def test_duplicate_name_last_wins(self, tmp_path, caplog):
        agents = tmp_path / ".claude" / "agents"
        _write_agent(agents, "a.md", "---\nname: dup\ntools: first\n---\n")
        _write_agent(agents, "b.md", "---\nname: dup\ntools: second\n---\n")

        with caplog.at_level(logging.WARNING):
            parsed = parse_agent_frontmatter(tmp_path)

        assert parsed["dup"]["tools"] == "second"
        assert any("dup" in r.message for r in caplog.records)
