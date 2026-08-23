"""Tests for the shared ``.claude/agents/*.md`` frontmatter reader.

The dispatch worker and ``submit_response`` both read what an agent declares
about itself from the rendered agent file. One parser, so the two cannot
disagree about which files are agents or what their names are.
"""

import logging

from osprey.mcp_server.agent_frontmatter import (
    parse_agent_frontmatter,
    results_category_for,
)


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
        _write_agent(agents, "a.md", "---\nname: dup\nresults_category: first\n---\n")
        _write_agent(agents, "b.md", "---\nname: dup\nresults_category: second\n---\n")

        with caplog.at_level(logging.WARNING):
            parsed = parse_agent_frontmatter(tmp_path)

        assert parsed["dup"]["results_category"] == "second"
        assert any("dup" in r.message for r in caplog.records)


class TestResultsCategoryFor:
    def test_declared(self, tmp_path):
        agents = tmp_path / ".claude" / "agents"
        _write_agent(
            agents,
            "pyat-specialist.md",
            "---\nname: pyat-specialist\nresults_category: lattice_analysis\n---\n",
        )
        assert results_category_for("pyat-specialist", tmp_path) == "lattice_analysis"

    def test_undeclared_agent_and_unknown_agent(self, tmp_path):
        agents = tmp_path / ".claude" / "agents"
        _write_agent(agents, "logbook-search.md", "---\nname: logbook-search\n---\n")

        assert results_category_for("logbook-search", tmp_path) is None
        assert results_category_for("nobody", tmp_path) is None

    def test_blank_or_non_string_declaration_is_none(self, tmp_path):
        agents = tmp_path / ".claude" / "agents"
        _write_agent(agents, "a.md", "---\nname: a\nresults_category: ''\n---\n")
        _write_agent(agents, "b.md", "---\nname: b\nresults_category: [x]\n---\n")

        assert results_category_for("a", tmp_path) is None
        assert results_category_for("b", tmp_path) is None
