"""Tests for ClaudeCodeFileService."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.audit.protected import SURFACE_CLAUDE_SETUP
from osprey.cli.profile_conventions import is_reserved_write
from osprey.interfaces.web_terminal.claude_code_files import (
    PROFILE_EDIT_NOTICE,
    ClaudeCodeFileService,
    ProtectedWriteError,
)
from osprey.interfaces.web_terminal.ownership import reserved_write_channel
from osprey.interfaces.web_terminal.routes.config import router as config_router
from osprey.utils.identity import acting_identity


@pytest.fixture
def project_dir(tmp_path):
    """Create a minimal project directory with Claude Code files."""
    # Root files
    (tmp_path / "CLAUDE.md").write_text("# Test CLAUDE.md\n")
    (tmp_path / ".mcp.json").write_text('{"servers": {}}\n')

    # .claude subdirectories
    claude = tmp_path / ".claude"
    claude.mkdir()
    (claude / "settings.json").write_text('{"permissions": {}}\n')

    rules = claude / "rules"
    rules.mkdir()
    (rules / "safety.md").write_text("# Safety rules\n")

    agents = claude / "agents"
    agents.mkdir()
    (agents / "test-agent.md").write_text("# Test agent\n")

    hooks = claude / "hooks"
    hooks.mkdir()
    (hooks / "pre-check.sh").write_text("#!/bin/bash\necho ok\n")

    # A writable JSON file: the root-level JSON files (.mcp.json,
    # .claude/settings.json) are all in the protected set now, so syntax
    # validation needs a target the panel is still allowed to write.
    commands = claude / "commands"
    commands.mkdir()
    (commands / "params.json").write_text('{"limit": 1}\n')

    return tmp_path


@pytest.fixture
def service(project_dir):
    return ClaudeCodeFileService(project_dir)


@pytest.fixture(autouse=True)
def audit_dir(tmp_path, monkeypatch):
    """Redirect the audit zone out of the real deployment.

    ``writer.audit_dir`` is the ledger's single test seam: every surface's
    path is derived from it, so patching it here catches the record without
    standing up a project root.

    Autouse, because every refusal in this file records whether or not the
    test reads the record back: a suite that appends refusals nobody caused to
    the deployment's own ledger makes the ledger unusable as evidence.
    """
    from osprey.audit import writer

    target = tmp_path / "audit-zone"
    monkeypatch.setattr(writer, "audit_dir", lambda: target)
    return target


def _audit_records(audit_dir):
    """Every ``claude_setup`` record written under *audit_dir*, oldest first."""
    log = audit_dir / acting_identity() / f"{SURFACE_CLAUDE_SETUP}.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


class TestListFiles:
    def test_discovers_expected_files(self, service):
        files = service.list_files()
        paths = {f["path"] for f in files}

        assert "CLAUDE.md" in paths
        assert ".mcp.json" in paths
        assert ".claude/settings.json" in paths
        assert ".claude/rules/safety.md" in paths
        assert ".claude/agents/test-agent.md" in paths
        assert ".claude/hooks/pre-check.sh" in paths

    def test_categories_assigned(self, service):
        files = service.list_files()
        by_path = {f["path"]: f for f in files}

        assert by_path["CLAUDE.md"]["category"] == "System Prompt"
        assert by_path[".mcp.json"]["category"] == "MCP Servers"
        assert by_path[".claude/settings.json"]["category"] == "Permissions"
        assert by_path[".claude/rules/safety.md"]["category"] == "Safety"
        assert by_path[".claude/agents/test-agent.md"]["category"] == "Agents"
        assert by_path[".claude/hooks/pre-check.sh"]["category"] == "Hooks"

    def test_languages_detected(self, service):
        files = service.list_files()
        by_path = {f["path"]: f for f in files}

        assert by_path["CLAUDE.md"]["language"] == "markdown"
        assert by_path[".mcp.json"]["language"] == "json"
        assert by_path[".claude/hooks/pre-check.sh"]["language"] == "shell"

    def test_skips_hidden_files(self, project_dir, service):
        (project_dir / ".claude" / "rules" / ".hidden").write_text("secret\n")
        files = service.list_files()
        paths = {f["path"] for f in files}
        assert ".claude/rules/.hidden" not in paths

    def test_list_files_read_only_agrees_with_write_file(self, service):
        """``read_only`` is the protected set, computed the same way as the refusal."""
        entries = {f["path"]: f for f in service.list_files()}

        assert entries["CLAUDE.md"]["read_only"] is True
        assert entries[".mcp.json"]["read_only"] is True
        assert entries[".claude/settings.json"]["read_only"] is True
        assert entries[".claude/rules/safety.md"]["read_only"] is True
        assert entries[".claude/agents/test-agent.md"]["read_only"] is False
        assert entries[".claude/hooks/pre-check.sh"]["read_only"] is False
        assert entries[".claude/commands/params.json"]["read_only"] is False

        # The flag is only useful if it predicts what the writer does, so pin
        # the two against each other rather than against a second hand-list.
        for path, entry in entries.items():
            try:
                service.write_file(path, entry["content"])
                refused = False
            except ProtectedWriteError:
                refused = True
            assert entry["read_only"] is refused, path

    def test_empty_project(self, tmp_path):
        svc = ClaudeCodeFileService(tmp_path)
        files = svc.list_files()
        assert files == []


class TestReadFile:
    def test_reads_existing_file(self, service):
        result = service.read_file("CLAUDE.md")
        assert result["content"] == "# Test CLAUDE.md\n"
        assert result["name"] == "CLAUDE.md"
        assert result["language"] == "markdown"

    def test_file_not_found(self, service):
        with pytest.raises(FileNotFoundError):
            service.read_file("nonexistent.md")


class TestWriteFile:
    def test_writes_existing_file(self, service, project_dir):
        result = service.write_file(".claude/agents/test-agent.md", "# Updated\n")
        assert result["status"] == "saved"
        assert (project_dir / ".claude" / "agents" / "test-agent.md").read_text() == "# Updated\n"

    def test_write_file_refuses_a_path_outside_the_project(self, service):
        """A climbing path is judged by the protected set, which owns it first."""
        with pytest.raises(ProtectedWriteError, match="not project-relative"):
            service.write_file("../../etc/passwd", "hacked")

    def test_write_file_refuses_a_link_that_escapes_the_project(
        self, service, project_dir, tmp_path
    ):
        """A link out of the project is refused by the reserved check, which resolves.

        It answers first and answers about the file the bytes would land on, so
        an escaping link never reaches ``_validate_path``; that check stays as
        the second gate. Either way the refusal is a ``PermissionError``, which
        is what the route maps to 403.
        """
        outside = tmp_path.parent / "outside-the-project.md"
        outside.write_text("# Untouched\n")
        (project_dir / ".claude" / "agents" / "escape.md").symlink_to(outside)

        with pytest.raises(ProtectedWriteError, match="not project-relative"):
            service.write_file(".claude/agents/escape.md", "hacked")
        assert outside.read_text() == "# Untouched\n"

    def test_validates_json(self, service):
        with pytest.raises(ValueError, match="Invalid JSON"):
            service.write_file(".claude/commands/params.json", "not valid json {{{")

    def test_valid_json_accepted(self, service, project_dir):
        new_content = '{"limit": 5}'
        result = service.write_file(".claude/commands/params.json", new_content)
        assert result["status"] == "saved"
        assert (project_dir / ".claude" / "commands" / "params.json").read_text() == new_content

    def test_file_not_found_returns_error(self, service):
        with pytest.raises(FileNotFoundError):
            service.write_file("does-not-exist.md", "content")

    def test_markdown_no_validation(self, service, project_dir):
        """Markdown files should not be syntax-checked."""
        weird_content = "{{{{not yaml not json\n"
        result = service.write_file(".claude/agents/test-agent.md", weird_content)
        assert result["status"] == "saved"


class TestWriteFileProtectedSet:
    """``write_file`` consults the protected set before it touches anything."""

    def test_write_file_refuses_reserved_settings_json(self, service, project_dir, audit_dir):
        before = (project_dir / ".claude" / "settings.json").read_bytes()

        with pytest.raises(ProtectedWriteError) as excinfo:
            service.write_file(".claude/settings.json", '{"permissions": {"allow": ["Bash"]}}')

        message = str(excinfo.value)
        # The refusal names the channel that does own the file...
        assert "`config:`" in message
        assert "claude_code.permissions" in message
        # ...says plainly that the file is untouched...
        assert "nothing was written" in message.lower()
        # ...and is not the create-file allowlist message.
        assert "New files must be in" not in message
        assert (project_dir / ".claude" / "settings.json").read_bytes() == before

    @pytest.mark.parametrize(
        ("rel_path", "channel_phrase"),
        [
            ("CLAUDE.md", "claude_md_template:"),
            (".mcp.json", "mcp_servers:"),
            (".claude/rules/safety.md", "rules/"),
        ],
    )
    def test_write_file_refusal_names_the_owning_channel(
        self, service, project_dir, audit_dir, rel_path, channel_phrase
    ):
        before = (project_dir / rel_path).read_bytes()

        with pytest.raises(ProtectedWriteError) as excinfo:
            service.write_file(rel_path, "# rewritten by the agent\n")

        assert channel_phrase in str(excinfo.value)
        assert (project_dir / rel_path).read_bytes() == before

    def test_write_file_allows_an_agent_definition(self, service, project_dir):
        """Authoring subagents is the point of the panel, so that subtree stays open."""
        result = service.write_file(".claude/agents/test-agent.md", "# Reworked agent\n")

        assert result["status"] == "saved"
        assert result["category"] == "Agents"
        assert (
            project_dir / ".claude" / "agents" / "test-agent.md"
        ).read_text() == "# Reworked agent\n"

    def test_write_file_refusal_is_audited(self, service, audit_dir):
        with pytest.raises(ProtectedWriteError):
            service.write_file(".claude/settings.json", "{}")

        records = _audit_records(audit_dir)
        assert len(records) == 1
        record = records[0]
        assert record["surface"] == "claude_setup"
        assert record["subject"] == ".claude/settings.json"
        assert "target=.claude/settings.json" in record["detail"]
        assert is_reserved_write(".claude/settings.json") in record["detail"]
        assert record["reason"] == "reserved path"

    def test_write_file_success_is_not_audited(self, service, audit_dir):
        service.write_file(".claude/agents/test-agent.md", "# Fine\n")
        assert _audit_records(audit_dir) == []

    def test_write_file_refusal_precedes_content_validation(self, service, project_dir, audit_dir):
        """Invalid JSON aimed at a reserved file is refused as reserved, not as syntax."""
        before = (project_dir / ".claude" / "settings.json").read_bytes()

        with pytest.raises(ProtectedWriteError):
            service.write_file(".claude/settings.json", "not valid json {{{")

        assert (project_dir / ".claude" / "settings.json").read_bytes() == before

    def test_protected_write_error_is_a_permission_error(self):
        """The route maps ``PermissionError`` to 403; the refusal must ride that."""
        assert issubclass(ProtectedWriteError, PermissionError)


class TestClaudeSetupRoutes:
    """How ``/api/claude-setup`` surfaces the refusal to the browser."""

    @pytest.fixture
    def app(self, project_dir):
        app = FastAPI()
        app.include_router(config_router)
        app.state.project_cwd = str(project_dir)
        app.state.agent_activity_ring = []
        return app

    def test_write_file_refusal_returns_403_naming_the_channel(self, app, project_dir, audit_dir):
        before = (project_dir / ".claude" / "settings.json").read_bytes()

        with TestClient(app) as client:
            resp = client.put(
                "/api/claude-setup",
                json={"path": ".claude/settings.json", "content": "{}"},
            )

        assert resp.status_code == 403
        assert "`config:`" in resp.json()["detail"]
        assert (project_dir / ".claude" / "settings.json").read_bytes() == before

    def test_write_file_refusal_publishes_agent_activity(self, app, audit_dir):
        with TestClient(app) as client:
            client.put(
                "/api/claude-setup",
                json={"path": ".claude/settings.json", "content": "{}"},
            )

        assert len(app.state.agent_activity_ring) == 1
        frame = app.state.agent_activity_ring[0]
        assert frame["tool"] == "claude_setup_refused"
        assert frame["target"]["kind"] == "config"
        assert ".claude/settings.json" in frame["target"]["detail"]

    def test_write_file_success_publishes_no_activity(self, app):
        with TestClient(app) as client:
            resp = client.put(
                "/api/claude-setup",
                json={"path": ".claude/agents/test-agent.md", "content": "# ok\n"},
            )

        assert resp.status_code == 200
        assert app.state.agent_activity_ring == []

    def test_create_file_refusal_returns_403_and_publishes_activity(
        self, app, project_dir, audit_dir
    ):
        with TestClient(app) as client:
            resp = client.post(
                "/api/claude-setup",
                json={"path": ".claude/skills/new/SKILL.md", "content": "# Skill\n"},
            )

        assert resp.status_code == 403
        assert "`skills/`" in resp.json()["detail"]
        assert not (project_dir / ".claude" / "skills").exists()

        assert len(app.state.agent_activity_ring) == 1
        frame = app.state.agent_activity_ring[0]
        assert frame["tool"] == "claude_setup_refused"
        assert frame["target"]["kind"] == "config"
        assert ".claude/skills/new/SKILL.md" in frame["target"]["detail"]

    def test_create_file_success_publishes_no_activity(self, app):
        with TestClient(app) as client:
            resp = client.post(
                "/api/claude-setup",
                json={"path": ".claude/agents/new.md", "content": "# ok\n"},
            )

        assert resp.status_code == 200
        assert app.state.agent_activity_ring == []

    def test_listing_carries_the_profile_notice_and_read_only_flags(self, app):
        with TestClient(app) as client:
            body = client.get("/api/claude-setup").json()

        assert body["notice"] == PROFILE_EDIT_NOTICE
        assert "profile" in body["notice"]
        by_path = {f["path"]: f for f in body["files"]}
        assert by_path[".claude/settings.json"]["read_only"] is True
        assert by_path[".claude/agents/test-agent.md"]["read_only"] is False


class TestCreateFile:
    def test_create_in_allowed_dir(self, service, project_dir):
        result = service.create_file(".claude/commands/new-command.md", "# New command\n")
        assert result["status"] == "created"
        assert (project_dir / ".claude" / "commands" / "new-command.md").exists()
        assert result["category"] == "Commands"

    def test_create_in_agents_dir(self, service, project_dir):
        result = service.create_file(".claude/agents/my-agent.md", "# Agent\n")
        assert result["status"] == "created"
        assert result["category"] == "Agents"

    def test_create_outside_allowed_dir(self, service):
        with pytest.raises(PermissionError, match="must be in .claude"):
            service.create_file("src/malicious.py", "import os\n")

    def test_create_in_root(self, service):
        with pytest.raises(PermissionError, match="must be in .claude"):
            service.create_file("evil.md", "# Evil\n")

    def test_create_file_already_exists(self, service):
        """A writable file that exists is still a conflict, not a refusal."""
        with pytest.raises(FileExistsError):
            service.create_file(".claude/agents/test-agent.md", "# Duplicate\n")

    def test_create_validates_json(self, service):
        with pytest.raises(ValueError, match="Invalid JSON"):
            service.create_file(".claude/commands/bad.json", "not json")

    def test_create_subdirectory(self, service, project_dir):
        """Should create parent directories if needed."""
        result = service.create_file(".claude/commands/sub/deep.md", "# Deep\n")
        assert result["status"] == "created"
        assert (project_dir / ".claude" / "commands" / "sub" / "deep.md").exists()

    def test_create_file_refuses_a_path_outside_the_project(self, service):
        """A climbing path is judged by the protected set, which owns it first."""
        with pytest.raises(ProtectedWriteError, match="not project-relative"):
            service.create_file("../../etc/evil.md", "hacked")

    def test_create_file_refuses_a_link_that_escapes_the_project(
        self, service, project_dir, tmp_path
    ):
        """Same on the create route: the resolving reserved check answers first."""
        outside = tmp_path.parent / "outside-the-create-target.md"
        outside.write_text("# Untouched\n")
        (project_dir / ".claude" / "agents" / "escape.md").symlink_to(outside)

        with pytest.raises(ProtectedWriteError, match="not project-relative"):
            service.create_file(".claude/agents/escape.md", "hacked")
        assert outside.read_text() == "# Untouched\n"


class TestCreateFileProtectedSet:
    """``create_file`` consults the protected set before it touches anything.

    Creation is the other half of the same hole: a subtree closed to rewrites
    is not closed at all while a *new* file may be dropped into it.
    """

    @pytest.mark.parametrize(
        ("rel_path", "channel_phrase"),
        [
            (".claude/skills/new/SKILL.md", "`skills/`"),
            (".claude/rules/new.md", "`rules/`"),
        ],
    )
    def test_create_file_refuses_a_new_file_in_a_reserved_subtree(
        self, service, project_dir, audit_dir, rel_path, channel_phrase
    ):
        with pytest.raises(ProtectedWriteError) as excinfo:
            service.create_file(rel_path, "# Authored by the agent\n")

        message = str(excinfo.value)
        # The refusal names the channel that does own the subtree...
        assert channel_phrase in message
        # ...says plainly that nothing landed...
        assert "nothing was written" in message.lower()
        # ...and is not the allowlist message, which would send an operator off
        # to pick a different directory rather than to the profile.
        assert "New files must be in" not in message
        assert not (project_dir / rel_path).exists()

    def test_create_file_allows_an_agent_definition(self, service, project_dir):
        """Authoring subagents is the point of the panel, so that subtree stays open."""
        result = service.create_file(".claude/agents/new.md", "# New agent\n")

        assert result["status"] == "created"
        assert result["category"] == "Agents"
        assert (project_dir / ".claude" / "agents" / "new.md").read_text() == "# New agent\n"

    def test_create_file_refusal_is_audited(self, service, audit_dir):
        with pytest.raises(ProtectedWriteError):
            service.create_file(".claude/skills/new/SKILL.md", "# Skill\n")

        records = _audit_records(audit_dir)
        assert len(records) == 1
        record = records[0]
        assert record["surface"] == "claude_setup"
        assert record["subject"] == ".claude/skills/new/SKILL.md"
        assert "target=.claude/skills/new/SKILL.md" in record["detail"]
        assert is_reserved_write(".claude/skills/new/SKILL.md") in record["detail"]
        assert record["reason"] == "reserved path"

    def test_create_file_success_is_not_audited(self, service, audit_dir):
        service.create_file(".claude/agents/new.md", "# Fine\n")
        assert _audit_records(audit_dir) == []

    def test_create_file_refusal_precedes_the_allowed_dir_check(
        self, service, project_dir, audit_dir
    ):
        """A reserved root file is refused as reserved, not as "wrong directory"."""
        before = (project_dir / ".mcp.json").read_bytes()

        with pytest.raises(ProtectedWriteError) as excinfo:
            service.create_file(".mcp.json", '{"servers": {}}')

        assert "mcp_servers:" in str(excinfo.value)
        assert "New files must be in" not in str(excinfo.value)
        assert (project_dir / ".mcp.json").read_bytes() == before

    def test_create_file_refusal_precedes_content_validation(self, service, project_dir, audit_dir):
        """Invalid JSON aimed at a reserved subtree is refused as reserved, not as syntax."""
        with pytest.raises(ProtectedWriteError):
            service.create_file(".claude/skills/new/skill.json", "not valid json {{{")

        # Not even the parent directory was made on the way to the refusal.
        assert not (project_dir / ".claude" / "skills").exists()


class TestCategorize:
    def test_known_files(self):
        assert ClaudeCodeFileService.categorize("CLAUDE.md", "CLAUDE.md") == "System Prompt"
        assert ClaudeCodeFileService.categorize(".mcp.json", ".mcp.json") == "MCP Servers"

    def test_agent_file(self):
        assert (
            ClaudeCodeFileService.categorize(
                "resolver-agent.md", ".claude/agents/resolver-agent.md"
            )
            == "Agents"
        )

    def test_hooks_file(self):
        assert (
            ClaudeCodeFileService.categorize("pre-check.sh", ".claude/hooks/pre-check.sh")
            == "Hooks"
        )

    def test_commands_file(self):
        assert (
            ClaudeCodeFileService.categorize("deploy.md", ".claude/commands/deploy.md")
            == "Commands"
        )

    def test_unknown_file(self):
        assert ClaudeCodeFileService.categorize("random.txt", "random.txt") == "Other"


class TestDetectLanguage:
    def test_markdown(self):
        assert ClaudeCodeFileService.detect_language("file.md") == "markdown"

    def test_json(self):
        assert ClaudeCodeFileService.detect_language("config.json") == "json"

    def test_yaml_variants(self):
        assert ClaudeCodeFileService.detect_language("config.yml") == "yaml"
        assert ClaudeCodeFileService.detect_language("config.yaml") == "yaml"

    def test_shell(self):
        assert ClaudeCodeFileService.detect_language("script.sh") == "shell"

    def test_python(self):
        assert ClaudeCodeFileService.detect_language("script.py") == "python"

    def test_unknown(self):
        assert ClaudeCodeFileService.detect_language("file.txt") == "text"


class TestSymlinkedReservedTargets:
    """A link is judged by the file it lands on, not by the name it wears.

    Every writer here reaches disk through the filesystem, which follows
    links: a save opens the link's *target*. So an unprotected name pointing
    into a reserved subtree (``.claude/agents/x.md`` -> ``../rules/safety.md``)
    is lexically an agent and physically a rule, and the panel must judge it as
    the latter — the same question ``ownership.reserved_write_channel`` already
    answers for the scaffold gallery.
    """

    @pytest.fixture
    def hook(self, project_dir):
        """An ``osprey_`` hook — the write-safety layer's own protected file."""
        target = project_dir / ".claude" / "hooks" / "osprey_writes_check.py"
        target.write_text("# the write-safety hook\n")
        return target

    @pytest.mark.parametrize(
        ("link_rel", "target_rel", "channel_phrase"),
        [
            (".claude/agents/x.md", ".claude/rules/safety.md", "`rules/`"),
            (
                ".claude/agents/hook-alias.py",
                ".claude/hooks/osprey_writes_check.py",
                "`hooks/`",
            ),
        ],
    )
    def test_write_file_refuses_a_link_onto_a_reserved_file(
        self, service, project_dir, audit_dir, hook, link_rel, target_rel, channel_phrase
    ):
        target = project_dir / target_rel
        before = target.read_bytes()
        (project_dir / link_rel).symlink_to(target)

        with pytest.raises(ProtectedWriteError) as excinfo:
            service.write_file(link_rel, "PWNED")

        message = str(excinfo.value)
        assert channel_phrase in message
        assert "nothing was written" in message.lower()
        assert target.read_bytes() == before

    def test_write_file_refusal_through_a_link_is_audited(self, service, project_dir, audit_dir):
        link_rel = ".claude/agents/x.md"
        (project_dir / link_rel).symlink_to(project_dir / ".claude" / "rules" / "safety.md")

        with pytest.raises(ProtectedWriteError):
            service.write_file(link_rel, "PWNED")

        records = _audit_records(audit_dir)
        assert len(records) == 1
        record = records[0]
        assert record["surface"] == "claude_setup"
        # The record names the path the operator typed, and the channel that
        # owns the file it would have landed on.
        assert f"target={link_rel}" in record["detail"]
        assert reserved_write_channel(project_dir, link_rel) in record["detail"]
        assert record["reason"] == "reserved path"

    def test_create_file_refuses_a_dangling_link_into_a_reserved_subtree(
        self, service, project_dir, audit_dir
    ):
        """Create is the other half: a link with no target yet still lands in ``rules/``."""
        link_rel = ".claude/agents/new-rule.md"
        landing = project_dir / ".claude" / "rules" / "authored-by-the-agent.md"
        (project_dir / link_rel).symlink_to(landing)

        with pytest.raises(ProtectedWriteError) as excinfo:
            service.create_file(link_rel, "# Authored by the agent\n")

        assert "`rules/`" in str(excinfo.value)
        assert not landing.exists()

    def test_list_files_marks_a_link_onto_a_reserved_file_read_only(self, service, project_dir):
        """The badge is the same one a direct reserved file gets."""
        (project_dir / ".claude" / "agents" / "x.md").symlink_to(
            project_dir / ".claude" / "rules" / "safety.md"
        )

        entries = {f["path"]: f for f in service.list_files()}
        assert entries[".claude/rules/safety.md"]["read_only"] is True
        assert entries[".claude/agents/x.md"]["read_only"] is True

    def test_a_link_onto_an_unreserved_file_is_still_writable(self, service, project_dir):
        """The twin: resolution decides, so an ordinary link keeps working.

        Without this the gate could pass by refusing every symlink, which
        would be a different feature — and a broken panel.
        """
        (project_dir / ".claude" / "agents" / "alias.md").symlink_to(
            project_dir / ".claude" / "commands" / "note.md"
        )
        (project_dir / ".claude" / "commands" / "note.md").write_text("# Note\n")

        entries = {f["path"]: f for f in service.list_files()}
        assert entries[".claude/agents/alias.md"]["read_only"] is False

        result = service.write_file(".claude/agents/alias.md", "# Reworked\n")
        assert result["status"] == "saved"
        assert (project_dir / ".claude" / "commands" / "note.md").read_text() == "# Reworked\n"
