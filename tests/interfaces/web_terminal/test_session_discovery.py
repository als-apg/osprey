"""Tests for SessionDiscovery."""

from __future__ import annotations

from pathlib import Path

from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery


class TestResolveSessionsDir:
    def test_honours_claude_config_dir(self, tmp_path, monkeypatch):
        """``CLAUDE_CONFIG_DIR`` names the state root; ``~/.claude`` is only the fallback.

        The per-user web-terminal container sets both ``CLAUDE_CONFIG_DIR``
        and ``HOME`` to the mounted volume, so a ``~/.claude/projects`` spelling
        looks one ``.claude`` too deep, never sees the session Claude Code
        writes, and the terminal never learns its own session id.
        """
        config_dir = tmp_path / "data" / "claude-config"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: config_dir))

        sessions_dir = SessionDiscovery("/app/project/build")._resolve_sessions_dir()

        assert sessions_dir == config_dir / "projects" / "-app-project-build"


class TestSnapshotSessionIds:
    def test_snapshot_missing_dir(self, tmp_path, monkeypatch):
        """Snapshot on missing dir returns empty set."""
        discovery = SessionDiscovery("/test")
        monkeypatch.setattr(
            discovery,
            "_resolve_sessions_dir",
            lambda: tmp_path / "no-such-dir",
        )
        assert discovery.snapshot_session_ids() == set()
