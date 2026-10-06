"""Tests for SessionDiscovery."""

from __future__ import annotations

import json
import logging
import time
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


class TestListSessions:
    def test_empty_dir(self, tmp_path, monkeypatch):
        """Missing sessions dir returns empty list."""
        discovery = SessionDiscovery("/nonexistent/project")
        monkeypatch.setattr(
            discovery,
            "_resolve_sessions_dir",
            lambda: tmp_path / "no-such-dir",
        )
        assert discovery.list_sessions() == []

    def test_parses_jsonl(self, tmp_path, monkeypatch):
        """JSONL files are parsed into SessionInfo objects."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        # Create a mock session file
        session_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        session_file = sessions_dir / f"{session_id}.jsonl"
        lines = [
            json.dumps({"type": "queue-operation", "sessionId": session_id}),
            json.dumps(
                {
                    "type": "user",
                    "message": {"content": "Hello, can you help me with beam tuning?"},
                }
            ),
            json.dumps({"type": "assistant", "message": {"content": "Sure!"}}),
        ]
        session_file.write_text("\n".join(lines))

        discovery = SessionDiscovery("/test")
        monkeypatch.setattr(discovery, "_resolve_sessions_dir", lambda: sessions_dir)

        sessions = discovery.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].session_id == session_id
        assert sessions[0].first_message == "Hello, can you help me with beam tuning?"
        assert sessions[0].message_count == 3

    def test_files_without_a_readable_record_are_skipped(self, tmp_path, monkeypatch, caplog):
        """A file no line of which parses as a JSON object is no session."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        valid_id = "11111111-2222-3333-4444-555555555555"
        (sessions_dir / f"{valid_id}.jsonl").write_text(
            json.dumps({"type": "user", "message": {"content": "hi"}})
        )
        corrupt_id = "corrupt-id-aaaa-bbbb-cccc-dddd"
        (sessions_dir / f"{corrupt_id}.jsonl").write_text("{bad json\n{also bad")
        empty_name = "empty-id-aaaa-bbbb-cccc-dddddddd.jsonl"
        (sessions_dir / empty_name).write_text("")

        discovery = SessionDiscovery("/test")
        monkeypatch.setattr(discovery, "_resolve_sessions_dir", lambda: sessions_dir)

        with caplog.at_level(
            logging.DEBUG, logger="osprey.interfaces.web_terminal.session_discovery"
        ):
            sessions = {s.session_id: s for s in discovery.list_sessions()}

        assert set(sessions) == {valid_id}
        messages = [r.getMessage() for r in caplog.records]
        corrupt_logs = [
            m for m in messages if f"{corrupt_id}.jsonl" in m and "no readable record" in m
        ]
        assert len(corrupt_logs) == 1
        assert not any(empty_name in m for m in messages)

    def test_message_count_counts_only_readable_records(self, tmp_path, monkeypatch):
        """Unparseable and blank lines are not counted; the readable record is."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        mixed_id = "mixed-id-aaaa-bbbb-cccc-dddddddddddd"
        (sessions_dir / f"{mixed_id}.jsonl").write_text(
            json.dumps({"type": "user", "message": {"content": "tune the orbit"}})
            + "\n{bad json\n\n"
        )

        discovery = SessionDiscovery("/test")
        monkeypatch.setattr(discovery, "_resolve_sessions_dir", lambda: sessions_dir)

        sessions = discovery.list_sessions()
        assert [s.session_id for s in sessions] == [mixed_id]
        assert sessions[0].message_count == 1
        assert sessions[0].first_message == "tune the orbit"

    def test_json_values_that_are_not_objects_are_not_records(self, tmp_path, monkeypatch):
        """A line that parses to a number, string or list is not a transcript record."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        (sessions_dir / "scalar-id-aaaa-bbbb-cccc-dddddddddddd.jsonl").write_text(
            '42\n"text"\n[1, 2]\n'
        )

        discovery = SessionDiscovery("/test")
        monkeypatch.setattr(discovery, "_resolve_sessions_dir", lambda: sessions_dir)

        assert discovery.list_sessions() == []

    def test_sorted_by_mtime(self, tmp_path, monkeypatch):
        """Sessions are sorted newest-first."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        for i, name in enumerate(["older", "newer"]):
            sid = f"{name}-aaa-bbbb-cccc-dddd-eeeeeeee{i:04d}"
            f = sessions_dir / f"{sid}.jsonl"
            f.write_text(json.dumps({"type": "user", "message": {"content": name}}))
            # Set mtime — older file gets earlier time
            import os

            mtime = time.time() - (100 - i * 50)
            os.utime(f, (mtime, mtime))

        discovery = SessionDiscovery("/test")
        monkeypatch.setattr(discovery, "_resolve_sessions_dir", lambda: sessions_dir)

        sessions = discovery.list_sessions()
        assert len(sessions) == 2
        # Newest (higher mtime) should be first
        assert "newer" in sessions[0].session_id

    def test_multipart_content(self, tmp_path, monkeypatch):
        """Multi-part content (list) extracts first text block."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        session_id = "multi-part-cccc-dddd-eeeeeeeeeeee"
        session_file = sessions_dir / f"{session_id}.jsonl"
        line = json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "text", "text": "Multi-part message here"},
                        {"type": "image", "source": "..."},
                    ],
                },
            }
        )
        session_file.write_text(line)

        discovery = SessionDiscovery("/test")
        monkeypatch.setattr(discovery, "_resolve_sessions_dir", lambda: sessions_dir)

        sessions = discovery.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].first_message == "Multi-part message here"


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
