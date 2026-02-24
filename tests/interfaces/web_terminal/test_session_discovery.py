"""Tests for SessionDiscovery."""

from __future__ import annotations

import json
import threading
import time

from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery


class TestResolveSessionsDir:
    def test_path_encoding(self, tmp_path):
        """Verify /Users/x/proj encodes to -Users-x-proj."""
        discovery = SessionDiscovery("/Users/x/proj")
        sessions_dir = discovery._resolve_sessions_dir()
        assert sessions_dir.name == "-Users-x-proj"
        assert sessions_dir.parent.name == "projects"

    def test_leading_dash_preserved(self, tmp_path):
        """Leading - from / replacement is preserved (matches Claude Code)."""
        discovery = SessionDiscovery("/foo/bar")
        sessions_dir = discovery._resolve_sessions_dir()
        # /foo/bar -> -foo-bar (leading - kept)
        assert sessions_dir.name == "-foo-bar"


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
            json.dumps({
                "type": "user",
                "message": {"content": "Hello, can you help me with beam tuning?"},
            }),
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

    def test_skips_corrupt_files(self, tmp_path, monkeypatch):
        """Corrupt JSONL files are skipped gracefully."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        # Valid file
        valid_id = "11111111-2222-3333-4444-555555555555"
        valid_file = sessions_dir / f"{valid_id}.jsonl"
        valid_file.write_text(json.dumps({"type": "user", "message": {"content": "hi"}}))

        # Corrupt file (not valid JSON)
        corrupt_file = sessions_dir / "corrupt-id-aaaa-bbbb-cccc-dddd.jsonl"
        corrupt_file.write_text("{bad json\n{also bad")

        # Empty file
        empty_file = sessions_dir / "empty-id-aaaa-bbbb-cccc-dddddddd.jsonl"
        empty_file.write_text("")

        discovery = SessionDiscovery("/test")
        monkeypatch.setattr(discovery, "_resolve_sessions_dir", lambda: sessions_dir)

        sessions = discovery.list_sessions()
        # Should have valid + corrupt (corrupt has 2 non-empty lines but no valid user msg)
        valid_ids = {s.session_id for s in sessions}
        assert valid_id in valid_ids
        # Empty file should be skipped (0 bytes)

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
        line = json.dumps({
            "type": "user",
            "message": {
                "content": [
                    {"type": "text", "text": "Multi-part message here"},
                    {"type": "image", "source": "..."},
                ],
            },
        })
        session_file.write_text(line)

        discovery = SessionDiscovery("/test")
        monkeypatch.setattr(discovery, "_resolve_sessions_dir", lambda: sessions_dir)

        sessions = discovery.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].first_message == "Multi-part message here"


class TestSnapshotAndDiscover:
    def test_snapshot_and_discover(self, tmp_path, monkeypatch):
        """Snapshot before, create file, discover returns new UUID."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        # Pre-existing session
        existing = sessions_dir / "existing-aaa-bbbb-cccc-ddddddddddd.jsonl"
        existing.write_text(json.dumps({"type": "user", "message": {"content": "old"}}))

        discovery = SessionDiscovery("/test")
        monkeypatch.setattr(discovery, "_resolve_sessions_dir", lambda: sessions_dir)

        before = discovery.snapshot_session_ids()
        assert "existing-aaa-bbbb-cccc-ddddddddddd" in before

        # Simulate Claude creating a new session file
        new_id = "new-sess-bbbb-cccc-dddd-eeeeeeeeeeee"
        new_file = sessions_dir / f"{new_id}.jsonl"

        def create_after_delay():
            time.sleep(0.3)
            new_file.write_text(json.dumps({"type": "user", "message": {"content": "hi"}}))

        t = threading.Thread(target=create_after_delay)
        t.start()

        result = discovery.discover_new_session(before, timeout=5.0)
        t.join()
        assert result == new_id

    def test_discover_timeout(self, tmp_path, monkeypatch):
        """No new file within timeout returns None."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        discovery = SessionDiscovery("/test")
        monkeypatch.setattr(discovery, "_resolve_sessions_dir", lambda: sessions_dir)

        before = discovery.snapshot_session_ids()
        result = discovery.discover_new_session(before, timeout=1.0)
        assert result is None

    def test_snapshot_missing_dir(self, tmp_path, monkeypatch):
        """Snapshot on missing dir returns empty set."""
        discovery = SessionDiscovery("/test")
        monkeypatch.setattr(
            discovery,
            "_resolve_sessions_dir",
            lambda: tmp_path / "no-such-dir",
        )
        assert discovery.snapshot_session_ids() == set()


class TestAllowedIdsFilter:
    def test_filters_to_allowed_ids(self, tmp_path, monkeypatch):
        """list_sessions only returns sessions in allowed_ids."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        id_a = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        id_b = "11111111-2222-3333-4444-555555555555"
        for sid in [id_a, id_b]:
            f = sessions_dir / f"{sid}.jsonl"
            f.write_text(json.dumps({"type": "user", "message": {"content": sid[:8]}}))

        discovery = SessionDiscovery("/test")
        monkeypatch.setattr(discovery, "_resolve_sessions_dir", lambda: sessions_dir)

        # No filter — returns both
        assert len(discovery.list_sessions()) == 2

        # Filter to just id_a
        result = discovery.list_sessions(allowed_ids={id_a})
        assert len(result) == 1
        assert result[0].session_id == id_a

    def test_empty_allowed_ids_returns_empty(self, tmp_path, monkeypatch):
        """Empty allowed_ids set returns no sessions."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        sid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        f = sessions_dir / f"{sid}.jsonl"
        f.write_text(json.dumps({"type": "user", "message": {"content": "hi"}}))

        discovery = SessionDiscovery("/test")
        monkeypatch.setattr(discovery, "_resolve_sessions_dir", lambda: sessions_dir)

        result = discovery.list_sessions(allowed_ids=set())
        assert result == []

    def test_none_allowed_ids_returns_all(self, tmp_path, monkeypatch):
        """None allowed_ids (default) returns all sessions."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        sid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        f = sessions_dir / f"{sid}.jsonl"
        f.write_text(json.dumps({"type": "user", "message": {"content": "hi"}}))

        discovery = SessionDiscovery("/test")
        monkeypatch.setattr(discovery, "_resolve_sessions_dir", lambda: sessions_dir)

        result = discovery.list_sessions(allowed_ids=None)
        assert len(result) == 1


