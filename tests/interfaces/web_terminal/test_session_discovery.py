"""Tests for SessionDiscovery."""

from __future__ import annotations

import json
import logging
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


class TestParseSessionFile:
    def _parse(self, path):
        return SessionDiscovery._parse_session_file(path)

    def test_parses_jsonl(self, tmp_path):
        """A JSONL file parses into a SessionInfo."""
        session_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        path = tmp_path / f"{session_id}.jsonl"
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
        path.write_text("\n".join(lines))

        info = self._parse(path)
        assert info is not None
        assert info.session_id == session_id
        assert info.first_message == "Hello, can you help me with beam tuning?"
        assert info.message_count == 3

    def test_files_without_a_readable_record_are_skipped(self, tmp_path, caplog):
        """A file no line of which parses as a JSON object is no session."""
        corrupt = tmp_path / "corrupt-id-aaaa-bbbb-cccc-dddd.jsonl"
        corrupt.write_text("{bad json\n{also bad")
        empty = tmp_path / "empty-id-aaaa-bbbb-cccc-dddddddd.jsonl"
        empty.write_text("")

        with caplog.at_level(
            logging.DEBUG, logger="osprey.interfaces.web_terminal.session_discovery"
        ):
            assert self._parse(corrupt) is None
            assert self._parse(empty) is None

        messages = [r.getMessage() for r in caplog.records]
        corrupt_logs = [m for m in messages if corrupt.name in m and "no readable record" in m]
        assert len(corrupt_logs) == 1
        assert not any(empty.name in m for m in messages)

    def test_message_count_counts_only_readable_records(self, tmp_path):
        """Unparseable and blank lines are not counted; the readable record is."""
        path = tmp_path / "mixed-id-aaaa-bbbb-cccc-dddddddddddd.jsonl"
        path.write_text(
            json.dumps({"type": "user", "message": {"content": "tune the orbit"}})
            + "\n{bad json\n\n"
        )

        info = self._parse(path)
        assert info is not None
        assert info.message_count == 1
        assert info.first_message == "tune the orbit"

    def test_json_values_that_are_not_objects_are_not_records(self, tmp_path):
        """A line that parses to a number, string or list is not a transcript record."""
        path = tmp_path / "scalar-id-aaaa-bbbb-cccc-dddddddddddd.jsonl"
        path.write_text('42\n"text"\n[1, 2]\n')

        assert self._parse(path) is None

    def test_multipart_content(self, tmp_path):
        """Multi-part content (list) extracts first text block."""
        path = tmp_path / "multi-part-cccc-dddd-eeeeeeeeeeee.jsonl"
        path.write_text(
            json.dumps(
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
        )

        info = self._parse(path)
        assert info is not None
        assert info.first_message == "Multi-part message here"


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
