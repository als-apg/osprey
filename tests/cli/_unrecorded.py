"""A host-local record, and a child that would record into it, for the unrecorded-run tests.

``osprey audit`` and the channel-finder benchmark hold their runs in
``unrecorded_run``; these helpers stand up what a host launch publishes and
play the Claude Code child that would otherwise write a spool line, a
transcript and an OTLP batch.
"""

from __future__ import annotations

import os
from pathlib import Path


def files_under(root) -> list:
    return [path for path in root.rglob("*") if path.is_file()] if root.exists() else []


def a_host_local_record_is_up(monkeypatch, tmp_path):
    """Publish what a host launch hands its children; return the transcript root.

    The spool directory is the test's own (set by the suite's spool zone); the
    record URL, an OTLP exporter and Claude Code's config dir are set beside it.
    """
    from osprey.agent_record.client import URL_ENV

    transcript_root = tmp_path / "claude-config"
    transcript_root.mkdir()
    monkeypatch.setenv(URL_ENV, "http://127.0.0.1:1")
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_TELEMETRY", "1")
    monkeypatch.setenv("OTEL_LOGS_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(transcript_root))
    return transcript_root


def act_as_a_recording_child(spool_root) -> dict:
    """Do what a Claude Code child and its hooks would do with the inherited env.

    It writes an audit line through the record writer, writes its transcript
    under ``$CLAUDE_CONFIG_DIR``, and, when any OTLP exporter variable reached
    it, drops an OTLP batch under the spool root. Returns what it saw.
    """
    from osprey.agent_record import writer
    from osprey.agent_record.contract import KIND_AUDIT

    entry_id = writer.write(KIND_AUDIT, {"tool": "Read"})
    transcripts = Path(os.environ["CLAUDE_CONFIG_DIR"]) / "projects" / "run"
    transcripts.mkdir(parents=True, exist_ok=True)
    (transcripts / "session.jsonl").write_text("{}\n")
    otel = sorted(
        key
        for key in os.environ
        if key.startswith("OTEL_") or key == "CLAUDE_CODE_ENABLE_TELEMETRY"
    )
    if otel:
        batch = spool_root / "otlp" / "alice"
        batch.mkdir(parents=True, exist_ok=True)
        (batch / "batch.json").write_text("{}")
    return {
        "entry_id": entry_id,
        "otel": otel,
        "spool_dir": os.environ.get("OSPREY_SPOOL_DIR"),
        "config_dir": os.environ["CLAUDE_CONFIG_DIR"],
    }
