"""The opt-in full tool-call record: its settings, its payload bound, its shape."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from osprey.audit import tool_call, writer


def test_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    assert tool_call.settings() == (False, tool_call.DEFAULT_MAX_INLINE_BYTES)

    monkeypatch.setenv("OSPREY_CONFIG", "relative/config.yml")
    assert tool_call.settings() == (False, tool_call.DEFAULT_MAX_INLINE_BYTES)


def test_the_documented_keys_and_default() -> None:
    assert tool_call.ENABLED_KEY == "audit.tool_call.enabled"
    assert tool_call.MAX_INLINE_KEY == "audit.tool_call.max_inline_bytes"
    assert tool_call.DEFAULT_MAX_INLINE_BYTES == 262144
    assert tool_call.SURFACE_TOOL_CALL == "tool_call"


def test_settings_read_from_osprey_config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "build" / "config.yml"
    config.parent.mkdir()
    config.write_text("audit:\n  tool_call:\n    enabled: true\n    max_inline_bytes: 1000\n")
    monkeypatch.setenv("OSPREY_CONFIG", str(config))

    assert tool_call.settings() == (True, 1000)


def test_an_unusable_max_falls_back_to_the_default(tmp_path, monkeypatch) -> None:
    config = tmp_path / "config.yml"
    config.write_text("audit:\n  tool_call:\n    enabled: true\n    max_inline_bytes: -5\n")
    monkeypatch.setenv("OSPREY_CONFIG", str(config))

    assert tool_call.settings() == (True, tool_call.DEFAULT_MAX_INLINE_BYTES)


def test_a_small_payload_stays_inline() -> None:
    value = {"channel": "A:1", "value": 3}
    assert tool_call.capped(
        value, label="arguments", subject="s", tool_use_id="toolu_1", max_inline=1000
    ) == (value, None)


class _Store:
    def __init__(self, fail: bool = False) -> None:
        self.saved: list[dict] = []
        self.fail = fail

    def save_file(self, **kwargs):
        if self.fail:
            raise OSError("disk full")
        self.saved.append(kwargs)
        return SimpleNamespace(id="art-42")


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _Store:
    fake = _Store()
    monkeypatch.setattr("osprey.stores.artifact_store.get_artifact_store", lambda: fake)
    return fake


@pytest.mark.usefixtures("store")
def test_a_large_payload_becomes_a_reference_with_its_sha256() -> None:
    value = {"data": "x" * 500}
    encoded = json.dumps(value, separators=(",", ":")).encode()

    inline, reference = tool_call.capped(
        value,
        label="result",
        subject="mcp__controls__channel_read",
        tool_use_id="toolu_1",
        max_inline=100,
    )

    assert inline is None
    assert reference == {
        "size": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "artifact_id": "art-42",
    }


def test_the_reference_names_the_saved_artifact(store: _Store) -> None:
    tool_call.capped(
        "y" * 300, label="arguments", subject="mcp__x__t", tool_use_id="toolu_9", max_inline=10
    )

    (saved,) = store.saved
    assert saved["filename"] == "toolu_9-arguments.json"
    assert saved["origin"] == tool_call.ARTIFACT_ORIGIN
    assert saved["tool_source"] == "audit.tool_call"
    assert saved["mime_type"] == "application/json"
    assert saved["metadata"]["tool_use_id"] == "toolu_9"
    assert json.loads(saved["file_content"]) == "y" * 300


def test_a_failed_save_keeps_size_and_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "osprey.stores.artifact_store.get_artifact_store", lambda: _Store(fail=True)
    )
    _inline, reference = tool_call.capped(
        "z" * 300, label="result", subject="s", tool_use_id=None, max_inline=10
    )

    assert reference is not None
    assert reference["artifact_id"] is None
    assert reference["artifact_error"] == "OSError"
    assert reference["size"] == len(json.dumps("z" * 300).encode())
    assert len(reference["sha256"]) == 64


def test_a_tool_result_serializes_its_blocks() -> None:
    from fastmcp.tools.base import ToolResult
    from mcp.types import TextContent

    result = ToolResult(
        content=[TextContent(type="text", text="hello")], structured_content={"a": 1}
    )

    assert tool_call.serialize_result(result) == {
        "content": [{"type": "text", "text": "hello"}],
        "structured_content": {"a": 1},
    }
    assert tool_call.serialize_result({"plain": [1, 2]}) == {"plain": [1, 2]}


def test_append_record_is_one_line(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(writer, "audit_dir", lambda: tmp_path / "var" / "audit")
    monkeypatch.setenv("OSPREY_AUDIT_IDENTITY", "alice")
    monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
    monkeypatch.delenv(writer.AUDIT_WRITER_ENV, raising=False)
    big = {"surface": "tool_call", "result": "r" * (writer.MAX_RECORD_BYTES * 4)}

    path = writer.append_record(tool_call.SURFACE_TOOL_CALL, big)

    assert path == tmp_path / "var" / "audit" / "alice" / "tool_call.jsonl"
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == big


def test_append_record_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken():
        raise OSError("gone")

    monkeypatch.setattr(writer, "audit_dir", broken)
    assert writer.append_record("tool_call", {"a": 1}) is None


def test_the_record_keeps_its_documented_order() -> None:
    record = tool_call.build_record(
        ts="t",
        actor="a",
        posture="writes",
        posture_source="process",
        session=None,
        session_id="c",
        tool_use_id="u",
        server="controls",
        subject="s",
        decision="allowed",
        reason="tool_call",
        approval=None,
        target="va",
        generation=3,
        arguments={"x": 1},
        arguments_ref=None,
        result=None,
        result_ref={"size": 1},
        error=None,
        is_error=False,
        facts={},
        duration_ms=1.0,
    )
    assert list(record) == [
        "ts",
        "surface",
        "actor",
        "posture",
        "posture_source",
        "session",
        "session_id",
        "tool_use_id",
        "server",
        "subject",
        "decision",
        "reason",
        "approval",
        "target",
        "generation",
        "arguments",
        "result_ref",
        "error",
        "is_error",
        "facts",
        "duration_ms",
    ]
