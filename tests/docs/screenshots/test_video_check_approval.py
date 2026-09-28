"""Tests for the approval-log checks of the demo video recorder.

Pure file parsing over temporary ``hook_debug.jsonl`` fixtures; safe for CI.
"""

from __future__ import annotations

import json
from pathlib import Path

from docs.screenshots import video_check_approval as vca
from docs.screenshots.video_check_approval import Ask


def _record(tool: str, status: str, detail: str | None = None, tool_use_id: str | None = None):
    rec = {"ts": "2026-09-24T10:00:00", "hook": "approval", "tool": tool, "status": status}
    if tool_use_id is not None:
        rec["tool_use_id"] = tool_use_id
    if detail is not None:
        rec["detail"] = detail
    return json.dumps(rec) + "\n"


def _append(path: Path, *lines: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.writelines(lines)


def test_log_offset_missing_file_is_zero(tmp_path):
    assert vca.log_offset(tmp_path / "hook_debug.jsonl") == 0


def test_log_offset_is_file_size(tmp_path):
    path = tmp_path / "hook_debug.jsonl"
    _append(path, _record("mcp__ariel__entry_create", "allow"))
    assert vca.log_offset(path) == path.stat().st_size


def test_find_asks_missing_file_is_empty(tmp_path):
    assert vca.find_asks(tmp_path / "hook_debug.jsonl", 0) == []


def test_file_appearing_after_offset(tmp_path):
    path = tmp_path / "hook_debug.jsonl"
    offset = vca.log_offset(path)
    assert offset == 0
    _append(
        path,
        _record(
            "mcp__ariel__entry_create",
            "ask",
            detail="policy=always tool=entry_create",
            tool_use_id="toolu_1",
        ),
    )
    assert vca.find_asks(path, offset) == [Ask(tool_use_id="toolu_1", name="entry_create")]


def test_only_records_after_offset_are_returned(tmp_path):
    path = tmp_path / "hook_debug.jsonl"
    _append(
        path, _record("mcp__ariel__entry_create", "ask", "policy=always tool=entry_create", "old")
    )
    offset = vca.log_offset(path)
    _append(
        path, _record("mcp__ariel__entry_publish", "ask", "policy=always tool=entry_publish", "new")
    )
    assert vca.find_asks(path, offset) == [Ask(tool_use_id="new", name="entry_publish")]


def test_entry_create_then_entry_publish(tmp_path):
    path = tmp_path / "hook_debug.jsonl"
    offset = vca.log_offset(path)
    _append(
        path,
        _record("mcp__ariel__entry_create", "ask", "policy=always tool=entry_create", "toolu_c"),
        _record("mcp__ariel__entry_create", "allow", None, "toolu_x"),
        _record("mcp__ariel__entry_publish", "ask", "policy=always tool=entry_publish", "toolu_p"),
    )
    asks = vca.find_asks(path, offset)
    assert [a.name for a in asks] == ["entry_create", "entry_publish"]
    assert [a.tool_use_id for a in asks] == ["toolu_c", "toolu_p"]


def test_execute_selective_takes_name_from_tool_field(tmp_path):
    path = tmp_path / "hook_debug.jsonl"
    _append(path, _record("mcp__python__execute", "ask", "execute_selective", "toolu_e"))
    assert vca.find_asks(path, 0) == [Ask(tool_use_id="toolu_e", name="execute")]


def test_channel_write_selective_takes_name_from_tool_field(tmp_path):
    path = tmp_path / "hook_debug.jsonl"
    _append(
        path, _record("mcp__controls__channel_write", "ask", "channel_write_selective", "toolu_w")
    )
    assert vca.find_asks(path, 0) == [Ask(tool_use_id="toolu_w", name="channel_write")]


def test_non_mcp_tool_name_is_kept(tmp_path):
    path = tmp_path / "hook_debug.jsonl"
    _append(path, _record("Bash", "ask", None, "toolu_b"))
    assert vca.find_asks(path, 0) == [Ask(tool_use_id="toolu_b", name="Bash")]


def test_missing_tool_use_id_is_none(tmp_path):
    path = tmp_path / "hook_debug.jsonl"
    _append(path, _record("mcp__ariel__entry_create", "ask", "policy=always tool=entry_create"))
    assert vca.find_asks(path, 0) == [Ask(tool_use_id=None, name="entry_create")]


def test_malformed_lines_are_skipped(tmp_path):
    path = tmp_path / "hook_debug.jsonl"
    _append(
        path,
        "not json at all\n",
        "\n",
        "[1, 2, 3]\n",
        '"just a string"\n',
        _record("mcp__ariel__entry_create", "ask", "policy=always tool=entry_create", "toolu_ok"),
        '{"status": "ask", "tool": 42}\n',
        '{"status": "ask"}\n',
        "\xff\xfe garbage\n",
        '{"status": "ask", "tool": "mcp__ariel__entry_publish", "tool_use_id": "tr',
    )
    assert vca.find_asks(path, 0) == [Ask(tool_use_id="toolu_ok", name="entry_create")]


def test_invalid_utf8_bytes_are_skipped(tmp_path):
    path = tmp_path / "hook_debug.jsonl"
    path.write_bytes(b"\xff\xfe\x00bad\n" + _record("Bash", "ask", None, "t").encode())
    assert vca.find_asks(path, 0) == [Ask(tool_use_id="t", name="Bash")]


def test_offset_beyond_file_size_is_empty(tmp_path):
    path = tmp_path / "hook_debug.jsonl"
    _append(path, _record("Bash", "ask", None, "t"))
    assert vca.find_asks(path, 10_000) == []


def test_hook_log_path(tmp_path):
    assert (
        vca.hook_log_path(tmp_path) == tmp_path / "build" / ".claude" / "hooks" / "hook_debug.jsonl"
    )
