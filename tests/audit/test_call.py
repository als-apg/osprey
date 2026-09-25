"""Tests for the tool-call scope every audit record reads its tool-use id from."""

from __future__ import annotations

import anyio
import pytest

from osprey.audit import call


def test_scope_sets_and_resets() -> None:
    assert call.current_call() is None
    with call.call_scope("toolu_1", "conv-1") as facts:
        assert call.current_call() is facts
        assert call.current_tool_use_id() == "toolu_1"
        assert facts.session_id == "conv-1"
        assert facts.facts == {}
    assert call.current_call() is None
    assert call.current_tool_use_id() is None


def test_scope_resets_on_exception() -> None:
    with pytest.raises(RuntimeError):
        with call.call_scope("toolu_1", None):
            raise RuntimeError("boom")
    assert call.current_call() is None


def test_nested_scope_restores_the_outer_one() -> None:
    with call.call_scope("toolu_outer", None) as outer:
        with call.call_scope("toolu_inner", None):
            assert call.current_tool_use_id() == "toolu_inner"
        assert call.current_call() is outer


@pytest.mark.parametrize("value", ["", "a/b", "x" * 129, None, 123, "toolu x", "../x"])
def test_invalid_ids_are_dropped(value: object) -> None:
    assert call.valid_tool_use_id(value) is None


@pytest.mark.parametrize("value", ["toolu_01AbC", "a", "A-b_9", "x" * 128])
def test_valid_ids_are_kept(value: str) -> None:
    assert call.valid_tool_use_id(value) == value


def test_facts_seen_across_to_thread() -> None:
    """A worker thread shares the holder, so what it notes is seen after."""

    seen: list[object] = []

    async def main() -> None:
        with call.call_scope("toolu_1", None) as facts:
            await anyio.to_thread.run_sync(lambda: seen.append(call.current_call()))
            assert seen == [facts]

    anyio.run(main)


def test_harness_session_id_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OSPREY_TELEMETRY_SESSION_ID", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    assert call.harness_session_id() is None

    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "harness-1")
    assert call.harness_session_id() == "harness-1"

    monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", "forced-1")
    assert call.harness_session_id() == "forced-1"

    monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", "")
    assert call.harness_session_id() == "harness-1"


def test_meta_key_is_the_harness_spelling() -> None:
    assert call.TOOL_USE_ID_META_KEY == "claudecode/toolUseId"
