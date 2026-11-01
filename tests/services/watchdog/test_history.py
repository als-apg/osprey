"""The watchdog's session history: a deadline-bound record read and a bounded view."""

from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest
from tests.agent_record.fakes import FakeRecordClient

from osprey.agent_record.client import URL_ENV, ClientAnswer, RecordClient
from osprey.agent_record.credentials import LOGIN_ENV
from osprey.services.watchdog import history
from osprey.services.watchdog.history import (
    SessionNodes,
    build_session_view_from_nodes,
    read_session_nodes,
    rough_tokens,
)


def _node(kind, key, **body):
    return {"kind": kind, "key": key, "ts": "2026-10-31T00:00:00Z", "body": body}


def _chars(text: str) -> int:
    return len(text)


# ---------------------------------------------------------------------------
# rough_tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "tokens"), [("", 0), ("a", 1), ("abcd", 1), ("abcde", 2)])
def test_rough_tokens_is_a_quarter_of_the_characters_rounded_up(text, tokens):
    assert rough_tokens(text) == tokens


# ---------------------------------------------------------------------------
# The view
# ---------------------------------------------------------------------------


class TestView:
    def test_prompts_answers_tool_calls_and_results_are_all_in_the_view(self):
        nodes = [
            _node("prompt", "l1:0", text="ramp the magnet"),
            _node("answer", "l2:0", text="reading its setpoint"),
            _node("tool_call", "l2:1", name="channel_read", input={"pv": "SR:MAG"}, id="t1"),
            _node("tool_result", "l3:0", content="SR:MAG = 12.5", tool_use_id="t1"),
            _node("thinking", "l4:0", text="private thought"),
        ]
        view = build_session_view_from_nodes(
            nodes, "now write it", max_tokens=10_000, count_tokens=_chars
        )
        assert "USER: ramp the magnet" in view.text
        assert "ASSISTANT: reading its setpoint" in view.text
        assert 'TOOL CALL channel_read({"pv": "SR:MAG"})' in view.text
        assert "TOOL RESULT: SR:MAG = 12.5" in view.text
        assert "private thought" not in view.text
        assert view.text.endswith("USER: now write it")
        assert view.entries_kept == 5
        assert view.entries_cut == 0
        assert view.tokens_estimate == len(view.text)

    def test_a_result_carried_on_its_call_and_an_error_result_are_included(self):
        call = _node("tool_call", "l1:0", name="channel_write", input={"pv": "X"})
        call["result"] = {
            "body": {"content": [{"type": "text", "text": "denied"}], "is_error": True}
        }
        view = build_session_view_from_nodes([call], "go", max_tokens=10_000, count_tokens=_chars)
        assert "TOOL ERROR: denied" in view.text

    def test_truncation_keeps_the_newest_entries_and_reports_the_cut(self):
        nodes = [_node("answer", f"l{i}:0", text=f"answer-{i:02d} " + "x" * 40) for i in range(20)]
        view = build_session_view_from_nodes(
            nodes, "the prompt", max_tokens=300, count_tokens=_chars
        )
        assert view.entries_cut > 0
        assert view.entries_kept + view.entries_cut == 21
        assert "answer-19" in view.text
        assert "answer-00" not in view.text
        kept = [i for i in range(20) if f"answer-{i:02d}" in view.text]
        assert kept == list(range(20 - len(kept), 20))
        assert view.text.startswith(f"[{view.entries_cut} earlier entries cut]")
        assert view.tokens_estimate <= 300

    def test_the_prompt_is_kept_even_when_it_alone_exceeds_the_budget(self):
        nodes = [_node("answer", "l1:0", text="older")]
        prompt = "p" * 500
        view = build_session_view_from_nodes(nodes, prompt, max_tokens=50, count_tokens=_chars)
        assert view.text.endswith(prompt)
        assert "older" not in view.text
        assert view.entries_kept == 1
        assert view.entries_cut == 1

    def test_the_prompt_is_not_doubled_when_the_record_already_has_it(self):
        nodes = [_node("prompt", "l1:0", text="do it")]
        view = build_session_view_from_nodes(nodes, "do it", max_tokens=1000, count_tokens=_chars)
        assert view.text.count("do it") == 1
        assert view.entries_kept == 1

    def test_no_nodes_yield_a_prompt_only_view(self):
        view = build_session_view_from_nodes(
            [], "hello", max_tokens=1000, count_tokens=rough_tokens
        )
        assert view.text == "USER: hello"
        assert (view.entries_kept, view.entries_cut) == (1, 0)
        assert view.tokens_estimate == rough_tokens("USER: hello")

    def test_the_injected_counter_decides_what_fits(self):
        nodes = [_node("answer", f"l{i}:0", text=f"a{i}") for i in range(5)]
        view = build_session_view_from_nodes(nodes, "p", max_tokens=4, count_tokens=lambda _t: 1)
        # prompt (1) + cut line (1) + two newest answers (1 each)
        assert view.entries_kept == 3
        assert view.entries_cut == 3
        assert "a4" in view.text and "a3" in view.text and "a2" not in view.text

    def test_bookkeeping_prompts_are_left_out(self):
        nodes = [_node("prompt", "l1:0", text="meta", is_meta=True)]
        view = build_session_view_from_nodes(nodes, "real", max_tokens=1000, count_tokens=_chars)
        assert "meta" not in view.text

    def test_one_huge_tool_result_is_clipped(self):
        nodes = [_node("tool_result", "l1:0", content="r" * 50_000)]
        view = build_session_view_from_nodes(nodes, "p", max_tokens=100_000, count_tokens=_chars)
        assert len(view.text) < history.MAX_ENTRY_CHARS + 200


# ---------------------------------------------------------------------------
# The read
# ---------------------------------------------------------------------------


class TestRead:
    def test_simple_mode_record_is_read_by_the_record_session_id(self):
        sdk_nodes = [
            _node("prompt", "sdk-sess:1:0", text="hi"),
            _node("answer", "sdk-sess:2:0", text="hello"),
        ]
        fake = FakeRecordClient({"timeline": ClientAnswer(data=sdk_nodes, complete=True)})
        timeouts: list[float] = []

        def make(timeout):
            timeouts.append(timeout)
            return fake

        got = read_session_nodes("sdk-sess", deadline=time.monotonic() + 5.0, make_client=make)
        assert got == SessionNodes(nodes=sdk_nodes)
        assert [c["session_id"] for c in fake.calls_to("timeline")] == ["sdk-sess"]
        assert fake.calls_to("timeline")[0]["wait"] is not None  # the first page drains
        assert 0 < timeouts[0] <= 5.0

    def test_each_call_waits_only_the_time_left_to_the_deadline(self):
        now = [0.0]

        def first_page(**_kwargs):
            now[0] = 3.0  # the first page took three seconds
            return ClientAnswer(data=[_node("answer", "a:0", text="one")], complete=True, next="c1")

        pages = [first_page, ClientAnswer(data=[_node("answer", "b:0", text="two")], complete=True)]
        fake = FakeRecordClient({"timeline": pages})
        clock = lambda: now[0]  # noqa: E731
        timeouts: list[float] = []

        def make(timeout):
            timeouts.append(timeout)
            return fake

        got = read_session_nodes("s", deadline=10.0, make_client=make, clock=clock)
        assert timeouts == [10.0, 7.0]
        assert len(got.nodes) == 2
        assert not got.drain_timed_out

    def test_a_slow_second_page_returns_within_the_deadline_with_drain_timed_out(self):
        first = ClientAnswer(data=[_node("answer", "a:0", text="one")], complete=True, next="c1")

        class SlowSecondPage:
            """Honours its timeout the way httpx does: waits it out, then fails."""

            def __init__(self, timeout):
                self.timeout = timeout
                self._unconfigured = ""

            def timeline(self, _session_id, *, after=None, **_kwargs):
                if after is None:
                    return first
                time.sleep(min(self.timeout, 5.0))
                return ClientAnswer.failed(f"agent-record did not answer within {self.timeout:g} s")

        started = time.monotonic()
        got = read_session_nodes("s", deadline=started + 0.3, make_client=SlowSecondPage)
        elapsed = time.monotonic() - started
        assert elapsed < 1.0
        assert got.drain_timed_out
        assert got.record_available
        assert [n["key"] for n in got.nodes] == ["a:0"]
        assert got.reason

    def test_a_past_deadline_makes_no_call(self):
        made: list[float] = []
        got = read_session_nodes(
            "s", deadline=time.monotonic() - 1.0, make_client=lambda t: made.append(t)
        )
        assert made == []
        assert got.drain_timed_out
        assert not got.record_available
        assert got.nodes == []

    def test_no_record_env_reports_the_record_unavailable(self, monkeypatch):
        monkeypatch.delenv(URL_ENV, raising=False)
        monkeypatch.delenv(LOGIN_ENV, raising=False)
        got = read_session_nodes("s", deadline=time.monotonic() + 5.0)
        assert got.record_available is False
        assert got.drain_timed_out is False
        assert got.nodes == []
        assert "not configured" in got.reason
        view = build_session_view_from_nodes(
            got.nodes, "the prompt", max_tokens=1000, count_tokens=rough_tokens
        )
        assert view.text == "USER: the prompt"

    def test_the_default_client_is_built_from_the_environment_with_the_remaining_time(
        self, monkeypatch
    ):
        built: list[float] = []
        real = RecordClient.from_env

        def spy(_env, *, timeout, **_kwargs):
            built.append(timeout)
            return real({}, timeout=timeout)

        monkeypatch.setattr(RecordClient, "from_env", staticmethod(spy))
        read_session_nodes("s", deadline=time.monotonic() + 2.0)
        assert len(built) == 1 and 0 < built[0] <= 2.0

    def test_an_unreachable_record_is_unavailable_but_not_timed_out(self):
        fake = FakeRecordClient(unavailable="agent-record is unreachable: ConnectError: boom")
        got = read_session_nodes("s", deadline=time.monotonic() + 5.0, make_client=lambda t: fake)
        assert not got.record_available
        assert not got.drain_timed_out

    def test_no_session_id_is_unavailable_without_a_call(self):
        got = read_session_nodes(None, deadline=time.monotonic() + 5.0, make_client=pytest.fail)
        assert not got.record_available


def test_history_imports_nothing_from_interfaces():
    source = Path(history.__file__).read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("osprey.interfaces")
        elif isinstance(node, ast.Import):
            assert not any(a.name.startswith("osprey.interfaces") for a in node.names)
