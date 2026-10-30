"""Tests for the prior_answer_read MCP tool.

The tool reads an earlier answer of this conversation back from the agent
record, narrowed to the runs the worker stamped into the environment at spawn,
and only for those runs.
"""

import json

import pytest
from fastmcp.exceptions import ToolError

from osprey.agent_record.client import ClientAnswer
from osprey.mcp_server.workspace.tools import prior_answer
from tests.agent_record.fakes import FakeRecordClient
from tests.mcp_server.conftest import get_tool_fn

_ENV = "OSPREY_DISPATCH_PRIOR_ANSWER_RUNS"
_RUN = "3f2b6c1e-8a4d-4f0e-9b1a-2c3d4e5f6a7b"
_OTHER = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"


def _fn():
    from osprey.mcp_server.workspace.tools.prior_answer import prior_answer_read

    return get_tool_fn(prior_answer_read)


def _error_type(exc: ToolError) -> str:
    return json.loads(str(exc))["error_type"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)


def _missing(run_id: str) -> ClientAnswer:
    """What the record answers for a run it holds no entry of."""
    return ClientAnswer(
        data=None, complete=False, reason=f"run {run_id} has no accept or terminal entry yet"
    )


@pytest.fixture
def run_store(monkeypatch):
    """The record the tool reads: each written run's answer text, by run id."""
    answers: dict[str, str | None] = {}

    def _run(run_id, **_kwargs):
        if run_id not in answers:
            return _missing(run_id)
        facts = {"run_id": run_id, "status": "completed", "answer": answers[run_id]}
        return ClientAnswer(data=facts, complete=True)

    fake = FakeRecordClient({"run": _run})
    monkeypatch.setattr(prior_answer, "record_client", lambda: fake)

    def _write(run_id: str, text: str | None) -> None:
        answers[run_id] = text

    _write.client = fake
    return _write


@pytest.mark.asyncio
async def test_a_permitted_run_returns_its_full_answer(monkeypatch, run_store):
    run_store(_RUN, "the whole table")
    monkeypatch.setenv(_ENV, _RUN)

    page = json.loads(await _fn()(run_id=_RUN))

    assert page == {
        "run_id": _RUN,
        "total_chars": 15,
        "offset": 0,
        "text": "the whole table",
        "next_offset": None,
    }


@pytest.mark.asyncio
async def test_the_read_is_narrowed_to_the_permitted_runs(monkeypatch, run_store):
    run_store(_RUN, "answer")
    monkeypatch.setenv(_ENV, f"{_RUN},{_OTHER}")

    await _fn()(run_id=_RUN)

    [call] = run_store.client.calls_to("run")
    assert call["run_id"] == _RUN
    assert set(call["narrow_runs"]) == {_RUN, _OTHER}


@pytest.mark.asyncio
async def test_a_run_outside_the_set_is_refused_before_anything_is_read(monkeypatch, run_store):
    monkeypatch.setenv(_ENV, _RUN)

    with pytest.raises(ToolError) as info:
        await _fn()(run_id=_OTHER)

    assert _error_type(info.value) == "not_permitted"
    assert run_store.client.calls == []


@pytest.mark.asyncio
async def test_outside_a_dispatched_run_every_run_is_refused(run_store):
    run_store(_RUN, "answer")

    with pytest.raises(ToolError) as info:
        await _fn()(run_id=_RUN)

    assert _error_type(info.value) == "not_permitted"


@pytest.mark.asyncio
@pytest.mark.usefixtures("run_store")
async def test_a_swept_run_says_the_answer_is_no_longer_available(monkeypatch):
    monkeypatch.setenv(_ENV, _RUN)

    with pytest.raises(ToolError) as info:
        await _fn()(run_id=_RUN)

    assert _error_type(info.value) == "no_longer_available"


@pytest.mark.asyncio
async def test_a_record_without_text_is_no_longer_available(monkeypatch, run_store):
    run_store(_RUN, "")
    monkeypatch.setenv(_ENV, _RUN)

    with pytest.raises(ToolError) as info:
        await _fn()(run_id=_RUN)

    assert _error_type(info.value) == "no_longer_available"


@pytest.mark.asyncio
async def test_a_long_answer_is_read_in_pages_that_join_to_the_whole(monkeypatch, run_store):
    original = "".join(chr(ord("a") + i % 26) for i in range(100_001))
    run_store(_RUN, original)
    monkeypatch.setenv(_ENV, _RUN)

    pages = []
    offset: int | None = 0
    while offset is not None:
        page = json.loads(await _fn()(run_id=_RUN, offset=offset))
        pages.append(page)
        offset = page["next_offset"]

    assert len(pages) == 3
    assert pages[-1]["next_offset"] is None
    assert all(p["total_chars"] == 100_001 for p in pages)
    assert "".join(p["text"] for p in pages) == original


@pytest.mark.asyncio
async def test_an_offset_past_the_end_is_refused(monkeypatch, run_store):
    run_store(_RUN, "short")
    monkeypatch.setenv(_ENV, _RUN)

    with pytest.raises(ToolError) as info:
        await _fn()(run_id=_RUN, offset=6)

    assert _error_type(info.value) == "bad_offset"


@pytest.mark.asyncio
async def test_an_unavailable_record_is_said_never_served_as_expired(monkeypatch):
    monkeypatch.setattr(prior_answer, "record_client", lambda: FakeRecordClient(unavailable="down"))
    monkeypatch.setenv(_ENV, _RUN)

    with pytest.raises(ToolError) as info:
        await _fn()(run_id=_RUN)

    assert _error_type(info.value) == "record_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        ClientAnswer(data=None, complete=False, reason="wait mark not passed within 10 s"),
        ClientAnswer(
            data={"run_id": _RUN, "answer": "the first half"},
            complete=False,
            reason=f"run {_RUN}'s answer is incomplete",
        ),
    ],
)
async def test_an_incomplete_read_is_said_never_served_as_the_answer(monkeypatch, answer):
    monkeypatch.setattr(prior_answer, "record_client", lambda: FakeRecordClient({"run": answer}))
    monkeypatch.setenv(_ENV, _RUN)

    with pytest.raises(ToolError) as info:
        await _fn()(run_id=_RUN)

    assert _error_type(info.value) == "incomplete"


def test_the_tool_needs_no_prompt():
    from osprey.registry.mcp import FRAMEWORK_SERVERS

    assert "prior_answer_read" in FRAMEWORK_SERVERS["osprey_workspace"].permissions_allow
