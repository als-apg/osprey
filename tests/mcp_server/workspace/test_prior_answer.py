"""Tests for the prior_answer_read MCP tool.

The tool reads an earlier answer of this conversation back from the dispatch
worker's run record, and only for the runs the worker stamped into the
environment at spawn.
"""

import json

import pytest
from fastmcp.exceptions import ToolError

from osprey.agent_runner import artifact_resolve
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


@pytest.fixture
def run_store(monkeypatch, tmp_path):
    monkeypatch.setattr(artifact_resolve, "dispatch_log_dir", lambda: tmp_path)

    def _write(run_id: str, record: dict) -> None:
        (tmp_path / f"{run_id}.json").write_text(json.dumps(record))

    return _write


@pytest.mark.asyncio
async def test_a_permitted_run_returns_its_full_answer(monkeypatch, run_store):
    run_store(_RUN, {"text_output": "the whole table"})
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
async def test_a_run_outside_the_set_is_refused_before_anything_is_read(monkeypatch):
    def _must_not_load(run_id):
        raise AssertionError(f"run store read for {run_id}")

    monkeypatch.setattr(artifact_resolve, "load_run_record", _must_not_load)
    monkeypatch.setenv(_ENV, _RUN)

    with pytest.raises(ToolError) as info:
        await _fn()(run_id=_OTHER)

    assert _error_type(info.value) == "not_permitted"


@pytest.mark.asyncio
async def test_outside_a_dispatched_run_every_run_is_refused(run_store):
    run_store(_RUN, {"text_output": "answer"})

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
    run_store(_RUN, {"status": "completed", "text_output": ""})
    monkeypatch.setenv(_ENV, _RUN)

    with pytest.raises(ToolError) as info:
        await _fn()(run_id=_RUN)

    assert _error_type(info.value) == "no_longer_available"


@pytest.mark.asyncio
async def test_a_long_answer_is_read_in_pages_that_join_to_the_whole(monkeypatch, run_store):
    original = "".join(chr(ord("a") + i % 26) for i in range(100_001))
    run_store(_RUN, {"text_output": original})
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
    run_store(_RUN, {"text_output": "short"})
    monkeypatch.setenv(_ENV, _RUN)

    with pytest.raises(ToolError) as info:
        await _fn()(run_id=_RUN, offset=6)

    assert _error_type(info.value) == "bad_offset"


def test_the_tool_needs_no_prompt():
    from osprey.registry.mcp import FRAMEWORK_SERVERS

    assert "prior_answer_read" in FRAMEWORK_SERVERS["osprey_workspace"].permissions_allow
