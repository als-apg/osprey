"""Run reads against the real record reader, under the worker's own scope.

The worker's routes and ``prior_answer_read`` tell "the record holds no such
run" from "the record may still be catching up" by the reader's answer. These
tests pin that reading against :class:`~osprey.agent_record.reader.RecordReader`
itself (through :mod:`tests.dispatch_worker.record_double`), so a change in how
the reader answers a missing run fails here rather than turning every unknown
run into a run that is forever pending.
"""

from __future__ import annotations

import json
import time

import pytest

from osprey.agent_runner.artifact_resolve import record_lacks_run
from osprey.mcp_server.dispatch_worker import dispatch_api
from osprey.mcp_server.workspace.tools import prior_answer
from tests.mcp_server.conftest import get_tool_fn

_RUN = "3f2b6c1e-8a4d-4f0e-9b1a-2c3d4e5f6a7b"
_OTHER = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"


def _finished_run(record, run_id: str, session: str, answer: str) -> None:
    created = time.time() - 5
    dispatch_api._write_run_accept(run_id, {"created_at": created, "session_id": session})
    dispatch_api._write_run_terminal(
        run_id, {"status": "completed", "completed_at": created + 1, "num_turns": 1}
    )
    record.answer(session, answer)


def test_a_run_the_record_never_saw_reads_as_lacking(worker_record):
    answer = worker_record.run(_RUN)

    assert answer.available and not answer.complete
    assert record_lacks_run(answer, _RUN)


def test_a_run_with_only_its_accept_is_not_lacking(worker_record):
    dispatch_api._write_run_accept(_RUN, {"created_at": time.time(), "session_id": "s-1"})

    answer = worker_record.run(_RUN)

    assert not answer.complete
    assert not record_lacks_run(answer, _RUN)


def test_an_unavailable_record_is_not_lacking():
    from osprey.agent_record.client import ClientAnswer

    answer = ClientAnswer.failed(f"run {_RUN} has no accept or terminal entry yet")

    assert not record_lacks_run(answer, _RUN)


def test_the_status_body_of_a_finished_run_read_from_the_reader(worker_record):
    _finished_run(worker_record, _RUN, "s-2", "the answer")

    body = dispatch_api._read_run_status(_RUN, {})

    assert body["status"] == "completed"
    assert body["text_output"] == "the answer"
    assert body["session_id"] == "s-2"
    assert body["num_turns"] == 1
    assert "owner" not in body


def test_an_unknown_run_reads_as_none(worker_record):  # noqa: ARG001 - the record is read
    assert dispatch_api._read_run_status(_RUN, {}) is None


@pytest.mark.asyncio
async def test_prior_answer_read_reads_the_answer_narrowed_to_its_runs(monkeypatch, worker_record):
    _finished_run(worker_record, _RUN, "s-3", "the whole table")
    monkeypatch.setattr(prior_answer, "record_client", worker_record.client)
    monkeypatch.setenv("OSPREY_DISPATCH_PRIOR_ANSWER_RUNS", f"{_RUN},{_OTHER}")

    page = json.loads(await get_tool_fn(prior_answer.prior_answer_read)(run_id=_RUN))

    assert page["text"] == "the whole table"
    [call] = worker_record.calls_to("run")
    assert call["narrow_runs"] == frozenset({_RUN, _OTHER})
