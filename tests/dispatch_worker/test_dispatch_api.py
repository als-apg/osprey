"""Unit tests for the dispatch-worker FastAPI app.

Exercises the HTTP surface of ``osprey.mcp_server.dispatch_worker.dispatch_api``
with a sync ``TestClient`` (lifespan runs inside the ``with`` block). The real
``sdk_runner.run_dispatch`` calls the Claude Agent SDK, so it is monkeypatched
with a fast canned coroutine — the background task then completes without the
SDK and tests never assert a timing-dependent terminal status. The worker reads
its runs back from an in-process agent record fed by its own spool
(:mod:`tests.dispatch_worker.record_double`); the canned coroutine records the
run's answer in it, as the agent's transcript would.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from osprey.agent_record import contract as record_contract
from osprey.agent_record.client import ClientAnswer
from osprey.agent_runner.tool_names import DISPATCH_DENIED_TOOLS
from osprey.mcp_server.dispatch_worker import dispatch_api
from tests.agent_record.fakes import FakeRecordClient

_TOKEN = "test-secret-token"

_CANNED_RESULT: dict[str, Any] = {
    "status": "completed",
    "text_output": "ok",
    "tool_calls": [],
    "error": None,
    "duration_sec": 0.01,
    "cost_usd": 0.0,
    "num_turns": 1,
}


@pytest.fixture(autouse=True)
def _no_managed_policy(monkeypatch: pytest.MonkeyPatch):
    """Pin the policy read to an empty policy.

    It reads OS-standard policy files, so on a machine that has one every
    provider-injection test here would refuse for a reason none of them is about.
    """
    monkeypatch.setattr(
        "osprey.agent_runner.provider_env.read_managed_policy_env",
        lambda paths=None: {},
    )


@pytest.fixture
def client(monkeypatch, worker_record):
    """A TestClient with auth configured and run_dispatch stubbed out.

    Resets the module-level in-memory stores so tests do not leak state into
    each other, and patches ``sdk_runner.run_dispatch`` with a fast async stub
    that records the canned answer as its session's transcript would.
    """
    monkeypatch.setenv("DISPATCH_WORKER_TOKEN", _TOKEN)

    # Isolate global state between tests.
    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_queues", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})

    async def _fake_run_dispatch(
        *,
        prompt,  # noqa: ARG001 - the patched run_dispatch is keyword-only
        allowed_tools,  # noqa: ARG001 - the patched run_dispatch is keyword-only
        max_turns,  # noqa: ARG001 - the patched run_dispatch is keyword-only
        event_queue,
        denied_tools=(),  # noqa: ARG001 - the patched run_dispatch is keyword-only
        run_id=None,  # noqa: ARG001 - the patched run_dispatch is keyword-only
        surface_prompt=None,  # noqa: ARG001 - the patched run_dispatch is keyword-only
        surface_tools=None,  # noqa: ARG001 - the patched run_dispatch is keyword-only
        owner=None,  # noqa: ARG001 - the patched run_dispatch is keyword-only
        prior_answer_runs=None,  # noqa: ARG001 - the patched run_dispatch is keyword-only
        session_id=None,
    ):
        if session_id is not None:
            worker_record.answer(session_id, _CANNED_RESULT["text_output"])
        if event_queue is not None:
            await event_queue.put({"type": "done"})
        return dict(_CANNED_RESULT)

    monkeypatch.setattr(dispatch_api.sdk_runner, "run_dispatch", _fake_run_dispatch)
    monkeypatch.setattr(dispatch_api, "describe_run_artifacts", lambda run_id: [])
    monkeypatch.setattr(dispatch_api, "describe_run_input_artifacts", lambda run_id: [])

    with TestClient(dispatch_api.app) as c:
        yield c


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {_TOKEN}"}


def _dispatch(client: TestClient, **fields: Any) -> str:
    body = {"prompt": "do it", "allowed_tools": ["Read"], **fields}
    resp = client.post("/dispatch", json=body, headers=_auth())
    assert resp.status_code == 202
    return resp.json()["run_id"]


def _wait_for_terminal(client: TestClient, run_id: str, timeout: float = 5.0) -> dict:
    """Poll a run until it leaves the ``pending`` state (or time out)."""
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        resp = client.get(f"/dispatch/{run_id}", headers=_auth())
        assert resp.status_code == 200
        last = resp.json()
        if last.get("status") != "pending":
            return last
        time.sleep(0.02)
    return last


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_no_auth(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    for key in ("pending_runs", "completed_runs", "error_runs", "total_runs"):
        assert key in body
        assert isinstance(body[key], int)


@pytest.mark.usefixtures("worker_record")
def test_health_counts_finished_runs_from_the_record(client):
    """A run this process no longer holds is still counted: the counts are the record's."""
    run_id = _dispatch(client)
    _wait_for_terminal(client, run_id)
    dispatch_api._runs.clear()

    body = client.get("/health").json()

    assert body["completed_runs"] == 1
    assert body["total_runs"] == 1
    assert body["pending_runs"] == 0
    assert "record" not in body


def test_health_answers_while_the_record_is_unavailable(client, monkeypatch):
    """The liveness check answers ok; the counts it cannot know are None, and it says why."""
    monkeypatch.setattr(
        dispatch_api, "record_client", lambda **_: FakeRecordClient(unavailable="down")
    )

    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["record"] == "unavailable"
    assert body["completed_runs"] is None
    assert body["error_runs"] is None
    assert body["total_runs"] is None
    assert body["pending_runs"] == 0


def test_health_reports_ca_user_in_process(client, monkeypatch):
    """``ca_user`` is the name the worker's uid resolves to, read live."""
    import pwd

    monkeypatch.delenv("OSPREY_CONTROL_IDENTITY_SKIPPED", raising=False)
    body = client.get("/health").json()
    assert body["ca_user"] == pwd.getpwuid(os.getuid()).pw_name
    assert body["control_identity_skipped"] is None


def test_health_reports_skipped_identity(client, monkeypatch):
    """The entrypoint's skip reason reaches ``osprey health`` unchanged."""
    monkeypatch.setenv("OSPREY_CONTROL_IDENTITY_SKIPPED", "non-root-start")
    body = client.get("/health").json()
    assert body["control_identity_skipped"] == "non-root-start"


# ---------------------------------------------------------------------------
# POST /dispatch
# ---------------------------------------------------------------------------


def test_dispatch_accepts_benign_tools(client):
    resp = client.post(
        "/dispatch",
        json={"prompt": "do it", "allowed_tools": ["Read"]},
        headers=_auth(),
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "accepted"
    assert isinstance(body["run_id"], str)
    assert body["run_id"]


def test_dispatch_rejects_denied_tool(client):
    resp = client.post(
        "/dispatch",
        json={"prompt": "fetch", "allowed_tools": ["Read", "WebFetch"]},
        headers=_auth(),
    )
    assert resp.status_code == 403
    assert "WebFetch" in resp.json()["detail"]


def test_dispatch_rejects_wildcard_denied_tool(client):
    """Denylist '*'-suffix entries block by prefix (e.g. all playwright tools)."""
    tool = "mcp__plugin_playwright_playwright__browser_click"
    resp = client.post(
        "/dispatch",
        json={"prompt": "click", "allowed_tools": [tool]},
        headers=_auth(),
    )
    assert resp.status_code == 403
    assert tool in resp.json()["detail"]


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("WebFetch", True),
        ("WebSearch", True),
        ("Bash", True),
        ("TaskStop", True),
        ("mcp__plugin_playwright_playwright__browser_click", True),
        ("mcp__plugin_playwright_playwright__", True),  # bare prefix still matches
        ("Read", False),
        ("Write", False),
        ("mcp__osprey_workspace__write_channel", False),
        ("WebFetcher", False),  # not an exact match, not a wildcard entry
        ("", False),
    ],
)
def test_is_denied_matrix(tool, expected):
    """The server-side denylist matcher: exact entries + '*'-suffix prefixes."""
    assert dispatch_api._is_denied(tool) is expected


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("mcp__plugin_anything_srv__x", True),
        ("mcp__plugin_playwright_playwright__browser_click", True),
        ("mcp__claude_ai_Gmail__search", True),
        ("mcp__controls__channel_read", False),
        ("mcp__pluginx__ping", False),
    ],
)
def test_plugin_and_connector_tools_are_denied(tool, expected):
    """Every plugin server's and claude.ai connector's tools are refused; a server
    whose name only resembles the namespace is not."""
    assert dispatch_api._is_denied(tool) is expected


@pytest.mark.parametrize("entry", sorted(DISPATCH_DENIED_TOOLS))
def test_the_worker_denies_every_entry_of_the_dispatch_floor(entry):
    """The worker reads the shared dispatch floor, not a copy of it."""
    probe = entry[:-1] + "probe" if entry.endswith("*") else entry
    assert dispatch_api._is_denied(probe)


def test_dispatch_denied_tool_schedules_no_run(client):
    """A denied tool 403s AND never creates a run (nothing enters the store)."""
    monkey_before = len(dispatch_api._runs)
    resp = client.post(
        "/dispatch",
        json={"prompt": "shell out", "allowed_tools": ["Read", "Bash"]},
        headers=_auth(),
    )
    assert resp.status_code == 403
    assert "Bash" in resp.json()["detail"]
    # No run was created and no task was scheduled.
    assert len(dispatch_api._runs) == monkey_before
    assert dispatch_api._tasks == {}


def test_dispatch_wrong_token(client):
    resp = client.post(
        "/dispatch",
        json={"prompt": "do it", "allowed_tools": ["Read"]},
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401


def test_dispatch_with_a_non_ascii_bearer_still_answers_401(client):
    """A header byte outside ASCII is a wrong credential, not a server fault.

    ``compare_digest`` refuses two ``str`` arguments unless both are ASCII-only,
    and Starlette decodes the header as latin-1, so comparing the decoded
    strings would turn a wrong bearer into a 500. The header goes on the wire as
    raw bytes because that is the only way to send one.
    """
    resp = client.post(
        "/dispatch",
        json={"prompt": "do it", "allowed_tools": ["Read"]},
        headers={"Authorization": "Bearer tökén-ünicode".encode("latin-1")},
    )
    assert resp.status_code == 401


def test_dispatch_no_auth_header(client):
    # HTTPBearer auto-error rejects a missing Authorization header. The exact
    # code depends on the FastAPI version (older: 403, current: 401); accept
    # either so the test tracks behavior without pinning a version-specific code.
    resp = client.post(
        "/dispatch",
        json={"prompt": "do it", "allowed_tools": ["Read"]},
    )
    assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Tests of GET /dispatch/{run_id}
# ---------------------------------------------------------------------------


def test_get_dispatch_unknown_id_404(client):
    resp = client.get("/dispatch/does-not-exist", headers=_auth())
    assert resp.status_code == 404


def test_get_dispatch_returns_status(client):
    post = client.post(
        "/dispatch",
        json={"prompt": "do it", "allowed_tools": ["Read"]},
        headers=_auth(),
    )
    run_id = post.json()["run_id"]

    result = _wait_for_terminal(client, run_id)
    assert result.get("status") in ("pending", "completed", "error")
    # With the fast stub the run should reach completion.
    assert result["status"] == "completed"
    assert result["text_output"] == "ok"


def test_the_answer_is_there_the_moment_the_status_turns_terminal(client):
    """The first terminal status a poller sees already carries the run's answer."""
    run_id = _dispatch(client)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        body = client.get(f"/dispatch/{run_id}", headers=_auth()).json()
        if body["status"] != "pending":
            break
        time.sleep(0.005)

    assert body["status"] == "completed"
    assert body["text_output"] == "ok"


def test_a_finished_run_reads_back_from_the_record_in_the_status_shape(client):
    """A run this process no longer holds answers from the record, in the shape bridges read."""
    run_id = _dispatch(client, owner="alice")
    live = _wait_for_terminal(client, run_id)
    dispatch_api._runs.clear()
    dispatch_api._settled.clear()

    resp = client.get(f"/dispatch/{run_id}", headers=_auth())

    assert resp.status_code == 200
    body = resp.json()
    assert body == live
    for key in (
        "status",
        "text_output",
        "run_id",
        "error",
        "artifacts",
        "input_artifacts",
        "failure_class",
        "num_tool_calls",
        "error_code",
    ):
        assert key in body
    assert body["status"] == "completed"
    assert body["text_output"] == "ok"
    assert body["owner"] == "alice"
    assert body["artifacts"] == [] and body["input_artifacts"] == []
    assert body["cost_usd"] == _CANNED_RESULT["cost_usd"]
    assert body["num_turns"] == _CANNED_RESULT["num_turns"]
    assert isinstance(body["created_at"], float)
    assert body["session_id"]


def test_a_run_in_flight_answers_from_memory_without_reading_the_record(client, monkeypatch):
    release = asyncio.Event()

    async def _blocking(**kwargs):
        await release.wait()
        return dict(_CANNED_RESULT)

    monkeypatch.setattr(dispatch_api.sdk_runner, "run_dispatch", _blocking)
    fake = FakeRecordClient()
    monkeypatch.setattr(dispatch_api, "record_client", lambda **_: fake)
    run_id = _dispatch(client, owner="alice")
    try:
        body = client.get(f"/dispatch/{run_id}", headers=_auth()).json()
    finally:
        client.portal.call(release.set)

    assert body["status"] == "pending"
    assert body["owner"] == "alice"
    assert "record" not in body
    assert fake.calls_to("run") == []


@pytest.mark.parametrize(
    ("fake", "state"),
    [
        (FakeRecordClient(unavailable="connection refused"), "unavailable"),
        (
            FakeRecordClient(
                {"run": ClientAnswer(data=None, complete=False, reason="wait mark not passed")}
            ),
            "incomplete",
        ),
    ],
)
def test_a_finished_run_the_record_cannot_settle_reads_pending(client, monkeypatch, fake, state):
    """Never terminal without a complete read, and never a 404 for a run that may exist."""
    run_id = _dispatch(client)
    _wait_for_terminal(client, run_id)
    dispatch_api._settled.clear()
    monkeypatch.setattr(dispatch_api, "record_client", lambda **_: fake)

    resp = client.get(f"/dispatch/{run_id}", headers=_auth())

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending"
    assert body["record"] == state
    assert "text_output" not in body
    assert body["run_id"] == run_id


def test_a_terminal_entry_without_its_whole_answer_reads_pending(client, monkeypatch):
    """An answer the record holds only part of is incomplete, so the status stays pending."""
    run_id = _dispatch(client)
    _wait_for_terminal(client, run_id)
    dispatch_api._settled.clear()
    facts = {
        "terminal": {"body": {"phase": "terminal", "status": "completed"}},
        "answer": "o",
    }
    fake = FakeRecordClient(
        {"run": ClientAnswer(data=facts, complete=False, reason="answer is incomplete")}
    )
    monkeypatch.setattr(dispatch_api, "record_client", lambda **_: fake)

    body = client.get(f"/dispatch/{run_id}", headers=_auth()).json()

    assert body["status"] == "pending"
    assert body["record"] == "incomplete"


def test_a_settled_run_is_read_from_the_record_once(client, worker_record):
    run_id = _dispatch(client)
    _wait_for_terminal(client, run_id)
    reads = len(worker_record.calls_to("run"))

    for _ in range(3):
        assert client.get(f"/dispatch/{run_id}", headers=_auth()).json()["status"] == "completed"

    assert len(worker_record.calls_to("run")) == reads


def test_a_run_id_that_is_no_record_key_is_404_without_a_read(client, monkeypatch):
    fake = FakeRecordClient(unavailable="down")
    monkeypatch.setattr(dispatch_api, "record_client", lambda **_: fake)

    resp = client.get("/dispatch/not a run id", headers=_auth())

    assert resp.status_code == 404
    assert fake.calls == []


# ---------------------------------------------------------------------------
# Tests of DELETE /dispatch/{run_id}
# ---------------------------------------------------------------------------


def test_cancel_unknown_id_404(client):
    resp = client.delete("/dispatch/does-not-exist", headers=_auth())
    assert resp.status_code == 404


def test_cancel_finished_run(client):
    post = client.post(
        "/dispatch",
        json={"prompt": "do it", "allowed_tools": ["Read"]},
        headers=_auth(),
    )
    run_id = post.json()["run_id"]
    # Let the stub finish so the run is no longer pending.
    _wait_for_terminal(client, run_id)

    resp = client.delete(f"/dispatch/{run_id}", headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert "cancelled" in body
    # A finished run cannot be cancelled.
    assert body["cancelled"] is False


def test_cancel_of_a_finished_run_this_process_no_longer_holds(client):
    run_id = _dispatch(client)
    _wait_for_terminal(client, run_id)
    dispatch_api._runs.clear()

    body = client.delete(f"/dispatch/{run_id}", headers=_auth()).json()

    assert body == {"run_id": run_id, "cancelled": False, "reason": "already finished"}


@pytest.mark.usefixtures("worker_record")
def test_cancel_of_a_run_accepted_by_another_process(client):
    """An accept with no terminal and no task here cannot be cancelled here."""
    dispatch_api._write_run_accept("elsewhere", {"created_at": time.time(), "session_id": "s-1"})

    body = client.delete("/dispatch/elsewhere", headers=_auth()).json()

    assert body["cancelled"] is False
    assert body["reason"] == "record incomplete"


def test_cancel_while_the_record_is_unavailable_says_so(client, monkeypatch):
    monkeypatch.setattr(
        dispatch_api, "record_client", lambda **_: FakeRecordClient(unavailable="down")
    )

    resp = client.delete("/dispatch/some-run", headers=_auth())

    assert resp.status_code == 200
    assert resp.json() == {
        "run_id": "some-run",
        "cancelled": False,
        "reason": "record unavailable",
    }


# ---------------------------------------------------------------------------
# The run history is the record's: nothing here deletes it
# ---------------------------------------------------------------------------


def test_the_worker_has_no_clear_history_route(client):
    """``DELETE /dispatch/runs`` is a cancel of a run named "runs", which the record lacks."""
    assert not hasattr(dispatch_api, "ClearRunsRequest")
    assert not hasattr(dispatch_api, "_persist_run")
    assert not hasattr(dispatch_api, "_load_persisted_runs")

    resp = client.delete("/dispatch/runs", headers=_auth())

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /dashboard/runs
# ---------------------------------------------------------------------------


def test_dashboard_runs_requires_auth(client):
    # The runs feed leaks full text_output/error, so it is token-gated like the
    # other worker endpoints. HTTPBearer auto-error rejects a missing header
    # (401 current FastAPI, 403 older) — accept either.
    resp = client.get("/dashboard/runs")
    assert resp.status_code in (401, 403)


def test_dashboard_runs_with_auth(client):
    # Seed one run via the API.
    client.post(
        "/dispatch",
        json={"prompt": "do it", "allowed_tools": ["Read"]},
        headers=_auth(),
    )
    resp = client.get("/dashboard/runs", headers=_auth())
    assert resp.status_code == 200
    runs = resp.json()
    assert isinstance(runs, list)
    assert len(runs) >= 1
    assert "run_id" in runs[0]
    assert "status" in runs[0]


def test_dashboard_runs_carries_the_session_and_the_owner(client):
    """Every row names its run's session (its history link) and who fired it.

    This endpoint is a hand-written projection, so a field absent from it is
    unavailable downstream; both keys are always present.
    """
    owned = _dispatch(client, owner="alice")
    unowned = _dispatch(client)
    for run_id in (owned, unowned):
        _wait_for_terminal(client, run_id)

    rows = {r["run_id"]: r for r in client.get("/dashboard/runs", headers=_auth()).json()}

    assert rows[owned]["owner"] == "alice"
    assert "owner" in rows[unowned] and rows[unowned]["owner"] is None
    for run_id in (owned, unowned):
        assert rows[run_id]["session_id"]
        assert rows[run_id]["text_output"] == "ok"


def test_dashboard_lists_the_runs_of_an_earlier_process(client):
    """The list is the record's: a restart loses no run."""
    run_id = _dispatch(client)
    _wait_for_terminal(client, run_id)
    dispatch_api._runs.clear()
    dispatch_api._settled.clear()

    rows = client.get("/dashboard/runs", headers=_auth()).json()

    assert [(r["run_id"], r["status"], r["text_output"]) for r in rows] == [
        (run_id, "completed", "ok")
    ]


def test_dashboard_lists_runs_within_the_records_retention(client, worker_record, monkeypatch):
    monkeypatch.setattr(dispatch_api, "deployed_record_retention_days", lambda: 3)

    client.get("/dashboard/runs", headers=_auth())

    since = worker_record.calls_to("entries")[0]["since"]
    expected = time.time() - 3 * 86400
    assert abs(since.timestamp() - expected) < 60
    assert worker_record.calls_to("entries")[0]["kind"] == record_contract.KIND_RUN


def test_dashboard_answers_503_while_the_record_is_unavailable(client, monkeypatch):
    monkeypatch.setattr(
        dispatch_api, "record_client", lambda **_: FakeRecordClient(unavailable="down")
    )

    resp = client.get("/dashboard/runs", headers=_auth())

    assert resp.status_code == 503
    assert "record unavailable" in resp.json()["detail"]


def test_dashboard_row_of_an_unsettled_run_is_pending(client, monkeypatch):
    """A row is terminal only from a complete read of its run."""
    run_id = _dispatch(client)
    _wait_for_terminal(client, run_id)
    dispatch_api._settled.clear()
    real = dispatch_api.record_client()

    class _RunsIncomplete:
        def entries(self, *args, **kwargs):
            return real.entries(*args, **kwargs)

        def run(self, run_id, **kwargs):  # noqa: ARG002 - the client's signature
            return ClientAnswer(data=None, complete=False, reason="still draining")

    monkeypatch.setattr(dispatch_api, "record_client", lambda **_: _RunsIncomplete())

    row = next(
        r for r in client.get("/dashboard/runs", headers=_auth()).json() if r["run_id"] == run_id
    )

    assert row["status"] == "pending"
    assert row["record"] == "incomplete"
    assert row["text_output"] == ""


def test_dashboard_runs_dates_and_orders_finished_runs_by_when_they_were_fired(client, monkeypatch):
    """Finished runs keep their accepted time, so the feed dates and orders them."""

    async def _run(**kwargs):
        await kwargs["event_queue"].put({"type": "done"})
        await asyncio.sleep(0.01)
        if kwargs["prompt"] == "fail":
            raise ValueError("boom")
        return dict(_CANNED_RESULT)

    monkeypatch.setattr(dispatch_api.sdk_runner, "run_dispatch", _run)

    completed_id = client.post(
        "/dispatch", json={"prompt": "ok", "allowed_tools": ["Read"]}, headers=_auth()
    ).json()["run_id"]
    _wait_for_terminal(client, completed_id)
    failed_id = client.post(
        "/dispatch", json={"prompt": "fail", "allowed_tools": ["Read"]}, headers=_auth()
    ).json()["run_id"]
    _wait_for_terminal(client, failed_id)

    resp = client.get("/dashboard/runs", headers=_auth())
    assert resp.status_code == 200
    rows = resp.json()
    assert [r["run_id"] for r in rows[:2]] == [failed_id, completed_id]
    assert [r["status"] for r in rows[:2]] == ["error", "completed"]
    for row in rows[:2]:
        created_at = row["created_at"]
        assert created_at == dispatch_api._runs[row["run_id"]]["created_at"]
        assert isinstance(created_at, float)
        assert created_at > 0
        assert isinstance(row["age_sec"], (int, float))
        assert row["age_sec"] >= 0


# ---------------------------------------------------------------------------
# Startup lifecycle: provider-env injection, no artifact regeneration
# ---------------------------------------------------------------------------
#
# The project image bakes .claude/ and data/ at build time (the render `osprey
# build` produces is COPYed in), so the worker does not regenerate those
# artifacts at runtime. Provider auth/model env injection still has to happen at
# process startup, since it depends on the mounted config.yml/environment.


@pytest.mark.asyncio
async def test_lifespan_injects_provider_env_once(monkeypatch):
    """Entering the lifespan calls provider-env injection exactly once."""
    import asyncio

    calls = {"n": 0}

    def _spy():
        calls["n"] += 1

    monkeypatch.setattr(dispatch_api, "_inject_provider_env_once", _spy)

    async with dispatch_api._lifespan(dispatch_api.app):
        pass
    await asyncio.sleep(0)  # let the cancelled cleanup task settle

    assert calls["n"] == 1


def test_no_artifact_regeneration_path():
    """The config.yml-only artifact-regen path has been removed (Task 1.5).

    Encodes the removal so a future change cannot silently reintroduce it.
    """
    assert not hasattr(dispatch_api, "_provision_claude_artifacts_once")


# ---------------------------------------------------------------------------
# Stale-run sweep cancels orphaned tasks
# ---------------------------------------------------------------------------


def _stale_created_at() -> float:
    return time.time() - (dispatch_api.DISPATCH_TIMEOUT_SEC + 30 + 60)


def _terminals(spool_zone, run_id: str) -> list:
    key = record_contract.run_key(run_id, record_contract.RUN_TERMINAL)
    return [e for e in spool_zone.recorded_entries(record_contract.KIND_RUN) if e.key == key]


@pytest.fixture
def empty_worker(monkeypatch, worker_record):
    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})
    monkeypatch.setattr(dispatch_api, "_queues", {})
    return worker_record


@pytest.mark.asyncio
@pytest.mark.usefixtures("empty_worker")
async def test_sweep_marks_stale_run_error_and_cancels_task(spool_zone):
    """A run pending past the cutoff is marked error AND its task is cancelled."""

    async def _long_runner():
        await asyncio.sleep(30)

    task = asyncio.create_task(_long_runner())
    run_id = "stale-1"
    accepted = {"status": "pending", "created_at": _stale_created_at(), "session_id": "s-1"}
    dispatch_api._runs[run_id] = accepted
    dispatch_api._write_run_accept(run_id, accepted)
    dispatch_api._tasks[run_id] = task

    dispatch_api._sweep_stale_runs()

    assert dispatch_api._runs[run_id]["status"] == "error"
    assert "Timed out" in dispatch_api._runs[run_id]["error"]
    [terminal] = _terminals(spool_zone, run_id)
    assert terminal.body["status"] == "error"
    assert terminal.body["failure_class"] == "infrastructure"
    # Let the cancellation propagate.
    await asyncio.sleep(0)
    assert task.cancelled()


@pytest.mark.usefixtures("empty_worker")
def test_sweep_ends_a_run_accepted_by_an_earlier_process(spool_zone):
    """An accept left with no terminal by a process that is gone gets its one terminal."""
    dispatch_api._write_run_accept(
        "orphan", {"created_at": _stale_created_at(), "session_id": "s-2", "owner": "alice"}
    )

    dispatch_api._sweep_stale_runs()
    dispatch_api._sweep_stale_runs()

    [terminal] = _terminals(spool_zone, "orphan")
    assert terminal.body["status"] == "error"
    assert terminal.body["failure_class"] == "infrastructure"
    assert "orphan" not in dispatch_api._runs


def test_sweep_drains_before_it_reads(empty_worker):
    dispatch_api._sweep_stale_runs()

    [call] = empty_worker.calls_to("entries")
    assert call["wait"] == record_contract.Drain()
    assert call["kind"] == record_contract.KIND_RUN


@pytest.mark.usefixtures("empty_worker")
def test_sweep_leaves_a_run_with_a_terminal_alone(spool_zone):
    created = _stale_created_at()
    dispatch_api._write_run_accept("done", {"created_at": created, "session_id": "s-3"})
    dispatch_api._write_run_terminal("done", {"status": "completed", "completed_at": created + 5})

    dispatch_api._sweep_stale_runs()

    [terminal] = _terminals(spool_zone, "done")
    assert terminal.body["status"] == "completed"


@pytest.mark.parametrize(
    "fake",
    [
        FakeRecordClient(unavailable="down"),
        FakeRecordClient({"entries": ClientAnswer(data=[], complete=False, reason="no drain")}),
    ],
)
@pytest.mark.usefixtures("empty_worker")
def test_sweep_ends_nothing_while_the_record_cannot_answer(monkeypatch, fake):
    """Without a drained read the sweep cannot tell a stale run from a slow ingest."""
    monkeypatch.setattr(dispatch_api, "record_client", lambda **_: fake)
    dispatch_api._runs["stale"] = {"status": "pending", "created_at": _stale_created_at()}
    written: list = []
    monkeypatch.setattr(dispatch_api, "_write_run_terminal", lambda *a: written.append(a))

    dispatch_api._sweep_stale_runs()

    assert dispatch_api._runs["stale"]["status"] == "pending"
    assert written == []


@pytest.mark.usefixtures("empty_worker")
def test_sweep_leaves_fresh_pending_run_alone():
    accepted = {"status": "pending", "created_at": time.time(), "session_id": "s-4"}
    dispatch_api._runs["fresh"] = accepted
    dispatch_api._write_run_accept("fresh", accepted)

    dispatch_api._sweep_stale_runs()

    assert dispatch_api._runs["fresh"]["status"] == "pending"


# ---------------------------------------------------------------------------
# Worker auth on the remaining gated endpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/dispatch/some-id"),
        ("get", "/dispatch/some-id/stream"),
        ("delete", "/dispatch/some-id"),
        ("get", "/dashboard/runs"),
    ],
)
def test_gated_endpoint_missing_auth_rejected(client, method, path):
    """Missing Authorization header is rejected (401 current FastAPI, 403 older)."""
    resp = getattr(client, method)(path)
    assert resp.status_code in (401, 403)


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/dispatch/some-id"),
        ("delete", "/dispatch/some-id"),
        ("get", "/dashboard/runs"),
    ],
)
def test_gated_endpoint_wrong_token_401(client, method, path):
    resp = getattr(client, method)(path, headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_unconfigured_worker_token_fails_closed_500(monkeypatch):
    """With DISPATCH_WORKER_TOKEN unset, the worker fails closed (500) on a token check."""
    monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)
    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_queues", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})
    with TestClient(dispatch_api.app) as c:
        resp = c.get("/dispatch/some-id", headers={"Authorization": "Bearer anything"})
    assert resp.status_code == 500
    assert "DISPATCH_WORKER_TOKEN" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# _inject_provider_env_once: ${VAR} expansion + proxy-start for the worker (#307)
# ---------------------------------------------------------------------------

_ARGO_CONFIG = """\
api:
  providers:
    argo:
      base_url: ${ARGO_PROD_URL}
      default_model: claudesonnet45
      models: [claudehaiku45, claudesonnet45, claudeopus41]
claude_code:
  provider: argo
"""

_CBORG_CONFIG = """\
claude_code:
  provider: cborg
"""

_ANTHROPIC_CONFIG = """\
claude_code:
  provider: anthropic
"""


def _isolated_environ(monkeypatch, tmp_path, **extra):
    """Swap os.environ for a throwaway dict so the function's mutations don't leak.

    Also strips any ambient telemetry vars: CI exports OTEL_*/
    CLAUDE_CODE_ENABLE_TELEMETRY for its own agent runs, and these tests assert on
    the telemetry env THIS function does (or does not) inject, so the baseline
    must start free of inherited telemetry state.
    """
    from osprey.build.claude_code_telemetry import TELEMETRY_ENV_VARS

    fake = dict(os.environ)
    # The repo root, with the render's config one level down — how the
    # dispatch-worker compose service wires the deployed worker.
    fake["OSPREY_PROJECT_DIR"] = str(tmp_path)
    fake["CONFIG_FILE"] = str(tmp_path / "build" / "config.yml")
    fake.pop("OSPREY_CONFIG", None)
    fake.pop("ANTHROPIC_BASE_URL", None)
    for _var in TELEMETRY_ENV_VARS:
        fake.pop(_var, None)
    fake.update(extra)
    monkeypatch.setattr(os, "environ", fake)
    return fake


def _render_config(tmp_path, body: str) -> None:
    """Write the rendered config where a three-zone deployment keeps it."""
    render = tmp_path / "build"
    render.mkdir(exist_ok=True)
    (render / "config.yml").write_text(body)


def test_inject_provider_env_expands_and_starts_proxy(tmp_path, monkeypatch):
    """Custom provider: ${VAR} base_url is expanded, proxy started, base URL repointed."""
    _render_config(tmp_path, _ARGO_CONFIG)
    (tmp_path / ".env").write_text("ARGO_PROD_URL=https://argo.example/v1\nARGO_API_KEY=sk-argo\n")
    fake = _isolated_environ(monkeypatch, tmp_path, ANTHROPIC_CUSTOM_HEADERS="X-Corp-Trace: abc123")
    proxy = MagicMock(return_value=7777)
    monkeypatch.setattr("osprey.infrastructure.proxy.lifecycle.start_proxy", proxy)

    dispatch_api._inject_provider_env_once()

    proxy.assert_called_once()
    upstream, api_key = proxy.call_args[0]
    assert upstream == "https://argo.example/v1"
    assert api_key == "sk-argo"
    assert proxy.call_args.kwargs["forward_headers"] == frozenset({"x-corp-trace"})
    # The argo entry declares nothing, so the adapter decides at proxy start.
    assert proxy.call_args.kwargs["supports_images"] is None
    assert fake["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:7777"


def test_inject_provider_env_no_proxy_for_native(tmp_path, monkeypatch):
    """Native provider (cborg): env injected but no translation proxy started."""
    _render_config(tmp_path, _CBORG_CONFIG)
    _isolated_environ(monkeypatch, tmp_path, CBORG_API_KEY="sk-cborg")
    proxy = MagicMock(return_value=1)
    monkeypatch.setattr("osprey.infrastructure.proxy.lifecycle.start_proxy", proxy)

    dispatch_api._inject_provider_env_once()

    proxy.assert_not_called()


def test_inject_provider_env_exports_otlp_to_the_record_with_the_workers_login(
    tmp_path, monkeypatch
):
    """The worker's agent sends its OTLP logs to agent-record under the worker's own login.

    The record address and login are the ones the worker's compose service
    hands it.
    """
    import base64

    _render_config(tmp_path, _ANTHROPIC_CONFIG)
    fake = _isolated_environ(
        monkeypatch,
        tmp_path,
        ANTHROPIC_API_KEY="sk-ant",
        OSPREY_AGENT_RECORD_URL="http://agent-record:8790",
        OSPREY_AGENT_RECORD_LOGIN="dispatch-worker-2:tok",
    )

    dispatch_api._inject_provider_env_once()

    assert fake.get("ANTHROPIC_MODEL"), "provider env is injected beside the OTLP block"
    assert fake["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"] == "http://agent-record:8790/v1/logs"
    assert fake["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://agent-record:8790"
    token = base64.b64encode(b"dispatch-worker-2:tok").decode()
    assert fake["OTEL_EXPORTER_OTLP_HEADERS"] == f"Authorization=Basic {token}"


def _pin_policy(monkeypatch, **env: str) -> None:
    """Make the managed-policy read return ``env`` from a fixed source file."""
    policy = {var: (value, "/etc/claude-code/managed-settings.json") for var, value in env.items()}
    monkeypatch.setattr(
        "osprey.agent_runner.provider_env.read_managed_policy_env",
        lambda paths=None: policy,
    )


def test_inject_provider_env_refuses_on_managed_policy_conflict(tmp_path, monkeypatch):
    """A managed-policy value that differs from the deployment's aborts worker
    startup rather than starting the agent against a backend the project did
    not configure.

    The refusal must propagate: it is raised outside the broad ``except`` that
    otherwise swallows provider-injection errors."""
    _render_config(tmp_path, _CBORG_CONFIG)
    _isolated_environ(monkeypatch, tmp_path, CBORG_API_KEY="sk-cborg")
    _pin_policy(monkeypatch, ANTHROPIC_BASE_URL="https://evil.example")

    with pytest.raises(RuntimeError, match="Refusing to start the dispatch worker"):
        dispatch_api._inject_provider_env_once()


def test_inject_provider_env_accepts_a_managed_policy_equal_to_the_launch_value(
    tmp_path, monkeypatch
):
    from osprey.agent_runner.provider_env import load_provider_spec

    _render_config(tmp_path, _CBORG_CONFIG)
    fake = _isolated_environ(monkeypatch, tmp_path, CBORG_API_KEY="sk-cborg")
    spec = load_provider_spec(tmp_path / "build", include_telemetry=False)
    assert spec is not None
    deployed = spec.env_block["ANTHROPIC_BASE_URL"]
    _pin_policy(monkeypatch, ANTHROPIC_BASE_URL=deployed)

    dispatch_api._inject_provider_env_once()

    assert fake["ANTHROPIC_BASE_URL"] == deployed


def test_inject_provider_env_compares_the_policy_with_the_proxy_loopback(tmp_path, monkeypatch):
    """The upstream origin the env holds before the proxy rewrite is not what the
    agent runs on, so a policy pinning it still refuses."""
    _render_config(tmp_path, _ARGO_CONFIG)
    (tmp_path / ".env").write_text("ARGO_PROD_URL=https://argo.example/v1\nARGO_API_KEY=sk-argo\n")
    _isolated_environ(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "osprey.infrastructure.proxy.lifecycle.start_proxy", MagicMock(return_value=7777)
    )
    _pin_policy(monkeypatch, ANTHROPIC_BASE_URL="https://argo.example")

    with pytest.raises(RuntimeError, match="Refusing to start the dispatch worker"):
        dispatch_api._inject_provider_env_once()


def test_inject_provider_env_refuses_on_presence_after_a_failed_injection(tmp_path, monkeypatch):
    """A failed injection compares against nothing, so any policy key refuses even
    though the broad ``except`` swallowed the injection error."""
    _render_config(tmp_path, _CBORG_CONFIG)
    _isolated_environ(monkeypatch, tmp_path, CBORG_API_KEY="sk-cborg")

    def _broken(*args, **kwargs):
        raise ValueError("unreadable provider")

    monkeypatch.setattr("osprey.agent_runner.provider_env.load_provider_spec", _broken)
    _pin_policy(monkeypatch, ANTHROPIC_MODEL="anything")

    with pytest.raises(RuntimeError, match="Refusing to start the dispatch worker"):
        dispatch_api._inject_provider_env_once()


def test_dispatch_request_max_turns_defaults_to_the_deployments_ceiling() -> None:
    """A request naming no ceiling takes DISPATCH_MAX_TURNS, this deployment's own.

    The env is read at import, so the assertion is against the module attribute
    rather than a literal — what matters is that the request model and the
    deployment's configured ceiling are one number.
    """
    request = dispatch_api.DispatchRequest(prompt="hi", allowed_tools=[])

    assert request.max_turns == dispatch_api.DISPATCH_MAX_TURNS


def test_dispatch_request_max_turns_is_taken_from_the_body_when_given() -> None:
    """A trigger that names its own ceiling overrides the deployment's."""
    request = dispatch_api.DispatchRequest(prompt="hi", allowed_tools=[], max_turns=60)

    assert request.max_turns == 60


# ---------------------------------------------------------------------------
# Owner on the wire
# ---------------------------------------------------------------------------


def _capture_run_dispatch(monkeypatch) -> dict[str, Any]:
    """Replace the runner with a stub that records the keywords it was called with."""
    captured: dict[str, Any] = {}

    async def _capturing_run_dispatch(**kwargs):
        captured.update(kwargs)
        queue = kwargs.get("event_queue")
        if queue is not None:
            await queue.put({"type": "done"})
        return dict(_CANNED_RESULT)

    monkeypatch.setattr(dispatch_api.sdk_runner, "run_dispatch", _capturing_run_dispatch)
    return captured


def test_dispatch_body_owner_reaches_the_runner(client, monkeypatch):
    """The owner the dispatcher put on the wire is carried into the run.

    The request model is the only thing standing between the dispatcher's
    payload and the runner: a field the model does not declare is dropped
    silently, and the run would be judged against no one's narrowing while the
    fire itself was attributed to a person.
    """
    captured = _capture_run_dispatch(monkeypatch)

    resp = client.post(
        "/dispatch",
        json={"prompt": "do it", "allowed_tools": ["Read"], "owner": "alice"},
        headers=_auth(),
    )
    assert resp.status_code == 202
    _wait_for_terminal(client, resp.json()["run_id"])

    assert captured["owner"] == "alice"


def test_dispatch_body_without_an_owner_runs_owner_less(client, monkeypatch):
    """A fire nobody is attributed to — cron — reaches the runner with no owner."""
    captured = _capture_run_dispatch(monkeypatch)

    resp = client.post(
        "/dispatch",
        json={"prompt": "do it", "allowed_tools": ["Read"]},
        headers=_auth(),
    )
    assert resp.status_code == 202
    _wait_for_terminal(client, resp.json()["run_id"])

    assert captured["owner"] is None


def test_dispatch_request_owner_is_additive() -> None:
    """A body naming no owner validates, so a dispatcher predating the field works.

    Worker and dispatcher are separately deployed images; a required field here
    would turn every fire from an older dispatcher into a 422.
    """
    request = dispatch_api.DispatchRequest(prompt="hi", allowed_tools=[])

    assert request.owner is None


def test_run_record_names_the_owner_from_the_body(client):
    """The finished run, in memory and in the record, names who fired it."""
    run_id = _dispatch(client, owner="alice")
    record = _wait_for_terminal(client, run_id)

    assert record["owner"] == "alice"
    assert dispatch_api._runs[run_id]["owner"] == "alice"


def test_owner_less_run_record_has_no_owner_key(client):
    """A run nobody is attributed to is answered without an owner key."""
    run_id = _dispatch(client)
    record = _wait_for_terminal(client, run_id)

    assert "owner" not in record
    assert "owner" not in dispatch_api._runs[run_id]


def test_pending_record_names_the_owner(client, monkeypatch):
    """The record of a run still in flight already names who fired it."""
    release = asyncio.Event()

    async def _blocking_run_dispatch(**kwargs):
        await release.wait()
        queue = kwargs.get("event_queue")
        if queue is not None:
            await queue.put({"type": "done"})
        return dict(_CANNED_RESULT)

    monkeypatch.setattr(dispatch_api.sdk_runner, "run_dispatch", _blocking_run_dispatch)

    resp = client.post(
        "/dispatch",
        json={"prompt": "do it", "allowed_tools": ["Read"], "owner": "alice"},
        headers=_auth(),
    )
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]
    try:
        assert dispatch_api._runs[run_id]["status"] == "pending"
        assert dispatch_api._runs[run_id]["owner"] == "alice"
    finally:
        client.portal.call(release.set)
    _wait_for_terminal(client, run_id)


@pytest.mark.parametrize("bad", ["${OSPREY_TERMINAL_USER}", "a/b", "x" * 65, ""])
def test_dispatch_request_refuses_a_malformed_owner_to_none(bad: str) -> None:
    """The worker routes the body's owner through the shared guard; a refusal names nobody."""
    request = dispatch_api.DispatchRequest(prompt="hi", allowed_tools=[], owner=bad)

    assert request.owner is None


# ---------------------------------------------------------------------------
# Earlier answers the run may read back
# ---------------------------------------------------------------------------

_RUN = "3f2b6c1e-8a4d-4f0e-9b1a-2c3d4e5f6a7b"


def test_dispatch_body_prior_answer_runs_reach_the_runner(client, monkeypatch):
    captured = _capture_run_dispatch(monkeypatch)

    resp = client.post(
        "/dispatch",
        json={"prompt": "do it", "allowed_tools": ["Read"], "prior_answer_runs": [_RUN]},
        headers=_auth(),
    )
    assert resp.status_code == 202
    _wait_for_terminal(client, resp.json()["run_id"])

    assert captured["prior_answer_runs"] == [_RUN]


def test_dispatch_request_prior_answer_runs_is_additive() -> None:
    request = dispatch_api.DispatchRequest(prompt="hi", allowed_tools=[])

    assert request.prior_answer_runs is None


def test_dispatch_request_drops_malformed_prior_answer_runs_instead_of_refusing(
    client, monkeypatch
):
    captured = _capture_run_dispatch(monkeypatch)

    resp = client.post(
        "/dispatch",
        json={
            "prompt": "do it",
            "allowed_tools": ["Read"],
            "prior_answer_runs": [_RUN, "../etc"],
        },
        headers=_auth(),
    )
    assert resp.status_code == 202
    _wait_for_terminal(client, resp.json()["run_id"])

    assert captured["prior_answer_runs"] == [_RUN]


# ---------------------------------------------------------------------------
# The Claude Code the worker's runs execute, named at startup
# ---------------------------------------------------------------------------


@pytest.fixture
def pinned_render(monkeypatch, tmp_path, real_agent_cli):  # noqa: ARG001 - opts into the real lookup
    """A render pinned to the build the image's global install reports."""
    from osprey.agent_runner import launcher

    render = tmp_path / "build"
    render.mkdir()
    (render / "config.yml").write_text("claude_code:\n  cli_version: '2.1.146'\n")
    monkeypatch.setenv("CONFIG_FILE", str(render / "config.yml"))
    monkeypatch.setattr("shutil.which", lambda *_a, **_k: "/usr/local/bin/claude")
    monkeypatch.setattr(launcher, "_probe_cli_version", lambda _exe: "2.1.146")
    monkeypatch.setattr(dispatch_api, "_inject_provider_env_once", lambda: None)
    return render


async def _start_and_stop_the_worker() -> None:
    async with dispatch_api._lifespan(dispatch_api.app):
        pass
    await asyncio.sleep(0)  # let the cancelled cleanup task settle


def _cli_lines(caplog) -> list:
    return [r for r in caplog.records if r.getMessage().startswith("Claude Code CLI")]


@pytest.mark.asyncio
async def test_the_worker_startup_line_names_the_build_headless_runs_execute(pinned_render, caplog):
    from osprey.agent_runner.primitives import build_agent_options

    with caplog.at_level("INFO", logger=dispatch_api.logger.name):
        await _start_and_stop_the_worker()

    (line,) = _cli_lines(caplog)
    assert line.levelname == "INFO"
    assert line.getMessage() == "Claude Code CLI: /usr/local/bin/claude (2.1.146)"
    # The line names the executable a run is actually pointed at.
    options = build_agent_options(pinned_render, disallowed_tools=[], env={})
    assert options.cli_path == "/usr/local/bin/claude"


@pytest.mark.parametrize(
    ("config", "found", "error", "expected"),
    [
        (
            "'2.1.146'",
            "2.1.999",
            "AgentCliUnavailableError",
            "npm install -g @anthropic-ai/claude-code@2.1.146",
        ),
        ("'2.1.146'", None, "AgentCliUnavailableError", "did not report a version"),
        ("'2.1.146'", "timeout", "AgentCliUnavailableError", "did not answer"),
        ("'latest'", "2.1.146", "ValueError", "exact version"),
    ],
    ids=["another-version", "no-version", "probe-timeout", "not-a-version"],
)
@pytest.mark.asyncio
async def test_a_worker_whose_pinned_build_is_not_installed_refuses_to_start(
    pinned_render, caplog, monkeypatch, config, found, error, expected
):
    """It could only refuse every job, so it accepts none and says why."""
    import subprocess

    from osprey.agent_runner import launcher

    (pinned_render / "config.yml").write_text(f"claude_code:\n  cli_version: {config}\n")

    def probe(exe):
        if found == "timeout":
            raise subprocess.TimeoutExpired([exe, "--version"], 15)
        return found

    monkeypatch.setattr(launcher, "_probe_cli_version", probe)
    reached: list[str] = []
    monkeypatch.setattr(
        dispatch_api, "_inject_provider_env_once", lambda: reached.append("provider env")
    )

    with pytest.raises(Exception) as refused:
        async with dispatch_api._lifespan(dispatch_api.app):
            reached.append("accepting jobs")

    assert type(refused.value).__name__ == error
    message = str(refused.value)
    assert message.startswith("Refusing to start the dispatch worker.\n")
    assert expected in message
    assert reached == [], "nothing else was started"
    assert refused.value.__cause__ is None
    assert not _cli_lines(caplog), "no line claims a build the worker does not have"


@pytest.mark.asyncio
@pytest.mark.usefixtures("pinned_render")
async def test_a_worker_with_no_claude_at_all_refuses_to_start(monkeypatch):
    from osprey.agent_runner.errors import AgentCliUnavailableError

    monkeypatch.setattr("shutil.which", lambda *_a, **_k: None)

    with pytest.raises(AgentCliUnavailableError, match="no `claude` was found on PATH"):
        async with dispatch_api._lifespan(dispatch_api.app):
            pytest.fail("the worker started")
