"""The owner survives the two paths that re-fire a trigger.

A dispatched run is checked against its owner's narrowing, so an owner dropped
anywhere between the fire and the worker turns a checked run into an unchecked
one — silently, since the dispatch itself still succeeds. Two places can drop
it, and each has a row here:

* The on-error retry recursion. Attempt 2 is the same fire as attempt 1; an
  owner left behind there would make a transient worker failure the way to get
  an unchecked run.
* ``POST /retry/{trigger_name}``. The dashboard reaches it through the panel
  proxy, which mints ``X-Osprey-Owner``, so a retry clicked there is attributed
  like a ``manual_fire``; a call straight at the dispatcher's port carries no
  header and re-fires owner-less, exactly as a cron tick does.

Both rows read the owner off the ``/dispatch`` request body the worker actually
receives, rather than off an argument on the way there — the body is what the
worker resolves the run's owner from.

Run: ``python3 -m pytest tests/dispatch/test_retry_owner.py -q``
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from osprey.dispatch import server
from osprey.dispatch.registry import TriggerRegistry
from osprey.dispatch.trigger_config import TriggerConfig
from osprey.utils.owner_header import OWNER_HEADER
from tests.conftest import dispatcher_route_registry

# The route rows build the app through ``create_server()``, which configures a
# process-wide FastMCP singleton, so this module must not be split across xdist
# workers mid-file.
pytestmark = pytest.mark.xdist_group("dispatch_retry_owner")

#: The bearer these rows configure as ``EVENT_DISPATCHER_TOKEN``.
_TOKEN = "dispatcher-retry-secret"

#: Where the dispatcher is told the worker lives. Nothing resolves it: every
#: request is answered by the mock transport below.
_TARGET = "http://worker.invalid:9000"


def _worker_transport(
    captured: list[dict[str, Any]],
    statuses: list[int],
    answered: asyncio.Event | None = None,
) -> httpx.MockTransport:
    """Answer each ``/dispatch`` POST with the next status, recording its body.

    ``statuses`` is consumed one per call and 200 is used once it runs out. A
    503 is the retryable failure the on-error policy re-dispatches after.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        status = statuses.pop(0) if statuses else 200
        body = (
            {"run_id": "abc123", "status": "accepted"}
            if status == 200
            else {"detail": "worker busy"}
        )
        if answered is not None:
            answered.set()
        return httpx.Response(
            status_code=status,
            content=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
            request=request,
        )

    return httpx.MockTransport(handler)


def _patched_worker_client(transport: httpx.MockTransport):
    """Route the worker client's requests through ``transport``.

    ``worker_client`` holds the ``httpx`` module itself, so patching the class
    there replaces it for every client built anywhere in the process while the
    patch is active — the rows below build one of their own to drive the ASGI
    app. Hence ``setdefault``: a caller that names its own transport keeps it,
    and only the worker client (which names none) is diverted.
    """
    original_cls = httpx.AsyncClient

    class PatchedClient(original_cls):
        def __init__(self, **kwargs):
            kwargs.setdefault("transport", transport)
            super().__init__(**kwargs)

    return patch("osprey.dispatch.worker_client.httpx.AsyncClient", PatchedClient)


def _retrying_trigger(name: str = "deploy") -> TriggerConfig:
    """A trigger whose policy re-dispatches once after a retryable failure."""
    return TriggerConfig(
        name=name,
        source="webhook",
        action={"prompt": "do it", "allowed_tools": []},
        on_error={"action": "retry", "max_retries": 1, "backoff_sec": 0.0},
    )


# ---------------------------------------------------------------------------
# The retry recursion
# ---------------------------------------------------------------------------


async def test_both_attempts_carry_the_same_owner():
    """A re-dispatch is the same fire, so it is checked against the same person."""
    registry = TriggerRegistry()
    trigger = _retrying_trigger()
    await registry.register(trigger)
    captured: list[dict[str, Any]] = []

    with _patched_worker_client(_worker_transport(captured, [503, 200])):
        result = await server._dispatch_with_policy(
            trigger, {}, registry, _TARGET, _TOKEN, owner="alice"
        )

    assert result == {"run_id": "abc123", "status": "accepted"}
    assert [body.get("owner") for body in captured] == ["alice", "alice"]


async def test_both_attempts_of_an_owner_less_fire_stay_owner_less():
    """A cron tick names nobody on either attempt — the key is simply absent."""
    registry = TriggerRegistry()
    trigger = _retrying_trigger()
    await registry.register(trigger)
    captured: list[dict[str, Any]] = []

    with _patched_worker_client(_worker_transport(captured, [503, 200])):
        await server._dispatch_with_policy(trigger, {}, registry, _TARGET, _TOKEN)

    assert len(captured) == 2
    assert all("owner" not in body for body in captured)


async def test_history_names_the_owner_on_both_attempts():
    """Every attempt of a retried fire is recorded under the same person."""
    registry = TriggerRegistry()
    trigger = _retrying_trigger()
    await registry.register(trigger)
    captured: list[dict[str, Any]] = []

    with _patched_worker_client(_worker_transport(captured, [503, 200])):
        await server._dispatch_with_policy(trigger, {}, registry, _TARGET, _TOKEN, owner="alice")

    history = await registry.get_history("deploy")
    assert len(history) == 2
    assert history[0]["result"].startswith("error")
    assert history[1]["result"] == "dispatched"
    assert [entry["owner"] for entry in history] == ["alice", "alice"]


async def test_owner_less_history_entries_carry_no_owner_key():
    """An owner-less fire is recorded with no owner key on either attempt."""
    registry = TriggerRegistry()
    trigger = _retrying_trigger()
    await registry.register(trigger)
    captured: list[dict[str, Any]] = []

    with _patched_worker_client(_worker_transport(captured, [503, 200])):
        await server._dispatch_with_policy(trigger, {}, registry, _TARGET, _TOKEN)

    history = await registry.get_history("deploy")
    assert len(history) == 2
    assert all("owner" not in entry for entry in history)


# ---------------------------------------------------------------------------
# POST /retry/{trigger_name}
# ---------------------------------------------------------------------------


@pytest.fixture
def triggers_yml(tmp_path):
    """A triggers file with one trigger the retry route can name."""
    path = tmp_path / "triggers.yml"
    path.write_text(
        "dispatcher:\n"
        f"  dispatch_target: {_TARGET}\n"
        "  max_concurrent_runs: 2\n"
        "  max_queue_depth: 10\n"
        "triggers:\n"
        "  - name: deploy\n"
        "    source: webhook\n"
        "    action:\n"
        "      prompt: do it\n"
        "      allowed_tools: []\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def dispatcher_app(triggers_yml, monkeypatch):
    """The dispatcher's real ASGI app, with only this module's routes live.

    ``dispatcher_route_registry`` owns the module-level FastMCP singleton here:
    the app answers from the routes this fixture registered, and the singleton is
    left carrying the ones it had before.
    """
    monkeypatch.setenv("TRIGGERS_YML", str(triggers_yml))
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", _TOKEN)

    with dispatcher_route_registry() as build:
        yield build().http_app()


async def _retry_and_record(app, owner_header: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """POST /retry/deploy with *owner_header*; return the worker body and history entry.

    The route answers 202 as soon as the pool accepts the fire, so the dispatch
    itself lands afterwards — the event the mock transport sets is what says the
    worker has been called. The history entry is written after the worker has
    answered, so it is polled for, bounded at 5 s.
    """
    captured: list[dict[str, Any]] = []
    answered = asyncio.Event()
    headers = {"Authorization": f"Bearer {_TOKEN}"}
    if owner_header is not None:
        headers[OWNER_HEADER] = owner_header

    with _patched_worker_client(_worker_transport(captured, [200], answered)):
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://dispatcher.invalid",
            ) as client:
                response = await client.post("/retry/deploy", json={}, headers=headers)
                assert response.status_code == 202, response.text
                await asyncio.wait_for(answered.wait(), timeout=5)
                history = await _wait_for_history(server.mcp._dispatcher_registry, "deploy")

    return captured[0], history[-1]


async def _wait_for_history(registry: TriggerRegistry, name: str) -> list[dict[str, Any]]:
    """Poll *name*'s history until it holds an entry, bounded at 5 s."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5
    while True:
        history = await registry.get_history(name)
        if history:
            return history
        assert loop.time() < deadline, f"no history entry for {name!r} within 5 s"
        await asyncio.sleep(0.01)


async def _retry_over_the_wire(app, owner_header: str | None) -> dict[str, Any]:
    """POST /retry/deploy with *owner_header*; return the worker's request body."""
    body, _entry = await _retry_and_record(app, owner_header)
    return body


async def test_retry_credits_the_owner_named_by_the_header(dispatcher_app):
    """A retry arriving through the panel proxy is attributed like a manual fire."""
    body = await _retry_over_the_wire(dispatcher_app, "alice")

    assert body["owner"] == "alice"


async def test_retry_without_the_header_is_owner_less(dispatcher_app):
    """A call straight at the dispatcher's port names nobody, as cron does."""
    body = await _retry_over_the_wire(dispatcher_app, None)

    assert "owner" not in body


async def test_retry_with_a_malformed_header_is_owner_less(dispatcher_app):
    """A header that names nobody costs the owner, never the re-fire.

    The route reads the header through the shared guard, whose refusals are
    pinned in ``tests/utils/test_owner_header.py``; what matters here is that a
    refusal degrades to an owner-less dispatch instead of failing the request.
    """
    body = await _retry_over_the_wire(dispatcher_app, "${OSPREY_TERMINAL_USER}")

    assert "owner" not in body


@pytest.mark.parametrize(
    ("owner_header", "expected"),
    [("alice", "alice"), (None, None)],
    ids=["header", "no-header"],
)
async def test_retry_route_records_the_header_owner_in_history(
    dispatcher_app, owner_header, expected
):
    """The re-fire's history entry names the owner the header carried, or nobody."""
    _body, entry = await _retry_and_record(dispatcher_app, owner_header)

    assert entry["result"] == "dispatched"
    if expected is None:
        assert "owner" not in entry
    else:
        assert entry["owner"] == expected


async def test_a_disabled_fire_records_its_owner(dispatcher_app):
    """A fire refused because the trigger is disabled is still recorded under its owner."""
    registry = server.mcp._dispatcher_registry
    async with dispatcher_app.router.lifespan_context(dispatcher_app):
        await registry.set_status("deploy", "disabled")
        trigger = registry._triggers["deploy"]

        assert await server.mcp._fire_callback(trigger, {}, "alice") is None

        (entry,) = await registry.get_history("deploy")
    assert entry["result"] == "ignored: disabled"
    assert entry["owner"] == "alice"
