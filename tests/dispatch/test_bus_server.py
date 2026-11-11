"""The dispatcher serving the bus: its own routes, and the facade the bridges read.

The app is the real dispatcher built by ``create_server`` with
``DISPATCH_BUS_BIND`` set, run through a Starlette ``TestClient`` (its lifespan
binds the bus). The worker is a :class:`~tests.dispatch._bus_fakes.FakeWorker`
on a real link, in a thread and event loop of its own, as a separate process
would be.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any

import pytest
from starlette.testclient import TestClient

from osprey.dispatch import server
from osprey.dispatch.bus import protocol
from osprey.dispatch.bus.link import WorkerLink
from osprey.dispatch.sources.webhook import WebhookSource
from tests._live import wait_until
from tests.conftest import dispatcher_route_registry
from tests.dispatch._bus_fakes import FakeWorker
from tests.dispatch._fakes import FakeEntryPoint

_AUTH = {"Authorization": "Bearer secret"}
_WORKER_AUTH = {"Authorization": "Bearer worker-secret"}


class LinkThread:
    """Run a worker's link in a thread with its own event loop, for a ``with`` block."""

    def __init__(self, endpoint: str, worker: FakeWorker) -> None:
        self._endpoint = endpoint
        self._worker = worker
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._future: concurrent.futures.Future[None] | None = None

    def __enter__(self) -> LinkThread:
        self._thread.start()
        link = WorkerLink(
            self._endpoint,
            self._worker.name,
            self._worker.handlers(),
            hello=self._worker.hello,
            keys=None,
            heartbeat_sec=0.1,
        )
        self._worker.link = link
        self._future = asyncio.run_coroutine_threadsafe(link.run(), self._loop)
        return self

    def __exit__(self, *_exc: Any) -> None:
        assert self._future is not None
        self._future.cancel()
        # Let the cancelled link close its socket before the loop stops.
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0.2), self._loop).result(5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)
        self._loop.close()


@pytest.fixture(autouse=True)
def _reset_mcp_routes():
    with dispatcher_route_registry():
        yield


@pytest.fixture
def bus_app(tmp_path, monkeypatch):
    """The dispatcher on a loopback bus, with one webhook trigger."""
    triggers = tmp_path / "triggers.yml"
    triggers.write_text(
        "dispatcher:\n"
        "  dispatch_target: http://localhost:9\n"
        "triggers:\n"
        "  - name: deploy\n"
        "    source: webhook\n"
        "    action:\n"
        "      prompt: do the thing\n"
        "      allowed_tools: []\n"
    )
    monkeypatch.setenv("TRIGGERS_YML", str(triggers))
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "secret")
    monkeypatch.setenv("DISPATCH_WORKER_TOKEN", "worker-secret")
    monkeypatch.setenv("DISPATCH_BUS_BIND", "tcp://127.0.0.1:*")
    monkeypatch.setenv("DISPATCH_BUS_WORKER_WAIT_SEC", "5")
    # Both are module state create_server sets; restored so no later case
    # inherits a bus.
    monkeypatch.setattr(server, "_bus_client", None)
    monkeypatch.setattr(server.mcp, "_worker_bus", None, raising=False)
    monkeypatch.setattr(
        "osprey.dispatch.source_registry.entry_points",
        lambda *, group: [FakeEntryPoint("webhook", WebhookSource)],
    )
    return server.create_server().http_app()


def _endpoint() -> str:
    return server.mcp._worker_bus.endpoint  # type: ignore[attr-defined]


def test_a_webhook_fire_reaches_a_worker_that_connected_to_the_dispatcher(bus_app):
    worker = FakeWorker("dispatch-worker-1")
    with TestClient(bus_app) as client, LinkThread(_endpoint(), worker):
        fired = client.post("/webhook/deploy", json={"ref": "main"}, headers=_AUTH)
        assert fired.status_code == 202
        dispatch_id = fired.json()["dispatch_id"]

        def settled() -> dict | None:
            body = client.get(f"/dispatch/{dispatch_id}", headers=_AUTH).json()
            return body if body.get("status") == "completed" else None

        body = wait_until(settled, timeout=10, what="the dispatch to complete")
        assert body["result"] == {"status": "accepted", "run_id": "dispatch-worker-1-run-1"}
        assert worker.asked(protocol.DISPATCH) == 1
        sent = dict(worker.calls)[protocol.DISPATCH]["request"]
        assert sent["allowed_tools"] == []
        assert sent["prompt"].startswith("do the thing")

        health = client.get("/health").json()
        assert [w["name"] for w in health["workers"]] == ["dispatch-worker-1"]


def test_the_facade_answers_for_the_worker_that_owns_the_run(bus_app):
    worker = FakeWorker("dispatch-worker-1")
    worker.statuses["r1"] = {"run_id": "r1", "status": "completed", "text_output": "ok"}
    with TestClient(bus_app) as client:
        assert client.get("/worker/health").status_code == 503

        with LinkThread(_endpoint(), worker):
            health = wait_until(
                lambda: (r := client.get("/worker/health")).status_code == 200 and r.json(),
                what="the worker to join",
            )
            assert health["status"] == "ok"
            assert health["capabilities"] == ["input_files", "prior_answers"]

            status = client.get("/worker/dispatch/r1", headers=_WORKER_AUTH)
            assert status.status_code == 200
            assert status.json()["text_output"] == "ok"
            assert client.get("/worker/dispatch/nope", headers=_WORKER_AUTH).status_code == 404

            artifact = client.get("/worker/dispatch/r1/artifacts/a1", headers=_WORKER_AUTH)
            assert artifact.status_code == 200
            assert artifact.content == worker.blob
            assert artifact.headers["content-type"] == "image/png"
            assert artifact.headers["content-disposition"] == 'attachment; filename="plot.png"'

            listed = client.get("/worker/dispatch/r1/artifacts", headers=_WORKER_AUTH)
            assert listed.json() == [{"artifact_id": "a1"}]


@pytest.mark.parametrize(
    "path",
    ["/worker/dispatch/r1", "/worker/dispatch/r1/artifacts", "/worker/dispatch/r1/artifacts/a1"],
)
@pytest.mark.parametrize("headers", [{}, _AUTH], ids=["no-token", "dispatcher-token"])
def test_the_facade_takes_the_worker_token_only(bus_app, path, headers):
    with TestClient(bus_app) as client:
        assert client.get(path, headers=headers).status_code == 401


def test_a_dispatcher_off_the_bus_serves_no_facade(tmp_path, monkeypatch):
    monkeypatch.delenv("DISPATCH_BUS_BIND", raising=False)
    monkeypatch.setenv("TRIGGERS_YML", str(tmp_path / "absent.yml"))
    monkeypatch.setattr(server, "_bus_client", None)
    app = server.create_server().http_app()
    with TestClient(app) as client:
        assert client.get("/worker/health").status_code == 404
        assert "workers" not in client.get("/health").json()
    assert server._bus_client is None
