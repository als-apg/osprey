"""The notebook tile's two hub routes: its server log and its restart.

A stub sidecar stands in for the real one (the launch harness of
``test_web_custom_panels``), so these tests pin what the routes answer and
which sidecar calls they make, not the process.
"""

from __future__ import annotations

import threading

import pytest

from osprey.interfaces.web_terminal.app import UNIVERSAL_PANELS
from tests._live import wait_until
from tests.interfaces.web_terminal.test_web_custom_panels import (
    SIDECAR_ID,
    SidecarScript,
    _make_client,
)

LOG = f"/api/panels/{SIDECAR_ID}/log"
RESTART = f"/api/panels/{SIDECAR_ID}/restart"


def _client(workspace_dir, script, enabled=True):
    panels = set(UNIVERSAL_PANELS) | ({SIDECAR_ID} if enabled else set())
    return _make_client(workspace_dir, enabled_panels=panels, sidecar_script=script)


def _settle(client, predicate):
    def settled():
        body = client.get(f"/api/panels/{SIDECAR_ID}/server").json()
        return body if predicate(body) else None

    return wait_until(settled, timeout=5.0, interval=0.02, what="the sidecar's config route")


@pytest.fixture
def running(workspace_dir):
    script = SidecarScript(
        preflight_error=None,
        stderr_tail="[I] Serving notebooks\n[I] http://127.0.0.1:9/lab?token=stub-token",
    )
    for client in _client(workspace_dir, script):
        yield client, script


@pytest.fixture
def failed(workspace_dir):
    yield from _client(workspace_dir, SidecarScript(preflight_error="boom"))


@pytest.fixture
def disabled(workspace_dir):
    yield from _client(workspace_dir, SidecarScript(), enabled=False)


class TestServerLog:
    def test_it_answers_the_running_sidecars_tail_without_its_token(self, running):
        client, _ = running

        resp = client.get(LOG)

        assert resp.status_code == 200
        text = resp.json()["text"]
        assert "Serving notebooks" in text
        assert "stub-token" not in text
        assert "token=<token>" in text

    def test_after_a_failed_start_it_answers_the_recorded_reason(self, failed):
        resp = failed.get(LOG)

        assert resp.status_code == 200
        assert resp.json() == {"text": "boom"}

    def test_it_is_empty_while_there_is_nothing_to_show(self, workspace_dir):
        for client in _client(workspace_dir, SidecarScript(preflight_error=None)):
            assert client.get(LOG).json() == {"text": ""}

    def test_it_is_404_when_the_panel_is_not_enabled(self, disabled):
        assert disabled.get(LOG).status_code == 404


class TestRestart:
    def test_it_stops_the_running_sidecar_and_starts_a_new_one(self, running):
        client, script = running
        state = client.app.state

        resp = client.post(RESTART)

        assert resp.status_code == 202
        assert resp.json()["state"] == "starting"
        body = _settle(client, lambda b: b["state"] == "running")
        assert body["available"] is True
        assert script.events == ["spawn", "stop", "spawn"]
        assert SIDECAR_ID in state.panel_auth_headers

    def test_the_panel_answers_unavailable_until_the_new_sidecar_is_up(self, running):
        client, script = running
        script.hold = threading.Event()

        client.post(RESTART)
        body = client.get(f"/api/panels/{SIDECAR_ID}/server").json()

        assert body["available"] is False
        assert body["state"] == "starting"
        script.hold.set()
        _settle(client, lambda b: b["state"] == "running")

    def test_a_second_request_while_one_runs_joins_it(self, running):
        client, script = running
        script.hold = threading.Event()

        assert client.post(RESTART).status_code == 202
        assert client.post(RESTART).status_code == 202
        script.hold.set()
        _settle(client, lambda b: b["state"] == "running")
        assert script.spawned == 2

    def test_it_starts_a_sidecar_whose_start_failed(self, workspace_dir):
        script = SidecarScript(preflight_error="boom")
        for client in _client(workspace_dir, script):
            script.preflight_error = None

            assert client.post(RESTART).status_code == 202
            body = _settle(client, lambda b: b["state"] == "running")
            assert body["available"] is True

    def test_it_is_404_when_the_panel_is_not_enabled(self, disabled):
        assert disabled.post(RESTART).status_code == 404
