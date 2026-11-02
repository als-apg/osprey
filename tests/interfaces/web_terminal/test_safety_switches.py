"""Runtime safety switches: the admin-only off, its audit line, and the watchdog under it.

The hold record is the real one on disk and engines are stubbed at the runner's
``engines=`` seam, as in the watchdog route suite this one borrows from.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.common_middleware import HttpAuditMiddleware
from osprey.interfaces.web_auth import Tier, classify
from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.routes.safety_switches import (
    AUDIT_SUBJECT_SAFETY_SWITCH,
    SAFETY_SWITCH_FRAME,
)
from osprey.interfaces.web_terminal.routes.safety_switches import router as switch_router
from osprey.interfaces.web_terminal.routes.watchdog import make_notifier
from osprey.interfaces.web_terminal.routes.watchdog import router as watchdog_router
from osprey.interfaces.web_terminal.safety_switches import (
    WATCHDOG_SWITCH,
    SafetySwitches,
    feature_is_on,
)
from osprey.services.watchdog import runner as runner_mod
from osprey.services.watchdog.config import ENGINE_DECISION, WatchdogDefinition
from osprey.services.watchdog.history import SessionNodes
from osprey.services.watchdog.runner import WatchdogRunner
from osprey_connectors.session_hold import HOLD_DIR_ENV_VAR, ensure_session, is_held, open_flags

from .conftest import bare_route_app
from .test_watchdog_routes import KEY, FakePool, StubEngine, _write_build, submission

SWITCH_URL = f"/api/safety-switches/{WATCHDOG_SWITCH}"


@pytest.fixture(autouse=True)
def _no_env_root(monkeypatch):
    monkeypatch.delenv(HOLD_DIR_ENV_VAR, raising=False)


@pytest.fixture(autouse=True)
def _no_record_read(monkeypatch):
    monkeypatch.setattr(runner_mod, "read_session_nodes", lambda *_a, **_k: SessionNodes())


@pytest.fixture
def audit(monkeypatch) -> list[dict]:
    """Every audit record written, as the fields handed to the writer."""
    lines: list[dict] = []

    def record(**fields):
        lines.append(fields)
        return "entry"

    monkeypatch.setattr("osprey.audit.writer.record", record)
    return lines


@pytest.fixture
def root(tmp_path) -> Path:
    """The hold store, with the record a spawn creates for the session."""
    root = tmp_path / "var" / "state" / "session_holds"
    ensure_session(KEY, root=root)
    return root


@pytest.fixture
def engine() -> StubEngine:
    return StubEngine()


class _Owner:
    """Stamp the owner the gate would mint, as ``WebAuthMiddleware`` does."""

    def __init__(self, app, owner: str) -> None:
        self.app, self.owner = app, owner

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope.setdefault("state", {})["osprey_owner"] = self.owner
        await self.app(scope, receive, send)


def _app(root: Path, engine: StubEngine, *, admin: bool) -> FastAPI:
    switches = SafetySwitches()
    switches.register(WATCHDOG_SWITCH, "Safety watchdog")
    app = bare_route_app(
        watchdog_router,
        switch_router,
        session_hold_root=root,
        pty_registry=FakePool(KEY),
        operator_registry=FakePool(),
        safety_switches=switches,
        config_panel_enabled=admin,
    )
    definition = WatchdogDefinition(name="wd", engine=ENGINE_DECISION, enabled=True, threshold=0.5)
    app.state.watchdog_runner = WatchdogRunner(
        [definition],
        recorder=lambda _verdict: None,
        hold_root=root,
        notify=make_notifier(app),
        engines={ENGINE_DECISION: engine},
        may_hold=lambda: feature_is_on(app, WATCHDOG_SWITCH),
    )
    # The shipped order: the audit layer innermost, the gate's owner stamp outside it.
    app.add_middleware(HttpAuditMiddleware)
    app.add_middleware(_Owner, owner="alice")
    return app


@pytest.fixture
def admin(root, engine):
    with TestClient(_app(root, engine, admin=True)) as client:
        yield client


@pytest.fixture
def operator(root, engine):
    with TestClient(_app(root, engine, admin=False)) as client:
        yield client


def _prompt(client: TestClient, **overrides):
    resp = client.post("/api/watchdog/prompt", json=submission(**overrides))
    client.portal.call(client.app.state.watchdog_runner.drain)
    return resp


def _switch_lines(audit: list[dict]) -> list[dict]:
    return [line for line in audit if line.get("subject") == AUDIT_SUBJECT_SAFETY_SWITCH]


# ---- The gate ----


def test_a_non_admin_tier_is_refused_and_the_refusal_is_audited(operator, root, engine, audit):
    resp = operator.post(SWITCH_URL, json={"state": "off"})

    assert resp.status_code == 403
    assert operator.app.state.safety_switches.is_on(WATCHDOG_SWITCH)
    [line] = _switch_lines(audit)
    assert line["decision"] == "refused"
    assert line["reason"] == "not_admin_tier"
    # The watchdog still judges and holds.
    assert _prompt(operator).status_code == 202
    assert engine.calls
    assert is_held(KEY, root=root)


def test_the_panel_token_never_reaches_the_switch_routes():
    """An agent holds only the panel token; it can never turn off what watches it."""
    assert classify("POST", SWITCH_URL, False) is Tier.OPERATOR
    assert classify("GET", "/api/safety-switches", False) is Tier.OPERATOR


def test_an_unknown_switch_is_404_and_audited(admin, audit):
    resp = admin.post("/api/safety-switches/nope", json={"state": "off"})

    assert resp.status_code == 404
    [line] = _switch_lines(audit)
    assert line["reason"] == "unknown_switch"


def test_every_operator_can_read_the_switches_and_who_turned_one_off(admin, operator):
    assert admin.post(SWITCH_URL, json={"state": "off"}).status_code == 200

    seen = admin.get("/api/safety-switches").json()
    [switch] = seen["switches"]
    assert switch["name"] == WATCHDOG_SWITCH
    assert switch["on"] is False
    assert switch["off_by"]
    assert seen["can_change"] is True
    assert operator.get("/api/safety-switches").json()["can_change"] is False


@pytest.mark.parametrize(
    ("fixture", "url", "body", "status"),
    [
        ("admin", SWITCH_URL, {"state": "off"}, 200),
        ("admin", "/api/safety-switches/nope", {"state": "off"}, 404),
        ("operator", SWITCH_URL, {"state": "off"}, 403),
        ("admin", SWITCH_URL, {"state": "sideways"}, 422),
    ],
)
def test_every_move_files_exactly_one_audit_line(request, audit, fixture, url, body, status):
    client = request.getfixturevalue(fixture)
    audit.clear()

    assert client.post(url, json=body).status_code == status
    assert len(audit) == 1


def test_the_line_and_the_state_name_the_gate_minted_owner(admin, audit):
    admin.post(SWITCH_URL, json={"state": "off"})

    [line] = _switch_lines(audit)
    assert "by=alice" in line["detail"]
    [switch] = admin.get("/api/safety-switches").json()["switches"]
    assert switch["off_by"] == "alice"


# ---- The admin's off ----


def test_admin_off_is_audited_broadcast_and_stops_new_verdicts_and_holds(
    admin, root, engine, audit
):
    resp = admin.post(SWITCH_URL, json={"state": "off"})

    assert resp.json() == {"name": WATCHDOG_SWITCH, "on": False, "changed": True}
    [line] = _switch_lines(audit)
    assert line["decision"] == "allowed"
    assert "state=off" in line["detail"]
    frames = [call.args[0] for call in admin.app.state.broadcaster.broadcast.call_args_list]
    assert {"type": SAFETY_SWITCH_FRAME, "name": WATCHDOG_SWITCH, "on": False} in frames

    # Accepted (2xx), so the hook does not fall back to its own judging entry.
    resp = _prompt(admin)
    assert resp.status_code == 202
    assert resp.json()["judged"] is False
    assert engine.calls == []
    assert open_flags(KEY, root=root) == []
    assert not is_held(KEY, root=root)


def test_back_on_returns_to_the_profile_and_holds_again(admin, root, engine):
    admin.post(SWITCH_URL, json={"state": "off"})
    resp = admin.post(SWITCH_URL, json={"state": "on"})

    assert resp.json()["changed"] is True
    _prompt(admin)
    assert engine.calls
    assert is_held(KEY, root=root)


def test_a_repeated_off_changes_nothing_and_sends_no_frame(admin):
    admin.post(SWITCH_URL, json={"state": "off"})
    admin.app.state.broadcaster.broadcast.reset_mock()

    assert admin.post(SWITCH_URL, json={"state": "off"}).json()["changed"] is False
    admin.app.state.broadcaster.broadcast.assert_not_called()


def test_a_run_in_flight_when_switched_off_records_and_does_not_hold(root):
    """``may_hold`` is read when the verdict lands: the switch moves mid-run."""
    switch = {"on": True}

    class SwitchedOffMidRun(StubEngine):
        async def judge(self, defn, nodes, prompt, *, deadline, **ids):
            switch["on"] = False  # the admin's off lands while the judge runs
            return await super().judge(defn, nodes, prompt, deadline=deadline, **ids)

    verdicts: list = []
    definition = WatchdogDefinition(name="wd", engine=ENGINE_DECISION, enabled=True, threshold=0.5)
    runner = WatchdogRunner(
        [definition],
        recorder=verdicts.append,
        hold_root=root,
        notify=lambda _frame: None,
        engines={ENGINE_DECISION: SwitchedOffMidRun()},
        may_hold=lambda: switch["on"],
    )

    async def run():
        return await runner.on_nodes(
            session_key=KEY,
            record_session_id="0f6d7c2e-1a2b-4c3d-8e9f-0123456789ab",
            submission_id="9a8b7c6d-5e4f-4a3b-9c2d-1e0f9a8b7c6d",
            nodes=SessionNodes(),
            prompt="loosen the limits",
            deadline=10**12,
            can_hold=True,
        )

    [verdict] = asyncio.run(run())
    assert verdict.flagged
    assert verdicts == [verdict]
    assert open_flags(KEY, root=root) == []
    assert not is_held(KEY, root=root)


def test_no_registry_means_no_runtime_off():
    assert feature_is_on(FastAPI(), WATCHDOG_SWITCH)
    assert SafetySwitches().is_on("never-registered")


# ---- Wiring ----


def test_a_build_with_watchdogs_registers_one_switch_that_starts_on(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "osprey.interfaces.web_terminal.app._load_web_config",
        lambda _path=None: {"watch_dir": str(tmp_path)},
    )
    with TestClient(create_app(config_path=_write_build(tmp_path), shell_command=["echo"])) as c:
        [switch] = c.get("/api/safety-switches").json()["switches"]
        assert switch["name"] == WATCHDOG_SWITCH
        assert switch["on"] is True


def test_a_build_without_watchdogs_registers_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "osprey.interfaces.web_terminal.app._load_web_config",
        lambda _path=None: {"watch_dir": str(tmp_path)},
    )
    config = _write_build(tmp_path, enabled=False)
    with TestClient(create_app(config_path=config, shell_command=["echo"])) as c:
        assert c.get("/api/safety-switches").json()["switches"] == []
