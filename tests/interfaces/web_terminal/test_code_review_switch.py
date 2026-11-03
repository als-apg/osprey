"""Code review's runtime safety switch: registered from the profile, moved by the admin only.

The consumers run in the agent's processes and read the switch from the terminal's own UNIX
socket; the admin moves it through the safety-switch route, which audits every move. The Review
tab reads the same row the Safety tab lists.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import stat
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces._serving import WEB_HOST_ENV
from osprey.interfaces.web_auth import Tier, classify
from osprey.interfaces.web_terminal import code_review_switch
from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.code_review_switch import serve_switch, switch_state
from osprey.interfaces.web_terminal.routes.code_review import router as review_router
from osprey.interfaces.web_terminal.routes.safety_switches import (
    AUDIT_SUBJECT_SAFETY_SWITCH,
    SAFETY_SWITCH_FRAME,
)
from osprey.interfaces.web_terminal.routes.safety_switches import router as switch_router
from osprey.interfaces.web_terminal.safety_switches import CODE_REVIEW_SWITCH, SafetySwitches
from osprey.services.code_review.switch import SWITCH_SOCKET_ENV

from .conftest import bare_route_app

MOVE_URL = f"/api/safety-switches/{CODE_REVIEW_SWITCH}"
RULES = "# Code reviewer\n\n- Read `SR:BEAM:USER_MODE` first.\n"


@pytest.fixture
def audit(monkeypatch) -> list[dict]:
    lines: list[dict] = []

    def record(**fields):
        lines.append(fields)
        return "entry"

    monkeypatch.setattr("osprey.audit.writer.record", record)
    return lines


def _render(tmp_path: Path, section: dict | None) -> Path:
    """A render at ``build/`` with the reviewer's rules in the source zone."""
    render = tmp_path / "build"
    render.mkdir(parents=True)
    prompt = tmp_path / "data" / "code_review" / "reviewer.md"
    prompt.parent.mkdir(parents=True)
    prompt.write_text(RULES, encoding="utf-8")
    config = {} if section is None else {"code_review": section}
    (render / "config.yml").write_text(json.dumps(config), encoding="utf-8")
    return render / "config.yml"


def _app(tmp_path: Path, *, admin: bool, registered: bool = True):
    switches = SafetySwitches()
    if registered:
        switches.register(CODE_REVIEW_SWITCH, "Code review", "Code runs unreviewed.")
    return bare_route_app(
        review_router,
        switch_router,
        config_path=_render(tmp_path, {"enabled": True}),
        safety_switches=switches,
        config_panel_enabled=admin,
    )


def _read(path: str) -> dict:
    """What the socket at *path* answers one connection."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(5)
        sock.connect(path)
        body = b""
        while chunk := sock.recv(4096):
            body += chunk
    return json.loads(body)


# ---- The consumers' read ----


def test_the_switch_has_no_http_route_and_the_move_is_operator_tier():
    """An agent's panel token can neither move the switch nor stand in for its socket."""
    assert classify("POST", MOVE_URL, False) is Tier.OPERATOR
    assert classify("GET", "/api/code-review/switch", False) is Tier.OPERATOR


def test_the_read_says_on_until_the_admin_turns_it_off(tmp_path):
    app = _app(tmp_path, admin=True)
    assert switch_state(app) == {"on": True, "off_by": None, "off_at": None}

    assert TestClient(app).post(MOVE_URL, json={"state": "off"}).status_code == 200

    state = switch_state(app)
    assert state["on"] is False
    assert state["off_by"]
    assert isinstance(state["off_at"], float)


@pytest.mark.parametrize("registry", ["none", "empty"])
def test_no_registry_or_no_switch_in_it_reads_on(registry):
    state = {} if registry == "none" else {"safety_switches": SafetySwitches()}
    app = bare_route_app(review_router, config_path=None, **state)

    assert switch_state(app)["on"] is True


# ---- The socket ----


@pytest.fixture
def unpublished(monkeypatch):
    """The listener publishes into ``os.environ``; leave it as it was found."""
    monkeypatch.delenv(SWITCH_SOCKET_ENV, raising=False)


@pytest.mark.usefixtures("unpublished")
async def test_the_socket_serves_the_row_each_connection_reads(tmp_path):
    app = _app(tmp_path, admin=True)
    listener = await serve_switch(app)
    try:
        assert os.environ[SWITCH_SOCKET_ENV] == listener.path
        on = await asyncio.to_thread(_read, listener.path)
        assert on == {"on": True, "off_by": None, "off_at": None}

        app.state.safety_switches.set(CODE_REVIEW_SWITCH, on=False, by="alice")
        off = await asyncio.to_thread(_read, listener.path)
        assert off["on"] is False
        assert off["off_by"] == "alice"
        assert isinstance(off["off_at"], float)
    finally:
        await listener.close()


@pytest.mark.usefixtures("unpublished")
async def test_the_socket_answers_on_without_a_registry():
    listener = await serve_switch(bare_route_app(review_router))
    try:
        assert (await asyncio.to_thread(_read, listener.path))["on"] is True
    finally:
        await listener.close()


@pytest.mark.usefixtures("unpublished")
async def test_only_the_owner_can_reach_the_socket(tmp_path):
    listener = await serve_switch(_app(tmp_path, admin=True))
    try:
        directory = os.stat(os.path.dirname(listener.path))
        sock = os.stat(listener.path)
        assert stat.S_IMODE(directory.st_mode) == 0o700
        assert stat.S_IMODE(sock.st_mode) == 0o600
        assert stat.S_ISSOCK(sock.st_mode)
        assert directory.st_uid == sock.st_uid == os.getuid()
        assert len(os.fsencode(listener.path)) <= 103
    finally:
        await listener.close()


@pytest.mark.usefixtures("unpublished")
async def test_closing_removes_the_socket_and_unpublishes_it(tmp_path):
    listener = await serve_switch(_app(tmp_path, admin=True))
    await listener.close()

    assert not os.path.exists(os.path.dirname(listener.path))
    assert SWITCH_SOCKET_ENV not in os.environ
    with pytest.raises(OSError):
        _read(listener.path)


# ---- The admin's move ----


def test_a_non_admin_cannot_turn_it_off_and_the_refusal_is_audited(tmp_path, audit):
    client = TestClient(_app(tmp_path, admin=False))

    assert client.post(MOVE_URL, json={"state": "off"}).status_code == 403
    assert switch_state(client.app)["on"] is True
    [line] = [x for x in audit if x.get("subject") == AUDIT_SUBJECT_SAFETY_SWITCH]
    assert line["decision"] == "refused"


def test_an_admin_off_is_audited_and_broadcast(tmp_path, audit):
    app = _app(tmp_path, admin=True)

    assert TestClient(app).post(MOVE_URL, json={"state": "off"}).status_code == 200

    [line] = [x for x in audit if x.get("subject") == AUDIT_SUBJECT_SAFETY_SWITCH]
    assert line["decision"] == "allowed"
    assert f"switch={CODE_REVIEW_SWITCH} state=off" in line["detail"]
    app.state.broadcaster.broadcast.assert_called_once_with(
        {"type": SAFETY_SWITCH_FRAME, "name": CODE_REVIEW_SWITCH, "on": False}
    )


# ---- The Review tab ----


def test_the_review_tab_carries_the_switch_row_for_every_tier(tmp_path):
    admin = TestClient(_app(tmp_path / "a", admin=True))
    admin.post(MOVE_URL, json={"state": "off"})
    row = admin.get("/api/code-review").json()["switch"]
    assert row["name"] == CODE_REVIEW_SWITCH
    assert row["label"] == "Code review"
    assert row["note"] == "Code runs unreviewed."
    assert row["on"] is False

    operator = TestClient(_app(tmp_path / "o", admin=False)).get("/api/code-review").json()
    assert operator["switch"]["on"] is True
    assert operator["can_change"] is False


def test_the_review_tab_has_no_switch_when_none_is_registered(tmp_path):
    body = TestClient(_app(tmp_path, admin=True, registered=False)).get("/api/code-review").json()

    assert body["switch"] is None


# ---- Wiring ----


@pytest.fixture
def web_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "osprey.interfaces.web_terminal.app._load_web_config",
        lambda _path=None: {"watch_dir": str(tmp_path)},
    )


def _names(config: Path) -> list[str]:
    with TestClient(create_app(config_path=config, shell_command=["echo"])) as client:
        return [s["name"] for s in client.get("/api/safety-switches").json()["switches"]]


@pytest.fixture
def bound(monkeypatch):
    """Bind the terminal as ``osprey web`` publishes it; loopback unless a test says otherwise."""

    def bind(host: str | None) -> None:
        if host is None:
            monkeypatch.delenv(WEB_HOST_ENV, raising=False)
        else:
            monkeypatch.setenv(WEB_HOST_ENV, host)

    bind("127.0.0.1")
    return bind


@pytest.mark.usefixtures("web_config", "bound", "unpublished")
def test_a_profile_with_code_review_on_registers_the_switch_on_and_serves_it(tmp_path):
    config = _render(tmp_path, {"enabled": True})
    with TestClient(create_app(config_path=config, shell_command=["echo"])) as client:
        [switch] = client.get("/api/safety-switches").json()["switches"]
        assert switch["name"] == CODE_REVIEW_SWITCH
        assert switch["on"] is True
        path = os.environ[SWITCH_SOCKET_ENV]
        assert _read(path)["on"] is True

        assert client.post(MOVE_URL, json={"state": "off"}).status_code == 200
        assert _read(path)["on"] is False
    assert not os.path.exists(path)
    assert SWITCH_SOCKET_ENV not in os.environ


@pytest.mark.parametrize(
    "section",
    [None, {"enabled": False}, {"enabled": "yes"}],
    ids=["no-section", "off", "unreadable"],
)
@pytest.mark.usefixtures("web_config", "bound")
def test_a_profile_without_code_review_on_registers_nothing(tmp_path, monkeypatch, section):
    """Off is the only widening: nothing is registered that could turn review on.

    A socket an outer terminal published is not passed on to this terminal's sessions.
    """
    monkeypatch.setenv(SWITCH_SOCKET_ENV, "/an/outer/terminal/s")
    assert _names(_render(tmp_path, section)) == []
    assert SWITCH_SOCKET_ENV not in os.environ


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "0.0.0.0", "::", "::1", "192.168.1.20", "terminal.example.org", None]
)
@pytest.mark.usefixtures("web_config", "unpublished")
def test_the_switch_is_registered_whatever_the_terminal_binds(tmp_path, bound, host):
    """The reviews ask the socket, not the bind address, so every bind can be switched off."""
    bound(host)
    assert _names(_render(tmp_path, {"enabled": True})) == [CODE_REVIEW_SWITCH]


@pytest.mark.usefixtures("web_config", "bound", "unpublished")
def test_a_terminal_that_cannot_serve_the_socket_registers_nothing_and_says_why(
    tmp_path, monkeypatch, caplog
):
    """An off switch nobody can read would never be heard, yet read 'Off'."""

    async def refuse(_app):
        raise OSError("no socket here")

    monkeypatch.setattr(code_review_switch, "serve_switch", refuse)
    with caplog.at_level("WARNING", logger="osprey.interfaces.web_terminal.app"):
        assert _names(_render(tmp_path, {"enabled": True})) == []
    assert any("no socket here" in r.getMessage() for r in caplog.records)
    assert SWITCH_SOCKET_ENV not in os.environ
