"""Code review's runtime safety switch: registered from the profile, moved by the admin only.

The consumers run in the agent's processes and read the switch through the panel-tier
``GET /api/code-review/switch``; the admin moves it through the safety-switch route, which
audits every move. The Review tab reads the same row the Safety tab lists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces._serving import WEB_HOST_ENV
from osprey.interfaces.web_auth import Tier, classify
from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.routes.code_review import SWITCH_ROUTE
from osprey.interfaces.web_terminal.routes.code_review import router as review_router
from osprey.interfaces.web_terminal.routes.safety_switches import (
    AUDIT_SUBJECT_SAFETY_SWITCH,
    SAFETY_SWITCH_FRAME,
)
from osprey.interfaces.web_terminal.routes.safety_switches import router as switch_router
from osprey.interfaces.web_terminal.safety_switches import CODE_REVIEW_SWITCH, SafetySwitches
from osprey.mcp_server.http import CODE_REVIEW_SWITCH_ROUTE

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


# ---- The consumers' read ----


def test_the_read_is_panel_tier_and_the_move_is_not():
    """An agent's panel token reads the switch and can never move it."""
    assert classify("GET", SWITCH_ROUTE, False) is Tier.PANEL
    assert classify("POST", MOVE_URL, False) is Tier.OPERATOR
    assert classify("POST", SWITCH_ROUTE, False) is Tier.OPERATOR
    assert SWITCH_ROUTE == CODE_REVIEW_SWITCH_ROUTE


def test_the_read_says_on_until_the_admin_turns_it_off(tmp_path):
    client = TestClient(_app(tmp_path, admin=True))
    assert client.get(SWITCH_ROUTE).json() == {"on": True, "off_by": None, "off_at": None}

    assert client.post(MOVE_URL, json={"state": "off"}).status_code == 200

    state = client.get(SWITCH_ROUTE).json()
    assert state["on"] is False
    assert state["off_by"]
    assert isinstance(state["off_at"], float)


@pytest.mark.parametrize("registry", ["none", "empty"])
def test_no_registry_or_no_switch_in_it_reads_on(registry):
    state = {} if registry == "none" else {"safety_switches": SafetySwitches()}
    app = bare_route_app(review_router, config_path=None, **state)

    assert TestClient(app).get(SWITCH_ROUTE).json()["on"] is True


# ---- The admin's move ----


def test_a_non_admin_cannot_turn_it_off_and_the_refusal_is_audited(tmp_path, audit):
    client = TestClient(_app(tmp_path, admin=False))

    assert client.post(MOVE_URL, json={"state": "off"}).status_code == 403
    assert client.get(SWITCH_ROUTE).json()["on"] is True
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


@pytest.mark.usefixtures("web_config", "bound")
def test_a_profile_with_code_review_on_registers_the_switch_on(tmp_path):
    config = _render(tmp_path, {"enabled": True})
    with TestClient(create_app(config_path=config, shell_command=["echo"])) as client:
        [switch] = client.get("/api/safety-switches").json()["switches"]
        assert switch["name"] == CODE_REVIEW_SWITCH
        assert switch["on"] is True
        assert client.get(SWITCH_ROUTE).json()["on"] is True


@pytest.mark.parametrize(
    "section",
    [None, {"enabled": False}, {"enabled": "yes"}],
    ids=["no-section", "off", "unreadable"],
)
@pytest.mark.usefixtures("web_config", "bound")
def test_a_profile_without_code_review_on_registers_nothing(tmp_path, section):
    """Off is the only widening: nothing is registered that could turn review on."""
    assert _names(_render(tmp_path, section)) == []


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "localhost", "0.0.0.0", "::"])
@pytest.mark.usefixtures("web_config")
def test_a_terminal_its_reviews_can_reach_registers_the_switch(tmp_path, bound, host):
    bound(host)
    assert _names(_render(tmp_path, {"enabled": True})) == [CODE_REVIEW_SWITCH]


@pytest.mark.parametrize("host", ["192.168.1.20", "::1", "terminal.example.org", None])
@pytest.mark.usefixtures("web_config")
def test_a_terminal_its_reviews_cannot_reach_registers_nothing_and_says_why(
    tmp_path, bound, host, caplog
):
    """The reviews dial 127.0.0.1; an off switch there would never be heard, yet read 'Off'."""
    bound(host)
    with caplog.at_level("WARNING", logger="osprey.interfaces.web_terminal.app"):
        assert _names(_render(tmp_path, {"enabled": True})) == []
    assert any("127.0.0.1" in r.getMessage() for r in caplog.records)
