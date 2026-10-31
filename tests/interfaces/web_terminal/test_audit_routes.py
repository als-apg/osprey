"""``GET /api/audit/recent`` — the gated read of this identity's audit entries.

The route serves the newest ``audit`` entries back to the Config panel, read
from the agent record through the record client. Three properties are what the
suite exists to hold:

* **It is a tier boundary, not a convenience.** The audit trail names every
  refusal a deployment produced, the subjects they were about and the sessions
  they happened in. A tier that may not open the Config panel may not read it
  either, so the route shares the Config panel's gate and its refusal verbatim,
  and the gate answers before any parameter check.
* **Only what this identity wrote.** The read asks the record for
  ``written_by="self"``; an entry another identity wrote is never served, and
  no request field names an identity.
* **Nothing here takes the web terminal down, and nothing hides a gap.** An
  unconfigured, unreachable or refusing record answers an empty list in the
  same shape marked ``"record": "unavailable"``; a partial page is marked
  ``"incomplete"``. Only a whole answer is ``"complete"``.

The read path's tests use :class:`~tests.agent_record.fakes.FakeRecordClient`;
one end-to-end test writes through :mod:`osprey.audit.writer` as two
identities, consumes the spool into a real agent-record service and reads it
back through the route.
"""

from __future__ import annotations

import base64
import json
from collections import deque
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from osprey.agent_record.client import ClientAnswer, RecordClient
from osprey.audit import writer
from osprey.audit.envelope import POSTURE_SOURCE_APP, AuditEnvelope
from osprey.interfaces.web_terminal.routes import audit as audit_routes
from osprey.interfaces.web_terminal.routes import router
from osprey.interfaces.web_terminal.routes.agent_activity import ACTIVITY_BUFFER_MAX
from osprey.interfaces.web_terminal.routes.config import _require_config_panel
from tests.agent_record.fakes import FakeRecordClient

RECENT = "/api/audit/recent"

#: The identity this process writes (and therefore reads) under.
OWN_IDENTITY = "alice"

BASE_CONFIG = {
    "project_name": "audit-routes",
    "control_system": {"writes_enabled": False},
    "claude_code": {"default_model": "claude-sonnet-5"},
}


def _body(ts: str, *, surface: str = "mcp", subject: str = "s", **extra) -> dict:
    """One audit entry body, in the shape :meth:`AuditEnvelope.to_dict` emits."""
    body = {
        "ts": ts,
        "surface": surface,
        "actor": OWN_IDENTITY,
        "posture": "readonly",
        "posture_source": "config",
        "session": None,
        "subject": subject,
        "decision": "refused",
        "reason": "denied",
    }
    body.update(extra)
    return body


def _item(body: dict | None, *, identity: str = OWN_IDENTITY, n: int = 0) -> dict:
    """One ``recent_audit`` item as the record serves it."""
    return {
        "key": f"sp:{identity}:{n}",
        "kind": "audit",
        "ts": body["ts"] if body else "2026-01-01T00:00:00+00:00",
        "identity": identity,
        "body": body,
        "incomplete": body is None,
    }


def _answer(*bodies: dict | None) -> ClientAnswer:
    return ClientAnswer(data=[_item(body, n=n) for n, body in enumerate(bodies)], complete=True)


@pytest.fixture
def project_dir(tmp_path) -> Path:
    """A project directory carrying a parseable ``config.yml``."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "config.yml").write_text(yaml.safe_dump(BASE_CONFIG, sort_keys=False))
    return project


@pytest.fixture
def record(monkeypatch) -> FakeRecordClient:
    """The record the route reads, answering an empty list until a test cans one."""
    fake = FakeRecordClient(login=OWN_IDENTITY)
    monkeypatch.setattr(audit_routes, "record_client", lambda: fake)
    monkeypatch.setattr(audit_routes, "acting_identity", lambda: OWN_IDENTITY)
    return fake


def _app(project_dir, *, config_panel_enabled: bool | None) -> FastAPI:
    """A routes-only app; ``None`` leaves the flag OFF ``app.state`` entirely."""
    app = FastAPI()
    app.include_router(router)
    app.state.config_path = project_dir / "config.yml"
    app.state.project_cwd = str(project_dir)
    app.state.agent_activity_buffer = deque(maxlen=ACTIVITY_BUFFER_MAX)
    if config_panel_enabled is not None:
        app.state.config_panel_enabled = config_panel_enabled
    return app


@pytest.fixture
def client(project_dir):
    """Config panel enabled — the tier that may read the audit trail."""
    with TestClient(_app(project_dir, config_panel_enabled=True)) as client:
        yield client


@pytest.fixture
def disabled_client(project_dir):
    with TestClient(_app(project_dir, config_panel_enabled=False)) as client:
        yield client


@pytest.fixture
def default_client(project_dir):
    """No flag on state at all — the absent flag is refused, never assumed on."""
    with TestClient(_app(project_dir, config_panel_enabled=None)) as client:
        yield client


def _subjects(response) -> list[str]:
    return [record["subject"] for record in response.json()["records"]]


class _FakeRequest:
    """Minimal stand-in carrying only ``app.state``, for the gate comparison."""

    def __init__(self, *, config_panel_enabled: bool) -> None:
        state = type("State", (), {"config_panel_enabled": config_panel_enabled})()
        self.app = type("App", (), {"state": state})()


# ---- the gate ---- #


class TestTierGate:
    """The route is behind the Config panel tier, with that tier's refusal."""

    @pytest.mark.usefixtures("record")
    def test_the_refusal_is_the_config_panels_own_wording(self, disabled_client):
        """Not a second spelling: the SAME gate produced it."""
        with pytest.raises(HTTPException) as excinfo:
            _require_config_panel(_FakeRequest(config_panel_enabled=False))
        assert disabled_client.get(RECENT).json()["detail"] == excinfo.value.detail

    def test_an_absent_flag_means_disabled(self, default_client, record):
        """The gate fails closed, and the record is never asked."""
        response = default_client.get(RECENT)
        assert response.status_code == 403
        assert "web.config_panel.enabled" in response.json()["detail"]
        assert record.calls == []

    @pytest.mark.parametrize(
        "params",
        [{}, {"surface": "../bob"}, {"surface": ""}, {"limit": "abc"}, {"limit": "1e9"}],
        ids=["no-params", "traversal-surface", "empty-surface", "non-numeric-limit", "float-limit"],
    )
    def test_the_gate_answers_before_any_parameter_check(self, disabled_client, record, params):
        """403, never 400 or 422 — the gate is the FIRST thing that answers.

        A 422 for ``limit=abc`` would confirm the route exists and that it
        takes an integer ``limit``, to a caller who may not read the audit
        trail at all. That is why ``limit`` arrives as a string.
        """
        response = disabled_client.get(RECENT, params=params)
        assert response.status_code == 403, (params, response.text)
        assert record.calls == []

    @pytest.mark.usefixtures("record")
    def test_a_non_numeric_limit_is_refused_once_past_the_gate(self, client):
        response = client.get(RECENT, params={"limit": "abc"})
        assert response.status_code == 400
        assert "limit" in response.json()["detail"]


# ---- the read ---- #


class TestTheRead:
    """One ``recent_audit`` read, for this identity's own entries."""

    def test_it_asks_for_this_identitys_own_entries(self, client, record):
        client.get(RECENT)

        (call,) = record.calls_to("recent_audit")
        assert call["written_by"] == "self"
        assert call["limit"] == audit_routes.DEFAULT_LIMIT
        assert call.get("surface") is None

    def test_records_are_the_entry_bodies_in_the_records_order(self, client, record):
        record.answers["recent_audit"] = _answer(
            *(_body(f"2026-01-01T00:00:0{n}Z", subject=f"s{n}") for n in (3, 2, 1, 0))
        )
        assert _subjects(client.get(RECENT)) == ["s3", "s2", "s1", "s0"]

    def test_records_are_served_verbatim(self, client, record):
        body = _body("2026-01-01T00:00:00Z", detail="why", role="operator")
        record.answers["recent_audit"] = _answer(body)

        assert client.get(RECENT).json()["records"] == [body]

    def test_the_response_keeps_its_shape(self, client, record):
        record.answers["recent_audit"] = _answer(
            _body("2026-01-01T00:00:02Z", surface="mcp"),
            _body("2026-01-01T00:00:01Z", surface="http_config"),
        )

        payload = client.get(RECENT).json()

        assert set(payload) == {"identity", "ledgers", "limit", "records", "record"}
        assert payload["record"] == "complete"
        assert payload["identity"] == OWN_IDENTITY
        assert payload["ledgers"] == ["http_config", "mcp"]
        assert payload["limit"] == audit_routes.DEFAULT_LIMIT

    def test_the_limit_is_passed_to_the_record(self, client, record):
        response = client.get(RECENT, params={"limit": 3})

        assert response.json()["limit"] == 3
        assert record.calls_to("recent_audit")[0]["limit"] == 3

    def test_an_oversized_limit_is_clamped_not_refused(self, client, record):
        response = client.get(RECENT, params={"limit": 10_000})

        assert response.status_code == 200
        assert response.json()["limit"] == audit_routes.MAX_LIMIT
        assert record.calls_to("recent_audit")[0]["limit"] == audit_routes.MAX_LIMIT

    @pytest.mark.parametrize("limit", [0, -1])
    def test_a_non_positive_limit_is_clamped_to_one(self, client, record, limit):
        response = client.get(RECENT, params={"limit": limit})

        assert response.json()["limit"] == 1
        assert record.calls_to("recent_audit")[0]["limit"] == 1

    def test_an_entry_without_a_body_is_skipped(self, client, record):
        """An incomplete entry carries no body; it is not a record to serve."""
        record.answers["recent_audit"] = _answer(
            None, _body("2026-01-01T00:00:00Z", subject="only")
        )
        assert _subjects(client.get(RECENT)) == ["only"]

    @pytest.mark.usefixtures("record")
    def test_no_entries_is_an_empty_list(self, client):
        payload = client.get(RECENT).json()
        assert payload["records"] == []
        assert payload["ledgers"] == []


class TestTheRecordUnavailable:
    """Nothing here takes the web terminal down, and nothing passes "the record
    could not answer" off as "there was nothing to answer"."""

    def test_an_unavailable_record_is_marked_not_a_500(self, client, record, caplog):
        record.unavailable = "agent-record is unreachable"

        with caplog.at_level("WARNING"):
            response = client.get(RECENT)

        assert response.status_code == 200
        assert response.json() == {
            "identity": OWN_IDENTITY,
            "ledgers": [],
            "limit": audit_routes.DEFAULT_LIMIT,
            "records": [],
            "record": "unavailable",
            "reason": "agent-record is unreachable",
        }
        assert "agent-record is unreachable" in caplog.text

    def test_an_answer_that_is_not_a_list_is_unavailable(self, client, record):
        record.answers["recent_audit"] = ClientAnswer(data={"oops": 1}, complete=True)

        payload = client.get(RECENT).json()

        assert payload["records"] == []
        assert payload["record"] == "unavailable"
        assert payload["reason"]

    def test_an_unconfigured_client_is_unavailable(self, client, monkeypatch):
        """With no record URL in the environment the real client answers
        unavailable, and the route says so rather than failing or serving an
        empty list as if it were whole."""
        monkeypatch.delenv("OSPREY_AGENT_RECORD_URL", raising=False)
        monkeypatch.delenv("OSPREY_AGENT_RECORD_LOGIN", raising=False)

        response = client.get(RECENT)

        assert response.status_code == 200
        assert response.json()["records"] == []
        assert response.json()["record"] == "unavailable"

    @pytest.mark.usefixtures("record")
    def test_a_complete_empty_answer_is_marked_complete(self, client):
        """The one empty list that means "nothing was written"."""
        payload = client.get(RECENT).json()

        assert payload["records"] == []
        assert payload["record"] == "complete"
        assert "reason" not in payload


class TestAnIncompleteAnswer:
    """A served page the record could not finish is never shown as whole."""

    def test_an_incomplete_page_is_marked_with_its_reason(self, client, record):
        record.answers["recent_audit"] = ClientAnswer(
            data=[_item(_body("2026-01-01T00:00:00Z", subject="seen"))],
            complete=False,
            reason="scan limit reached",
        )

        payload = client.get(RECENT).json()

        assert [r["subject"] for r in payload["records"]] == ["seen"]
        assert payload["record"] == "incomplete"
        assert payload["reason"] == "scan limit reached"

    def test_an_incomplete_empty_page_is_not_complete(self, client, record):
        record.answers["recent_audit"] = ClientAnswer(data=[], complete=False)

        payload = client.get(RECENT).json()

        assert payload["records"] == []
        assert payload["record"] == "incomplete"
        assert payload["reason"]


class TestToolCallEntriesDoNotDisplaceDecisions:
    """tool_call entries are never served; when they fill the newest page the
    route reads the record's whole row ceiling so decisions behind them show."""

    @staticmethod
    def _tool_calls(n: int, start: int = 0) -> list[dict]:
        return [
            _item(_body(f"2026-01-02T00:00:{i % 60:02d}Z", surface="tool_call"), n=start + i)
            for i in range(n)
        ]

    def test_a_decision_behind_a_page_of_tool_calls_is_served(self, client, record):
        limit = 5
        decision = _item(_body("2026-01-01T00:00:00Z", subject="refused"), n=99)
        first = ClientAnswer(data=self._tool_calls(limit), complete=True, more=True)
        whole = ClientAnswer(
            data=[*self._tool_calls(2 * limit), decision], complete=True, more=False
        )
        record.answers["recent_audit"] = [first, whole]

        payload = client.get(RECENT, params={"limit": limit}).json()

        assert [r["subject"] for r in payload["records"]] == ["refused"]
        assert payload["record"] == "complete"
        calls = record.calls_to("recent_audit")
        assert calls[0]["limit"] == limit
        assert calls[1].get("limit") is None
        assert all(c["written_by"] == "self" for c in calls)

    def test_decisions_still_crowded_out_are_marked_incomplete(self, client, record):
        limit = 5
        crowded = ClientAnswer(data=self._tool_calls(limit), complete=True, more=True)
        record.answers["recent_audit"] = crowded

        payload = client.get(RECENT, params={"limit": limit}).json()

        assert payload["records"] == []
        assert payload["record"] == "incomplete"
        assert "tool_call" in payload["reason"]

    def test_a_full_page_of_decisions_reads_once(self, client, record):
        record.answers["recent_audit"] = ClientAnswer(
            data=[_item(_body("2026-01-01T00:00:00Z", subject="d"), n=0)],
            complete=True,
            more=True,
        )

        payload = client.get(RECENT, params={"limit": 1}).json()

        assert [r["subject"] for r in payload["records"]] == ["d"]
        assert payload["record"] == "complete"
        assert len(record.calls_to("recent_audit")) == 1


class TestSurfaceFilter:
    """The one client-supplied string, passed as a value to compare."""

    def test_a_surface_filter_is_the_records(self, client, record):
        record.answers["recent_audit"] = _answer(_body("2026-01-01T00:00:00Z", subject="m"))

        response = client.get(RECENT, params={"surface": "mcp"})

        assert _subjects(response) == ["m"]
        assert response.json()["ledgers"] == ["mcp"]
        assert record.calls_to("recent_audit")[0]["surface"] == "mcp"

    @pytest.mark.parametrize(
        "value",
        [
            "../bob/mcp",
            "..",
            ".",
            "/etc/passwd",
            "mcp/../../bob/mcp",
            "a\\b",
            "mcp\x00",
            "",
            " ",
            "x" * 300,
        ],
    )
    def test_a_path_ish_surface_is_refused(self, client, record, value):
        response = client.get(RECENT, params={"surface": value})
        assert response.status_code == 400, value
        assert "surface" in response.json()["detail"]
        assert record.calls == []


def test_the_tool_call_surface_is_not_served(client, record):
    """Its entries carry payloads the recent-activity list is not for."""
    record.answers["recent_audit"] = _answer(
        _body("2026-01-01T00:00:01Z", surface="tool_call", subject="full"),
        _body("2026-01-01T00:00:00Z", subject="default"),
    )

    assert _subjects(client.get(RECENT)) == ["default"]
    assert _subjects(client.get(RECENT, params={"surface": "tool_call"})) == []
    assert len(record.calls_to("recent_audit")) == 1, "surface=tool_call asked the record"


# ---- end to end: writer -> spool -> agent-record -> route ---- #


def _token(login: str) -> str:
    return f"token-of-{login}-0123456789"


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def _envelope(subject: str, actor: str) -> AuditEnvelope:
    return AuditEnvelope(
        surface="http_config",
        actor=actor,
        posture="sandbox",
        posture_source=POSTURE_SOURCE_APP,
        session=None,
        subject=subject,
        decision="refused",
        reason="protected_key",
    )


def test_only_the_entries_this_identity_wrote_are_served(
    client, spool_zone, spool_for, monkeypatch, tmp_path
):
    """Two identities write through the audit writer; each reads back only its own."""
    from osprey.agent_record import credentials as creds
    from osprey.agent_record import identities as ids
    from osprey.agent_record import service
    from osprey.interfaces._serving import run_app_server

    spool_for("alice")
    alice = _envelope("alice-decision", "alice")
    assert writer.record_envelope(alice)
    spool_for("bob")
    assert writer.record_envelope(_envelope("bob-decision", "bob"))

    entries = spool_zone.recorded_entries("audit")
    assert {e.identity for e in entries} == {"alice", "bob"}

    config = spool_zone.config()
    config["project_root"] = str(tmp_path / "project")
    lines = "".join(
        creds.verifier_line(login, _token(login)) + "\n" for login in ids.credential_table(config)
    )
    env = {
        creds.VERIFIER_ENV: _b64(lines),
        creds.SECRETS_ENV: _b64(json.dumps(["s3cr3t-token-value-0123456789"])),
        service.SPOOL_ROOT_ENV: str(tmp_path / "svc" / "spool"),
        service.RECORD_DIR_ENV: str(tmp_path / "svc" / "record"),
        service.TRANSCRIPTS_DIR_ENV: str(tmp_path / "svc" / "transcripts"),
    }
    app = service.create_app(config=config, backend=spool_zone._backend, env=env)
    with run_app_server(app) as url:
        monkeypatch.setattr(
            audit_routes, "record_client", lambda: RecordClient(url, "alice", _token("alice"))
        )
        monkeypatch.setattr(audit_routes, "acting_identity", lambda: "alice")

        payload = client.get(RECENT).json()

    assert payload["records"] == [alice.to_dict()]
    assert payload["identity"] == "alice"
