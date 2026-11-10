"""The web terminal's approve card: ``/api/terminal/write-authorizations``.

A holder (an optimizer run) writes a request into its ledger directory; this
terminal reads it from the read-only bind ``OSPREY_WRITE_REQUESTS_DIR`` and the
signed-in login approves it, sees the authorization on the chip, and stops it.

What the tests below pin:

* **The card shows what is being approved**: channels, ranges, the limits
  bound each range sits inside, current values, steps, expiry, holder, label,
  completion, and that a write the limits refuse still costs its step.
* **An approval is refused** for a login whose ceiling is ``never``, a range
  outside the limits database, a channel that is not numeric (by the database
  or by its current value), a request that is missing or names another login;
  a grant file beside it that cannot be read is no veto.
* **The chip shows a machine an authorization holds**, its switch off
  beside it, and its off control revokes it: the next write refuses. One whose
  holder's ledger shows its steps spent or its lease lapsed is not shown held.
* **A program's self-issued grant at allowed** — in the holders' ledger
  directory, which the terminal binds read-only — is listed and shown on the
  chip as self-issued, without a revoke, and an approval refuses to overlap it.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from osprey.interfaces.web_terminal import control_context_owner
from osprey.interfaces.web_terminal.routes import write_authorizations as routes
from osprey_connectors import control_context
from osprey_connectors import write_authorization as wa
from osprey_connectors.identity import acting_identity
from osprey_connectors.write_authorization import Approver, ChannelRange, Grant
from osprey_connectors_tests._harness import write_control_context
from tests.interfaces.web_terminal._started_app import terminal_client

#: The real reader, kept before the autouse fixture below stands a fake in for it.
REAL_READ_CURRENT_VALUES = routes.read_current_values

SESSION = "aaaaaaaa-1111-2222-3333-444444444444"
TARGET = "standin"
FAR_FUTURE_NS = time.time_ns() + 12 * 3600 * 10**9

LIMITS = {
    "SR:A": {"writable": True, "min_value": 0.0, "max_value": 10.0},
    "SR:B": {"writable": True, "min_value": -5.0, "max_value": 5.0},
    "SR:ENUM": {"writable": True},
    "SR:RO": {"writable": False, "min_value": 0.0, "max_value": 1.0},
}


def render(tmp_path: Path, *, writes: str = "allowed", limits: Any = LIMITS) -> Path:
    gateway = {"address": "localhost", "port": 5074, "use_name_server": True}
    section: dict[str, Any] = {
        "type": "live_standin",
        "writes": writes,
        "limits_checking": {"enabled": True, "mode": "exclusive"},
        "connector": {
            "live_standin": {
                "probe_channel": "SR:PROBE",
                "gateways": {"read_only": dict(gateway), "write_access": dict(gateway)},
            },
            "virtual_accelerator": {"gateways": {"read_only": dict(gateway)}},
        },
    }
    if limits is not None:
        database = tmp_path / "channel_limits.json"
        database.write_text(json.dumps(limits), encoding="utf-8")
        section["limits_checking"]["database_path"] = str(database)
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "control_system": section,
                "archiver": {"type": "epics_archiver"},
                "services": {"live_standin": {"port": 5074}},
                "deployed_services": ["virtual_accelerator", "live_standin"],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture(autouse=True)
def requests_dir(tmp_path, monkeypatch) -> Path:
    directory = tmp_path / "write_ledger" / "requests"
    directory.mkdir(parents=True)
    monkeypatch.setenv(wa.REQUESTS_DIR_ENV_VAR, str(directory))
    for name in (
        wa.AUTHORIZATION_ENV_VAR,
        wa.LEDGER_DIR_ENV_VAR,
        wa.LEDGER_VIEW_DIR_ENV_VAR,
    ):
        monkeypatch.delenv(name, raising=False)
    return directory


@pytest.fixture(autouse=True)
def values(monkeypatch) -> dict[str, Any]:
    """What the card's connector reads: every channel at 1.0 unless a test says otherwise."""
    current: dict[str, Any] = {}

    async def fake(_config_path, _target, addresses):
        return {address: current.get(address, 1.0) for address in addresses}

    monkeypatch.setattr(routes, "read_current_values", fake)
    return current


@pytest.fixture
def make_client(agent_data_root, workspace_dir, tmp_path):
    write_control_context(agent_data_root, target="standin", generation=1)

    @contextmanager
    def _make(config_path: Path | None = None):
        with patch.object(
            control_context_owner.ControlContextOwnerTask, "start", lambda self: None
        ):
            with terminal_client(workspace_dir) as client:
                client.app.state.config_path = config_path or render(tmp_path)
                yield client

    return _make


@pytest.fixture
def client(make_client):
    with make_client() as c:
        yield c


def put_request(directory: Path, **fields: Any) -> wa.WriteRequest:
    """A holder's request, as ``write_request`` writes it from the holder's container."""
    payload: dict[str, Any] = {
        "schema": 1,
        "id": uuid.uuid4().hex,
        "owner": acting_identity(),
        "holder": "optimizer",
        "target": TARGET,
        "channels": {"SR:A": {"min": 1.0, "max": 9.0}, "SR:B": {"min": -1.0, "max": 1.0}},
        "steps": 12,
        "requested_at_ns": time.time_ns(),
        "expires_at_ns": FAR_FUTURE_NS,
        "lease_s": 30.0,
        "label": "run-42",
        "completion": "apply_best",
    }
    payload.update(fields)
    request = wa._parse_request(payload)
    (directory / f"{request.id}.json").write_text(json.dumps(payload), encoding="utf-8")
    return request


def approve(client, request_id: str):
    """Approve as the chip does."""
    return client.post(
        "/api/terminal/write-authorizations",
        json={"request_id": request_id, "session_id": SESSION},
    )


def own_grants() -> tuple[Grant, ...]:
    directory = wa.authorizations_dir()
    assert directory is not None
    return wa.list_grants(directory)[0]


def row(client, target: str = TARGET) -> dict[str, Any]:
    payload = client.get("/api/terminal/posture").json()
    return next(r for r in payload["targets"] if r["target"] == target) | {
        "_any_armed": payload["any_armed"]
    }


class TestTheCard:
    def test_a_pending_request_for_this_login_is_listed(self, client, requests_dir):
        mine = put_request(requests_dir)
        put_request(requests_dir, owner="someone-else")
        put_request(requests_dir, requested_at_ns=1, expires_at_ns=2)
        (requests_dir / f"{uuid.uuid4().hex}.json").write_text("{")
        listing = client.get("/api/terminal/write-authorizations").json()
        assert [r["id"] for r in listing["requests"]] == [mine.id]
        assert listing["unreadable"] == 1
        assert listing["authorizations"] == []

    def test_the_card_shows_what_is_approved(self, client, requests_dir, values):
        request = put_request(requests_dir)
        values["SR:A"] = 4.5
        card = client.get(f"/api/terminal/write-authorizations/requests/{request.id}").json()
        assert card["approvable"] is True and card["refusal"] is None
        assert (card["holder"], card["label"], card["steps"], card["completion"]) == (
            "optimizer",
            "run-42",
            12,
            "apply_best",
        )
        assert card["lease_s"] == 30.0 and card["expires_at"]
        assert card["channels"][0] == {
            "address": "SR:A",
            "min": 1.0,
            "max": 9.0,
            "limit_min": 0.0,
            "limit_max": 10.0,
            "problem": None,
            "current": 4.5,
        }
        assert "still costs its step" in card["limits_note"]

    def test_a_card_names_what_would_refuse_it(self, client, requests_dir):
        request = put_request(requests_dir, channels={"SR:A": {"min": -1.0, "max": 9.0}})
        card = client.get(f"/api/terminal/write-authorizations/requests/{request.id}").json()
        assert card["approvable"] is False
        assert card["refusal"]["reason"] == "outside_limits"
        assert card["channels"][0]["problem"] == "outside_limits"

    def test_a_card_names_every_channel_that_would_refuse_it(self, client, requests_dir):
        request = put_request(
            requests_dir,
            channels={"SR:A": {"min": -1.0, "max": 9.0}, "SR:RO": {"min": 0.0, "max": 1.0}},
        )
        card = client.get(f"/api/terminal/write-authorizations/requests/{request.id}").json()
        assert card["refusal"]["reason"] == "outside_limits"
        assert "SR:A" in card["refusal"]["phrase"] and "SR:RO" in card["refusal"]["phrase"]
        assert [c["problem"] for c in card["channels"]] == ["outside_limits", "not_writable"]

    def test_a_request_a_program_on_the_host_filed_is_listed_and_approved(self, client):
        """A program on the host with no ledger bound files in its owner's own directory."""
        own = wa.authorizations_dir()
        assert own is not None
        (own / wa.REQUESTS_DIR_NAME).mkdir(parents=True, exist_ok=True)
        request = put_request(own / wa.REQUESTS_DIR_NAME)
        listing = client.get("/api/terminal/write-authorizations").json()
        assert [r["id"] for r in listing["requests"]] == [request.id]
        assert approve(client, request.id).status_code == 200
        assert [grant.id for grant in own_grants()] == [request.id]

    def test_a_missing_request_is_404(self, client):
        response = client.get(f"/api/terminal/write-authorizations/requests/{uuid.uuid4().hex}")
        assert response.status_code == 404


class TestApproval:
    def test_an_approval_issues_the_grant_and_the_chip_shows_it(self, client, requests_dir):
        request = put_request(requests_dir)
        assert row(client)["program_hold"] is False
        response = approve(client, request.id)
        assert response.status_code == 200, response.text
        (grant,) = own_grants()
        assert grant.id == request.id
        assert (grant.approver.identity, grant.approver.surface) == (acting_identity(), "chip")
        assert grant.holder == "optimizer"
        chip = row(client)
        assert chip["switch"] == "off"
        assert chip["program_hold"] is True and chip["_any_armed"] is True
        assert chip["programs"][0]["holder"] == "optimizer"
        assert row(client, "va")["program_hold"] is False
        listing = client.get("/api/terminal/write-authorizations").json()
        assert listing["requests"] == []
        assert [(a["id"], a["in_force"]) for a in listing["authorizations"]] == [(request.id, True)]
        assert approve(client, request.id).json()["detail"]["reason"] == "already_approved"

    @pytest.mark.parametrize(
        ("channels", "reason"),
        [
            ({"SR:A": {"min": 0.0, "max": 11.0}}, "outside_limits"),
            ({"SR:NOPE": {"min": 0.0, "max": 1.0}}, "not_in_limits"),
            ({"SR:ENUM": {"min": 0.0, "max": 1.0}}, "not_numeric"),
            ({"SR:RO": {"min": 0.0, "max": 1.0}}, "not_writable"),
        ],
    )
    def test_a_range_the_limits_database_does_not_hold_is_refused(
        self, client, requests_dir, channels, reason
    ):
        request = put_request(requests_dir, channels=channels)
        response = approve(client, request.id)
        assert response.status_code == 422
        assert response.json()["detail"]["reason"] == reason
        assert response.json()["detail"]["phrase"]
        assert own_grants() == ()

    def test_a_channel_whose_value_is_not_a_number_is_refused(self, client, requests_dir, values):
        values["SR:B"] = "OPEN"
        response = approve(client, put_request(requests_dir).id)
        assert (response.status_code, response.json()["detail"]["reason"]) == (422, "not_numeric")
        assert own_grants() == ()

    def test_a_login_whose_ceiling_is_never_is_refused(self, make_client, requests_dir, tmp_path):
        with make_client(render(tmp_path, writes="never")) as client:
            response = approve(client, put_request(requests_dir).id)
        assert response.status_code == 403
        assert response.json()["detail"] == {"reason": "tier_caps", "ceiling": "never"}
        assert own_grants() == ()

    def test_no_limits_database_refuses(self, make_client, requests_dir, tmp_path):
        with make_client(render(tmp_path, limits=None)) as client:
            response = approve(client, put_request(requests_dir).id)
        assert (response.status_code, response.json()["detail"]["reason"]) == (
            503,
            "limits_unavailable",
        )

    def test_an_unreadable_grant_beside_it_does_not_veto_the_approval(self, client, requests_dir):
        directory = wa.authorizations_dir()
        assert directory is not None
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{uuid.uuid4().hex}.json").write_text("{")
        response = approve(client, put_request(requests_dir).id)
        assert response.status_code == 200, response.text

    def test_a_missing_request_and_another_logins_are_refused(self, client, requests_dir):
        assert approve(client, uuid.uuid4().hex).status_code == 404
        theirs = put_request(requests_dir, owner="someone-else")
        response = approve(client, theirs.id)
        assert (response.status_code, response.json()["detail"]["reason"]) == (
            403,
            "owner_mismatch",
        )
        assert approve(client, "../../etc/passwd").status_code in (400, 404)
        assert own_grants() == ()

    def test_a_machine_this_render_does_not_configure_is_refused(self, client, requests_dir):
        response = approve(client, put_request(requests_dir, target="live").id)
        assert (response.status_code, response.json()["detail"]["reason"]) == (
            400,
            "unknown_target",
        )


class TestStopping:
    def test_the_revoke_route_ends_it(self, client, requests_dir):
        request = put_request(requests_dir)
        assert approve(client, request.id).status_code == 200
        response = client.post(
            f"/api/terminal/write-authorizations/{request.id}/revoke",
            json={"session_id": SESSION},
        )
        assert response.json() == {"id": request.id, "revoked": True}
        (grant,) = own_grants()
        assert grant.revoked is not None and grant.revoked.by == acting_identity()
        assert row(client)["program_hold"] is False

    @pytest.mark.parametrize("target", [TARGET, "all"])
    def test_the_rows_off_control_revokes_it(self, client, requests_dir, target):
        request = put_request(requests_dir)
        assert approve(client, request.id).status_code == 200
        response = client.post(
            "/api/terminal/posture",
            json={"session_id": SESSION, "target": target, "state": "off", "epoch": None},
        )
        assert response.status_code == 200, response.text
        control_context.invalidate_cache()
        (grant,) = own_grants()
        assert grant.revoked is not None
        chip = row(client)
        assert chip["program_hold"] is False and chip["_any_armed"] is False


class TestEndedByTheLedger:
    """A grant whose holder's ledger ended it no longer shows as a program writing."""

    @staticmethod
    def _ledger(requests_dir: Path, monkeypatch, grant_id: str, *, steps: int, lease_ns: int):
        ledgers = requests_dir.parent
        monkeypatch.setenv(wa.LEDGER_VIEW_DIR_ENV_VAR, str(ledgers))
        ledger = wa.Ledger(
            authorization=grant_id,
            holder_pid=None,
            holder_start=None,
            pid_ns=None,
            lease_until_ns=lease_ns,
            steps=tuple(
                wa.LedgerStep(n=n, at_ns=n, channels=("SR:A",)) for n in range(1, steps + 1)
            ),
        )
        (ledgers / f"{grant_id}.ledger.json").write_text(json.dumps(ledger.to_payload()))

    def test_a_running_grant_shows_its_steps_used(self, client, requests_dir, monkeypatch):
        request = put_request(requests_dir)
        assert approve(client, request.id).status_code == 200
        self._ledger(requests_dir, monkeypatch, request.id, steps=5, lease_ns=FAR_FUTURE_NS)
        chip = row(client)
        assert chip["program_hold"] is True
        assert [(p["used"], p["steps"]) for p in chip["programs"]] == [(5, 12)]
        (entry,) = client.get("/api/terminal/write-authorizations").json()["authorizations"]
        assert (entry["in_force"], entry["used"], entry["steps"]) == (True, 5, 12)

    @pytest.mark.parametrize(
        ("steps", "lease_ns"), [(12, FAR_FUTURE_NS), (1, 1)], ids=["steps-spent", "lease-lapsed"]
    )
    def test_an_ended_grant_is_not_a_program_hold(
        self, client, requests_dir, monkeypatch, steps, lease_ns
    ):
        request = put_request(requests_dir)
        assert approve(client, request.id).status_code == 200
        self._ledger(requests_dir, monkeypatch, request.id, steps=steps, lease_ns=lease_ns)
        chip = row(client)
        assert chip["program_hold"] is False and chip["programs"] == []
        (entry,) = client.get("/api/terminal/write-authorizations").json()["authorizations"]
        assert (entry["in_force"], entry["used"]) == (False, steps)


class TestSelfIssued:
    @pytest.fixture
    def held(self, client, requests_dir, monkeypatch) -> Grant:  # noqa: ARG002 - the record the client writes
        """A program's grant self-issued at allowed, in the ledger the terminal sees read-only."""
        ledgers = requests_dir.parent
        monkeypatch.setenv(wa.LEDGER_VIEW_DIR_ENV_VAR, str(ledgers))
        control_context.invalidate_cache()
        record = control_context.read_record()
        assert record is not None
        epoch = record.switch_epoch
        if epoch is None:
            with pytest.raises(control_context.StoppedSince) as stamped:
                control_context.set_switch([TARGET], "asked", epoch=None)
            epoch = stamped.value.epoch
        control_context.set_switch([TARGET], "allowed", epoch=epoch)
        control_context.invalidate_cache()
        now = time.time_ns()
        grant = Grant(
            id=uuid.uuid4().hex,
            owner=acting_identity(),
            holder="tuning-scripts",
            target=TARGET,
            switch_epoch=epoch,
            channels={"SR:B": ChannelRange(0.0, 1.0)},
            steps=50,
            issued_at_ns=now,
            expires_at_ns=FAR_FUTURE_NS,
            lease_s=30.0,
            approver=Approver(identity="tuning-scripts", surface="write_switch", at_ns=now),
            label="scan",
        )
        wa._issue_grant(grant, directory=ledgers, peers=())
        return grant

    def test_it_is_listed_as_self_issued_and_not_revocable(self, client, held):
        listing = client.get("/api/terminal/write-authorizations").json()
        (entry,) = listing["authorizations"]
        assert entry["id"] == held.id
        assert (entry["in_force"], entry["self_issued"], entry["revocable"]) == (True, True, False)
        chip = row(client)
        assert chip["program_hold"] is True
        assert [(p["id"], p["self_issued"]) for p in chip["programs"]] == [(held.id, True)]

    def test_an_approval_refuses_to_overlap_it(self, client, requests_dir, held):
        response = approve(client, put_request(requests_dir).id)
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["reason"] == "auth_overlap"
        assert held.id in detail["phrase"]
        assert own_grants() == ()

    def test_moving_the_switch_off_allowed_ends_it_on_the_chip(self, client, held):
        response = client.post(
            "/api/terminal/posture",
            json={"session_id": SESSION, "target": TARGET, "state": "off", "epoch": None},
        )
        assert response.status_code == 200, response.text
        control_context.invalidate_cache()
        assert row(client)["program_hold"] is False
        (entry,) = client.get("/api/terminal/write-authorizations").json()["authorizations"]
        assert entry["id"] == held.id and entry["in_force"] is False


class _Pool:
    """A connector-host pool whose reads take *read_s* each and whose close takes *close_s*."""

    def __init__(self, read_s: float, close_s: float) -> None:
        self.read_s, self.close_s = read_s, close_s

    def __call__(self, *_args: Any, **_kwargs: Any) -> _Pool:
        return self

    async def __aenter__(self) -> _Pool:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        import asyncio

        await asyncio.sleep(self.close_s)

    async def connector(self, _target: str) -> _Pool:
        return self

    async def read_channel(self, address: str, timeout: float) -> Any:  # noqa: ARG002
        import asyncio
        from types import SimpleNamespace

        await asyncio.sleep(self.read_s)
        return SimpleNamespace(value=float(len(address)))


class TestCurrentValues:
    """The card's values: read together, and never waited on past the timeout."""

    @pytest.fixture
    def pool(self, monkeypatch):
        import osprey_connectors.ipc

        def install(read_s: float, close_s: float = 0.0) -> None:
            monkeypatch.setattr(osprey_connectors.ipc, "ConnectorHostPool", _Pool(read_s, close_s))

        monkeypatch.setattr(routes, "VALUES_TIMEOUT_S", 0.5)
        return install

    async def test_the_channels_are_read_together(self, tmp_path, pool):
        pool(read_s=0.3)
        started = time.monotonic()
        values = await REAL_READ_CURRENT_VALUES(render(tmp_path), TARGET, ["SR:A", "SR:BB", "SR:C"])
        assert values == {"SR:A": 4.0, "SR:BB": 5.0, "SR:C": 4.0}
        assert time.monotonic() - started < 0.5

    async def test_a_slow_machine_is_not_waited_on_past_the_timeout(self, tmp_path, pool):
        pool(read_s=5.0, close_s=5.0)
        started = time.monotonic()
        values = await REAL_READ_CURRENT_VALUES(render(tmp_path), TARGET, ["SR:A", "SR:B"])
        assert values == {"SR:A": None, "SR:B": None}
        assert time.monotonic() - started < 1.5
