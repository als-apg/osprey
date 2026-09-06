"""Tests for ``POST /api/terminal/posture`` — the operator's write-posture toggle.

The posture is the operator's per-target sandbox toggle, and it belongs to the
**deployment**: one control context, one ``posture`` field in its record, read
live by every write-time gate. There is no per-session store behind it any
more, so a ``session_id`` names who made the gesture and decides nothing about
what the gesture does.

What the tests below pin:

* **The record write is the commit point.** It happens inside the mutation
  primitive, so a toggle is always derived from the record it lands on, and a
  terminal that may not write the record refuses instead of half-applying.
* **Narrows, never widens.** The ceiling is read per target, so a deployment
  that arms only its simulator never offers a writes toggle on the facility's
  own machine; narrowing needs no ceiling at all.
* **Absence spells ``writes``.** An entry that narrows nothing removes the key.
* **Nothing is respawned.** Neither the PTY child nor the chat child is torn
  down: the gates read the record live.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

from osprey.audit import writer as audit_writer
from osprey.interfaces.web_terminal import control_context_owner
from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey.mcp_server.control_system import target_state
from osprey_connectors import control_context, posture_store
from tests._control_context_fixtures import owner, write_control_context

SESSION_A = "aaaaaaaa-1111-2222-3333-444444444444"
SESSION_B = "bbbbbbbb-1111-2222-3333-444444444444"
CHAT_A = "cccccccc-1111-2222-3333-444444444444"

STANDIN_PORT = 5074

#: A pid no kernel hands out: the largest a 32-bit ``pid_t`` holds.
DEAD_PID = 2_147_483_646

#: The terminal that owns the context in the follower cases. ``os.getppid()``
#: is a real, live process, so a follower test is refused for the reason it
#: claims rather than because the owner it names is a corpse.
OTHER_TERMINAL_PID = os.getppid()
OTHER_TERMINAL_PORT = 8090


# -- render shapes ----------------------------------------------------------


def _gateways(port):
    row = {"address": "localhost", "port": port, "use_name_server": True}
    return {"read_only": dict(row), "write_access": dict(row)}


def control_system_section(
    *,
    global_writes=False,
    va_writes=None,
    epics_writes=None,
    standin_writes=None,
    standin_gateways=None,
):
    """The three-target render every test here starts from.

    ``epics`` is the facility's own machine, ``live_standin`` the co-deployed
    stand-in, ``virtual_accelerator`` the simulator — three connector blocks,
    therefore three targets, which is what makes this deployment switchable and
    ``session_posture`` answer one ceiling per target. The virtual accelerator
    carries a gateway table because the build writes one for every project that
    deploys the service; a VA with no table would derive no endpoints at all.
    """
    epics = {"gateways": _gateways(5064)}
    standin = {
        "gateways": _gateways(STANDIN_PORT) if standin_gateways is None else standin_gateways
    }
    va = {"simulation_file": "data/sim.json", "gateways": _gateways(5064)}
    if epics_writes is not None:
        epics["writes_enabled"] = epics_writes
    if standin_writes is not None:
        standin["writes_enabled"] = standin_writes
    if va_writes is not None:
        va["writes_enabled"] = va_writes
    return {
        "type": "live_standin",
        "writes_enabled": global_writes,
        "connector": {"epics": epics, "live_standin": standin, "virtual_accelerator": va},
    }


def write_config(tmp_path, section=None, *, name="config.yml"):
    """Write a ``config.yml`` carrying *section* (default: the shape above)."""
    path = tmp_path / name
    path.write_text(
        yaml.safe_dump(
            {
                "control_system": control_system_section() if section is None else section,
                "services": {
                    "live_standin": {"port": STANDIN_PORT},
                    "virtual_accelerator": {"port": 5064},
                },
                "deployed_services": ["virtual_accelerator", "live_standin"],
            }
        ),
        encoding="utf-8",
    )
    return path


# -- fixtures ---------------------------------------------------------------


@pytest.fixture
def agent_data_root(tmp_path, monkeypatch):
    """Point every resolver at one throwaway agent-data root.

    ``OSPREY_AGENT_DATA_ROOT`` is the single stamp the record reader and
    ``posture_store`` both prefer — the stamp this feature puts in every
    session child's environment. Patching ``resolve_shared_data_root`` instead
    would redirect only one of them: ``posture_store`` binds the resolver at
    import, so the other half would write into the repository's own
    ``var/agent_data``.
    """
    root = tmp_path / "agent_data"
    (root / posture_store.STATE_DIR_NAME).mkdir(parents=True)
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(root))
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    posture_store.invalidate_cache()
    control_context.invalidate_cache()
    yield root
    posture_store.invalidate_cache()
    control_context.invalidate_cache()


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_watch"
    ws.mkdir()
    return ws


@pytest.fixture
def make_client(agent_data_root, workspace_dir, tmp_path):
    """Build an app + TestClient, repeatably, over the same stamped root.

    The record is written **before** the first app starts, so the lifespan
    claim merges into it rather than minting a fresh one from
    ``load_osprey_config``, which this process has no workspace for. The owner
    task's loop is stubbed out: every tick these tests need has happened by the
    time the client is handed over, and a loop ticking underneath the
    assertions would be a race rather than coverage.
    """
    write_control_context(agent_data_root, target="live", generation=1)

    @contextmanager
    def _make(config_path=None):
        with patch.object(
            control_context_owner.ControlContextOwnerTask, "start", lambda self: None
        ):
            with patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value={"watch_dir": str(workspace_dir)},
            ):
                app = create_app(shell_command="echo")
                with TestClient(app) as test_client:
                    test_client.app.state.config_path = (
                        write_config(tmp_path) if config_path is None else config_path
                    )
                    yield test_client

    return _make


@pytest.fixture
def client(make_client):
    with make_client() as c:
        yield c


@pytest.fixture
def ledger():
    """Capture every audit record this request would have written.

    Patches ``osprey.audit.writer.record``, which BOTH recorders resolve at
    call time — the route's ``record_and_mark`` and
    ``HttpAuditMiddleware._emit_audit_record`` — so the count is the true
    number of ledger lines one POST produces.
    """
    records: list[dict] = []

    def _record(**fields):
        records.append(fields)
        return Path("/dev/null/ledger.jsonl")

    with patch.object(audit_writer, "record", side_effect=_record):
        yield records


class _RecordingChatPool:
    """An operator registry that remembers what was asked of it."""

    def __init__(self) -> None:
        self.terminated: list[str] = []

    def has_chat_key(self, key: str) -> bool:
        return key == CHAT_A

    def get_chat_session(self, key: str):
        return object() if key == CHAT_A else None

    def terminate_chat_session(self, key: str) -> None:
        self.terminated.append(key)

    async def cleanup_all(self) -> None:
        """Deliberately not recorded: it runs when the lifespan exits, long
        after the assertion, and recording it would make the case fail."""


# -- harness ----------------------------------------------------------------


def recorded_posture():
    """The deployment's narrowings, straight off the record."""
    control_context.invalidate_cache()
    record = control_context.read_record()
    return {} if record is None else dict(record.posture)


def read_record():
    control_context.invalidate_cache()
    return control_context.read_record()


def write_inflight_marker(root: Path, *, pid, target="standin"):
    """Plant one execution marker, as the python executor writes it."""
    directory = root / posture_store.STATE_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    path = (
        directory / f"{target_state.INFLIGHT_FILE_PREFIX}{pid}{target_state.INFLIGHT_FILE_SUFFIX}"
    )
    path.write_text(
        json.dumps(
            {
                "pid": pid,
                "target": target,
                "generation": 1,
                "session": "another-session",
                "surface": "python_executor",
                "kernel_id": None,
                "started_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    return path


@contextmanager
def only_alive(*pids):
    """Report exactly *pids* as running processes.

    One patch reaches every reader in the request: ``control_context`` owns the
    predicate, and ``target_state.is_process_alive`` — which the marker sweep
    calls — delegates to it at call time.
    """
    wanted = {int(p) for p in pids}

    def alive(pid):
        try:
            return int(pid) in wanted
        except (TypeError, ValueError):
            return False

    with patch.object(control_context, "is_process_alive", side_effect=alive):
        yield


#: The 400's sentence for a target this render does not configure, spelled
#: once in ``_unknown_target_message`` and pinned on both gesture routes.
UNKNOWN_TARGET_SENTENCE = "This deployment configures no control target by that name."


def post_posture(client, *, session_id=SESSION_A, target="standin", posture="sandbox"):
    body = {"target": target, "posture": posture}
    if session_id is not None:
        body["session_id"] = session_id
    return client.post("/api/terminal/posture", json=body)


# -- the refusal ladder -----------------------------------------------------


class TestTheGetTakesTheSameOptionalId:
    """``GET /api/terminal/posture`` follows the two POSTs, or they drift apart."""

    def test_the_roster_answers_with_no_session_id_at_all(self, client):
        """The Lab page's bar has no terminal session to name, and needs none."""
        response = client.get("/api/terminal/posture")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["session_id"] is None
        assert [row["target"] for row in body["targets"]] == ["live", "va", "standin"]

    def test_a_malformed_id_is_still_the_400_the_posts_give(self, client):
        response = client.get("/api/terminal/posture", params={"session_id": "nope"})
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "invalid_session_id"

    def test_the_roster_reports_the_recorded_narrowing(self, client):
        """The posture the GET shows is the one the POST just wrote."""
        assert post_posture(client, target="standin", posture="sandbox").status_code == 200
        rows = client.get("/api/terminal/posture").json()["targets"]
        standin = next(row for row in rows if row["target"] == "standin")
        assert standin["posture"] == "sandbox"
        assert standin["effective"] is False


class TestGrammar:
    @pytest.mark.parametrize(
        "bad_id",
        ["../../etc/passwd", "operator-deadbeef", "", "AAAAAAAA-1111-2222-3333-444444444444"],
    )
    def test_an_id_outside_the_closed_grammar_is_400(self, client, bad_id):
        resp = post_posture(client, session_id=bad_id)
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "invalid_session_id"

    def test_a_body_with_no_session_id_is_accepted(self, client, agent_data_root):
        """The posture is the deployment's, so the gesture needs no session."""
        resp = post_posture(client, session_id=None)
        assert resp.status_code == 200, resp.text
        assert resp.json()["session_id"] is None
        assert recorded_posture() == {"standin": "sandbox"}

    @pytest.mark.parametrize("bad", ["readonly", "SANDBOX", "", "readwrite", "true"])
    def test_only_the_two_named_postures_are_accepted(self, client, bad):
        assert post_posture(client, posture=bad).status_code == 422

    def test_a_body_missing_a_field_is_422(self, client):
        for body in (
            {"session_id": SESSION_A, "target": "standin"},
            {"session_id": SESSION_A, "posture": "sandbox"},
        ):
            assert client.post("/api/terminal/posture", json=body).status_code == 422

    def test_a_target_this_deployment_does_not_configure_is_400(self, client):
        resp = post_posture(client, target="banana")
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "unknown_target"
        assert resp.json()["detail"]["message"].startswith(UNKNOWN_TARGET_SENTENCE + " It has: ")

    def test_all_plus_writes_is_400(self, client):
        """Widening is per target, always.

        ``[ Sandbox everything ]`` is one gesture because narrowing everything
        is unambiguous; there is no matching "arm everything", because each
        target's ceiling is its own and an operator arming three machines at
        once could not have meant all three.
        """
        resp = post_posture(client, target="all", posture="writes")
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "writes_requires_one_target"
        assert recorded_posture() == {}

    @pytest.mark.parametrize(
        ("target", "posture"),
        [("all", "sandbox"), ("standin", "sandbox"), ("standin", "writes")],
    )
    def test_an_unreadable_render_configures_no_target(self, client, target, posture):
        """No readable config, no vocabulary — and therefore nothing to toggle.

        ``configured_targets`` is what every row, probe and refusal here
        enumerates; a server that cannot read its own render does not know
        which machines exist. ``all`` in particular must not fall through: over
        an empty vocabulary it would CLEAR the narrowings rather than add one,
        which is the one direction this field must never move by accident.
        """
        client.app.state.config_path = None
        resp = post_posture(client, target=target, posture=posture)
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "unknown_target"
        assert resp.json()["detail"]["message"] == UNKNOWN_TARGET_SENTENCE + " It has: none."


class TestOwnership:
    """Who may write the record at all, before any judgement about the posture."""

    def test_a_terminal_with_no_owner_is_503(self, client):
        del client.app.state.control_context_owner
        resp = post_posture(client)
        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "store_unavailable"
        assert recorded_posture() == {}

    def test_a_follower_is_409_naming_the_owners_pid_and_port(self, client):
        client.app.state.control_context_follows = owner(
            pid=OTHER_TERMINAL_PID, port=OTHER_TERMINAL_PORT
        )
        resp = post_posture(client)
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["error"] == "context_owned_elsewhere"
        assert str(OTHER_TERMINAL_PID) in detail["message"]
        assert str(OTHER_TERMINAL_PORT) in detail["message"]
        assert recorded_posture() == {}

    def test_a_takeover_between_the_hint_and_the_write_is_the_same_409(
        self, client, agent_data_root
    ):
        """``app.state`` can be a tick stale; the primitive is the guard."""
        write_control_context(
            agent_data_root,
            target="live",
            generation=1,
            owned_by=owner(pid=OTHER_TERMINAL_PID, port=OTHER_TERMINAL_PORT),
        )
        resp = post_posture(client)
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "context_owned_elsewhere"
        assert recorded_posture() == {}

    def test_a_failing_write_is_503_and_narrows_nothing(self, client):
        with patch.object(
            control_context_owner, "write_record", side_effect=OSError("read-only volume")
        ):
            resp = post_posture(client)
        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "store_write_failed"
        assert recorded_posture() == {}


class TestCeiling:
    """403 ``writes_disabled``, per target, naming that target's own key."""

    def test_widening_an_unarmed_target_is_403_naming_its_key(self, client, tmp_path):
        resp = post_posture(client, target="standin", posture="writes")
        assert resp.status_code == 403
        detail = resp.json()["detail"]
        assert detail["error"] == "writes_disabled"
        assert "control_system.connector.live_standin.writes_enabled" in detail["message"]

    def test_a_mixed_render_arms_the_va_and_refuses_the_live_machine(self, client, tmp_path):
        """The union is true here and says nothing about where the operator is.

        This is the whole reason the ceiling is read per target: a deployment
        that arms only its simulator would otherwise offer a writes toggle on
        the facility's own machine, and every write through it would be refused
        one layer down.
        """
        client.app.state.config_path = write_config(
            tmp_path, control_system_section(va_writes=True)
        )
        armed = post_posture(client, target="va", posture="writes")
        unarmed = post_posture(client, target="live", posture="writes")
        assert armed.status_code == 200
        assert unarmed.status_code == 403
        assert (
            "control_system.connector.epics.writes_enabled" in unarmed.json()["detail"]["message"]
        )

    def test_a_va_plus_standin_render_arms_its_standin(self, client, tmp_path):
        """Two configured targets are switch-capable with no live machine at all.

        ``session_posture`` answers per target on any switch-capable render, so
        a deployment rehearsing on its stand-in beside the simulator gets the
        stand-in's own ceiling — not a 403 for want of an ``epics`` block it
        never claimed to have.
        """
        section = {
            "type": "virtual_accelerator",
            "writes_enabled": False,
            "connector": {
                "virtual_accelerator": {
                    "simulation_file": "data/sim.json",
                    "gateways": _gateways(5064),
                    "writes_enabled": True,
                },
                "live_standin": {
                    "gateways": _gateways(STANDIN_PORT),
                    "writes_enabled": True,
                },
            },
        }
        client.app.state.config_path = write_config(tmp_path, section)
        assert post_posture(client, target="standin", posture="writes").status_code == 200

    def test_narrowing_needs_no_ceiling(self, client):
        """A target nothing arms can still be narrowed; narrowing grants nothing."""
        assert post_posture(client, target="live", posture="sandbox").status_code == 200

    def test_all_sandbox_ignores_the_ceiling(self, client):
        assert post_posture(client, target="all", posture="sandbox").status_code == 200


class TestSelectedRoleMissing:
    """409 when narrowing would leave the target with no gateway to select."""

    def _write_access_only(self, client, tmp_path, *, name="config.yml", **kwargs):
        row = {"address": "localhost", "port": STANDIN_PORT, "use_name_server": True}
        client.app.state.config_path = write_config(
            tmp_path,
            control_system_section(standin_gateways={"write_access": row}, **kwargs),
            name=name,
        )

    def test_a_write_access_only_target_cannot_be_narrowed(self, client, tmp_path):
        self._write_access_only(client, tmp_path)
        resp = post_posture(client, target="standin", posture="sandbox")
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["error"] == "selected_role_missing"
        assert "control_system.connector.live_standin.gateways.read_only" in detail["message"]

    def test_the_other_targets_are_unaffected(self, client, tmp_path):
        self._write_access_only(client, tmp_path)
        assert post_posture(client, target="va", posture="sandbox").status_code == 200

    def test_sandbox_everything_narrows_the_rest_and_reports_what_it_skipped(
        self, client, tmp_path
    ):
        """ "Everything" means everything it can, and says so — not nothing.

        Refusing the whole gesture because one target is write_access-only
        would leave "Sandbox everything" doing nothing at all on that
        deployment, while the popover offers the button. The other machines
        are narrowed, and the one that stayed writable is named with its
        reason so the operator is never told a narrowing happened that did not.
        """
        self._write_access_only(client, tmp_path)
        resp = post_posture(client, target="all", posture="sandbox")

        assert resp.status_code == 200
        body = resp.json()
        assert body["entry"] == {"live": "sandbox", "va": "sandbox"}
        assert [row["target"] for row in body["skipped"]] == ["standin"]
        assert body["skipped"][0]["reason"] == "selected_role_missing"
        assert "gateways.read_only" in body["skipped"][0]["detail"]
        assert recorded_posture() == {"live": "sandbox", "va": "sandbox"}

    def test_an_already_sandboxed_target_does_not_veto_a_later_gesture(self, client, tmp_path):
        """``narrowing_refusal`` is record-blind, so only CHANGING targets are asked.

        A target narrowed while the deployment could still derive a read_only
        gateway keeps its narrowing when that gateway later disappears from the
        render. Asking about it again would have it refuse on behalf of a
        change nobody requested — freezing every later toggle.
        """
        assert post_posture(client, target="standin", posture="sandbox").status_code == 200

        # The stand-in's read_only gateway goes away under the running session.
        self._write_access_only(client, tmp_path, name="config2.yml")
        resp = post_posture(client, target="all", posture="sandbox")

        assert resp.status_code == 200
        assert resp.json()["skipped"] == []
        assert recorded_posture() == {
            "standin": "sandbox",
            "live": "sandbox",
            "va": "sandbox",
        }

    def test_a_clean_render_skips_nothing(self, client):
        resp = post_posture(client, target="all", posture="sandbox")
        assert resp.status_code == 200
        assert resp.json()["skipped"] == []

    def test_widening_is_not_checked(self, client, tmp_path):
        """The question is what NARROWING would cost; widening does not narrow."""
        row = {"address": "localhost", "port": STANDIN_PORT, "use_name_server": True}
        client.app.state.config_path = write_config(
            tmp_path,
            control_system_section(standin_writes=True, standin_gateways={"write_access": row}),
        )
        assert post_posture(client, target="standin", posture="writes").status_code == 200


class TestExecutionInFlight:
    """A run that started narrow is never widened out from under itself."""

    def test_widening_under_a_live_marker_is_409(self, client, tmp_path, agent_data_root):
        client.app.state.config_path = write_config(
            tmp_path, control_system_section(va_writes=True)
        )
        write_inflight_marker(agent_data_root, pid=4242, target="standin")
        with only_alive(4242):
            resp = post_posture(client, target="va", posture="writes")
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["error"] == "execution_in_flight"
        # The switch gate's own words, so the popover and the agent read alike.
        assert "in flight on target" in detail["message"].lower()
        assert "standin" in detail["message"]
        assert "wait" in detail["message"].lower()

    def test_narrowing_under_a_live_marker_still_lands(self, client, agent_data_root):
        """Narrowing is always safe; it is the gesture an operator needs most."""
        write_inflight_marker(agent_data_root, pid=4242, target="standin")
        with only_alive(4242):
            resp = post_posture(client, target="standin", posture="sandbox")
        assert resp.status_code == 200
        assert recorded_posture() == {"standin": "sandbox"}

    def test_a_marker_whose_writer_is_gone_does_not_block(self, client, tmp_path, agent_data_root):
        """A killed executor's residue is swept, not treated as a running run."""
        client.app.state.config_path = write_config(
            tmp_path, control_system_section(va_writes=True)
        )
        marker = write_inflight_marker(agent_data_root, pid=DEAD_PID, target="standin")
        with only_alive():
            resp = post_posture(client, target="va", posture="writes")
        assert resp.status_code == 200
        assert not marker.exists()

    def test_any_targets_marker_blocks_any_widening(self, client, tmp_path, agent_data_root):
        """ANY live marker, not just one on the target being widened.

        The marker says a run is going; the posture the run launched under is
        pinned into it, and widening any target while one is in flight is the
        surprise the refusal exists to prevent.
        """
        client.app.state.config_path = write_config(
            tmp_path, control_system_section(va_writes=True)
        )
        write_inflight_marker(agent_data_root, pid=4242, target="va")
        with only_alive(4242):
            assert post_posture(client, target="va", posture="writes").status_code == 409


# -- the accepted gesture ---------------------------------------------------


class TestAcceptedPosture:
    def test_a_narrowing_lands_in_the_record(self, client):
        resp = post_posture(client, target="standin", posture="sandbox")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == SESSION_A
        assert body["target"] == "standin"
        assert body["posture"] == "sandbox"
        assert body["entry"] == {"standin": "sandbox"}
        assert recorded_posture() == {"standin": "sandbox"}

    def test_the_record_is_co_sited_with_the_control_target_directory(self, client):
        """One directory for the record and the fleet's reports, by one rule."""
        post_posture(client)
        assert control_context.record_path().parent == target_state.state_dir()

    def test_the_toggle_moves_neither_target_nor_generation(self, client):
        before = read_record()
        post_posture(client, target="standin", posture="sandbox")
        after = read_record()
        assert (after.target, after.generation) == (before.target, before.generation)
        assert after.last_switch == before.last_switch

    def test_targets_narrow_independently(self, client):
        post_posture(client, target="standin", posture="sandbox")
        resp = post_posture(client, target="va", posture="sandbox")
        assert resp.json()["entry"] == {"standin": "sandbox", "va": "sandbox"}
        assert recorded_posture() == {"standin": "sandbox", "va": "sandbox"}

    def test_widening_removes_only_that_target(self, client, tmp_path):
        client.app.state.config_path = write_config(
            tmp_path, control_system_section(va_writes=True)
        )
        post_posture(client, target="standin", posture="sandbox")
        post_posture(client, target="va", posture="sandbox")
        resp = post_posture(client, target="va", posture="writes")
        assert resp.json()["entry"] == {"standin": "sandbox"}
        assert recorded_posture() == {"standin": "sandbox"}

    def test_the_last_widening_clears_the_field(self, client, tmp_path):
        """Absence is how this field spells ``writes``; a stored value is not."""
        client.app.state.config_path = write_config(
            tmp_path, control_system_section(standin_writes=True)
        )
        post_posture(client, target="standin", posture="sandbox")
        resp = post_posture(client, target="standin", posture="writes")
        assert resp.json()["entry"] == {}
        assert recorded_posture() == {}

    def test_sandbox_everything_narrows_every_configured_target(self, client):
        resp = post_posture(client, target="all", posture="sandbox")
        assert resp.status_code == 200
        assert resp.json()["entry"] == {
            "live": "sandbox",
            "va": "sandbox",
            "standin": "sandbox",
        }

    def test_two_sessions_toggle_one_deployment(self, client):
        """There is one posture, so the second gesture adds to the first."""
        post_posture(client, session_id=SESSION_A, target="standin")
        post_posture(client, session_id=SESSION_B, target="va")
        assert recorded_posture() == {"standin": "sandbox", "va": "sandbox"}

    def test_a_repeated_narrowing_writes_nothing_new(self, client):
        """A gesture asking for what is already stored moves no signature.

        Readers watch the record's ``(mtime_ns, size, ino)``; rewriting it for
        a change of nothing would wake every one of them for nothing.
        """
        post_posture(client, target="standin", posture="sandbox")
        before = control_context.file_signature(control_context.record_path())
        assert post_posture(client, target="standin", posture="sandbox").status_code == 200
        assert control_context.file_signature(control_context.record_path()) == before


class TestNoTermination:
    """The POST never respawns the session — the whole point of the feature."""

    def test_the_live_child_survives_a_narrowing(self, client):
        registry = client.app.state.pty_registry
        registry.get_or_create_session(SESSION_A, "echo")
        assert registry.get_session(SESSION_A) is not None

        assert post_posture(client, target="standin").status_code == 200

        assert registry.get_session(SESSION_A) is not None

    def test_the_route_never_reaches_for_the_pool_teardown(self, client):
        """Pinned on the registry, not on a helper name.

        The old route applied a posture by terminating the child so the next
        attach respawned it under a fresh environment. Asserting that some
        ``_terminate_for_respawn`` helper is not called would stop meaning
        anything the moment that helper is deleted. The durable statement is
        that this POST touches none of the ways the pool lets go of a live PTY
        session.
        """
        registry = client.app.state.pty_registry
        registry.get_or_create_session(SESSION_A, "echo")
        with (
            patch.object(registry, "detach_session") as detach,
            patch.object(registry, "terminate_session") as terminate,
            patch.object(registry, "terminate_session_if_owner") as terminate_if_owner,
        ):
            assert post_posture(client, target="standin").status_code == 200
        detach.assert_not_called()
        terminate.assert_not_called()
        terminate_if_owner.assert_not_called()

    def test_the_chat_child_survives_a_narrowing(self, client):
        """The chat-key analogue, and the half nothing else covers."""
        recorder = _RecordingChatPool()
        client.app.state.operator_registry = recorder

        resp = post_posture(client, session_id=CHAT_A, target="standin")

        assert resp.status_code == 200
        assert recorded_posture() == {"standin": "sandbox"}
        assert recorder.terminated == []


class TestPersistence:
    """The narrowing outlives the process that recorded it."""

    def test_a_narrowing_reloads_in_a_fresh_app(self, make_client):
        """A container recreation must not silently lift a narrowing."""
        with make_client() as first:
            assert post_posture(first, target="standin").status_code == 200

        with make_client() as second:
            assert websocket_routes._recorded_posture() == {"standin": "sandbox"}
            assert post_posture(second, target="va").status_code == 200
        assert recorded_posture() == {"standin": "sandbox", "va": "sandbox"}

    def test_a_corrupt_record_reads_as_no_narrowing(self, make_client, agent_data_root):
        control_context.record_path_under(agent_data_root).write_text("{not json", encoding="utf-8")
        control_context.invalidate_cache()
        with make_client():
            assert websocket_routes._recorded_posture() == {}

    def test_an_unknown_posture_value_is_dropped_on_read(self, make_client, agent_data_root):
        """``parse_posture`` runs the grammar: only narrowings survive."""
        write_control_context(
            agent_data_root,
            target="live",
            generation=1,
            posture={"standin": "readonly", "va": "sandbox"},
        )
        with make_client():
            assert websocket_routes._recorded_posture() == {"va": "sandbox"}


# -- audit ------------------------------------------------------------------


class TestAudit:
    def test_an_accepted_toggle_files_exactly_one_record(self, client, ledger):
        assert post_posture(client, target="standin").status_code == 200
        assert len(ledger) == 1
        record = ledger[0]
        assert record["subject"] == "session_posture_set"
        assert record["decision"] == "allowed"
        assert record["session"] == SESSION_A
        assert "standin" in record["detail"]
        assert "sandbox" in record["detail"]

    def test_a_refusal_files_exactly_one_record_too(self, client, ledger):
        assert post_posture(client, target="standin", posture="writes").status_code == 403
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "refused"
        assert ledger[0]["reason"] == "writes_disabled"

    def test_a_malformed_session_id_still_leaves_exactly_one_record(self, client, ledger):
        """The one refusal the route does NOT file itself is still filed once.

        ``_require_session_uuid`` runs before there is a legitimate key to put
        in the envelope's ``session`` field, so the route does not record it and
        ``HttpAuditMiddleware`` files its own ``route_refused`` line instead.
        The count is what matters: a refused request leaves one line, never two
        and never none, whichever layer wrote it.
        """
        assert post_posture(client, session_id="../../etc/passwd").status_code == 400
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "refused"

    def test_the_record_joins_on_the_session_key(self, client, ledger):
        """A toggle is filed under the key the session is pooled on."""
        assert post_posture(client, target="standin").status_code == 200
        assert ledger[0]["session"] == SESSION_A

    def test_a_session_less_gesture_is_filed_on_the_lab_surface(self, client, ledger):
        assert post_posture(client, session_id=None, target="standin").status_code == 200
        assert ledger[0]["session"] is None
        assert ledger[0]["surface"] == websocket_routes.LAB_MUTATION_SURFACE
