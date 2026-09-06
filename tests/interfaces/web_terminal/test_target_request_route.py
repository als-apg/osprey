"""Tests for ``POST /api/terminal/target`` — the operator's switch gesture.

A deployment has ONE control context: a record naming the target, the
generation the fleet coordinates on, and the terminus of the last switch. The
web terminal owns that record while it runs, so this route does not file
desired state for somebody else to apply — it takes the record, runs the switch
gate against it, and writes the answer, both halves inside one mutation.

Four consequences the tests below pin:

* **The verdict is taken against the record the answer is written into.** The
  facts hop gathers the render, the fleet's reports and the execution markers;
  the gate runs inside the mutation, so no verdict can be about a deployment
  state that has since moved.
* **A refusal is a record write that moves nothing else.** ``last_switch``
  carries the gate's reason and sentence; target and generation stay where they
  were, and no generation is minted for a switch that did not happen.
* **A terminal that does not own the context refuses.** No owner is
  ``503 store_unavailable``; following another terminal is
  ``409 context_owned_elsewhere`` naming its pid and port — as a fast rung off
  ``app.state`` and again, for the race, from the mutation primitive itself.
* **One audit record per POST**, carrying the *spawn* session key — or no
  session at all, on the ``jupyter_lab`` surface, for the Lab bar's gesture.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

from osprey.audit import writer as audit_writer
from osprey.interfaces.web_terminal import control_context_owner
from osprey.interfaces.web_terminal import jupyter_sidecar as jupyter_sidecar_module
from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey.mcp_server.control_system import target_eligibility, target_state
from osprey_connectors import control_context, posture_store
from tests._control_context_fixtures import owner, write_control_context, write_server_report

SESSION_A = "aaaaaaaa-1111-2222-3333-444444444444"
SESSION_B = "bbbbbbbb-1111-2222-3333-444444444444"

#: The PTY process the terminal card is attached to.
PTY_PID = 7000

#: A controls server that is running, and one that is not.
SERVER_PID = 5150
DEAD_SERVER_PID = 5151

#: The terminal that owns the context in the follower cases. ``os.getppid()``
#: is a real, live process, so the owner task follows it for the same reason
#: the route does rather than claiming over a pid nothing answers to.
OTHER_TERMINAL_PID = os.getppid()
OTHER_TERMINAL_PORT = 8090


# -- fixtures ---------------------------------------------------------------


@pytest.fixture
def agent_data_root(tmp_path, monkeypatch):
    """Point every resolver at one throwaway agent-data root.

    ``OSPREY_AGENT_DATA_ROOT`` is the single stamp the record reader and
    ``posture_store`` both prefer, which is exactly why the feature stamps it
    into every session child. Patching ``resolve_shared_data_root`` instead
    would redirect only one of the two — ``posture_store`` binds the resolver
    at import — and the other half would write into the repository's own
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
def config_path(tmp_path):
    """A render whose ``va`` target is eligible from a ``live`` baseline.

    Both connector blocks carry gateways and a probe channel, so the gate gets
    past eligibility and the tests below can reach the reachability rung — the
    one the fleet's reports decide. ``va`` is the wanted target throughout
    because it carries none of FR-8's posture gates, which have their own
    suite.
    """
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(render()), encoding="utf-8")
    return path


def render() -> dict:
    """The rendered config the suite is built on."""
    return {
        "control_system": {
            "type": "epics",
            "writes_enabled": False,
            "connector": {
                "epics": {
                    "timeout": 5.0,
                    "probe_channel": "LIVE:PROBE:CHANNEL",
                    "gateways": {
                        "read_only": {
                            "address": "gw.example.org",
                            "port": 5064,
                            "use_name_server": False,
                        },
                        "write_access": {
                            "address": "gw.example.org",
                            "port": 5084,
                            "use_name_server": False,
                        },
                    },
                },
                "virtual_accelerator": {
                    "timeout": 5.0,
                    "probe_channel": "VA:PROBE:CHANNEL",
                    "gateways": {
                        "read_only": {
                            "address": "localhost",
                            "port": 5074,
                            "use_name_server": True,
                        },
                        "write_access": {
                            "address": "localhost",
                            "port": 5074,
                            "use_name_server": True,
                        },
                    },
                },
            },
            "target_switch": {target_eligibility.ACK_LEAF: "gw.example.org"},
        },
        "archiver": {"type": "epics_archiver"},
    }


@pytest.fixture
def client(agent_data_root, workspace_dir, config_path):
    """A terminal that owns the control context, with the owner task quiesced.

    The record is written **before** the app starts, so the lifespan claim
    merges into it rather than minting a fresh one from ``load_osprey_config``,
    which this process has no workspace for. ``start`` is stubbed out for the
    same reason a test never wants a second writer: every tick this suite needs
    has already happened by the time the client is handed over, and a loop
    ticking underneath the assertions would be a race, not coverage.
    """
    write_control_context(agent_data_root, target="live", generation=1)
    with patch.object(control_context_owner.ControlContextOwnerTask, "start", lambda self: None):
        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ):
            app = create_app(shell_command="echo")
            with TestClient(app) as test_client:
                test_client.app.state.config_path = config_path
                yield test_client


@pytest.fixture
def ledger():
    """Capture every audit record the request would have written.

    Patches ``osprey.audit.writer.record``, which BOTH recorders resolve at
    call time — the route's ``record_and_mark`` and
    ``HttpAuditMiddleware._emit_audit_record`` — so the count is the true
    number of ledger lines this POST produces, not just the route's own.
    """
    records: list[dict] = []

    def _record(**fields):
        records.append(fields)
        return Path("/dev/null/ledger.jsonl")

    with patch.object(audit_writer, "record", side_effect=_record):
        yield records


# -- harness ----------------------------------------------------------------


@contextmanager
def attached_pty(client, session_id, pid=PTY_PID):
    """Make the registry report a PTY with *pid* for *session_id*."""
    registry = client.app.state.pty_registry
    pty = SimpleNamespace(pid=pid, is_alive=True, exit_code=None)
    with patch.object(
        registry,
        "get_session",
        side_effect=lambda sid: pty if sid == session_id else None,
    ):
        yield


@contextmanager
def only_alive(*pids):
    """Report exactly *pids* as running processes.

    One patch reaches every reader in the request: ``control_context`` owns the
    predicate, and ``target_state.is_process_alive`` — which the route hands to
    the report filter and the marker sweep — delegates to it at call time.
    """
    wanted = {int(p) for p in pids}

    def alive(pid):
        try:
            return int(pid) in wanted
        except (TypeError, ValueError):
            return False

    with patch.object(control_context, "is_process_alive", side_effect=alive):
        yield


def state_dir() -> Path:
    return control_context.state_dir()


def write_inflight_marker(
    root: Path, *, pid, target="live", session="s", surface="cli", kernel_id=None
):
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
                "session": session,
                "surface": surface,
                "kernel_id": kernel_id,
                "started_at": "2026-08-30T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return path


@contextmanager
def jupyter_sidecar(client, *, url="http://127.0.0.1:9999/panel/jupyter", notebook=None):
    """A Jupyter panel whose sidecar answers *notebook* for any kernel id.

    ``notebook=None`` is a sidecar that cannot name the kernel — no session row,
    a 403, a body in an unexpected shape — which is the same answer as no
    sidecar at all as far as the refusal is concerned.
    """
    client.app.state.jupyter_server_url = url
    client.app.state.panel_auth_headers = {"jupyter": {"authorization": "Bearer t"}}
    with patch.object(
        jupyter_sidecar_module, "kernel_notebook_path", side_effect=lambda *_: notebook
    ):
        yield


def reachable(target="va", state=None):
    """A report's reachability block saying *target* answered its last probe."""
    return {
        "targets": {
            target: {
                "read_only": {
                    "state": state or target_eligibility.REACHABILITY_REACHED,
                    "probed_at": datetime.now(UTC).isoformat(),
                }
            }
        }
    }


def applying_block(generation=1, *, expires_in_s=30.0):
    """A server's ``last_switch`` while it is mid-swap, with its own bound."""
    return {
        "generation": generation,
        "status": control_context.REPORT_APPLYING,
        "at": datetime.now(UTC).isoformat(),
        "expires_at": (datetime.now(UTC) + timedelta(seconds=expires_in_s)).isoformat(),
    }


def read_record():
    control_context.invalidate_cache()
    return control_context.read_record()


#: The 400's sentence for a target this render does not configure, spelled
#: once in ``_unknown_target_message`` and pinned on both gesture routes.
UNKNOWN_TARGET_SENTENCE = "This deployment configures no control target by that name."


def post_target(client, session_id=SESSION_A, target="va"):
    body = {"target": target}
    if session_id is not None:
        body["session_id"] = session_id
    return client.post("/api/terminal/target", json=body)


# -- the refusal ladder -----------------------------------------------------


class TestGrammar:
    """400 before anything is opened: the id and the target name are identifiers."""

    @pytest.mark.parametrize(
        "bad_id",
        ["", "not-a-uuid", "operator-abc12345", SESSION_A.upper(), SESSION_A + "x"],
    )
    def test_a_session_id_outside_the_closed_grammar_is_400(self, client, bad_id):
        response = post_target(client, session_id=bad_id)
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "invalid_session_id"

    def test_a_body_with_no_session_id_is_accepted(self, client):
        """The gesture needs no session: the control context is the deployment's."""
        response = post_target(client, session_id=None)
        assert response.status_code == 202, response.text
        assert response.json()["session_id"] is None

    def test_a_target_this_deployment_did_not_configure_is_400(self, client):
        response = post_target(client, target="standin")
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["error"] == "unknown_target"
        assert UNKNOWN_TARGET_SENTENCE in detail["message"]
        assert "live, va" in detail["message"]

    def test_a_body_missing_a_field_is_422(self, client):
        response = client.post("/api/terminal/target", json={"session_id": SESSION_A})
        assert response.status_code == 422

    def test_a_refused_grammar_moves_no_record(self, client):
        before = read_record()
        post_target(client, session_id="nope")
        assert read_record() == before


class TestOwnership:
    """Who may write the record at all, before any judgement about the switch."""

    def test_a_terminal_with_no_owner_is_503(self, client):
        del client.app.state.control_context_owner
        response = post_target(client)
        assert response.status_code == 503
        assert response.json()["detail"]["error"] == "store_unavailable"

    def test_a_follower_is_409_naming_the_owners_pid_and_port(self, client):
        client.app.state.control_context_follows = owner(
            pid=OTHER_TERMINAL_PID, port=OTHER_TERMINAL_PORT
        )
        response = post_target(client)
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["error"] == "context_owned_elsewhere"
        assert str(OTHER_TERMINAL_PID) in detail["message"]
        assert str(OTHER_TERMINAL_PORT) in detail["message"]

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
        response = post_target(client)
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["error"] == "context_owned_elsewhere"
        assert str(OTHER_TERMINAL_PID) in detail["message"]

    def test_a_failing_write_is_503_and_moves_nothing(self, client):
        before = read_record()
        with patch.object(
            control_context_owner, "write_record", side_effect=OSError("read-only volume")
        ):
            response = post_target(client)
        assert response.status_code == 503
        assert response.json()["detail"]["error"] == "store_write_failed"
        assert read_record() == before


class TestSwitchInProgress:
    """A fleet mid-swap owns ``last_switch``; nothing else may write it."""

    def test_a_live_server_applying_this_generation_is_409(self, client, agent_data_root):
        write_server_report(agent_data_root, SERVER_PID, last_switch=applying_block(1))
        with only_alive(SERVER_PID, os.getpid()):
            response = post_target(client)
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["error"] == "switch_in_progress"
        assert str(SERVER_PID) in detail["message"]

    def test_the_refusal_writes_nothing(self, client, agent_data_root):
        write_server_report(agent_data_root, SERVER_PID, last_switch=applying_block(1))
        before = read_record()
        with only_alive(SERVER_PID, os.getpid()):
            post_target(client)
        assert read_record() == before

    def test_a_swap_past_its_own_bound_no_longer_blocks(self, client, agent_data_root):
        write_server_report(
            agent_data_root,
            SERVER_PID,
            reachability=reachable(),
            last_switch=applying_block(1, expires_in_s=-1.0),
        )
        with only_alive(SERVER_PID, os.getpid()):
            response = post_target(client)
        assert response.status_code == 202, response.text

    def test_a_dead_servers_applying_block_does_not_block(self, client, agent_data_root):
        write_server_report(agent_data_root, DEAD_SERVER_PID, last_switch=applying_block(1))
        with only_alive(os.getpid()):
            response = post_target(client)
        assert response.status_code == 202, response.text


class TestTheGate:
    """The switch gate's own refusals, in the gate's own words."""

    def test_a_read_only_run_is_refused(self, client, monkeypatch):
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        response = post_target(client)
        assert response.status_code == 409
        assert response.json()["detail"]["error"] == target_eligibility.REASON_READONLY_RUN

    def test_an_execution_in_flight_names_the_busy_client(self, client, agent_data_root):
        write_inflight_marker(agent_data_root, pid=9100, session="other-session")
        with only_alive(9100, os.getpid()):
            response = post_target(client)
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["error"] == target_eligibility.REASON_EXECUTION_IN_FLIGHT
        # One fact and one action: who holds the target, and what to do. The
        # gate's headline says neither without repeating both, so it is dropped.
        assert "belongs to session other-se" in detail["message"]
        assert "Wait for it to finish, or stop it, then switch again." in detail["message"]
        assert "wait or stop it" not in detail["message"]

    def test_a_busy_notebook_kernel_is_named_by_its_notebook(self, client, agent_data_root):
        """The one refusal only this surface can sharpen.

        A kernel's session key is ``kernel:<id>``, which names no window an
        operator can go and look at. The terminal owns the sidecar, so it hands
        the gate a resolver and the refusal says which notebook to interrupt.
        """
        write_inflight_marker(
            agent_data_root,
            pid=9200,
            session="kernel:abcdef0123456789",
            surface=target_eligibility.SURFACE_NOTEBOOK_KERNEL,
            kernel_id="abcdef0123456789",
        )
        with only_alive(9200, os.getpid()), jupyter_sidecar(client, notebook="studies/orbit.ipynb"):
            response = post_target(client)
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["error"] == target_eligibility.REASON_EXECUTION_IN_FLIGHT
        assert "notebook studies/orbit.ipynb" in detail["message"]
        assert "Interrupt that kernel to proceed." in detail["message"]
        # The kernel remedy stands alone: the gate's headline would have put
        # the vaguer "wait or stop it" beside it.
        assert "wait or stop it" not in detail["message"]

    def test_a_kernel_the_sidecar_cannot_name_falls_back_to_its_id(self, client, agent_data_root):
        """A resolver that answers nothing is not a refusal that fails."""
        write_inflight_marker(
            agent_data_root,
            pid=9200,
            session="kernel:abcdef0123456789",
            surface=target_eligibility.SURFACE_NOTEBOOK_KERNEL,
            kernel_id="abcdef0123456789",
        )
        with only_alive(9200, os.getpid()), jupyter_sidecar(client, notebook=None):
            response = post_target(client)
        assert response.status_code == 409
        assert "notebook kernel abcdef01" in response.json()["detail"]["message"]

    def test_no_jupyter_panel_asks_no_sidecar(self, client, agent_data_root):
        """A panel that is off or retracted leaves the URL unset; nothing is called."""
        write_inflight_marker(
            agent_data_root,
            pid=9200,
            session="kernel:abcdef0123456789",
            surface=target_eligibility.SURFACE_NOTEBOOK_KERNEL,
            kernel_id="abcdef0123456789",
        )
        with (
            only_alive(9200, os.getpid()),
            patch.object(jupyter_sidecar_module, "kernel_notebook_path") as query,
        ):
            response = post_target(client)
        query.assert_not_called()
        assert response.status_code == 409
        assert "notebook kernel abcdef01" in response.json()["detail"]["message"]

    def test_a_sidecar_that_raises_degrades_to_the_id(self, client, agent_data_root):
        """Naming the busy client must never turn a refusal into a stack trace."""
        write_inflight_marker(
            agent_data_root,
            pid=9200,
            session="kernel:abcdef0123456789",
            surface=target_eligibility.SURFACE_NOTEBOOK_KERNEL,
            kernel_id="abcdef0123456789",
        )
        client.app.state.jupyter_server_url = "http://127.0.0.1:9999/panel/jupyter"
        client.app.state.panel_auth_headers = {"jupyter": {}}
        with (
            only_alive(9200, os.getpid()),
            patch.object(
                jupyter_sidecar_module,
                "kernel_notebook_path",
                side_effect=RuntimeError("sidecar exploded"),
            ),
        ):
            response = post_target(client)
        assert response.status_code == 409
        assert "notebook kernel abcdef01" in response.json()["detail"]["message"]

    def test_the_gate_runs_no_network_call_under_the_record_lock(self, client, agent_data_root):
        """The sidecar is asked in the facts hop, never inside the mutation.

        A two-second sidecar timeout taken under the record lock would stall
        the owner task's tick and every other write behind it, so the lookup
        the gate is handed must already be resolved. Pinned by asserting the
        query happens before the mutation starts, not during it.
        """
        write_inflight_marker(
            agent_data_root,
            pid=9200,
            session="kernel:abcdef0123456789",
            surface=target_eligibility.SURFACE_NOTEBOOK_KERNEL,
            kernel_id="abcdef0123456789",
        )
        order: list[str] = []
        real_apply = control_context_owner.ControlContextOwner._apply

        def tracking_apply(self, fn, verify_owner):
            order.append("mutation")
            return real_apply(self, fn, verify_owner)

        with (
            only_alive(9200, os.getpid()),
            jupyter_sidecar(client, notebook="studies/orbit.ipynb"),
            patch.object(
                jupyter_sidecar_module,
                "kernel_notebook_path",
                side_effect=lambda *_: order.append("sidecar") or "studies/orbit.ipynb",
            ),
            patch.object(control_context_owner.ControlContextOwner, "_apply", tracking_apply),
        ):
            response = post_target(client)
        assert response.status_code == 409
        assert order == ["sidecar", "mutation"]

    def test_an_ineligible_target_is_not_told_to_read_the_roster(self, client, tmp_path):
        """The popover IS the roster; "ask for the target roster" is agent copy."""
        render_without_va_gateways = render()
        render_without_va_gateways["control_system"]["connector"]["virtual_accelerator"].pop(
            "gateways"
        )
        path = tmp_path / "no-va-gateways.yml"
        path.write_text(yaml.safe_dump(render_without_va_gateways), encoding="utf-8")
        client.app.state.config_path = path

        response = post_target(client)
        assert response.status_code == 409
        message = response.json()["detail"]["message"]
        assert "Ask for the target roster" not in message
        assert "gateways" in message

    def test_a_live_server_that_has_published_no_reachability_refuses(
        self, client, agent_data_root
    ):
        write_server_report(agent_data_root, SERVER_PID)
        with only_alive(SERVER_PID, os.getpid()):
            response = post_target(client)
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["error"] == target_eligibility.REASON_REACHABILITY_UNKNOWN
        assert str(SERVER_PID) in detail["message"]

    def test_a_target_the_fleet_reports_down_refuses(self, client, agent_data_root):
        write_server_report(
            agent_data_root,
            SERVER_PID,
            reachability={
                "targets": {
                    "va": {
                        "read_only": {
                            "state": target_eligibility.REACHABILITY_DOWN,
                            "probed_at": datetime.now(UTC).isoformat(),
                        }
                    }
                }
            },
        )
        with only_alive(SERVER_PID, os.getpid()):
            response = post_target(client)
        assert response.status_code == 409
        assert response.json()["detail"]["error"] == target_eligibility.REASON_TARGET_UNREACHABLE

    def test_zero_live_servers_allows_the_switch(self, client):
        assert post_target(client).status_code == 202

    def test_a_refusal_is_a_record_write_that_moves_nothing_else(self, client, monkeypatch):
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        post_target(client)
        record = read_record()
        assert record.target == "live"
        assert record.generation == 1
        assert record.last_switch["status"] == control_context.SWITCH_REFUSED
        assert record.last_switch["reason"] == target_eligibility.REASON_READONLY_RUN
        assert record.last_switch["generation"] is None


class TestApplied:
    """What the record carries once a switch is granted."""

    def test_the_target_and_the_generation_move_together(self, client):
        response = post_target(client)
        assert response.status_code == 202, response.text
        record = read_record()
        assert record.target == "va"
        assert record.generation == 2
        assert response.json()["generation"] == 2

    def test_the_terminus_names_the_request(self, client):
        request_id = post_target(client).json()["request_id"]
        block = read_record().last_switch
        assert block["request_id"] == request_id
        assert block["status"] == control_context.SWITCH_APPLIED
        assert block["target"] == "va"
        assert block["generation"] == 2

    def test_the_requester_is_the_session_when_there_is_one(self, client):
        post_target(client, session_id=SESSION_A)
        assert read_record().last_switch["requested_by"] == SESSION_A

    def test_a_session_less_gesture_is_recorded_against_this_process(self, client):
        post_target(client, session_id=None)
        assert read_record().last_switch["requested_by"] == f"pid:{os.getpid()}"

    def test_the_target_it_is_already_on_succeeds_without_minting(self, client):
        before = read_record()
        response = post_target(client, target="live")
        assert response.status_code == 202, response.text
        assert response.json()["generation"] == 1
        assert "already" in response.json()["detail"]
        assert read_record() == before

    def test_every_request_id_is_fresh(self, client):
        first = post_target(client).json()["request_id"]
        second = post_target(client, target="live").json()["request_id"]
        assert first != second

    def test_the_owner_stamp_survives_the_write(self, client):
        post_target(client)
        record = read_record()
        assert record.owner is not None
        assert record.owner.pid == os.getpid()
        assert record.owner.kind == control_context.OWNER_WEB_TERMINAL


class TestAudit:
    """One ledger line per POST, joinable to the session that made it."""

    def test_an_accepted_gesture_files_exactly_one_record(self, client, ledger):
        assert post_target(client).status_code == 202
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "allowed"
        assert ledger[0]["subject"] == websocket_routes.AUDIT_SUBJECT_TARGET_SET

    def test_the_record_names_the_target_and_the_generation(self, client, ledger):
        request_id = post_target(client).json()["request_id"]
        detail = ledger[0]["detail"]
        assert "target=va" in detail
        assert f"request_id={request_id}" in detail
        assert "generation=2" in detail

    def test_a_refusal_files_exactly_one_record_too(self, client, ledger):
        assert post_target(client, target="standin").status_code == 400
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "refused"
        assert ledger[0]["reason"] == "unknown_target"

    def test_a_malformed_session_id_still_leaves_exactly_one_record(self, client, ledger):
        assert post_target(client, session_id="nope").status_code == 400
        # The grammar check raises before the route's own recorder, so this is
        # the middleware's line — one, and refused.
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "refused"

    def test_the_record_joins_on_the_session_key(self, client, ledger):
        """The ledger line and the record's ``requested_by`` name one key."""
        with attached_pty(client, SESSION_A):
            assert post_target(client).status_code == 202
        assert ledger[0]["session"] == SESSION_A
        assert read_record().last_switch["requested_by"] == SESSION_A

    def test_a_session_less_gesture_is_filed_on_the_lab_surface(self, client, ledger):
        assert post_target(client, session_id=None).status_code == 202
        assert ledger[0]["session"] is None
        assert ledger[0]["surface"] == websocket_routes.LAB_MUTATION_SURFACE

    def test_a_session_gesture_keeps_the_http_surface(self, client, ledger):
        assert post_target(client).status_code == 202
        assert ledger[0]["surface"] == websocket_routes.HTTP_MUTATION_SURFACE
