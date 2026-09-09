"""Tests for ``GET /api/terminal/posture`` — the deployment's control roster.

One route answers the whole of what the header chip and its popover render.
There is one control context per deployment, so every answer here is the
deployment's: which target it is standing on, at which generation, who owns the
record, what each running controls server has actually bound, what is executing
right now, and one row per configured target carrying that machine's identity,
its reachability, the persona's ceiling, the operator's narrowing, the effective
answer the connector will apply and whether a switch is offered.

Four properties shape every test below.

* **No session is involved.** ``session_id`` names who is asking. It selects no
  record, resolves no PTY and changes nothing in the payload; the roster of a
  chat key, a terminal key and no key at all is one roster.
* **The server decides, not the browser.** The refusal word under a missing
  Switch button is the switch tool's own; ``age_s`` and ``stale`` are computed
  here because the browser's clock is not the one the stamps were written on;
  and the collapse from "two gateway roles were probed" to "this row is
  reachable" follows the role the connector will actually select.
* **Ceiling, posture and effective are three separate columns.** The ceiling is
  the deployment's, the posture is the operator's, and the effective answer is
  the rule the connector applies. Collapsing them would leave the popover unable
  to say whether a locked toggle is the persona's doing or the operator's.
* **The fleet is published, not collapsed.** A switch has landed when every live
  server reports the generation it was asked for, so the rows say which server
  is where instead of offering one verdict that could not name a stuck pid.

Harness mirrors ``test_posture_routes.py``: one app per test through
``create_app``, entered as a ``TestClient`` context manager so the lifespan
claims the record, over an ``OSPREY_AGENT_DATA_ROOT`` stamped at a throwaway
directory. The two cases about the id itself — no id is a 200, a malformed id is
a 400 — live in that file with the grammar the three routes share; this file
owns the FIELD SET.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal import control_context_owner
from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey.mcp_server.control_system import target_state
from osprey_connectors import control_context, posture_store
from tests._control_context_fixtures import owner, write_control_context, write_server_report

SESSION_A = "aaaaaaaa-1111-2222-3333-444444444444"
CHAT_A = "cccccccc-1111-2222-3333-444444444444"

#: The Channel Access port a co-deployed stand-in serves on.
STANDIN_PORT = 5074

#: Two live controls servers, which is the ordinary shape: one per agent
#: session on the deployment.
SERVER_A = 5150
SERVER_B = 5151

#: A pid no kernel hands out: the largest a 32-bit ``pid_t`` holds.
DEAD_PID = 2_147_483_646

#: A real, live process that is not this one. A record owned by a corpse is a
#: record this terminal claims over, and a test wanting a foreign owner has to
#: name one that is actually running or it passes for the wrong reason.
OTHER_TERMINAL_PID = os.getppid()
OTHER_TERMINAL_PORT = 8090

#: Per-target display metadata in the shape a controls server publishes.
TARGET_META = {
    "live": {"label": "LIVE MACHINE", "endpoint": "gw:5064", "real_machine": True},
    "va": {"label": "virtual accelerator (simulation)", "endpoint": "localhost:5064"},
    "standin": {
        "label": "LIVE MACHINE (stand-in)",
        "endpoint": f"localhost:{STANDIN_PORT}",
        "real_machine": True,
    },
}

#: The keys every row carries. Pinned as a SET, so a field silently dropped and
#: a field silently added are both a red test — the chip reads this contract.
ROW_FIELDS = {
    "target",
    "label",
    "display_name",
    "short_label",
    "kind",
    "endpoint",
    "real_machine",
    "active",
    "is_baseline",
    "available_now",
    "reason",
    "reason_detail",
    "ceiling_writes",
    "posture",
    "effective",
    "narrowing_refusal",
    "reachability",
}

REACHABILITY_FIELDS = {"state", "role", "probed_at", "age_s", "role_detail"}

#: One row per running controls server.
SERVER_FIELDS = {
    "pid",
    "session",
    "applied_target",
    "applied_generation",
    "children",
    "last_switch",
    "last_posture_realign",
    "updated_at",
}

#: One row per live execution marker.
EXECUTION_FIELDS = {"pid", "target", "session", "surface", "kernel_id", "started_at", "age_s"}

OWNER_FIELDS = {"kind", "pid", "port", "self"}

TOP_LEVEL_FIELDS = {
    "session_id",
    "control_target",
    "generation",
    "store_available",
    "readonly_run",
    "owner",
    "servers",
    "execution_in_flight",
    "last_switch",
    "last_posture_realign",
    "targets",
}

#: Fields two earlier eras of this payload carried. The badge era answered
#: session-wide questions the per-target rows replaced; the session era resolved
#: one controls server per PTY, which the deployment-wide record replaced, and
#: spelled the target key ``session_target`` before it was named for the control
#: context. A client still reading any of them would be reading a fact this
#: route no longer has.
RETIRED_FIELDS = (
    "posture",
    "rendered_writes_enabled",
    "session_target",
    "session_target_label",
    "target_writes_enabled",
    "target_source",
)


# -- render -----------------------------------------------------------------


def render(
    *,
    global_writes=False,
    va_writes=None,
    standin_roles=("read_only", "write_access"),
    display_names=None,
    live_write_port=None,
    name="config.yml",
    tmp_path=None,
):
    """A render carrying all three control targets.

    ``epics`` is the facility's own machine, ``live_standin`` the co-deployed
    stand-in, ``virtual_accelerator`` the simulator — three connector blocks,
    therefore three targets, which is what makes ``configured_targets`` answer
    with three names and ``session_posture`` with one ceiling each. The
    deployment's baseline is the stand-in, so a record on ``live`` is a
    deployment that has switched.

    ``standin_roles`` narrows the stand-in's gateway table. Dropping
    ``read_only`` from it is the deployment shape that makes narrowing that
    target cost something: the session would select a role the config does not
    configure and the target would stop being usable at all.

    ``display_names`` is written verbatim as
    ``control_system.target_display_names``, the deployment's renaming of the
    chip's per-target words.

    ``live_write_port`` puts the facility machine's write gateway on a port of
    its own. It is what makes a posture observable in an endpoint at all: with
    one gateway row serving both roles the selected role moves under a narrowing
    and the host:port the row reports does not, so a render that named the wrong
    gateway would look identical to one that named the right one.
    """
    gateway = {"address": "gw", "port": 5064, "use_name_server": True}
    standin_gateway = {"address": "localhost", "port": STANDIN_PORT, "use_name_server": True}
    va: dict = {
        "simulation_file": "data/sim.json",
        "probe_channel": "SIM:PROBE",
        "gateways": {"read_only": dict(gateway)},
    }
    if va_writes is not None:
        va["writes_enabled"] = va_writes
    config = {
        "control_system": {
            "type": "live_standin",
            "writes_enabled": global_writes,
            # The keys a switch is judged on besides the gateways: a channel to
            # probe, strict limits (required toward the live family) and the
            # operator's acknowledgement of the live gateway. Without them every
            # row would answer with an eligibility refusal and the roster could
            # not be exercised.
            "limits_checking": {"enabled": True, "allow_unlisted_channels": False},
            "target_switch": {"live_gateway_acknowledged": "operator@example"},
            "connector": {
                "epics": {
                    "probe_channel": "SR:PROBE",
                    "gateways": {
                        "read_only": dict(gateway),
                        "write_access": dict(gateway)
                        | ({} if live_write_port is None else {"port": live_write_port}),
                    },
                },
                "live_standin": {
                    "probe_channel": "SR:PROBE",
                    "gateways": {role: dict(standin_gateway) for role in standin_roles},
                },
                "virtual_accelerator": va,
            },
        },
        # A real archiver, because a simulated target paired with a mock one is
        # refused as ``invented_history`` and every row would answer with that
        # instead of the roster fact under test.
        "archiver": {"type": "epics_archiver"},
        "services": {"live_standin": {"port": STANDIN_PORT}},
        "deployed_services": ["virtual_accelerator", "live_standin"],
    }
    if display_names is not None:
        config["control_system"]["target_display_names"] = display_names
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


# -- fixtures ---------------------------------------------------------------


@pytest.fixture
def agent_data_root(tmp_path, monkeypatch):
    """Point every resolver at one throwaway agent-data root.

    ``OSPREY_AGENT_DATA_ROOT`` is the single stamp the record reader and
    ``posture_store`` both prefer, and the ONLY way this route reaches either —
    patching ``resolve_shared_data_root`` would redirect one of them and leave
    the other reading the repository's own ``var/agent_data``.
    """
    root = tmp_path / "agent_data"
    (root / posture_store.STATE_DIR_NAME).mkdir(parents=True)
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(root))
    # A read-only *run* is a deployment-wide fact this process must not inherit
    # from whatever ran before it: it would zero every ``effective`` below.
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv(control_context_owner.WEB_PORT_ENV, raising=False)
    posture_store.invalidate_cache()
    control_context.invalidate_cache()
    websocket_routes._reset_rendered_config_memo()
    yield root
    posture_store.invalidate_cache()
    control_context.invalidate_cache()
    websocket_routes._reset_rendered_config_memo()


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_watch"
    ws.mkdir()
    return ws


@pytest.fixture
def make_client(agent_data_root, workspace_dir, tmp_path):
    """Build an app + TestClient over the stamped root.

    The record is written **before** the app starts, so the lifespan claim
    merges into it rather than minting a fresh one from ``load_osprey_config``,
    which this process has no workspace for. The owner task's loop is stubbed
    out: every tick these tests need has happened by the time the client is
    handed over, and a loop ticking underneath the assertions would be a race
    rather than coverage.
    """

    @contextmanager
    def _make(config_path=None, *, target="live", generation=4, posture=None):
        write_control_context(
            agent_data_root, target=target, generation=generation, posture=posture
        )
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
                        render(tmp_path=tmp_path) if config_path is None else config_path
                    )
                    yield test_client

    return _make


@pytest.fixture
def client(make_client):
    with make_client() as c:
        yield c


# -- harness ----------------------------------------------------------------


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


def stamp(age_s=0.0):
    """An ISO-8601 stamp *age_s* seconds in the past."""
    return (datetime.now(UTC) - timedelta(seconds=age_s)).isoformat()


def at_offset(age_s, hours):
    """The instant *age_s* seconds ago, written in a fixed UTC offset.

    Two servers on one deployment need not run in one timezone, and the same
    moment written at ``-11:00`` and at ``+13:00`` is a full day apart in wall
    time. That is what makes a lexical comparison of ISO-8601 stamps report the
    wrong sweep as the newer one.
    """
    moment = datetime.now(UTC) - timedelta(seconds=age_s)
    return moment.astimezone(timezone(timedelta(hours=hours))).isoformat()


def probed(state, *, age_s=0.0, gateway="gw:5064"):
    """One published probe row, *age_s* seconds old."""
    return {"state": state, "probed_at": stamp(age_s), "gateway": gateway}


def sweep(**targets):
    """A published reachability block, in the shape the prober normalises to."""
    return {"published_at": stamp(), "targets": dict(targets)}


def write_marker(root: Path, *, pid, target="live", session=None, surface=None, kernel_id=None):
    """Plant one execution marker, as the executor and the kernel write it."""
    directory = root / posture_store.STATE_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    path = (
        directory / f"{target_state.INFLIGHT_FILE_PREFIX}{pid}{target_state.INFLIGHT_FILE_SUFFIX}"
    )
    path.write_text(
        json.dumps(
            {
                "pid": pid,
                "session": session,
                "surface": surface or "python_executor",
                "kernel_id": kernel_id,
                "target": target,
                "launch_posture": "*=writes",
                "started_at": stamp(),
            }
        ),
        encoding="utf-8",
    )
    return path


def get_posture(client, session_id=None):
    params = {} if session_id is None else {"session_id": session_id}
    resp = client.get("/api/terminal/posture", params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def row_for(payload, target):
    rows = [row for row in payload["targets"] if row["target"] == target]
    assert len(rows) == 1, f"expected exactly one {target!r} row, got {rows}"
    return rows[0]


def server_row(payload, pid):
    rows = [row for row in payload["servers"] if row["pid"] == pid]
    assert len(rows) == 1, f"expected exactly one row for server {pid}, got {payload['servers']}"
    return rows[0]


# -- the field set ----------------------------------------------------------


class TestTheFieldSet:
    """What the payload carries, pinned as sets so a drift in either direction fails."""

    def test_the_top_level_keys_are_exactly_these(self, client, agent_data_root):
        write_server_report(agent_data_root, SERVER_A, applied_target="live")
        with only_alive(SERVER_A):
            payload = get_posture(client, SESSION_A)
        assert set(payload) == TOP_LEVEL_FIELDS

    def test_every_target_row_carries_the_same_keys(self, client):
        payload = get_posture(client)
        assert payload["targets"], "the render configures three targets"
        for row in payload["targets"]:
            assert set(row) == ROW_FIELDS
            assert set(row["reachability"]) == REACHABILITY_FIELDS

    def test_every_server_row_carries_the_same_keys(self, client, agent_data_root):
        write_server_report(agent_data_root, SERVER_A, session="s-1", applied_target="live")
        write_server_report(agent_data_root, SERVER_B)
        with only_alive(SERVER_A, SERVER_B):
            payload = get_posture(client)
        assert len(payload["servers"]) == 2
        for row in payload["servers"]:
            assert set(row) == SERVER_FIELDS

    def test_every_execution_row_carries_the_same_keys(self, client, agent_data_root):
        write_marker(agent_data_root, pid=DEAD_PID - 1)
        with only_alive(DEAD_PID - 1):
            payload = get_posture(client)
        assert len(payload["execution_in_flight"]) == 1
        assert set(payload["execution_in_flight"][0]) == EXECUTION_FIELDS

    def test_the_owner_row_carries_the_same_keys(self, client):
        assert set(get_posture(client)["owner"]) == OWNER_FIELDS

    @pytest.mark.parametrize("field", RETIRED_FIELDS)
    def test_a_retired_field_is_not_answered(self, client, field):
        assert field not in get_posture(client)

    def test_the_session_id_is_echoed_and_nothing_else_reads_it(self, client):
        with_id = get_posture(client, SESSION_A)
        without = get_posture(client)
        assert with_id["session_id"] == SESSION_A
        assert without["session_id"] is None
        assert {k: v for k, v in with_id.items() if k != "session_id"} == {
            k: v for k, v in without.items() if k != "session_id"
        }


# -- the deployment's target ------------------------------------------------


class TestTheTarget:
    """``control_target`` and ``generation`` are the record's, not a session's."""

    def test_they_come_from_the_record(self, make_client):
        with make_client(target="va", generation=9) as client:
            payload = get_posture(client, SESSION_A)
        assert payload["control_target"] == "va"
        assert payload["generation"] == 9

    def test_the_active_row_is_the_recorded_target(self, make_client):
        with make_client(target="va", generation=2) as client:
            payload = get_posture(client)
        assert [row["target"] for row in payload["targets"] if row["active"]] == ["va"]

    def test_the_baseline_row_is_the_renders_own_target(self, client):
        """A ``live_standin`` deployment's baseline is the stand-in, not ``live``."""
        payload = get_posture(client)
        assert [row["target"] for row in payload["targets"] if row["is_baseline"]] == ["standin"]

    def test_no_record_falls_back_to_the_baseline_at_no_generation(self, client):
        with patch.object(control_context, "read_record", return_value=None):
            payload = get_posture(client)
        assert payload["control_target"] == "standin"
        assert payload["generation"] is None


# -- ownership --------------------------------------------------------------


class TestTheOwner:
    """``owner {kind, pid, port, self}`` — who may write, and whether it is us."""

    def test_this_terminal_owning_reports_itself(self, client, monkeypatch):
        monkeypatch.setenv(control_context_owner.WEB_PORT_ENV, "8080")
        payload = get_posture(client)
        assert payload["owner"] == {
            "kind": "web_terminal",
            "pid": os.getpid(),
            "port": 8080,
            "self": True,
        }

    def test_an_unset_port_is_null_rather_than_a_guess(self, client):
        assert get_posture(client)["owner"]["port"] is None

    def test_a_follower_names_the_other_terminal(self, client):
        client.app.state.control_context_follows = owner(
            pid=OTHER_TERMINAL_PID, port=OTHER_TERMINAL_PORT
        )
        payload = get_posture(client)
        assert payload["owner"] == {
            "kind": "web_terminal",
            "pid": OTHER_TERMINAL_PID,
            "port": OTHER_TERMINAL_PORT,
            "self": False,
        }

    def test_with_no_owner_task_the_record_answers(self, client, agent_data_root):
        """A deployment whose context a controls server owns, read by a terminal
        that has not got far enough to claim anything."""
        write_control_context(
            agent_data_root,
            target="live",
            generation=4,
            owned_by=owner(kind=control_context.OWNER_CONTROLS_SERVER, pid=SERVER_A),
        )
        del client.app.state.control_context_owner
        payload = get_posture(client)
        assert payload["owner"] == {
            "kind": "controls_server",
            "pid": SERVER_A,
            "port": None,
            "self": False,
        }

    def test_a_record_this_process_owns_is_self_even_with_no_task(self, client, agent_data_root):
        del client.app.state.control_context_owner
        assert get_posture(client)["owner"]["self"] is True

    def test_an_unowned_context_reports_no_owner(self, client, agent_data_root):
        write_control_context(agent_data_root, target="live", generation=4, owned_by=None)
        del client.app.state.control_context_owner
        assert get_posture(client)["owner"] is None


# -- the fleet --------------------------------------------------------------


class TestTheServers:
    """One row per running controls server — the fleet, published not collapsed."""

    def test_a_dead_servers_report_is_not_a_row(self, client, agent_data_root):
        write_server_report(agent_data_root, SERVER_A, applied_target="live")
        write_server_report(agent_data_root, DEAD_PID, applied_target="va")
        with only_alive(SERVER_A):
            payload = get_posture(client)
        assert [row["pid"] for row in payload["servers"]] == [SERVER_A]

    def test_no_live_server_is_an_empty_list(self, client, agent_data_root):
        write_server_report(agent_data_root, DEAD_PID, applied_target="va")
        with only_alive():
            assert get_posture(client)["servers"] == []

    def test_a_row_reports_what_that_server_has_bound(self, client, agent_data_root):
        write_server_report(
            agent_data_root,
            SERVER_A,
            session="agent-7",
            applied_target="live",
            applied_generation=4,
            updated_at=stamp(),
        )
        with only_alive(SERVER_A):
            row = server_row(get_posture(client), SERVER_A)
        assert row["session"] == "agent-7"
        assert row["applied_target"] == "live"
        assert row["applied_generation"] == 4

    def test_an_unbound_server_reports_null_not_the_baseline(self, client, agent_data_root):
        """Null is "this server has not got there yet", never "it is on live"."""
        write_server_report(agent_data_root, SERVER_A)
        with only_alive(SERVER_A):
            row = server_row(get_posture(client), SERVER_A)
        assert row["applied_target"] is None
        assert row["applied_generation"] is None
        assert row["session"] is None

    def test_a_row_names_the_connector_children_it_holds(self, client, agent_data_root):
        """The chip's convergence wait is scoped to rows that hold a connector.

        ``children`` is the row's word for that: an explicit empty list is a
        server serving nothing — nothing it runs can still touch the old
        target — and the chip stops waiting on it.
        """
        write_server_report(agent_data_root, SERVER_A, children=[5001, 5002])
        with only_alive(SERVER_A):
            row = server_row(get_posture(client), SERVER_A)
        assert row["children"] == [5001, 5002]

    def test_a_server_with_no_child_reports_an_empty_list(self, client, agent_data_root):
        write_server_report(agent_data_root, SERVER_A)
        with only_alive(SERVER_A):
            row = server_row(get_posture(client), SERVER_A)
        assert row["children"] == []

    def test_a_servers_switch_progress_is_aged_here(self, client, agent_data_root):
        write_server_report(
            agent_data_root,
            SERVER_A,
            last_switch={"generation": 4, "status": "applying", "at": stamp(age_s=30)},
        )
        with only_alive(SERVER_A):
            row = server_row(get_posture(client), SERVER_A)
        assert row["last_switch"]["status"] == "applying"
        assert row["last_switch"]["generation"] == 4
        assert 25 <= row["last_switch"]["age_s"] <= 45

    def test_the_rows_are_ordered_oldest_report_first(self, client, agent_data_root):
        write_server_report(agent_data_root, SERVER_A, updated_at=stamp(age_s=1))
        write_server_report(agent_data_root, SERVER_B, updated_at=stamp(age_s=90))
        with only_alive(SERVER_A, SERVER_B):
            payload = get_posture(client)
        assert [row["pid"] for row in payload["servers"]] == [SERVER_B, SERVER_A]

    def test_a_stuck_server_is_nameable(self, client, agent_data_root):
        """The whole reason the fleet is published: a failed swap names its pid."""
        write_server_report(
            agent_data_root,
            SERVER_A,
            applied_target="live",
            applied_generation=4,
            updated_at=stamp(age_s=5),
        )
        write_server_report(
            agent_data_root,
            SERVER_B,
            applied_generation=3,
            last_switch={"generation": 4, "status": "failed", "at": stamp()},
            updated_at=stamp(),
        )
        with only_alive(SERVER_A, SERVER_B):
            payload = get_posture(client)
        stuck = [row["pid"] for row in payload["servers"] if row["applied_generation"] != 4]
        assert stuck == [SERVER_B]
        assert server_row(payload, SERVER_B)["last_switch"]["status"] == "failed"


# -- reachability -----------------------------------------------------------


class TestReachability:
    """Per target, the sweep of whichever live server looked at it last."""

    def test_the_newest_probe_of_a_target_wins(self, client, agent_data_root):
        write_server_report(
            agent_data_root,
            SERVER_A,
            reachability=sweep(live={"read_only": probed("down", age_s=120)}),
        )
        write_server_report(
            agent_data_root,
            SERVER_B,
            reachability=sweep(live={"read_only": probed("reached", age_s=1)}),
        )
        with only_alive(SERVER_A, SERVER_B):
            payload = get_posture(client)
        assert row_for(payload, "live")["reachability"]["state"] == "reached"

    def test_the_newest_is_read_per_target_not_per_server(self, client, agent_data_root):
        """Two servers, each fresher about a different machine."""
        write_server_report(
            agent_data_root,
            SERVER_A,
            reachability=sweep(
                live={"read_only": probed("reached", age_s=1)},
                va={"read_only": probed("down", age_s=300)},
            ),
        )
        write_server_report(
            agent_data_root,
            SERVER_B,
            reachability=sweep(
                live={"read_only": probed("down", age_s=300)},
                va={"read_only": probed("reached", age_s=1)},
            ),
        )
        with only_alive(SERVER_A, SERVER_B):
            payload = get_posture(client)
        assert row_for(payload, "live")["reachability"]["state"] == "reached"
        assert row_for(payload, "va")["reachability"]["state"] == "reached"

    def test_a_dead_servers_sweep_is_not_consulted(self, client, agent_data_root):
        write_server_report(
            agent_data_root,
            DEAD_PID,
            reachability=sweep(live={"read_only": probed("reached")}),
        )
        with only_alive(SERVER_A):
            payload = get_posture(client)
        assert row_for(payload, "live")["reachability"]["state"] == "unknown"

    def test_no_live_sweep_at_all_is_unknown(self, client):
        payload = get_posture(client)
        for row in payload["targets"]:
            assert row["reachability"]["state"] == "unknown"
            assert row["reachability"]["probed_at"] is None

    def test_a_row_older_than_the_probers_own_threshold_is_stale(self, client, agent_data_root):
        write_server_report(
            agent_data_root,
            SERVER_A,
            reachability=sweep(live={"read_only": probed("reached", age_s=10_000)}),
        )
        with only_alive(SERVER_A):
            payload = get_posture(client)
        assert row_for(payload, "live")["reachability"]["state"] == "stale"

    def test_the_row_reports_the_role_the_connector_would_select(self, client, agent_data_root):
        write_server_report(
            agent_data_root,
            SERVER_A,
            reachability=sweep(
                live={"read_only": probed("reached"), "write_access": probed("down")}
            ),
        )
        with only_alive(SERVER_A):
            reach = row_for(get_posture(client), "live")["reachability"]
        # Writes are off across this render, so the connector opens the read
        # gateway and a down write gateway says nothing about the row.
        assert reach["role"] == "read_only"
        assert reach["state"] == "reached"
        assert reach["role_detail"] == {"write_access": "down"}


# -- how two stamps are ordered ---------------------------------------------


class TestStampOrdering:
    """Which of two published stamps is newer, and what an absent one does.

    The route compares moments, not the strings they were written as. Two
    servers on one deployment need not run in one timezone and need not write
    a stamp at all, and both of those turn a lexical comparison into a wrong
    answer rather than an error anybody would notice.
    """

    def test_a_stamp_in_another_zone_is_ordered_by_its_moment(self, client, agent_data_root):
        """``-11:00`` and ``+13:00`` are a full day apart in wall time, so the
        fresher sweep's ISO string sorts BEFORE the staler one. Comparing the
        strings would report the machine that was measured a minute ago as the
        one nobody has looked at."""
        stale = at_offset(60, 13)
        fresh = at_offset(1, -11)
        assert fresh < stale, "the trap this test exists for has stopped being one"

        write_server_report(
            agent_data_root,
            SERVER_A,
            reachability=sweep(live={"read_only": {"state": "down", "probed_at": stale}}),
        )
        write_server_report(
            agent_data_root,
            SERVER_B,
            reachability=sweep(live={"read_only": {"state": "reached", "probed_at": fresh}}),
        )
        with only_alive(SERVER_A, SERVER_B):
            payload = get_posture(client)
        assert row_for(payload, "live")["reachability"]["state"] == "reached"

    def test_a_stamp_with_no_zone_is_read_as_utc(self, client, agent_data_root):
        """A writer that stamped naively is not thereby the oldest writer."""
        naive_fresh = (datetime.now(UTC) - timedelta(seconds=1)).replace(tzinfo=None).isoformat()
        write_server_report(
            agent_data_root,
            SERVER_A,
            reachability=sweep(live={"read_only": probed("down", age_s=120)}),
        )
        write_server_report(
            agent_data_root,
            SERVER_B,
            reachability=sweep(live={"read_only": {"state": "reached", "probed_at": naive_fresh}}),
        )
        with only_alive(SERVER_A, SERVER_B):
            payload = get_posture(client)
        assert row_for(payload, "live")["reachability"]["state"] == "reached"

    def test_a_sweep_with_no_probed_at_loses_to_one_that_has_one(self, client, agent_data_root):
        """A sweep with no time on it must never outrank one that carries its own."""
        write_server_report(
            agent_data_root,
            SERVER_A,
            reachability=sweep(live={"read_only": {"state": "reached"}}),
        )
        write_server_report(
            agent_data_root,
            SERVER_B,
            reachability=sweep(live={"read_only": probed("down", age_s=5)}),
        )
        with only_alive(SERVER_A, SERVER_B):
            payload = get_posture(client)
        assert row_for(payload, "live")["reachability"]["state"] == "down"

    def test_an_unparseable_probed_at_loses_the_same_way(self, client, agent_data_root):
        write_server_report(
            agent_data_root,
            SERVER_A,
            reachability=sweep(live={"read_only": {"state": "reached", "probed_at": "soon"}}),
        )
        write_server_report(
            agent_data_root,
            SERVER_B,
            reachability=sweep(live={"read_only": probed("down", age_s=5)}),
        )
        with only_alive(SERVER_A, SERVER_B):
            payload = get_posture(client)
        assert row_for(payload, "live")["reachability"]["state"] == "down"

    def test_a_sweep_with_no_probed_at_still_renders_when_it_is_all_there_is(
        self, client, agent_data_root
    ):
        """Losing every comparison is not the same as being dropped."""
        write_server_report(
            agent_data_root,
            SERVER_A,
            reachability=sweep(live={"read_only": {"state": "reached"}}),
        )
        with only_alive(SERVER_A):
            reach = row_for(get_posture(client), "live")["reachability"]
        assert reach["state"] == "reached"
        assert reach["probed_at"] is None
        assert reach["age_s"] is None

    def test_a_report_with_no_update_stamp_sorts_first_and_does_not_crash(
        self, client, agent_data_root
    ):
        write_server_report(agent_data_root, SERVER_A, updated_at=stamp())
        write_server_report(agent_data_root, SERVER_B)
        with only_alive(SERVER_A, SERVER_B):
            payload = get_posture(client)
        assert [row["pid"] for row in payload["servers"]] == [SERVER_B, SERVER_A]
        assert server_row(payload, SERVER_B)["updated_at"] is None

    def test_a_stamped_publisher_outranks_an_unstamped_one(self, client, agent_data_root):
        write_server_report(
            agent_data_root,
            SERVER_A,
            targets={"live": {"label": "NAMED BY A DATED REPORT", "real_machine": True}},
            updated_at=stamp(),
        )
        write_server_report(
            agent_data_root,
            SERVER_B,
            targets={"live": {"label": "NAMED BY AN UNDATED ONE", "real_machine": True}},
        )
        with only_alive(SERVER_A, SERVER_B):
            payload = get_posture(client)
        assert row_for(payload, "live")["label"] == "NAMED BY A DATED REPORT"


# -- executions in flight ---------------------------------------------------


class TestExecutionsInFlight:
    """Rows, not a boolean: the surface has to be able to say WHOSE run it is."""

    def test_no_marker_is_an_empty_list(self, client):
        assert get_posture(client)["execution_in_flight"] == []

    def test_a_notebook_kernels_run_names_its_kernel(self, client, agent_data_root):
        write_marker(
            agent_data_root,
            pid=SERVER_A,
            target="live",
            session="kernel:abc123",
            surface="notebook_kernel",
            kernel_id="abc123",
        )
        with only_alive(SERVER_A):
            rows = get_posture(client)["execution_in_flight"]
        assert len(rows) == 1
        assert rows[0]["session"] == "kernel:abc123"
        assert rows[0]["surface"] == "notebook_kernel"
        assert rows[0]["kernel_id"] == "abc123"
        assert rows[0]["pid"] == SERVER_A
        assert rows[0]["target"] == "live"
        assert rows[0]["age_s"] is not None

    def test_an_executors_run_carries_no_kernel(self, client, agent_data_root):
        write_marker(agent_data_root, pid=SERVER_A, session="agent-7")
        with only_alive(SERVER_A):
            rows = get_posture(client)["execution_in_flight"]
        assert rows[0]["surface"] == "python_executor"
        assert rows[0]["kernel_id"] is None

    def test_a_dead_writers_marker_is_not_reported(self, client, agent_data_root):
        write_marker(agent_data_root, pid=DEAD_PID)
        with only_alive(SERVER_A):
            assert get_posture(client)["execution_in_flight"] == []

    def test_every_live_run_is_reported_not_only_the_first(self, client, agent_data_root):
        write_marker(agent_data_root, pid=SERVER_A, session="agent-7")
        write_marker(agent_data_root, pid=SERVER_B, session="kernel:z", surface="notebook_kernel")
        with only_alive(SERVER_A, SERVER_B):
            rows = get_posture(client)["execution_in_flight"]
        assert {row["pid"] for row in rows} == {SERVER_A, SERVER_B}


# -- the record's terminus and the fleet's realignment ----------------------


class TestLastSwitchAndRealign:
    """``last_switch`` is the record's; ``last_posture_realign`` is the fleet's."""

    def test_the_terminus_is_the_records_and_is_aged(self, make_client, agent_data_root):
        with make_client() as client:
            write_control_context(
                agent_data_root,
                target="live",
                generation=4,
                last_switch={
                    "request_id": "req-1",
                    "target": "live",
                    "status": "applied",
                    "generation": 4,
                    "at": stamp(age_s=12),
                },
            )
            payload = get_posture(client)
        assert payload["last_switch"]["request_id"] == "req-1"
        assert payload["last_switch"]["status"] == "applied"
        assert 8 <= payload["last_switch"]["age_s"] <= 25

    def test_a_record_with_no_terminus_answers_null(self, client):
        assert get_posture(client)["last_switch"] is None

    def test_a_servers_own_switch_progress_is_not_the_top_level_terminus(
        self, client, agent_data_root
    ):
        """The record and a report use the same words for different things."""
        write_server_report(
            agent_data_root,
            SERVER_A,
            last_switch={"generation": 4, "status": "applying", "at": stamp()},
        )
        with only_alive(SERVER_A):
            payload = get_posture(client)
        assert payload["last_switch"] is None
        assert server_row(payload, SERVER_A)["last_switch"]["status"] == "applying"

    def test_no_realignment_anywhere_answers_null(self, client):
        assert get_posture(client)["last_posture_realign"] is None

    def test_a_pending_realignment_wins_over_a_newer_settled_one(self, client, agent_data_root):
        write_server_report(
            agent_data_root,
            SERVER_A,
            last_posture_realign={"state": "pending", "at": stamp(age_s=60)},
        )
        write_server_report(
            agent_data_root,
            SERVER_B,
            last_posture_realign={"state": "done", "at": stamp(age_s=1)},
        )
        with only_alive(SERVER_A, SERVER_B):
            payload = get_posture(client)
        assert payload["last_posture_realign"]["state"] == "pending"

    def test_with_nothing_pending_the_newest_answers(self, client, agent_data_root):
        write_server_report(
            agent_data_root,
            SERVER_A,
            last_posture_realign={"state": "done", "at": stamp(age_s=600)},
        )
        write_server_report(
            agent_data_root,
            SERVER_B,
            last_posture_realign={"state": "done", "at": stamp(age_s=1)},
        )
        with only_alive(SERVER_A, SERVER_B):
            payload = get_posture(client)
        assert payload["last_posture_realign"]["state"] == "done"
        assert (
            payload["last_posture_realign"]["at"]
            == server_row(payload, SERVER_B)["last_posture_realign"]["at"]
        )


# -- identity ---------------------------------------------------------------


class TestTargetIdentity:
    """What each machine is called, and which derivation named it."""

    def test_a_published_label_wins_over_the_render(self, client, agent_data_root):
        write_server_report(agent_data_root, SERVER_A, targets=TARGET_META)
        with only_alive(SERVER_A):
            payload = get_posture(client)
        assert row_for(payload, "live")["label"] == "LIVE MACHINE"
        assert row_for(payload, "standin")["label"] == "LIVE MACHINE (stand-in)"

    def test_the_newest_publisher_wins_per_target(self, client, agent_data_root):
        write_server_report(
            agent_data_root,
            SERVER_A,
            targets={"live": {"label": "OLD NAME", "real_machine": True}},
            updated_at=stamp(age_s=600),
        )
        write_server_report(
            agent_data_root,
            SERVER_B,
            targets={"live": {"label": "NEW NAME", "real_machine": True}},
            updated_at=stamp(age_s=1),
        )
        with only_alive(SERVER_A, SERVER_B):
            payload = get_posture(client)
        assert row_for(payload, "live")["label"] == "NEW NAME"

    def test_a_target_no_server_published_still_renders(self, client, agent_data_root):
        write_server_report(
            agent_data_root,
            SERVER_A,
            targets={"live": {"label": "LIVE MACHINE", "real_machine": True}},
        )
        with only_alive(SERVER_A):
            payload = get_posture(client)
        assert row_for(payload, "va")["label"]
        assert row_for(payload, "va")["kind"] == "virtual accelerator"

    def test_the_short_word_comes_from_the_label_not_the_target_name(self, client, agent_data_root):
        write_server_report(agent_data_root, SERVER_A, targets=TARGET_META)
        with only_alive(SERVER_A):
            payload = get_posture(client)
        assert row_for(payload, "live")["short_label"] == "LIVE"
        assert row_for(payload, "standin")["short_label"] == "STAND-IN"
        assert row_for(payload, "va")["short_label"] == "VIRTUAL"

    def test_a_deployment_may_rename_the_rows(self, make_client, tmp_path):
        config = render(tmp_path=tmp_path, display_names={"va": "Digital Twin"})
        with make_client(config) as client:
            payload = get_posture(client)
        assert row_for(payload, "va")["display_name"] == "Digital Twin"


# -- ceiling, posture, effective -------------------------------------------


class TestTheWriteColumns:
    """Three separate answers, never collapsed into one."""

    def test_absence_in_the_record_spells_writes(self, client):
        for row in get_posture(client)["targets"]:
            assert row["posture"] == "writes"

    def test_a_recorded_narrowing_shows_and_zeroes_the_effective_answer(self, make_client):
        with make_client(posture={"va": "sandbox"}) as client:
            payload = get_posture(client)
        assert row_for(payload, "va")["posture"] == "sandbox"
        assert row_for(payload, "va")["effective"] is False

    def test_the_ceiling_is_reported_per_target(self, make_client, tmp_path):
        config = render(tmp_path=tmp_path, va_writes=True)
        with make_client(config) as client:
            payload = get_posture(client)
        assert row_for(payload, "va")["ceiling_writes"] is True
        assert row_for(payload, "live")["ceiling_writes"] is False

    def test_an_armed_target_with_no_narrowing_is_effective(self, make_client, tmp_path):
        config = render(tmp_path=tmp_path, va_writes=True)
        with make_client(config) as client:
            payload = get_posture(client)
        assert row_for(payload, "va")["effective"] is True

    def test_a_ceiling_off_target_is_never_effective(self, client):
        for row in get_posture(client)["targets"]:
            if not row["ceiling_writes"]:
                assert row["effective"] is False

    def test_a_read_only_run_is_stated_outright(self, make_client, tmp_path, monkeypatch):
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        config = render(tmp_path=tmp_path, va_writes=True)
        with make_client(config) as client:
            payload = get_posture(client)
        assert payload["readonly_run"] is True
        assert row_for(payload, "va")["effective"] is False
        assert row_for(payload, "va")["ceiling_writes"] is True

    def test_narrowing_a_target_that_would_strand_it_is_flagged(self, make_client, tmp_path):
        """A stand-in configured for writes alone has no read gateway to fall to."""
        config = render(tmp_path=tmp_path, standin_roles=("write_access",))
        with make_client(config) as client:
            payload = get_posture(client)
        assert row_for(payload, "standin")["narrowing_refusal"] == "selected_role_missing"

    def test_an_already_narrowed_row_is_not_flagged(self, make_client, tmp_path):
        """That toggle brings the target back; nothing about it can strand anything."""
        config = render(tmp_path=tmp_path, standin_roles=("write_access",))
        with make_client(config, posture={"standin": "sandbox"}) as client:
            payload = get_posture(client)
        assert row_for(payload, "standin")["narrowing_refusal"] is None


# -- where a narrowing can be recorded --------------------------------------


class TestWhereItLands:
    """``store_available`` answers "is there anywhere to record a narrowing"."""

    def test_a_resolvable_root_answers_yes(self, client):
        assert get_posture(client)["store_available"] is True

    def test_no_location_for_the_record_answers_no(self, client):
        with patch.object(control_context, "record_path", return_value=None):
            assert get_posture(client)["store_available"] is False


# -- nothing here is session-scoped -----------------------------------------


class TestNoSessionScope:
    """The roster is the deployment's; a key names who is asking and no more."""

    def test_a_chat_key_gets_the_ordinary_roster(self, client):
        client.app.state.operator_registry = SimpleNamespace(
            has_chat_key=lambda key: key == CHAT_A,
            get_chat_session=lambda key: object() if key == CHAT_A else None,
            cleanup_all=_noop_cleanup,
        )
        chat = get_posture(client, CHAT_A)
        terminal = get_posture(client, SESSION_A)
        assert chat["targets"] == terminal["targets"]

    def test_a_chat_key_is_offered_the_switch_a_terminal_key_is(self, client, agent_data_root):
        """The refusal this feature removes: every chat row used to be
        ``available_now: false`` with reason ``chat_session``, because a chat had
        no controls server of its own to address a request to. The request goes
        to the deployment's owner now, so a chat asks for a switch like anyone."""
        client.app.state.operator_registry = SimpleNamespace(
            has_chat_key=lambda key: key == CHAT_A,
            get_chat_session=lambda key: object() if key == CHAT_A else None,
            cleanup_all=_noop_cleanup,
        )
        write_server_report(
            agent_data_root,
            SERVER_A,
            reachability=sweep(standin={"read_only": probed("reached")}),
        )
        with only_alive(SERVER_A):
            payload = get_posture(client, CHAT_A)
        assert row_for(payload, "standin")["available_now"] is True
        assert all(row["reason"] != "chat_session" for row in payload["targets"])

    def test_the_pty_registry_is_never_asked(self, client):
        """A roster that consulted a PTY would be answering a session question."""
        registry = client.app.state.pty_registry
        with patch.object(registry, "get_session", side_effect=AssertionError("asked")) as getter:
            get_posture(client, SESSION_A)
        assert getter.call_count == 0

    def test_the_reason_word_is_gone_from_the_module(self):
        assert not hasattr(websocket_routes, "REASON_CHAT_SESSION")

    def test_the_session_record_resolver_is_gone_from_the_module(self):
        for name in ("_session_record", "_reset_session_record_memo", "_state_dir_names"):
            assert not hasattr(websocket_routes, name)


async def _noop_cleanup() -> None:
    """The lifespan's shutdown calls this on whatever registry it finds."""
