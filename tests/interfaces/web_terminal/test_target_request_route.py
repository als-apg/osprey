"""Tests for ``POST /api/terminal/target`` — the operator's switch gesture.

The web server never switches a control target. It has no handle on the
connector: the controls MCP server owns it, and that server is a stdio child of
the Claude process inside the PTY with no inbound channel. So this route writes
**desired state** — one request file, addressed by ``server_pid`` to exactly the
controls server this session runs — and answers ``202``. The reconciler inside
that server picks it up, re-evaluates the same gate ``control_target_set``
applies, and publishes the outcome back into the state file.

Three consequences the tests below pin:

* **No availability pre-check.** Whether the target is eligible, reachable, or
  already active is the reconciler's question, evaluated immediately before the
  switch. A route that pre-judged it would answer from a snapshot taken a
  moment earlier and disagree with the gate that actually decides.
* **The address is the record's ``server_pid``**, never this process's pid and
  never a guess: a request written for a server that has since been replaced is
  dropped by its successor rather than honoured.
* **One audit record per POST**, carrying the *spawn* session key, so the
  gesture joins on the same key every record the session's own child emits
  carries.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

from osprey.audit import writer as audit_writer
from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey.mcp_server.control_system import target_banner, target_state
from osprey_connectors import session_store

SESSION_A = "aaaaaaaa-1111-2222-3333-444444444444"
SESSION_B = "bbbbbbbb-1111-2222-3333-444444444444"

#: The PTY process the terminal card is attached to, and the Claude Code
#: process inside it (a descendant when ``claude_code.cli_version`` pins the
#: CLI and the PTY child is ``npx``).
PTY_PID = 7000
CLAUDE_PID = 7001

#: The controls MCP server's own pid — the address every request file carries.
SERVER_PID = 5150

PARENT_MAP = {CLAUDE_PID: PTY_PID, PTY_PID: 6000, 6000: 1}

#: The Channel Access port a co-deployed stand-in serves on.
STANDIN_PORT = 5074

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


# -- fixtures ---------------------------------------------------------------


@pytest.fixture
def agent_data_root(tmp_path, monkeypatch):
    """Point every resolver at one throwaway agent-data root.

    ``OSPREY_AGENT_DATA_ROOT`` is the single stamp ``target_state.state_dir()``
    and ``session_store`` both prefer, which is exactly why the feature stamps
    it into every session child. Patching ``resolve_shared_data_root`` instead
    would redirect only one of the two — ``session_store`` binds the resolver at
    import — and the other half would write into the repository's own
    ``var/agent_data``.
    """
    root = tmp_path / "agent_data"
    root.mkdir()
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(root))
    session_store.invalidate_cache()
    websocket_routes._reset_session_record_memo()
    yield root
    session_store.invalidate_cache()
    websocket_routes._reset_session_record_memo()


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_watch"
    ws.mkdir()
    return ws


@pytest.fixture
def config_path(tmp_path):
    """A render carrying all three control targets.

    ``epics`` is the facility's own machine, ``live_standin`` the co-deployed
    stand-in, ``virtual_accelerator`` the simulator — three separate connector
    blocks, therefore three separate targets, which is what makes
    ``configured_targets`` answer with three names.
    """
    gateway = {"address": "gw", "port": 5064, "use_name_server": True}
    standin_gateway = {"address": "localhost", "port": STANDIN_PORT, "use_name_server": True}
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "control_system": {
                    "type": "live_standin",
                    "writes_enabled": False,
                    "connector": {
                        "epics": {
                            "gateways": {
                                "read_only": dict(gateway),
                                "write_access": dict(gateway),
                            }
                        },
                        "live_standin": {
                            "gateways": {
                                "read_only": dict(standin_gateway),
                                "write_access": dict(standin_gateway),
                            }
                        },
                        "virtual_accelerator": {"simulation_file": "data/sim.json"},
                    },
                },
                "services": {"live_standin": {"port": STANDIN_PORT}},
                "deployed_services": ["virtual_accelerator", "live_standin"],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def client(agent_data_root, workspace_dir, config_path):
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
    """Make the registry report a running PTY with *pid* for *session_id*."""
    registry = client.app.state.pty_registry
    with patch.object(
        registry,
        "get_session",
        side_effect=lambda sid: SimpleNamespace(pid=pid) if sid == session_id else None,
    ):
        yield


@contextmanager
def synthetic_process_tree(parent_map=None):
    """Replace the ancestor walk's one syscall seam with a fixed parent map."""
    tree = PARENT_MAP if parent_map is None else parent_map
    with patch.object(target_banner, "_parent_pid", side_effect=lambda pid: tree.get(int(pid))):
        yield


@contextmanager
def only_alive(*pids):
    """Report exactly *pids* as running processes."""
    wanted = {int(p) for p in pids}
    with patch.object(target_state, "is_process_alive", side_effect=lambda pid: int(pid) in wanted):
        yield


@contextmanager
def live_session(client, session_id=SESSION_A, *, target="standin", server_pid=SERVER_PID):
    """A session whose controls server has published a live state record."""
    write_target_state(target=target, owner_ppid=PTY_PID, server_pid=server_pid)
    with attached_pty(client, session_id), synthetic_process_tree(), only_alive(server_pid):
        yield


def state_dir() -> Path:
    return target_state.state_dir()


def write_target_state(*, target, owner_ppid, server_pid, targets=TARGET_META):
    """Publish one controls-server state record under the stamped root."""
    directory = state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{target_state.STATE_FILE_PREFIX}{server_pid}.json"
    path.write_text(
        json.dumps(
            {
                "target": target,
                "generation": 1,
                "server_pid": server_pid,
                "owner_ppid": owner_ppid,
                "targets": targets,
                "children": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def request_path(server_pid=SERVER_PID) -> Path:
    return state_dir() / f"{target_state.REQUEST_FILE_PREFIX}{server_pid}.json"


def write_request_file(*, server_pid=SERVER_PID, target="va", age_s=0.0, request_id="pending-1"):
    """Plant a request file *age_s* seconds old."""
    created = datetime.now(UTC) - timedelta(seconds=age_s)
    path = request_path(server_pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "request_id": request_id,
                "target": target,
                "server_pid": server_pid,
                "created_at": created.isoformat(),
                "requested_by": "someone",
            }
        ),
        encoding="utf-8",
    )
    return path


def post_target(client, session_id=SESSION_A, target="va"):
    return client.post(
        "/api/terminal/target",
        json={"session_id": session_id, "target": target},
    )


# -- the refusal ladder -----------------------------------------------------


class TestGrammar:
    """400 before anything is read: the id and the target name are identifiers."""

    @pytest.mark.parametrize(
        "bad_id",
        ["../../etc/passwd", "operator-deadbeef", "", "AAAAAAAA-1111-2222-3333-444444444444"],
    )
    def test_a_session_id_outside_the_closed_grammar_is_400(self, client, bad_id):
        resp = post_target(client, session_id=bad_id)
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "invalid_session_id"

    def test_a_target_this_build_never_heard_of_is_400(self, client):
        with live_session(client):
            resp = post_target(client, target="banana")
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "unknown_target"

    def test_a_target_this_deployment_did_not_configure_is_400(self, client, tmp_path):
        """``va`` is a real target name; a render without its block has no such row.

        The vocabulary is ``configured_targets``, never ``CONTROL_TARGETS`` —
        writing a request for a machine the deployment never described would
        put the reconciler in the position of refusing a switch nobody could
        have meant.
        """
        path = tmp_path / "va-less.yml"
        path.write_text(
            yaml.safe_dump({"control_system": {"type": "epics", "connector": {"epics": {}}}}),
            encoding="utf-8",
        )
        client.app.state.config_path = path
        with live_session(client, target="live"):
            resp = post_target(client, target="va")
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "unknown_target"

    def test_a_body_missing_a_field_is_422(self, client):
        assert client.post("/api/terminal/target", json={"target": "va"}).status_code == 422
        assert (
            client.post("/api/terminal/target", json={"session_id": SESSION_A}).status_code == 422
        )

    def test_a_refused_grammar_writes_no_request(self, client):
        with live_session(client):
            post_target(client, target="banana")
        assert not request_path().exists()


class TestSessionNotStarted:
    """409 when nothing has published a record this session owns."""

    def test_no_pty_at_all_is_409(self, client):
        with synthetic_process_tree(), only_alive():
            resp = post_target(client)
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "session_not_started"

    def test_a_pty_with_no_published_record_is_409(self, client):
        with attached_pty(client, SESSION_A), synthetic_process_tree(), only_alive():
            resp = post_target(client)
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "session_not_started"

    def test_a_record_owned_by_another_session_is_409(self, client):
        """The ancestor walk is the ownership rule; a stranger's record is not ours."""
        write_target_state(target="standin", owner_ppid=999_001, server_pid=SERVER_PID)
        with attached_pty(client, SESSION_A), synthetic_process_tree(), only_alive(SERVER_PID):
            resp = post_target(client)
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "session_not_started"

    def test_a_dead_controls_server_is_409(self, client):
        write_target_state(target="standin", owner_ppid=PTY_PID, server_pid=SERVER_PID)
        with attached_pty(client, SESSION_A), synthetic_process_tree(), only_alive():
            resp = post_target(client)
        assert resp.status_code == 409

    def test_a_refused_session_writes_no_request(self, client):
        with synthetic_process_tree(), only_alive():
            post_target(client)
        assert not request_path().exists()


class TestStoreUnavailable:
    """503 when the request has nowhere to land — never a silent success."""

    def test_an_unresolvable_state_dir_is_503(self, client):
        with (
            live_session(client),
            patch.object(target_state, "state_dir", side_effect=RuntimeError("no root")),
        ):
            resp = post_target(client)
        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "store_unavailable"

    def test_a_failing_write_is_503(self, client):
        with (
            live_session(client),
            patch.object(target_state, "write_request", side_effect=OSError("read-only fs")),
        ):
            resp = post_target(client)
        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "store_write_failed"
        assert not request_path().exists()


class TestRequestPending:
    """409 while one gesture is still outstanding; a stale one is not a gesture."""

    def test_a_fresh_request_blocks_a_second_one(self, client):
        write_request_file(age_s=1.0, target="live")
        with live_session(client):
            resp = post_target(client, target="va")
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "request_pending"
        # The pending request is untouched — this route never overwrites one.
        assert json.loads(request_path().read_text(encoding="utf-8"))["request_id"] == "pending-1"

    def test_an_expired_request_does_not_block(self, client):
        write_request_file(age_s=target_state.REQUEST_TTL_S + 5, target="live")
        with live_session(client):
            resp = post_target(client, target="va")
        assert resp.status_code == 202
        written = json.loads(request_path().read_text(encoding="utf-8"))
        assert written["target"] == "va"
        assert written["request_id"] == resp.json()["request_id"]

    def test_a_request_addressed_to_another_server_does_not_block(self, client):
        """Requests are addressed by pid; another server's file is not ours."""
        write_request_file(server_pid=SERVER_PID + 1, age_s=1.0)
        with live_session(client):
            resp = post_target(client, target="va")
        assert resp.status_code == 202

    def test_a_request_overwritten_by_a_concurrent_post_is_409(self, client):
        """The freshness read and the write are not one atomic step.

        Two operators clicking Switch in the same moment both pass the "is one
        pending?" read — there is an ``await`` between it and the write — and
        ``os.replace`` cannot refuse a slot somebody else owns. So the write is
        read back, and the one that did not survive is told what the earlier
        read would have told it: a request is pending. Exactly one request
        lives, and neither operator is left watching a ``request_id`` no file
        carries.
        """
        real_write = target_state._atomic_write_json

        def _overwritten(path, payload):
            real_write(path, payload)
            # The competing POST's write lands last, exactly as it would.
            real_write(path, {**payload, "request_id": "the-other-operator"})

        with (
            live_session(client),
            patch.object(target_state, "_atomic_write_json", side_effect=_overwritten),
        ):
            resp = post_target(client, target="va")

        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "request_pending"
        # The winner's request is intact — the loser removed nothing.
        assert json.loads(request_path().read_text(encoding="utf-8"))["request_id"] == (
            "the-other-operator"
        )

    def test_an_uncontested_write_reads_itself_back(self, client):
        """The read-back must not turn ordinary writes into false collisions."""
        with live_session(client):
            resp = post_target(client, target="va")
        assert resp.status_code == 202
        assert (
            json.loads(request_path().read_text(encoding="utf-8"))["request_id"]
            == (resp.json()["request_id"])
        )


# -- the accepted gesture ---------------------------------------------------


class TestAcceptedRequest:
    def test_a_switch_is_accepted_with_a_request_id(self, client):
        with live_session(client):
            resp = post_target(client, target="va")
        assert resp.status_code == 202
        body = resp.json()
        assert body["session_id"] == SESSION_A
        assert body["target"] == "va"
        assert body["request_id"]

    def test_the_written_file_is_addressed_by_the_records_server_pid(self, client):
        """The address is the record's pid, never this web server's own."""
        with live_session(client, server_pid=6321):
            resp = post_target(client, target="va")
        assert resp.status_code == 202
        path = request_path(6321)
        assert path.exists()
        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["server_pid"] == 6321
        assert written["request_id"] == resp.json()["request_id"]
        assert written["target"] == "va"

    def test_the_request_carries_a_created_at_and_a_requester(self, client):
        """Both are what makes the request ageable and attributable."""
        with live_session(client):
            post_target(client, target="va")
        written = json.loads(request_path().read_text(encoding="utf-8"))
        assert target_state.is_request_fresh(written)
        assert written["requested_by"]

    def test_every_request_id_is_fresh(self, client):
        with live_session(client):
            first = post_target(client, target="va").json()["request_id"]
            request_path().unlink()
            second = post_target(client, target="live").json()["request_id"]
        assert first != second

    def test_the_active_target_is_not_pre_refused(self, client):
        """No availability pre-check: ``already_active`` is the reconciler's word.

        The gate is re-evaluated immediately before the switch, inside the
        server that owns the connector. Answering here from a snapshot taken a
        moment earlier is how a route and a gate come to disagree.
        """
        with live_session(client, target="standin"):
            resp = post_target(client, target="standin")
        assert resp.status_code == 202
        assert json.loads(request_path().read_text(encoding="utf-8"))["target"] == "standin"

    def test_an_unreachable_target_is_not_pre_refused(self, client):
        """Nothing here probes; ``live`` is accepted with no gateway reachable."""
        with live_session(client):
            resp = post_target(client, target="live")
        assert resp.status_code == 202


# -- audit ------------------------------------------------------------------


class TestAudit:
    def test_an_accepted_gesture_files_exactly_one_record(self, client, ledger):
        with live_session(client):
            assert post_target(client, target="va").status_code == 202
        assert len(ledger) == 1
        record = ledger[0]
        assert record["subject"] == "control_target_set"
        assert record["decision"] == "allowed"
        assert record["session"] == SESSION_A

    def test_the_record_names_the_target_and_the_request(self, client, ledger):
        with live_session(client):
            request_id = post_target(client, target="va").json()["request_id"]
        detail = ledger[0]["detail"]
        assert "va" in detail
        assert request_id in detail

    def test_a_refusal_files_exactly_one_record_too(self, client, ledger):
        write_request_file(age_s=1.0)
        with live_session(client):
            assert post_target(client, target="va").status_code == 409
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "refused"
        assert ledger[0]["reason"] == "request_pending"

    def test_a_malformed_session_id_still_leaves_exactly_one_record(self, client, ledger):
        """The one refusal the route does NOT file itself is still filed once.

        ``_require_session_uuid`` runs before there is a legitimate key to put
        in the envelope's ``session`` field, so the route does not record it and
        ``HttpAuditMiddleware`` files its own ``route_refused`` line instead.
        A refused request leaves one line, never two and never none.
        """
        with live_session(client):
            assert post_target(client, session_id="../../etc/passwd").status_code == 400
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "refused"

    def test_the_record_joins_on_the_spawn_key(self, client, ledger):
        """A rekeyed session's gesture is filed under the key its child exported.

        The running child cannot have ``OSPREY_POSTURE_SESSION`` rewritten
        without being killed, so every record it emits carries the spawn key. A
        gesture filed under the *current* key would split one session into two
        unrelated actors in the ledger.
        """
        registry = client.app.state.pty_registry
        with (
            patch.object(registry, "audit_session_key", side_effect=lambda key: SESSION_B),
            live_session(client),
        ):
            assert post_target(client, target="va").status_code == 202
        assert ledger[0]["session"] == SESSION_B
