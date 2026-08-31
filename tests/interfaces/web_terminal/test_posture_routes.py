"""Tests for ``POST /api/terminal/posture`` — the per-target write posture.

The posture is the operator's per-(session, target) write toggle: narrow the
stand-in to read-only while the virtual accelerator keeps writing, and have the
session's *running* agent obey it on its very next write. Three properties
shape everything below.

* **It is enforced at write time, not at spawn time.** The store is the truth,
  read live by the connector's reference monitor, the executor's gates and the
  write hook. Nothing here terminates or respawns anything — a POST that killed
  the child to apply a posture would throw away the conversation for a toggle.
* **It narrows and never widens.** ``writes`` on a target the render does not
  arm is a ``403`` naming that target's own ``writes_enabled`` key, not the
  deployment-wide union: on a mixed render the union is true while the machine
  the operator is pointed at refuses every write.
* **The persist is the commit point.** The file lands first; memory follows
  only once it has. A store that could not be written is a ``503`` and a
  posture that did not change, never a badge that shows a narrowing the agent
  is not in.

Harness mirrors ``test_target_request_route.py``: one app per test through
``create_app``, entered as a ``TestClient`` context manager so the lifespan
runs, over an ``OSPREY_AGENT_DATA_ROOT`` stamped at a throwaway directory.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

from osprey.audit import writer as audit_writer
from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey.mcp_server.control_system import target_state
from osprey_connectors import session_store

SESSION_A = "aaaaaaaa-1111-2222-3333-444444444444"
SESSION_B = "bbbbbbbb-1111-2222-3333-444444444444"
#: A chat-pool key, minted the way the shipped client mints one
#: (``crypto.randomUUID()`` in static/js/chat.js): a bare lowercase UUID.
CHAT_A = "cccccccc-1111-2222-3333-444444444444"

PTY_PID = 7000

#: The Channel Access port a co-deployed stand-in serves on.
STANDIN_PORT = 5074

#: A pid no kernel hands out: the largest a 32-bit ``pid_t`` holds.
DEAD_PID = 2_147_483_646


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

    ``OSPREY_AGENT_DATA_ROOT`` is the single stamp ``session_store`` and
    ``target_state.state_dir()`` both prefer — the stamp this feature puts in
    every session child's environment. Patching ``resolve_shared_data_root``
    instead would redirect only one of them: ``session_store`` binds the
    resolver at import, so the other half would write into the repository's own
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
def make_client(agent_data_root, workspace_dir, tmp_path):
    """Build an app + TestClient, repeatably, over the same stamped root."""

    @contextmanager
    def _make(config_path=None):
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

    Patches ``osprey.audit.writer.record``, which both recorders resolve at
    call time — the route's ``record_and_mark`` and
    ``HttpAuditMiddleware._emit_audit_record`` — so the count is the true
    number of ledger lines a POST produces.
    """
    records: list[dict] = []

    def _record(**fields):
        records.append(fields)
        return Path("/dev/null/ledger.jsonl")

    with patch.object(audit_writer, "record", side_effect=_record):
        yield records


# -- harness ----------------------------------------------------------------


@contextmanager
def known_sessions(*session_ids):
    """Make ``SessionDiscovery`` report *session_ids* as started on disk.

    The posture route no longer consults the discovery walk, but the GET
    surface and the terminate/respawn paths still do; pinning it keeps every
    test's disk state explicit either way.
    """
    with patch(
        "osprey.interfaces.web_terminal.session_discovery.SessionDiscovery.snapshot_session_ids",
        return_value=set(session_ids),
    ):
        yield


class _RecordingChatPool:
    """An ``operator_registry`` that answers to one chat key and records teardown.

    The three facades the posture surface reaches for on a chat key —
    ``has_chat_key`` and ``get_chat_session`` to decide the key is addressable,
    ``terminate_chat_session`` to tear the child down. Only the last is a
    regression: it is recorded rather than raising, so a POST that calls it
    fails on the assertion that names the contract instead of on a stray error.
    """

    def __init__(self, key: str = CHAT_A):
        self.key = key
        self.terminated: list[str] = []

    def has_chat_key(self, session_id: str) -> bool:
        return session_id == self.key

    def get_chat_session(self, session_id: str):
        return object() if session_id == self.key else None

    async def terminate_chat_session(self, session_id: str) -> None:
        self.terminated.append(session_id)

    async def cleanup_all(self) -> None:
        """The app's own shutdown hook, not a teardown the POST could reach.

        Deliberately not recorded: it runs when the TestClient's lifespan
        exits, long after the assertion, and recording it would make every
        case fail.
        """


def store_file(root: Path) -> Path:
    return root / session_store.STATE_DIR_NAME / session_store.STORE_FILENAME


def read_store(root: Path):
    path = store_file(root)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_inflight_marker(root: Path, *, pid, target="standin"):
    """Plant one execution marker, as the python executor writes it."""
    directory = root / target_state.STATE_DIR_NAME
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
                "owner_ppid": PTY_PID,
                "started_at": "2026-08-30T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return path


@contextmanager
def store_outage(monkeypatch):
    """Break the posture store's location the ONE way that counts.

    The store path is ``session_store.store_path()``, and ``session_store``
    resolves it from :data:`session_store.AGENT_DATA_ROOT_ENV_VAR` first and
    its OWN import-bound ``resolve_shared_data_root`` second. An outage helper
    that patched ``osprey_connectors.workspace.resolve_shared_data_root``
    instead would patch a name this module never consults and simulate nothing
    at all — which is exactly what the previous version of this file did, while
    its outage test passed.

    So the helper **asserts its own premise**: inside the block the store must
    genuinely have no location. A future refactor that moves the resolution
    breaks this loudly instead of quietly turning every test below into a
    no-op. It also guarantees no test here can fall through to the repository's
    own ``var/agent_data`` — with the resolver raising, there is nothing to
    fall through to.
    """
    stamped = os.environ.get(session_store.AGENT_DATA_ROOT_ENV_VAR)
    monkeypatch.delenv(session_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    session_store.invalidate_cache()
    try:
        with patch.object(
            session_store,
            "resolve_shared_data_root",
            side_effect=RuntimeError("config unreadable"),
        ):
            assert session_store.store_path() is None, (
                "store_outage() simulated nothing: the store still resolves a path"
            )
            yield
    finally:
        # The stamp is restored HERE, not left to monkeypatch's teardown. An
        # outage that ends at the end of the *test* rather than at the end of
        # the block would leave everything after it resolving through the real
        # ``resolve_shared_data_root`` — i.e. writing the recovery half of
        # these tests into the repository's own ``var/agent_data``. The
        # recovery assertions are the whole point, so the recovery must land
        # back on the tmp root.
        if stamped is not None:
            monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, stamped)
        session_store.invalidate_cache()


def reset_posture_memory(app):
    """Forget the loaded store so the next access re-reads it from disk."""
    app.state.session_postures = None
    app.state.session_postures_provisional = False


@contextmanager
def only_alive(*pids):
    """Report exactly *pids* as running processes."""
    wanted = {int(p) for p in pids}
    with patch.object(target_state, "is_process_alive", side_effect=lambda pid: int(pid) in wanted):
        yield


def post_posture(client, *, session_id=SESSION_A, target="standin", posture="sandbox"):
    return client.post(
        "/api/terminal/posture",
        json={"session_id": session_id, "target": target, "posture": posture},
    )


# -- the refusal ladder -----------------------------------------------------


class TestGrammar:
    @pytest.mark.parametrize(
        "bad_id",
        ["../../etc/passwd", "operator-deadbeef", "", "AAAAAAAA-1111-2222-3333-444444444444"],
    )
    def test_an_id_outside_the_closed_grammar_is_400(self, client, bad_id):
        with known_sessions(SESSION_A):
            resp = post_posture(client, session_id=bad_id)
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "invalid_session_id"

    @pytest.mark.parametrize("bad", ["readonly", "SANDBOX", "", "readwrite", "true"])
    def test_only_the_two_named_postures_are_accepted(self, client, bad):
        with known_sessions(SESSION_A):
            assert post_posture(client, posture=bad).status_code == 422

    def test_a_body_missing_a_field_is_422(self, client):
        for body in (
            {"posture": "sandbox", "target": "standin"},
            {"session_id": SESSION_A, "target": "standin"},
            {"session_id": SESSION_A, "posture": "sandbox"},
        ):
            assert client.post("/api/terminal/posture", json=body).status_code == 422

    def test_a_target_this_deployment_does_not_configure_is_400(self, client):
        with known_sessions(SESSION_A):
            resp = post_posture(client, target="banana")
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "unknown_target"

    def test_all_plus_writes_is_400(self, client, agent_data_root):
        """Widening is per target, always.

        ``[ Sandbox everything ]`` is one gesture because narrowing everything
        is unambiguous; there is no matching "arm everything", because each
        target's ceiling is its own and an operator arming three machines at
        once could not have meant all three.
        """
        with known_sessions(SESSION_A):
            resp = post_posture(client, target="all", posture="writes")
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "writes_requires_one_target"
        assert read_store(agent_data_root) is None

    @pytest.mark.parametrize(
        ("target", "posture"),
        [("all", "sandbox"), ("standin", "sandbox"), ("standin", "writes")],
    )
    def test_an_unreadable_render_configures_no_target(self, client, target, posture):
        """No readable config, no vocabulary — and therefore nothing to toggle.

        ``configured_targets`` is what every row, probe and refusal here
        enumerates; a server that cannot read its own render does not know
        which machines exist. ``all`` in particular must not fall through: over
        an empty vocabulary it would CLEAR the entry rather than narrow it,
        which is the one direction this store must never move by accident.
        """
        client.app.state.config_path = None
        with known_sessions(SESSION_A):
            resp = post_posture(client, target=target, posture=posture)
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "unknown_target"


class TestUnspokenSessions:
    """The posture never depends on whether the operator has talked to the agent.

    Both spawn paths read the store at spawn, so a narrowing recorded before
    the first prompt binds that session's very first write — and the store
    only narrows, so an entry under a key nothing spawns is inert. The gate
    that refused these gestures with "send one prompt first" guarded nothing
    and is gone.
    """

    def test_a_key_with_no_session_behind_it_still_narrows(self, client, agent_data_root):
        with known_sessions():  # nothing on disk, nothing pooled
            resp = post_posture(client)
        assert resp.status_code == 200
        assert read_store(agent_data_root) == {SESSION_A: {"standin": "sandbox"}}

    def test_a_fresh_chat_key_narrows_before_its_first_prompt(self, client, agent_data_root):
        """The page mints its chat id at load; the toggle must work from then on."""
        with known_sessions():
            resp = post_posture(client, session_id=CHAT_A, target="va")
        assert resp.status_code == 200
        assert read_store(agent_data_root) == {CHAT_A: {"va": "sandbox"}}


class TestStoreUnavailable:
    """503, and a posture that did not change — never a badge that lies."""

    def test_an_unresolvable_store_is_503(self, client, monkeypatch):
        with known_sessions(SESSION_A), store_outage(monkeypatch):
            resp = post_posture(client)
        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "store_unavailable"

    def test_a_failing_write_is_503_and_leaves_memory_alone(self, client, agent_data_root):
        """The write is the commit point: a failed one changes nothing at all."""
        with known_sessions(SESSION_A):
            assert post_posture(client, target="standin").status_code == 200
            before = dict(client.app.state.session_postures)
            with patch.object(websocket_routes, "_write_store", side_effect=OSError("read-only")):
                resp = post_posture(client, target="va")
        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "store_write_failed"
        assert client.app.state.session_postures == before
        assert read_store(agent_data_root) == {SESSION_A: {"standin": "sandbox"}}


class TestStoreOutageAndRecovery:
    """A load taken with no location is provisional, and recovery merges.

    ``_session_postures`` caches the store on ``app.state`` after the first
    read. Caching a load taken while the store had *no location* would let one
    transient config failure outlive itself: every later read would serve an
    empty store and report a narrowed session as unnarrowed — a silent revert
    to writes, which is the exact failure persisting the store exists to
    prevent. So that load is marked provisional and retried.
    """

    def test_a_load_with_no_location_is_provisional_and_empty(
        self, client, agent_data_root, monkeypatch
    ):
        path = store_file(agent_data_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({SESSION_A: {"standin": "sandbox"}}), encoding="utf-8")
        reset_posture_memory(client.app)

        with store_outage(monkeypatch):
            assert websocket_routes._session_postures(client.app) == {}
            assert client.app.state.session_postures_provisional is True

    def test_the_provisional_load_is_retried_once_the_store_comes_back(
        self, client, agent_data_root, monkeypatch
    ):
        """The narrowing on disk must not stay invisible after the outage ends."""
        path = store_file(agent_data_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({SESSION_A: {"standin": "sandbox"}}), encoding="utf-8")
        reset_posture_memory(client.app)

        with store_outage(monkeypatch):
            websocket_routes._session_postures(client.app)

        assert websocket_routes._session_postures(client.app) == {SESSION_A: {"standin": "sandbox"}}
        assert client.app.state.session_postures_provisional is False
        assert websocket_routes._posture_entry(client.app, SESSION_A) == {"standin": "sandbox"}

    def test_a_narrowing_held_only_in_memory_wins_over_the_recovery_read(
        self, client, agent_data_root, monkeypatch
    ):
        """Memory wins on overlap; everything it was never told about survives.

        The recovery read is authoritative for keys memory has not heard of,
        but a narrowing that exists only in memory would be quietly undone by
        it — and undoing a narrowing is the one direction this store must never
        move on its own.
        """
        path = store_file(agent_data_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({SESSION_A: {"va": "sandbox"}, SESSION_B: {"live": "sandbox"}}),
            encoding="utf-8",
        )
        reset_posture_memory(client.app)

        with store_outage(monkeypatch):
            held = websocket_routes._session_postures(client.app)
            held[SESSION_A] = {"standin": "sandbox"}

        assert websocket_routes._session_postures(client.app) == {
            SESSION_A: {"standin": "sandbox"},
            SESSION_B: {"live": "sandbox"},
        }

    def test_a_toggle_refused_during_the_outage_lands_after_it(
        self, client, agent_data_root, monkeypatch
    ):
        """End to end: 503 while it lasts, 200 after, disk intact throughout."""
        path = store_file(agent_data_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({SESSION_B: {"live": "sandbox"}}), encoding="utf-8")
        reset_posture_memory(client.app)

        with known_sessions(SESSION_A), store_outage(monkeypatch):
            assert post_posture(client, target="standin").status_code == 503
        assert read_store(agent_data_root) == {SESSION_B: {"live": "sandbox"}}

        with known_sessions(SESSION_A):
            assert post_posture(client, target="standin").status_code == 200
        assert read_store(agent_data_root) == {
            SESSION_A: {"standin": "sandbox"},
            SESSION_B: {"live": "sandbox"},
        }


class TestCeiling:
    """403 ``writes_disabled``, per target, naming that target's own key."""

    def test_widening_an_unarmed_target_is_403_naming_its_key(self, client, tmp_path):
        client.app.state.config_path = write_config(tmp_path, control_system_section())
        with known_sessions(SESSION_A):
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
        with known_sessions(SESSION_A):
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
        with known_sessions(SESSION_A):
            resp = post_posture(client, target="standin", posture="writes")
        assert resp.status_code == 200

    def test_narrowing_needs_no_ceiling(self, client, tmp_path):
        """A target nothing arms can still be narrowed; narrowing grants nothing."""
        client.app.state.config_path = write_config(tmp_path, control_system_section())
        with known_sessions(SESSION_A):
            assert post_posture(client, target="live", posture="sandbox").status_code == 200

    def test_all_sandbox_ignores_the_ceiling(self, client, tmp_path):
        client.app.state.config_path = write_config(tmp_path, control_system_section())
        with known_sessions(SESSION_A):
            assert post_posture(client, target="all", posture="sandbox").status_code == 200


class TestSelectedRoleMissing:
    """409 when narrowing would leave the target with no gateway to select."""

    def test_a_write_access_only_target_cannot_be_narrowed(self, client, tmp_path):
        row = {"address": "localhost", "port": STANDIN_PORT, "use_name_server": True}
        client.app.state.config_path = write_config(
            tmp_path, control_system_section(standin_gateways={"write_access": row})
        )
        with known_sessions(SESSION_A):
            resp = post_posture(client, target="standin", posture="sandbox")
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["error"] == "selected_role_missing"
        assert "control_system.connector.live_standin.gateways.read_only" in detail["message"]

    def test_the_other_targets_are_unaffected(self, client, tmp_path):
        row = {"address": "localhost", "port": STANDIN_PORT, "use_name_server": True}
        client.app.state.config_path = write_config(
            tmp_path, control_system_section(standin_gateways={"write_access": row})
        )
        with known_sessions(SESSION_A):
            assert post_posture(client, target="va", posture="sandbox").status_code == 200

    def test_sandbox_everything_narrows_the_rest_and_reports_what_it_skipped(
        self, client, tmp_path, agent_data_root
    ):
        """ "Everything" means everything it can, and says so — not nothing.

        Refusing the whole gesture because one target is write_access-only
        would leave "Sandbox everything" doing nothing at all on that
        deployment, while the popover offers the button. The other machines
        are narrowed, and the one that stayed writable is named with its
        reason so the operator is never told a narrowing happened that did not.
        """
        row = {"address": "localhost", "port": STANDIN_PORT, "use_name_server": True}
        client.app.state.config_path = write_config(
            tmp_path, control_system_section(standin_gateways={"write_access": row})
        )
        with known_sessions(SESSION_A):
            resp = post_posture(client, target="all", posture="sandbox")

        assert resp.status_code == 200
        body = resp.json()
        assert body["entry"] == {"live": "sandbox", "va": "sandbox"}
        assert [row_["target"] for row_ in body["skipped"]] == ["standin"]
        assert body["skipped"][0]["reason"] == "selected_role_missing"
        assert "gateways.read_only" in body["skipped"][0]["detail"]
        assert read_store(agent_data_root) == {SESSION_A: {"live": "sandbox", "va": "sandbox"}}

    def test_a_named_target_that_cannot_narrow_still_refuses(self, client, tmp_path):
        """The single-target 409 is unchanged: that one target IS the request."""
        row = {"address": "localhost", "port": STANDIN_PORT, "use_name_server": True}
        client.app.state.config_path = write_config(
            tmp_path, control_system_section(standin_gateways={"write_access": row})
        )
        with known_sessions(SESSION_A):
            resp = post_posture(client, target="standin", posture="sandbox")
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "selected_role_missing"

    def test_an_already_sandboxed_target_does_not_veto_a_later_gesture(
        self, client, tmp_path, agent_data_root
    ):
        """``narrowing_refusal`` is store-blind, so only CHANGING targets are asked.

        A target narrowed while the deployment could still derive a read_only
        gateway keeps its entry when that gateway later disappears from the
        render. Asking about it again would have it refuse on behalf of a
        change nobody requested — freezing every later toggle on the session.
        """
        client.app.state.config_path = write_config(tmp_path, control_system_section())
        with known_sessions(SESSION_A):
            assert post_posture(client, target="standin", posture="sandbox").status_code == 200

        # The stand-in's read_only gateway goes away under the running session.
        row = {"address": "localhost", "port": STANDIN_PORT, "use_name_server": True}
        client.app.state.config_path = write_config(
            tmp_path,
            control_system_section(standin_gateways={"write_access": row}),
            name="config2.yml",
        )
        with known_sessions(SESSION_A):
            resp = post_posture(client, target="all", posture="sandbox")

        assert resp.status_code == 200
        assert resp.json()["skipped"] == []
        assert read_store(agent_data_root) == {
            SESSION_A: {"standin": "sandbox", "live": "sandbox", "va": "sandbox"}
        }

    def test_a_clean_render_skips_nothing(self, client):
        with known_sessions(SESSION_A):
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
        with known_sessions(SESSION_A):
            assert post_posture(client, target="standin", posture="writes").status_code == 200


class TestExecutionInFlight:
    """A run that started narrow is never widened out from under itself."""

    def test_widening_under_a_live_marker_is_409(self, client, tmp_path, agent_data_root):
        client.app.state.config_path = write_config(
            tmp_path, control_system_section(va_writes=True)
        )
        write_inflight_marker(agent_data_root, pid=4242, target="standin")
        with known_sessions(SESSION_A), only_alive(4242):
            resp = post_posture(client, target="va", posture="writes")
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["error"] == "execution_in_flight"
        # The switch tool's own words, so the popover and the agent's answer read alike.
        assert "in flight on target" in detail["message"].lower()
        assert "standin" in detail["message"]
        assert "wait" in detail["message"].lower()

    def test_the_refusal_does_not_guess_whose_run_it_is(self, client, tmp_path, agent_data_root):
        """The tool decides "whose" from os.getppid(), which the web server is not.

        Off the agent's process tree that comparison always answers "another
        session", so the operator whose OWN session is running the execution
        would be told it belongs to somebody else. The clause is dropped rather
        than answered wrongly.
        """
        client.app.state.config_path = write_config(
            tmp_path, control_system_section(va_writes=True)
        )
        write_inflight_marker(agent_data_root, pid=4242, target="standin")
        with known_sessions(SESSION_A), only_alive(4242):
            resp = post_posture(client, target="va", posture="writes")
        message = resp.json()["detail"]["message"]
        assert "belongs to" not in message
        assert "another session" not in message.lower()

    def test_narrowing_under_a_live_marker_still_lands(self, client, agent_data_root):
        """Narrowing is always safe; it is the gesture an operator needs most."""
        write_inflight_marker(agent_data_root, pid=4242, target="standin")
        with known_sessions(SESSION_A), only_alive(4242):
            resp = post_posture(client, target="standin", posture="sandbox")
        assert resp.status_code == 200
        assert read_store(agent_data_root) == {SESSION_A: {"standin": "sandbox"}}

    def test_a_marker_whose_writer_is_gone_does_not_block(self, client, tmp_path, agent_data_root):
        """A killed executor's residue is swept, not treated as a running run."""
        client.app.state.config_path = write_config(
            tmp_path, control_system_section(va_writes=True)
        )
        marker = write_inflight_marker(agent_data_root, pid=DEAD_PID, target="standin")
        with known_sessions(SESSION_A), only_alive():
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
        with known_sessions(SESSION_A), only_alive(4242):
            assert post_posture(client, target="va", posture="writes").status_code == 409


# -- the accepted gesture ---------------------------------------------------


class TestAcceptedPosture:
    def test_a_narrowing_lands_in_memory_and_on_disk(self, client, agent_data_root):
        with known_sessions(SESSION_A):
            resp = post_posture(client, target="standin", posture="sandbox")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == SESSION_A
        assert body["target"] == "standin"
        assert body["posture"] == "sandbox"
        assert body["entry"] == {"standin": "sandbox"}
        assert client.app.state.session_postures[SESSION_A] == {"standin": "sandbox"}
        assert read_store(agent_data_root) == {SESSION_A: {"standin": "sandbox"}}

    def test_the_store_is_co_sited_with_the_state_file(self, client, agent_data_root):
        """One directory for the store and the target state, by one rule."""
        with known_sessions(SESSION_A):
            post_posture(client)
        assert store_file(agent_data_root).parent == target_state.state_dir()

    def test_targets_narrow_independently(self, client, agent_data_root):
        with known_sessions(SESSION_A):
            post_posture(client, target="standin", posture="sandbox")
            resp = post_posture(client, target="va", posture="sandbox")
        assert resp.json()["entry"] == {"standin": "sandbox", "va": "sandbox"}
        assert read_store(agent_data_root) == {SESSION_A: {"standin": "sandbox", "va": "sandbox"}}

    def test_widening_removes_only_that_target(self, client, tmp_path, agent_data_root):
        client.app.state.config_path = write_config(
            tmp_path, control_system_section(va_writes=True)
        )
        with known_sessions(SESSION_A):
            post_posture(client, target="standin", posture="sandbox")
            post_posture(client, target="va", posture="sandbox")
            resp = post_posture(client, target="va", posture="writes")
        assert resp.json()["entry"] == {"standin": "sandbox"}
        assert read_store(agent_data_root) == {SESSION_A: {"standin": "sandbox"}}

    def test_the_last_widening_clears_the_key(self, client, tmp_path, agent_data_root):
        """Absence is how this store spells ``writes``; a stored ``{}`` is not."""
        client.app.state.config_path = write_config(
            tmp_path, control_system_section(standin_writes=True)
        )
        with known_sessions(SESSION_A):
            post_posture(client, target="standin", posture="sandbox")
            resp = post_posture(client, target="standin", posture="writes")
        assert resp.json()["entry"] == {}
        assert read_store(agent_data_root) == {}

    def test_sandbox_everything_narrows_every_configured_target(self, client, agent_data_root):
        with known_sessions(SESSION_A):
            resp = post_posture(client, target="all", posture="sandbox")
        assert resp.status_code == 200
        assert resp.json()["entry"] == {
            "live": "sandbox",
            "va": "sandbox",
            "standin": "sandbox",
        }

    def test_sessions_do_not_disturb_each_other(self, client, agent_data_root):
        with known_sessions(SESSION_A, SESSION_B):
            post_posture(client, session_id=SESSION_A, target="standin")
            post_posture(client, session_id=SESSION_B, target="va")
        assert read_store(agent_data_root) == {
            SESSION_A: {"standin": "sandbox"},
            SESSION_B: {"va": "sandbox"},
        }


class TestNoTermination:
    """The POST never respawns the session — the whole point of the feature."""

    def test_the_live_child_survives_a_narrowing(self, client):
        registry = client.app.state.pty_registry
        registry.get_or_create_session(SESSION_A, "echo")
        assert registry.get_session(SESSION_A) is not None

        with known_sessions(SESSION_A):
            assert post_posture(client, target="standin").status_code == 200

        assert registry.get_session(SESSION_A) is not None

    def test_the_route_never_reaches_for_the_pool_teardown(self, client):
        """Pinned on the registry, not on a helper name.

        The old route applied a posture by terminating the child so the next
        attach respawned it under a fresh environment. Asserting that some
        ``_terminate_for_respawn`` helper is not called would stop meaning
        anything the moment that helper is deleted — which is exactly what the
        env retirement does to it. The durable statement is that this POST
        touches none of the ways the pool lets go of a live PTY session.

        ``terminate_session`` is the one the retired route actually called, so
        it is the one that matters most here; ``terminate_session_if_owner`` is
        pinned beside it because it is the other public teardown a re-adding
        change would reach for, and ``detach_session`` because it drops the
        client's hold.
        """
        registry = client.app.state.pty_registry
        registry.get_or_create_session(SESSION_A, "echo")
        with (
            known_sessions(SESSION_A),
            patch.object(registry, "detach_session") as detach,
            patch.object(registry, "terminate_session") as terminate,
            patch.object(registry, "terminate_session_if_owner") as terminate_if_owner,
        ):
            assert post_posture(client, target="standin").status_code == 200
        detach.assert_not_called()
        terminate.assert_not_called()
        terminate_if_owner.assert_not_called()

    def test_the_chat_child_survives_a_narrowing(self, client, agent_data_root):
        """The chat-key analogue, and the half nothing else covers.

        The retired route also called ``terminate_chat_session`` — a chat has no
        PTY, so the PTY assertions above say nothing about it, and the sibling
        that pins a live chat child (``test_terminate_respawn.py::
        test_a_narrowing_does_not_rebuild_the_chat_child``) writes the store
        directly rather than going through this POST. Without this case a
        regression re-adding a chat teardown to the route would pass the suite.
        """
        recorder = _RecordingChatPool()
        client.app.state.operator_registry = recorder

        resp = post_posture(client, session_id=CHAT_A, target="standin")

        assert resp.status_code == 200
        assert read_store(agent_data_root) == {CHAT_A: {"standin": "sandbox"}}
        assert recorder.terminated == []


class TestPersistence:
    def test_a_narrowing_reloads_in_a_fresh_app(self, make_client, agent_data_root):
        """A container recreation must not silently lift a narrowing."""
        with make_client() as first:
            with known_sessions(SESSION_A):
                assert post_posture(first, target="standin").status_code == 200

        with make_client() as second:
            assert websocket_routes._posture_entry(second.app, SESSION_A) == {"standin": "sandbox"}

    def test_a_corrupt_store_reads_as_no_narrowing(self, make_client, agent_data_root):
        path = store_file(agent_data_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        with make_client() as fresh:
            assert websocket_routes._posture_entry(fresh.app, SESSION_A) == {}
            with known_sessions(SESSION_A):
                assert post_posture(fresh, target="standin").status_code == 200
        assert read_store(agent_data_root) == {SESSION_A: {"standin": "sandbox"}}

    def test_an_unknown_posture_value_is_dropped_on_load(self, make_client, agent_data_root):
        path = store_file(agent_data_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({SESSION_A: {"standin": "readonly", "va": "sandbox"}}), encoding="utf-8"
        )
        with make_client() as fresh:
            assert websocket_routes._posture_entry(fresh.app, SESSION_A) == {"va": "sandbox"}


class TestTopologies:
    """Both spawn surfaces are addressable; neither is respawned."""

    def test_a_live_chat_pool_key_is_addressable(self, client, agent_data_root):
        """A chat session honours the store on its next turn, so it may be set.

        Checking only the on-disk session stems is what made a chat session's
        posture unsettable — a badge the operator could not act on.
        """
        with patch.object(
            websocket_routes, "_chat_pool_answers_to", side_effect=lambda app, sid: sid == CHAT_A
        ):
            with known_sessions():
                resp = post_posture(client, session_id=CHAT_A, target="standin")
        assert resp.status_code == 200
        assert read_store(agent_data_root) == {CHAT_A: {"standin": "sandbox"}}

    def test_a_prompt_less_pty_session_narrows(self, client, agent_data_root):
        """A just-opened terminal, zero prompts sent: the toggle works.

        The regression that motivated dropping the started-session gate — an
        operator opened a terminal and was told to talk to the agent before
        they could take writes away from it.
        """
        registry = client.app.state.pty_registry
        registry.get_or_create_session(SESSION_A, "echo")
        with known_sessions():  # nothing on disk: no prompt has been sent
            resp = post_posture(client, target="standin")
        assert resp.status_code == 200
        assert read_store(agent_data_root) == {SESSION_A: {"standin": "sandbox"}}

    def test_a_key_the_store_already_holds_stays_addressable(self, client, agent_data_root):
        """Otherwise a chat sandboxed once could never be brought back out."""
        with patch.object(
            websocket_routes, "_chat_pool_answers_to", side_effect=lambda app, sid: sid == CHAT_A
        ):
            with known_sessions():
                assert post_posture(client, session_id=CHAT_A, target="va").status_code == 200

        # The pool has dropped it; the entry is what keeps it reachable. The
        # answer is the render's 403, never the 409 that would strand it.
        with known_sessions():
            resp = post_posture(client, session_id=CHAT_A, target="va", posture="writes")
        assert resp.status_code == 403
        assert resp.json()["detail"]["error"] == "writes_disabled"


# -- audit ------------------------------------------------------------------


class TestAudit:
    def test_an_accepted_toggle_files_exactly_one_record(self, client, ledger):
        with known_sessions(SESSION_A):
            assert post_posture(client, target="standin").status_code == 200
        assert len(ledger) == 1
        record = ledger[0]
        assert record["subject"] == "session_posture_set"
        assert record["decision"] == "allowed"
        assert record["session"] == SESSION_A
        assert "standin" in record["detail"]
        assert "sandbox" in record["detail"]

    def test_a_refusal_files_exactly_one_record_too(self, client, ledger):
        with known_sessions(SESSION_A):
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
        with known_sessions(SESSION_A):
            assert post_posture(client, session_id="../../etc/passwd").status_code == 400
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "refused"

    def test_the_record_joins_on_the_spawn_key(self, client, ledger):
        """A rekeyed session's toggle is filed under the key its child exported."""
        registry = client.app.state.pty_registry
        with (
            patch.object(registry, "audit_session_key", side_effect=lambda key: SESSION_B),
            known_sessions(SESSION_A),
        ):
            assert post_posture(client, target="standin").status_code == 200
        assert ledger[0]["session"] == SESSION_B
