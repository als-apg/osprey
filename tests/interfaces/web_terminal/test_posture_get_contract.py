"""Tests for ``GET /api/terminal/posture`` — the header chip's roster.

One route answers the whole of what the chip and its popover render: which
control target this session is standing on, and one row per configured target
carrying that machine's identity, its reachability, the persona's ceiling, this
session's own narrowing, the effective answer the connector will apply, and
whether a switch is offered.

Three properties shape every test below.

* **The server decides, not the browser.** The refusal word under a missing
  Switch button is the switch tool's own; ``age_s`` and ``stale`` are computed
  here because the browser's clock is not the one the stamps were written on;
  and the collapse from "two gateway roles were probed" to "this row is
  reachable" follows the role the connector will actually select.
* **Ceiling, posture and effective are three separate columns.** The ceiling is
  the deployment's, the posture is the operator's, and the effective answer is
  the rule the connector applies. Collapsing them would leave the popover unable
  to say whether a locked toggle is the persona's doing or the operator's.
* **A read grants nothing.** Unlike POST, a key that names no session is
  answered rather than refused — the chip renders with the page, before any
  prompt has written a session file.

Harness mirrors ``test_target_request_route.py``: one app per test through
``create_app``, entered as a ``TestClient`` context manager so the lifespan
runs, over an ``OSPREY_AGENT_DATA_ROOT`` stamped at a throwaway directory.
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

from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey.mcp_server.control_system import target_banner, target_state
from osprey.mcp_server.control_system.connector_host_manager import _label
from osprey_connectors import session_store

SESSION_A = "aaaaaaaa-1111-2222-3333-444444444444"
CHAT_A = "cccccccc-1111-2222-3333-444444444444"

PTY_PID = 7000
CLAUDE_PID = 7001
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

TOP_LEVEL_FIELDS = {
    "session_id",
    "session_target",
    "store_available",
    "enforceable",
    "enforceable_reason",
    "execution_in_flight",
    "last_switch",
    "last_posture_realign",
    "targets",
}

#: The badge-era payload. Every one of these was a session-wide answer, and the
#: per-target rows replaced all of them; a client still reading one would be
#: reading a fact this route no longer has.
RETIRED_FIELDS = (
    "posture",
    "rendered_writes_enabled",
    "session_target_label",
    "target_writes_enabled",
    "target_source",
)


# -- fixtures ---------------------------------------------------------------


@pytest.fixture
def agent_data_root(tmp_path, monkeypatch):
    """Point every resolver at one throwaway agent-data root.

    ``OSPREY_AGENT_DATA_ROOT`` is the single stamp ``target_state.state_dir()``
    and ``session_store`` both prefer, and the ONLY way this route reaches
    either — patching ``resolve_shared_data_root`` would redirect one of them
    and leave the other writing into the repository's own ``var/agent_data``.

    Both of the module's memos are cleared around every test: they are keyed on
    a pid and on a file signature, and a tmp path reused across tests could
    otherwise serve one test's render or record to the next.
    """
    root = tmp_path / "agent_data"
    root.mkdir()
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(root))
    # A read-only *run* is a deployment-wide fact this process must not inherit
    # from whatever ran before it: it would zero every ``effective`` below.
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    session_store.invalidate_cache()
    websocket_routes._reset_session_record_memo()
    websocket_routes._reset_rendered_config_memo()
    yield root
    session_store.invalidate_cache()
    websocket_routes._reset_session_record_memo()
    websocket_routes._reset_rendered_config_memo()


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_watch"
    ws.mkdir()
    return ws


def render(
    *,
    global_writes=False,
    va_writes=None,
    standin_roles=("read_only", "write_access"),
    display_names=None,
    name="config.yml",
    tmp_path=None,
):
    """A render carrying all three control targets.

    ``epics`` is the facility's own machine, ``live_standin`` the co-deployed
    stand-in, ``virtual_accelerator`` the simulator — three connector blocks,
    therefore three targets, which is what makes ``configured_targets`` answer
    with three names and ``session_posture`` with one ceiling each.

    ``standin_roles`` narrows the stand-in's gateway table. Dropping
    ``read_only`` from it is the deployment shape that makes narrowing that
    target cost something: the session would select a role the config does not
    configure and the target would stop being usable at all.

    ``display_names`` is written verbatim as
    ``control_system.target_display_names``, the deployment's renaming of the
    chip's per-target words.
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
            # The keys a switch is judged on besides the gateways: a
            # channel to probe, strict limits (required toward the live
            # family) and the operator's acknowledgement of the live
            # gateway. Without them every row would answer with an
            # eligibility refusal and the roster could not be exercised.
            "limits_checking": {"enabled": True, "allow_unlisted_channels": False},
            "target_switch": {"live_gateway_acknowledged": "operator@example"},
            "connector": {
                "epics": {
                    "probe_channel": "SR:PROBE",
                    "gateways": {
                        "read_only": dict(gateway),
                        "write_access": dict(gateway),
                    },
                },
                "live_standin": {
                    "probe_channel": "SR:PROBE",
                    "gateways": {role: dict(standin_gateway) for role in standin_roles},
                },
                "virtual_accelerator": va,
            },
        },
        "services": {"live_standin": {"port": STANDIN_PORT}},
        "deployed_services": ["virtual_accelerator", "live_standin"],
    }
    if display_names is not None:
        config["control_system"]["target_display_names"] = display_names
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def render_not_switch_capable(*, tmp_path, name="single.yml"):
    """A render that configures more targets than its persona can arm.

    ``type: mock`` with no ``target_switch`` block is not switch-capable, so
    ``session_posture`` answers for the BASELINE alone — one key — while
    ``configured_targets`` still lists every connector block. The simulator is
    armed and the deployment is not, which is the exact shape where the row's
    two ceilings (``session_posture`` for the column, ``target_writes_enabled``
    inside the store) can disagree about the same target.
    """
    path = tmp_path / name
    path.write_text(
        yaml.safe_dump(
            {
                "control_system": {
                    "type": "mock",
                    "writes_enabled": False,
                    "connector": {
                        "mock": {},
                        "epics": {
                            "probe_channel": "SR:PROBE",
                            "gateways": {"read_only": {"address": "gw", "port": 5064}},
                        },
                        "virtual_accelerator": {
                            "simulation_file": "data/sim.json",
                            "probe_channel": "SIM:PROBE",
                            "writes_enabled": True,
                            "gateways": {"read_only": {"address": "gw", "port": 5064}},
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def config_path(tmp_path):
    return render(tmp_path=tmp_path)


@pytest.fixture
def make_client(agent_data_root, workspace_dir):
    @contextmanager
    def _make(config_path):
        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ):
            app = create_app(shell_command="echo")
            with TestClient(app) as test_client:
                test_client.app.state.config_path = config_path
                yield test_client

    return _make


@pytest.fixture
def client(make_client, config_path):
    with make_client(config_path) as c:
        yield c


# -- harness ----------------------------------------------------------------


@contextmanager
def attached_pty(client, session_id=SESSION_A, pid=PTY_PID):
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


def write_target_state(
    *,
    target="standin",
    owner_ppid=PTY_PID,
    server_pid=SERVER_PID,
    targets=None,
    reachability=None,
    last_switch=None,
    last_posture_realign=None,
):
    """Publish one controls-server state record under the stamped root."""
    directory = target_state.state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{target_state.STATE_FILE_PREFIX}{server_pid}.json"
    path.write_text(
        json.dumps(
            {
                "target": target,
                "generation": 1,
                "server_pid": server_pid,
                "owner_ppid": owner_ppid,
                "targets": TARGET_META if targets is None else targets,
                "children": [],
                "reachability": reachability,
                "last_switch": last_switch,
                "last_posture_realign": last_posture_realign,
            }
        ),
        encoding="utf-8",
    )
    return path


@contextmanager
def live_session(client, session_id=SESSION_A, **record):
    """A session whose controls server has published a live state record."""
    write_target_state(**record)
    with (
        attached_pty(client, session_id),
        synthetic_process_tree(),
        only_alive(record.get("server_pid", SERVER_PID)),
    ):
        yield


def probed(state, *, age_s=0.0, gateway="gw:5064"):
    """One published probe row, *age_s* seconds old."""
    return {
        "state": state,
        "probed_at": (datetime.now(UTC) - timedelta(seconds=age_s)).isoformat(),
        "gateway": gateway,
    }


def narrow(root: Path, session_id, entry):
    """Write a narrowing straight into the store, under the stamped root."""
    path = root / session_store.STATE_DIR_NAME / session_store.STORE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({session_id: entry}), encoding="utf-8")
    session_store.invalidate_cache()


def get_posture(client, session_id=SESSION_A):
    resp = client.get("/api/terminal/posture", params={"session_id": session_id})
    assert resp.status_code == 200, resp.text
    return resp.json()


def row_for(payload, target):
    rows = [row for row in payload["targets"] if row["target"] == target]
    assert len(rows) == 1, f"expected exactly one {target!r} row, got {rows}"
    return rows[0]


# -- the contract -----------------------------------------------------------


class TestGrammar:
    @pytest.mark.parametrize(
        "bad_id",
        ["../../etc/passwd", "operator-deadbeef", "", "AAAAAAAA-1111-2222-3333-444444444444"],
    )
    def test_an_id_outside_the_closed_grammar_is_400(self, client, bad_id):
        """One error contract with POST: the same grammar, the same slug."""
        resp = client.get("/api/terminal/posture", params={"session_id": bad_id})
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "invalid_session_id"

    def test_a_key_that_names_no_session_is_still_answered(self, client):
        """A read grants nothing, so it is never a 409."""
        payload = get_posture(client)
        assert payload["session_id"] == SESSION_A
        assert [row["target"] for row in payload["targets"]] == ["live", "va", "standin"]


class TestTheShape:
    """The payload the chip reads, pinned field by field."""

    def test_the_top_level_fields_are_exactly_these(self, client):
        assert set(get_posture(client)) == TOP_LEVEL_FIELDS

    def test_every_row_carries_exactly_these_fields(self, client):
        for row in get_posture(client)["targets"]:
            assert set(row) == ROW_FIELDS, row["target"]

    def test_every_reachability_block_carries_exactly_these_fields(self, client):
        for row in get_posture(client)["targets"]:
            assert set(row["reachability"]) == REACHABILITY_FIELDS, row["target"]

    @pytest.mark.parametrize("field", RETIRED_FIELDS)
    def test_the_badge_era_fields_are_gone(self, client, field):
        """Each was a session-wide answer the per-target rows replaced."""
        assert field not in get_posture(client)

    def test_one_row_per_configured_target_in_the_configured_order(self, client):
        """``configured_targets`` and nothing else decides which rows exist."""
        from osprey_connectors.types import configured_targets

        section = yaml.safe_load(Path(client.app.state.config_path).read_text())["control_system"]
        assert [row["target"] for row in get_posture(client)["targets"]] == list(
            configured_targets(section)
        )


class TestIdentity:
    """What a row calls its machine, and where it says it points."""

    def test_a_published_record_names_the_targets(self, client):
        """The controls server mints the label; every reader renders that one."""
        with live_session(client):
            payload = get_posture(client)

        assert row_for(payload, "standin")["label"] == "LIVE MACHINE (stand-in)"
        assert row_for(payload, "standin")["endpoint"] == f"localhost:{STANDIN_PORT}"
        assert row_for(payload, "live")["real_machine"] is True
        assert row_for(payload, "va")["real_machine"] is False

    def test_without_a_record_the_render_names_them(self, client):
        """A card whose session has not started still has to name its targets.

        Derived through the same ``target_display_metadata`` the state file's
        writer uses, so the two cannot disagree about which machine is which.
        """
        payload = get_posture(client)
        assert row_for(payload, "standin")["label"] == "LIVE MACHINE (stand-in)"
        assert row_for(payload, "live")["label"] == "LIVE MACHINE"
        assert row_for(payload, "va")["label"] == "virtual accelerator (simulation)"
        assert row_for(payload, "live")["display_name"] == "Real machine"
        assert row_for(payload, "standin")["display_name"] == "Rehearsal"
        assert row_for(payload, "va")["display_name"] == "Simulator"

    def test_an_older_published_record_does_not_erase_the_display_name(self, client):
        """``TARGET_META`` predates the key; the derived word must survive it.

        A record's fields win where they exist — the label above proves that —
        but a writer from an older build publishes no ``display_name`` at all,
        and the merge in ``_target_display`` keeps the derived one rather than
        rendering an unnamed chip.
        """
        with live_session(client):
            payload = get_posture(client)

        assert row_for(payload, "standin")["display_name"] == "Rehearsal"
        assert row_for(payload, "live")["display_name"] == "Real machine"
        assert row_for(payload, "va")["display_name"] == "Simulator"

    def test_a_configured_display_name_reaches_the_row(self, make_client, tmp_path):
        """``control_system.target_display_names`` renames the chip's word only."""
        path = render(
            tmp_path=tmp_path,
            display_names={"standin": "Shadow ring", "va": ""},
        )
        with make_client(path) as client:
            payload = get_posture(client)

        assert row_for(payload, "standin")["display_name"] == "Shadow ring"
        # The label the safety surfaces render is not the operator's to rename.
        assert row_for(payload, "standin")["label"] == "LIVE MACHINE (stand-in)"
        # An empty override is no name at all: the default answers.
        assert row_for(payload, "va")["display_name"] == "Simulator"
        assert row_for(payload, "live")["display_name"] == "Real machine"

    def test_active_follows_the_published_target_and_baseline_does_not(self, client):
        with live_session(client, target="va"):
            payload = get_posture(client)

        assert payload["session_target"] == "va"
        assert row_for(payload, "va")["active"] is True
        assert row_for(payload, "standin")["active"] is False
        # The deployment's own baseline is the stand-in; being switched away
        # from it does not move it.
        assert row_for(payload, "standin")["is_baseline"] is True
        assert row_for(payload, "va")["is_baseline"] is False

    def test_no_record_falls_back_to_the_baseline(self, client):
        payload = get_posture(client)
        assert payload["session_target"] == "standin"
        assert row_for(payload, "standin")["active"] is True


class TestShortLabelAndKind:
    """The chip's word for a row comes from what it IS, never from its name."""

    @pytest.mark.parametrize(
        ("label", "real_machine", "short_label", "kind"),
        [
            (_label("live", "epics"), True, "LIVE", "live machine"),
            (_label("standin", "epics", standin=True), True, "STAND-IN", "stand-in"),
            (_label("va", "virtual_accelerator"), False, "VIRTUAL", "virtual accelerator"),
            (_label("live", "mock"), False, "SIMULATED", "simulated"),
        ],
    )
    def test_every_label_the_writer_can_mint_has_a_word(
        self, label, real_machine, short_label, kind
    ):
        """Over ``_label``'s own outputs, so a new label shape breaks here.

        The four shapes are the four the state file's single writer produces.
        Pinning against that function rather than against strings typed here is
        what makes this a contract with the writer instead of a copy of it.
        """
        assert websocket_routes._short_label_and_kind(label, real_machine) == (short_label, kind)

    def test_an_unrecognised_real_machine_label_stays_loud(self):
        """The direction this stack must fail in: never quietly downgrade."""
        assert websocket_routes._short_label_and_kind("something new", True) == (
            "LIVE",
            "live machine",
        )

    def test_the_rows_carry_the_words(self, client):
        with live_session(client):
            payload = get_posture(client)

        assert row_for(payload, "live")["short_label"] == "LIVE"
        assert row_for(payload, "standin")["short_label"] == "STAND-IN"
        assert row_for(payload, "va")["short_label"] == "VIRTUAL"
        assert row_for(payload, "standin")["kind"] == "stand-in"
        assert row_for(payload, "va")["kind"] == "virtual accelerator"

    def test_a_published_row_missing_real_machine_falls_back_to_the_derivation(self, client):
        """A published label MERGES over the derivation; it does not replace it.

        ``real_machine`` decides which half of ``_short_label_and_kind``
        answers, so a writer from an older build — one that publishes a label
        and not that flag, and whose file a newer web server goes on reading
        because the record is not rewritten on upgrade — would otherwise render
        the stand-in as a muted ``SIMULATED`` simulator. The published label
        still wins; only the keys it omits fall back.
        """
        with live_session(client, targets={"standin": {"label": "LIVE MACHINE (stand-in)"}}):
            payload = get_posture(client)

        standin = row_for(payload, "standin")
        assert standin["label"] == "LIVE MACHINE (stand-in)"
        assert standin["real_machine"] is True
        assert standin["short_label"] == "STAND-IN"
        assert standin["kind"] == "stand-in"
        assert standin["endpoint"] == f"localhost:{STANDIN_PORT}"


class TestTheThreePostureColumns:
    """Ceiling, narrowing and effective answer, kept apart on purpose."""

    def test_an_unarmed_render_is_ceiling_off_and_effective_off(self, client):
        payload = get_posture(client)
        for row in payload["targets"]:
            assert row["ceiling_writes"] is False, row["target"]
            assert row["effective"] is False, row["target"]
            assert row["posture"] == "writes", row["target"]

    def test_an_armed_render_is_ceiling_on_and_effective_on(self, make_client, tmp_path):
        with make_client(render(tmp_path=tmp_path, global_writes=True)) as client:
            payload = get_posture(client)

        for row in payload["targets"]:
            assert row["ceiling_writes"] is True, row["target"]
            assert row["effective"] is True, row["target"]

    def test_a_narrowing_shows_as_posture_and_clears_effective(
        self, make_client, tmp_path, agent_data_root
    ):
        """The operator's own column, and its consequence, side by side."""
        narrow(agent_data_root, SESSION_A, {"standin": "sandbox"})
        with make_client(render(tmp_path=tmp_path, global_writes=True)) as client:
            payload = get_posture(client)

        standin = row_for(payload, "standin")
        assert standin["posture"] == "sandbox"
        assert standin["ceiling_writes"] is True
        assert standin["effective"] is False
        # The other machines are untouched — that is the whole feature.
        assert row_for(payload, "va")["posture"] == "writes"
        assert row_for(payload, "va")["effective"] is True

    def test_a_narrowing_cannot_lift_an_unarmed_ceiling(
        self, make_client, tmp_path, agent_data_root
    ):
        """The store only ever narrows; absence is not an assertion of writes."""
        narrow(agent_data_root, SESSION_A, {"standin": "sandbox"})
        with make_client(render(tmp_path=tmp_path, global_writes=False)) as client:
            payload = get_posture(client)

        assert row_for(payload, "va")["posture"] == "writes"
        assert row_for(payload, "va")["effective"] is False

    def test_a_mixed_render_arms_one_machine_and_not_the_other(self, make_client, tmp_path):
        """Why the union cannot answer this: one row true, another false."""
        config = render(tmp_path=tmp_path, global_writes=False, va_writes=True)
        with make_client(config) as client:
            payload = get_posture(client)

        assert row_for(payload, "va")["ceiling_writes"] is True
        assert row_for(payload, "live")["ceiling_writes"] is False

    def test_the_store_is_read_under_the_stamped_root_only(
        self, make_client, tmp_path, agent_data_root
    ):
        """A narrowing written anywhere else governs nothing and shows nothing."""
        elsewhere = tmp_path / "not_the_root" / session_store.STATE_DIR_NAME
        elsewhere.mkdir(parents=True)
        (elsewhere / session_store.STORE_FILENAME).write_text(
            json.dumps({SESSION_A: {"standin": "sandbox"}}), encoding="utf-8"
        )
        with make_client(render(tmp_path=tmp_path, global_writes=True)) as client:
            assert row_for(get_posture(client), "standin")["posture"] == "writes"

    def test_one_row_never_reports_two_ceilings(self, make_client, tmp_path):
        """``effective`` answers under the ceiling the row itself shows.

        On a non-switch-capable render the two ceiling predicates diverge: the
        column reads ``session_posture``, which keys the baseline alone, while
        the store's own ceiling is ``target_writes_enabled`` for every
        configured target — armed here for ``va``. Left ungated the row would
        render ceiling-off beside effective-on: a filled dot promising writes on
        a target no connector on this deployment is ever built for, the chip
        disagreeing with the runtime on the one surface that answers "will a
        write be refused?".
        """
        config = render_not_switch_capable(tmp_path=tmp_path)
        section = yaml.safe_load(config.read_text(encoding="utf-8"))["control_system"]

        # The divergence this test exists for, asserted at the predicate level so
        # it cannot quietly go vacuous if the render stops being mixed.
        from osprey_connectors.types import session_posture, target_writes_enabled

        assert "va" not in session_posture(section)
        assert target_writes_enabled(section, "va") is True

        with make_client(config) as client:
            payload = get_posture(client)

        assert {row["target"] for row in payload["targets"]} == {"live", "va"}
        for row in payload["targets"]:
            if row["ceiling_writes"] is False:
                assert row["effective"] is False, row["target"]
        assert row_for(payload, "va")["ceiling_writes"] is False
        assert row_for(payload, "va")["effective"] is False

    def test_a_va_plus_standin_render_reports_the_standins_own_ceiling(self, make_client, tmp_path):
        """Two configured targets are switch-capable with no live machine at all.

        The stand-in row carries its own armed ceiling, which is what keeps the
        popover's posture toggle on that row unlocked — a render rehearsing on
        its stand-in beside the simulator must not lock the one toggle it
        exists to offer.
        """
        config = tmp_path / "va_standin.yml"
        config.write_text(
            yaml.safe_dump(
                {
                    "control_system": {
                        "type": "virtual_accelerator",
                        "writes_enabled": False,
                        "connector": {
                            "virtual_accelerator": {
                                "simulation_file": "data/sim.json",
                                "probe_channel": "SIM:PROBE",
                                "writes_enabled": True,
                                "gateways": {"read_only": {"address": "gw", "port": 5064}},
                            },
                            "live_standin": {
                                "probe_channel": "SR:PROBE",
                                "writes_enabled": True,
                                "gateways": {
                                    "read_only": {"address": "localhost", "port": STANDIN_PORT},
                                    "write_access": {"address": "localhost", "port": STANDIN_PORT},
                                },
                            },
                        },
                    },
                    "services": {"live_standin": {"port": STANDIN_PORT}},
                    "deployed_services": ["virtual_accelerator", "live_standin"],
                }
            ),
            encoding="utf-8",
        )
        with make_client(config) as client:
            payload = get_posture(client)

        assert {row["target"] for row in payload["targets"]} == {"va", "standin"}
        standin = row_for(payload, "standin")
        assert standin["ceiling_writes"] is True
        assert standin["effective"] is True
        assert standin["narrowing_refusal"] is None


class TestNarrowingRefusal:
    """What narrowing a row would cost — the one lock reason the rest of the
    payload cannot be derived from."""

    def test_an_ordinary_render_costs_nothing_to_narrow(self, client):
        """The ordinary case: anything that can be read from has a read gateway."""
        for row in get_posture(client)["targets"]:
            assert row["narrowing_refusal"] is None, row["target"]

    def test_a_write_only_gateway_table_strands_the_target(self, make_client, tmp_path):
        """The toggle is locked BEFORE the click, with the POST's own word.

        A stand-in whose block configures ``write_access`` alone has nothing for
        a narrowed session to select: it would land on ``read_only``, find no
        such gateway, and stop being usable. That is what the POST answers its
        409 with, and the row says it first.
        """
        config = render(tmp_path=tmp_path, standin_roles=("write_access",))
        with make_client(config) as client:
            payload = get_posture(client)

        assert row_for(payload, "standin")["narrowing_refusal"] == "selected_role_missing"
        # Only the target that would be stranded; its neighbours are unaffected.
        assert row_for(payload, "live")["narrowing_refusal"] is None
        assert row_for(payload, "va")["narrowing_refusal"] is None

    def test_an_already_narrowed_row_reports_nothing(self, make_client, tmp_path, agent_data_root):
        """Narrowing cannot strand a target that is already narrowed.

        That row's toggle brings the target BACK, and locking it on a refusal
        the operator has already lived through would trap the session in the
        sandbox. Mirrors the POST's "only the targets this request would
        change" rule.
        """
        narrow(agent_data_root, SESSION_A, {"standin": "sandbox"})
        config = render(tmp_path=tmp_path, standin_roles=("write_access",))
        with make_client(config) as client:
            payload = get_posture(client)

        standin = row_for(payload, "standin")
        assert standin["posture"] == "sandbox"
        assert standin["narrowing_refusal"] is None

    def test_the_word_comes_from_the_shared_verdict(self, make_client, tmp_path):
        """Pinned against target_eligibility.narrowing_refusal, not a literal."""
        from osprey.mcp_server.control_system.target_eligibility import narrowing_refusal

        config = render(tmp_path=tmp_path, standin_roles=("write_access",))
        rendered = yaml.safe_load(Path(config).read_text())
        expected = narrowing_refusal(rendered, "standin")

        with make_client(config) as client:
            payload = get_posture(client)

        assert expected is not None
        assert row_for(payload, "standin")["narrowing_refusal"] == expected.reason

    def test_an_unreadable_render_fails_open(self, make_client):
        """No render, no rows — and nothing claiming a narrowing would fail."""
        with make_client(None) as client:
            assert get_posture(client)["targets"] == []


class TestAvailability:
    """Whether a switch is offered, in the switch tool's own words."""

    def test_the_active_target_reports_already_active(self, client):
        with live_session(client, target="va"):
            payload = get_posture(client)

        va = row_for(payload, "va")
        assert va["available_now"] is False
        assert va["reason"] == "already_active"

    def test_another_target_is_offered(self, client):
        with live_session(client, target="va"):
            payload = get_posture(client)

        assert row_for(payload, "live")["available_now"] is True
        assert row_for(payload, "live")["reason"] is None

    def test_a_refused_target_names_a_reason_from_the_tools_vocabulary(self, client):
        """The word under a missing Switch button is the switch tool's own.

        Which refusal this render earns is the eligibility rules' business and
        not this route's; what is pinned here is that the word comes from that
        closed set rather than from a sentence invented for the popover.
        """
        from osprey.mcp_server.control_system import target_eligibility

        vocabulary = {
            value
            for name, value in vars(target_eligibility).items()
            if name.startswith("REASON_") and isinstance(value, str)
        }

        with live_session(client, target="va"):
            payload = get_posture(client)

        standin = row_for(payload, "standin")
        assert standin["available_now"] is False
        assert standin["reason"] in vocabulary

    def test_a_refusal_carries_the_verdicts_own_sentence(self, client):
        """``reason_detail`` is the eligibility detail, not a second opinion.

        The popover renders a short phrase keyed on the code and puts this
        sentence on the tooltip; pinning it to the shared verdict keeps the
        tooltip, the 409 and the switch tool's answer one wording.
        """
        from osprey.mcp_server.control_system.target_eligibility import target_availability

        with live_session(client, target="va"):
            payload = get_posture(client)

        standin = row_for(payload, "standin")
        expected = target_availability(
            yaml.safe_load(Path(client.app.state.config_path).read_text(encoding="utf-8")),
            "standin",
            "va",
            "standin",
            writes_enabled=False,
        )
        assert standin["reason_detail"] == expected.detail
        assert standin["reason_detail"]

    def test_an_offered_target_carries_no_detail(self, client):
        with live_session(client, target="va"):
            payload = get_posture(client)

        assert row_for(payload, "live")["reason_detail"] is None

    def test_an_unreadable_render_offers_nothing(self, make_client):
        with make_client(None) as client:
            payload = get_posture(client)
        assert payload["targets"] == []


class TestChatSessions:
    """A chat has no controls server, and says so without losing its toggles."""

    @contextmanager
    def _chat(self, client, chat_id=CHAT_A):
        async def _cleanup_all():  # the lifespan's shutdown calls this
            return None

        client.app.state.operator_registry = SimpleNamespace(
            has_chat_key=lambda key: key == chat_id,
            get_chat_session=lambda key: SimpleNamespace() if key == chat_id else None,
            cleanup_all=_cleanup_all,
        )
        yield

    def test_no_switch_is_offered_and_the_reason_says_why(self, client):
        with self._chat(client):
            payload = get_posture(client, CHAT_A)

        for row in payload["targets"]:
            assert row["available_now"] is False, row["target"]
            assert row["reason"] == "chat_session", row["target"]

    def test_reachability_is_unknown(self, client):
        """No PTY, so no record, so nothing has been probed on its behalf."""
        with self._chat(client):
            payload = get_posture(client, CHAT_A)

        for row in payload["targets"]:
            assert row["reachability"]["state"] == "unknown", row["target"]

    def test_nothing_is_active_beyond_the_baseline(self, client):
        with self._chat(client):
            payload = get_posture(client, CHAT_A)

        assert payload["session_target"] == "standin"
        assert [row["target"] for row in payload["targets"] if row["active"]] == ["standin"]

    def test_the_toggles_stay_live(self, make_client, tmp_path):
        """The store is keyed on the session, not on the topology."""
        with make_client(render(tmp_path=tmp_path, global_writes=True)) as client:
            with self._chat(client):
                payload = get_posture(client, CHAT_A)

        for row in payload["targets"]:
            assert row["ceiling_writes"] is True, row["target"]
            assert row["effective"] is True, row["target"]

    def test_a_chat_session_is_enforceable(self, client):
        """Its spawn seam stamps the store key, which is the whole question."""
        with self._chat(client):
            payload = get_posture(client, CHAT_A)

        assert payload["enforceable"] is True
        assert payload["enforceable_reason"] is None


class TestEnforceable:
    """Whether a narrowing recorded here would be read by anybody."""

    def test_a_started_pty_with_no_record_is_not_enforceable(self, client):
        """Its controls server is outside this process tree, or there is none."""
        with attached_pty(client), synthetic_process_tree(), only_alive():
            payload = get_posture(client)

        assert payload["enforceable"] is False
        assert payload["enforceable_reason"] == "no_session_record"

    def test_a_session_owned_record_is_enforceable(self, client):
        with live_session(client):
            payload = get_posture(client)

        assert payload["enforceable"] is True
        assert payload["enforceable_reason"] is None

    def test_a_card_that_has_not_started_is_enforceable(self, client):
        """No PTY yet, so nothing is running that could be missing a record.

        The spawn seam stamps the store key, so whatever starts under this key
        will read the narrowing. Reporting ``false`` here would lock the toggles
        on every freshly opened card.
        """
        payload = get_posture(client)
        assert payload["enforceable"] is True
        assert payload["enforceable_reason"] is None


class TestReachability:
    """One row, one role: the one the connector will actually select."""

    def test_the_selected_role_under_an_unarmed_posture_is_the_read_gateway(self, client):
        reach = {
            "published_at": datetime.now(UTC).isoformat(),
            "targets": {
                "standin": {
                    "read_only": probed("reached"),
                    "write_access": probed("down"),
                }
            },
        }
        with live_session(client, reachability=reach):
            payload = get_posture(client)

        block = row_for(payload, "standin")["reachability"]
        assert block["role"] == "read_only"
        assert block["state"] == "reached"
        # The other role is named, never merged: an OR would call this target
        # reachable through a gateway it will not open.
        assert block["role_detail"] == {"write_access": "down"}

    def test_the_selected_role_follows_the_effective_posture(self, make_client, tmp_path):
        """An armed target selects its write gateway, and reports that one."""
        reach = {
            "published_at": datetime.now(UTC).isoformat(),
            "targets": {
                "standin": {
                    "read_only": probed("reached"),
                    "write_access": probed("down"),
                }
            },
        }
        with make_client(render(tmp_path=tmp_path, global_writes=True)) as client:
            with live_session(client, reachability=reach):
                payload = get_posture(client)

        block = row_for(payload, "standin")["reachability"]
        assert block["role"] == "write_access"
        assert block["state"] == "down"
        assert block["role_detail"] == {"read_only": "reached"}

    def test_a_narrowing_moves_the_role_the_row_reports(
        self, make_client, tmp_path, agent_data_root
    ):
        """The collapse follows the operator's own narrowing, not the config."""
        narrow(agent_data_root, SESSION_A, {"standin": "sandbox"})
        reach = {
            "published_at": datetime.now(UTC).isoformat(),
            "targets": {
                "standin": {
                    "read_only": probed("reached"),
                    "write_access": probed("down"),
                }
            },
        }
        with make_client(render(tmp_path=tmp_path, global_writes=True)) as client:
            with live_session(client, reachability=reach):
                payload = get_posture(client)

        block = row_for(payload, "standin")["reachability"]
        assert block["role"] == "read_only"
        assert block["state"] == "reached"

    def test_a_row_older_than_three_intervals_is_stale(self, client):
        """The prober's own interval decides, not a number invented here."""
        from osprey.mcp_server.control_system.endpoint_prober import (
            DEFAULT_PROBE_INTERVAL_S,
            STALENESS_INTERVALS,
        )

        too_old = DEFAULT_PROBE_INTERVAL_S * STALENESS_INTERVALS + 5
        reach = {
            "published_at": datetime.now(UTC).isoformat(),
            "targets": {"standin": {"read_only": probed("reached", age_s=too_old)}},
        }
        with live_session(client, reachability=reach):
            payload = get_posture(client)

        block = row_for(payload, "standin")["reachability"]
        assert block["state"] == "stale"
        assert block["age_s"] >= too_old

    def test_a_fresh_row_is_not_stale(self, client):
        from osprey.mcp_server.control_system.endpoint_prober import (
            DEFAULT_PROBE_INTERVAL_S,
            STALENESS_INTERVALS,
        )

        recent = DEFAULT_PROBE_INTERVAL_S * STALENESS_INTERVALS - 5
        reach = {
            "published_at": datetime.now(UTC).isoformat(),
            "targets": {"standin": {"read_only": probed("reached", age_s=recent)}},
        }
        with live_session(client, reachability=reach):
            payload = get_posture(client)

        assert row_for(payload, "standin")["reachability"]["state"] == "reached"

    def test_not_applicable_survives_ageing(self, client):
        """It is a decision, not a measurement; ageing it out would lie."""
        reach = {
            "published_at": datetime.now(UTC).isoformat(),
            "targets": {"standin": {"read_only": probed("not_applicable", age_s=100_000)}},
        }
        with live_session(client, reachability=reach):
            payload = get_posture(client)

        assert row_for(payload, "standin")["reachability"]["state"] == "not_applicable"

    def test_an_absent_block_is_unknown(self, client):
        """No sweep has landed. ``unknown`` — never "down"."""
        with live_session(client):
            payload = get_posture(client)

        for row in payload["targets"]:
            block = row["reachability"]
            assert block["state"] == "unknown", row["target"]
            assert block["probed_at"] is None
            assert block["age_s"] is None

    def test_a_target_with_no_row_is_unknown_while_another_is_reached(self, client):
        reach = {
            "published_at": datetime.now(UTC).isoformat(),
            "targets": {"standin": {"read_only": probed("reached")}},
        }
        with live_session(client, reachability=reach):
            payload = get_posture(client)

        assert row_for(payload, "standin")["reachability"]["state"] == "reached"
        assert row_for(payload, "live")["reachability"]["state"] == "unknown"

    def test_the_age_is_computed_server_side(self, client):
        reach = {
            "published_at": datetime.now(UTC).isoformat(),
            "targets": {"standin": {"read_only": probed("reached", age_s=12.0)}},
        }
        with live_session(client, reachability=reach):
            payload = get_posture(client)

        block = row_for(payload, "standin")["reachability"]
        assert 11.0 <= block["age_s"] <= 20.0
        assert block["probed_at"].startswith("20")


class TestPublishedBlocks:
    """``last_switch`` and ``last_posture_realign``, as the writer left them."""

    def test_a_last_switch_is_returned_with_its_age(self, client):
        switch = {
            "request_id": "req-1",
            "target": "va",
            "status": "success",
            "reason": None,
            "detail": "standin → va",
            "at": (datetime.now(UTC) - timedelta(seconds=4)).isoformat(),
        }
        with live_session(client, last_switch=switch):
            payload = get_posture(client)

        assert payload["last_switch"]["request_id"] == "req-1"
        assert payload["last_switch"]["status"] == "success"
        assert 3.0 <= payload["last_switch"]["age_s"] <= 20.0

    def test_the_target_the_switch_named_surfaces(self, client):
        """The popover renders an outcome on the row it names, so the key the
        writer published has to survive the route rather than be filtered out
        by an allow-list of fields this render happens to know about."""
        switch = {
            "request_id": "req-3",
            "target": "standin",
            "status": "refused",
            "reason": "target_not_configured",
            "detail": "no stand-in here",
            "at": datetime.now(UTC).isoformat(),
        }
        with live_session(client, last_switch=switch):
            payload = get_posture(client)

        assert payload["last_switch"]["target"] == "standin"
        assert payload["last_switch"]["reason"] == "target_not_configured"

    def test_an_unstamped_switch_is_returned_unaged(self, client):
        """``age_s`` is ``None``, never a fabricated zero."""
        with live_session(client, last_switch={"request_id": "req-2", "status": "success"}):
            payload = get_posture(client)

        assert payload["last_switch"]["age_s"] is None

    def test_a_pending_realignment_is_passed_through(self, client):
        realign = {"state": "pending", "at": datetime.now(UTC).isoformat()}
        with live_session(client, last_posture_realign=realign):
            payload = get_posture(client)

        assert payload["last_posture_realign"]["state"] == "pending"

    def test_absent_blocks_are_null(self, client):
        with live_session(client):
            payload = get_posture(client)

        assert payload["last_switch"] is None
        assert payload["last_posture_realign"] is None


class TestSessionWideFlags:
    def test_the_store_resolves(self, client):
        assert get_posture(client)["store_available"] is True

    def test_an_unresolvable_store_is_reported(self, client, monkeypatch):
        """The popover head-note says so rather than offering dead toggles."""
        monkeypatch.delenv(session_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)
        session_store.invalidate_cache()
        with patch.object(
            session_store, "resolve_shared_data_root", side_effect=RuntimeError("no config")
        ):
            assert session_store.store_path() is None, "the outage simulated nothing"
            payload = get_posture(client)

        assert payload["store_available"] is False

    def test_no_execution_in_flight_by_default(self, client):
        assert get_posture(client)["execution_in_flight"] is False

    def test_a_live_marker_is_reported(self, client, agent_data_root):
        """The popover's head-note needs it: a widening would be refused now."""
        directory = agent_data_root / target_state.STATE_DIR_NAME
        directory.mkdir(parents=True, exist_ok=True)
        marker = (
            directory
            / f"{target_state.INFLIGHT_FILE_PREFIX}{PTY_PID}{target_state.INFLIGHT_FILE_SUFFIX}"
        )
        marker.write_text(
            json.dumps(
                {
                    "pid": PTY_PID,
                    "target": "standin",
                    "generation": 1,
                    "owner_ppid": PTY_PID,
                    "started_at": datetime.now(UTC).isoformat(),
                }
            ),
            encoding="utf-8",
        )
        with only_alive(PTY_PID):
            assert get_posture(client)["execution_in_flight"] is True


class TestTheRenderMemo:
    """The render is parsed once per version of the file, and no longer."""

    def test_a_repeat_request_does_not_reparse(self, client):
        get_posture(client)
        with patch.object(yaml, "safe_load", side_effect=AssertionError("reparsed")) as parse:
            get_posture(client)
        parse.assert_not_called()

    def test_a_rewritten_config_is_picked_up(self, client, tmp_path):
        """A re-render moves the signature, so the memo answers with the new file."""
        assert row_for(get_posture(client), "va")["ceiling_writes"] is False

        client.app.state.config_path = render(
            tmp_path=tmp_path, name="config.yml", global_writes=False, va_writes=True
        )

        assert row_for(get_posture(client), "va")["ceiling_writes"] is True

    def test_the_memo_hands_back_a_copy(self, client):
        """A caller may keep or mutate the render; the next request must not
        inherit it."""
        first = websocket_routes._rendered_config(client.app.state.config_path)
        first["control_system"]["connector"].clear()

        second = websocket_routes._rendered_config(client.app.state.config_path)
        assert set(second["control_system"]["connector"]) == {
            "epics",
            "live_standin",
            "virtual_accelerator",
        }


class TestTheRecordMemo:
    """A memo hit is never a weaker answer than a miss."""

    #: A switched-away-from-baseline record. ``standin`` is this render's
    #: baseline, so a record naming it could not be told apart from the
    #: fallback a dropped record lands on.
    SWITCHED = {
        "target": "va",
        "last_switch": {"request_id": "req-memo", "status": "success", "detail": "standin → va"},
    }

    def test_a_live_writer_is_answered_from_the_memo(self, client):
        """The fast path stays fast: no second walk of the process table."""
        write_target_state(**self.SWITCHED)
        with attached_pty(client), synthetic_process_tree(), only_alive(SERVER_PID):
            assert get_posture(client)["session_target"] == "va"

            with patch.object(
                target_banner,
                "session_record_for_pid",
                side_effect=AssertionError("re-resolved"),
            ) as resolve:
                assert get_posture(client)["session_target"] == "va"
            resolve.assert_not_called()

    def test_a_dead_writer_is_dropped_on_the_cheapest_hit(self, client):
        """A controls server that dies leaves its file behind untouched.

        Its signature therefore never moves, so the memo's signature-hit path
        never expires on its own: without a liveness check the chip would go on
        naming that server's target and its switch outcome long after the
        process was gone. The full resolver drops a dead writer, and the memo's
        cheapest path must reach the same answer.
        """
        write_target_state(**self.SWITCHED)
        with attached_pty(client), synthetic_process_tree(), only_alive(SERVER_PID):
            assert get_posture(client)["session_target"] == "va"
            assert get_posture(client)["last_switch"]["request_id"] == "req-memo"

        # Same file, same signature — only the writer is gone.
        with attached_pty(client), synthetic_process_tree(), only_alive():
            payload = get_posture(client)

        assert payload["session_target"] == "standin"  # the deployment baseline
        assert payload["last_switch"] is None


class TestOneResolutionPerRequest:
    def test_the_process_table_is_walked_at_most_once(self, client):
        """Two resolutions could straddle a switch and describe two machines."""
        write_target_state()
        with (
            attached_pty(client),
            synthetic_process_tree(),
            only_alive(SERVER_PID),
            patch.object(
                target_banner,
                "session_record_for_pid",
                wraps=target_banner.session_record_for_pid,
            ) as resolve,
        ):
            get_posture(client)

        assert resolve.call_count <= 1
