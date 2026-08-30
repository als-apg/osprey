"""Eligibility, the roster and the switch, read through the session posture store.

The deployment's ``writes_enabled`` keys are not the last word on whether a
target is armed. An operator narrows one target for one session from the header
chip, that narrowing lands in the per-(session, target) posture store, and the
connector-host child selects its gateway from it — so every parent-side answer
about "which gateway will this session use" has to read the same store or stop
being about this session.

Why this file exists at all is one bug: ``derive_endpoints`` defaults its
``writes_enabled`` to config alone. A parent that took that default derived
``write_access`` for a target the operator had just narrowed, the child came up
on ``read_only`` because it read the store, and ``verify_child_report`` — doing
exactly its job — aborted the switch on a ``selected_role`` mismatch neither
side had got wrong. It fails closed, but it refuses a switch that is correctly
configured, and on the shipped virtual-accelerator preset it is invisible
because both gateway rows name the same endpoint.

So the load-bearing test here is the one with real children and a target whose
two gateway roles point at DIFFERENT ports: it is the only shape in which
parent and child can disagree about where the session landed.

``tests/connectors/test_session_store.py`` owns the store's own contract — where
the file is, what its shapes mean, how a lookup combines. Nothing here re-tests
that; these tests write the file the way the store defines it and assert on what
this stack derives from it.
"""

from __future__ import annotations

import contextlib
import json
import os
from typing import Any

import pytest

from osprey.mcp_server.control_system import target_eligibility as te
from osprey.mcp_server.control_system.connector_host_manager import (
    ConnectorHostManager,
)
from osprey.mcp_server.control_system.server_context import MCPServerConfig
from osprey.mcp_server.control_system.tools import control_target
from osprey_connectors import session_store
from osprey_connectors.control_system.base import ChannelValue
from osprey_connectors.types import VIRTUAL_ACCELERATOR

# The live-child half runs on the switch suite's fixture connector: a mock
# variant whose connect() applies the real gateway-role selection, reading the
# real per-type posture, with no Channel Access anywhere. Imported rather than
# restated — a second copy would be a second rule to keep in step with
# EPICSConnector.connect(). Importing the module applies none of its fixtures.
from tests.mcp_server.test_switch_lifecycle import (
    FIXTURE_MODULE,
    GATEWAY_HOST,
    REPO_PATHS,
    SITECUSTOMIZE,
    SPAWN_TIMEOUT_S,
    VA_PROBE,
    VA_READ_GATEWAY_PORT,
    VA_WRITE_GATEWAY_PORT,
    gateway_config,
    project_config,
)

LIVE = "live"
VA = "va"

EPICS_TYPE = "epics"
VA_TYPE = "virtual_accelerator"

#: The session this process is stamped as. Any non-blank string is a key; the
#: store's grammar belongs to the web server, not to a reader.
SESSION = "11111111-2222-3333-4444-555555555555"


# ---------------------------------------------------------------------------
# Config builders — one deployment, two targets, distinct gateway roles
# ---------------------------------------------------------------------------


def _gateways(read_port: int | None, write_port: int | None) -> dict[str, Any]:
    """A gateways table with only the roles whose port is named."""
    table: dict[str, Any] = {}
    if read_port is not None:
        table["read_only"] = {"address": "gw.example.org", "port": read_port}
    if write_port is not None:
        table["write_access"] = {"address": "gw.example.org", "port": write_port}
    return table


def _config(
    *,
    live_gateways: dict[str, Any] | None = None,
    va_gateways: dict[str, Any] | None = None,
    live_writes: bool = True,
    va_writes: bool = True,
) -> dict[str, Any]:
    """A switch-capable config whose two targets are both armed by default.

    Armed on purpose: a narrowing can only ever take arming away, so a config
    that never armed anything could not tell a working store read from a
    no-op.
    """
    return {
        "control_system": {
            "type": EPICS_TYPE,
            "writes_enabled": False,
            "limits_checking": {"enabled": True, "allow_unlisted_channels": False},
            "target_switch": {te.ACK_LEAF: "gw.example.org"},
            "connector": {
                EPICS_TYPE: {
                    "probe_channel": "LIVE:PROBE",
                    "writes_enabled": live_writes,
                    "gateways": (_gateways(5064, 5084) if live_gateways is None else live_gateways),
                },
                VA_TYPE: {
                    "probe_channel": "VA:PROBE",
                    "writes_enabled": va_writes,
                    "gateways": _gateways(5074, 5075) if va_gateways is None else va_gateways,
                },
            },
        },
        "archiver": {"type": "epics_archiver"},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    """A scratch agent-data root this process is stamped at, caches cleared.

    Both stamps together, never one without the other: a process that knows
    the session key and not the root reads a store nobody writes.
    """
    monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION)
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    session_store.invalidate_cache()
    yield tmp_path
    session_store.invalidate_cache()


def narrow(root, *targets: str, session: str = SESSION) -> None:
    """Record the operator's narrowing of *targets* for *session*."""
    path = root / session_store.STATE_DIR_NAME / session_store.STORE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({session: dict.fromkeys(targets, session_store.POSTURE_SANDBOX)}),
        encoding="utf-8",
    )
    session_store.invalidate_cache()


# ---------------------------------------------------------------------------
# The helper every live-session caller routes through
# ---------------------------------------------------------------------------


class TestEffectiveWritesForTarget:
    def test_an_unnarrowed_target_keeps_the_deployments_own_answer(self, store_root):
        section = _config()["control_system"]

        assert te.effective_writes_for_target(section, LIVE) is True
        assert te.effective_writes_for_target(section, VA) is True

    def test_a_narrowing_takes_arming_away_from_that_target_alone(self, store_root):
        section = _config()["control_system"]
        narrow(store_root, VA)

        assert te.effective_writes_for_target(section, VA) is False
        assert te.effective_writes_for_target(section, LIVE) is True

    def test_the_store_never_widens_a_deployment_that_arms_nothing(self, store_root):
        """Nothing in the store can arm a target the config left unarmed."""
        section = _config(live_writes=False, va_writes=False)["control_system"]

        assert te.effective_writes_for_target(section, LIVE) is False
        assert te.effective_writes_for_target(section, VA) is False

    def test_a_readonly_run_answers_no_before_the_store_is_consulted(self, store_root, monkeypatch):
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        section = _config()["control_system"]

        assert te.effective_writes_for_target(section, VA) is False


# ---------------------------------------------------------------------------
# Eligibility and the roster
# ---------------------------------------------------------------------------


class TestEligibilityFollowsTheStore:
    def test_a_narrowed_target_derives_the_read_gateway(self, store_root):
        """The whole point: eligibility answers for the session, not the file."""
        config = _config()
        narrow(store_root, VA)

        derivation = te.derive_endpoints(
            config, VA, writes_enabled=te.effective_writes_for_target(config["control_system"], VA)
        )

        assert derivation.selected_role == "read_only"
        assert derivation.selected_endpoint().port == 5074

    def test_evaluate_eligibility_defaults_to_this_sessions_posture(self, store_root):
        config = _config()
        narrow(store_root, VA)

        assert te.evaluate_eligibility(config, VA).eligible is True
        # Named explicitly, the caller's value wins over the store's.
        armed = te.evaluate_eligibility(config, VA, writes_enabled=True, readonly_run=False)
        assert armed.eligible is True

    def test_a_sandboxed_target_can_still_be_switched_to_and_from(self, store_root):
        """A narrowing is a write posture, never a switch gate.

        Both roles are configured, so the narrowed target selects ``read_only``
        and stays perfectly switchable — and the session sitting on it can
        still leave for a target it has not narrowed.
        """
        config = _config()
        narrow(store_root, VA)

        toward = te.target_availability(config, VA, session_target=LIVE, baseline_target=LIVE)
        away = te.target_availability(config, LIVE, session_target=VA, baseline_target=LIVE)

        assert toward.available_now is True
        assert toward.reason is None
        assert away.available_now is True

    def test_a_write_only_gateway_narrowed_reports_selected_role_missing(self, store_root):
        """The one case where narrowing really does cost the target.

        A block that configures ``write_access`` alone has no ``read_only`` row
        to fall back to, so the narrowed session selects a role this deployment
        never configured — and that is the reason it is told, in the eligibility
        module's own words.
        """
        config = _config(va_gateways=_gateways(None, 5075))
        narrow(store_root, VA)

        verdict = te.evaluate_eligibility(config, VA)

        assert verdict.eligible is False
        assert verdict.reason == te.REASON_SELECTED_ROLE_MISSING
        assert "'read_only'" in verdict.detail
        assert "control_system.connector.virtual_accelerator.gateways.read_only" in verdict.detail

    def test_the_same_write_only_block_is_eligible_while_the_session_is_armed(self, store_root):
        """Nothing about the config changed — only who is asking."""
        config = _config(va_gateways=_gateways(None, 5075))

        assert te.evaluate_eligibility(config, VA).eligible is True


class TestNarrowingRefusal:
    """The pre-flight a surface offering the narrowing owes the operator."""

    def test_it_names_what_a_write_only_block_would_lose(self, store_root):
        config = _config(va_gateways=_gateways(None, 5075))

        refusal = te.narrowing_refusal(config, VA)

        assert refusal is not None
        assert refusal.reason == te.REASON_SELECTED_ROLE_MISSING
        assert "'read_only'" in refusal.detail

    def test_an_ordinary_block_loses_nothing(self, store_root):
        assert te.narrowing_refusal(_config(), VA) is None
        assert te.narrowing_refusal(_config(), LIVE) is None

    def test_it_answers_the_narrowing_and_not_the_sessions_current_posture(self, store_root):
        """Hypothetical by construction: it reads no store and needs none.

        The target here is already narrowed, and the answer is the same one an
        armed session gets — otherwise a surface could only warn about a
        narrowing after the operator had already taken it.
        """
        config = _config(va_gateways=_gateways(None, 5075))
        narrow(store_root, VA)

        assert te.narrowing_refusal(config, VA).reason == te.REASON_SELECTED_ROLE_MISSING

    def test_a_readonly_run_does_not_answer_it_for_every_target(self, store_root, monkeypatch):
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

        assert te.narrowing_refusal(_config(), VA) is None

    def test_an_underivable_target_answers_with_that_reason(self, store_root):
        # A deployment that never named its real machine: the section's own
        # type is absent (so it resolves to the mock) and no connector block
        # names a machine either, which is what `live` cannot be derived from.
        config = _config()
        config["control_system"].pop("type")
        config["control_system"]["connector"].pop(EPICS_TYPE)

        refusal = te.narrowing_refusal(config, LIVE)

        assert refusal is not None
        assert refusal.reason == te.REASON_TARGET_UNRESOLVABLE


class TestRosterRows:
    def test_a_narrowed_rows_role_and_flag_are_the_same_answer(self, store_root):
        config = _config()
        narrow(store_root, VA)

        rows = control_target.target_rows(config, session_target=LIVE, baseline=LIVE)

        assert rows[VA]["writes_permitted"] is False
        assert rows[VA]["selected_role"] == "read_only"
        assert rows[VA]["endpoints"]["read_only"]["port"] == 5074
        # The target the operator left alone is untouched by the narrowing.
        assert rows[LIVE]["writes_permitted"] is True
        assert rows[LIVE]["selected_role"] == "write_access"

    def test_an_unnarrowed_roster_is_the_deployments_own_picture(self, store_root):
        rows = control_target.target_rows(_config(), session_target=LIVE, baseline=LIVE)

        assert rows[VA]["writes_permitted"] is True
        assert rows[VA]["selected_role"] == "write_access"

    def test_writes_permitted_routes_through_the_store(self, store_root):
        """The roster's flag is the store-aware helper, not a second reading."""
        config = _config()

        assert control_target._writes_permitted(config, VA) is True
        narrow(store_root, VA)
        assert control_target._writes_permitted(config, VA) is False
        assert control_target._writes_permitted(config, LIVE) is True

    def test_writes_permitted_still_refuses_a_readonly_run(self, store_root, monkeypatch):
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

        assert control_target._writes_permitted(_config(), VA) is False


# ---------------------------------------------------------------------------
# The switch itself, with real children
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    """A scratch directory the children import their VA connector from."""
    directory = tmp_path_factory.mktemp("store_switch_fixture")
    (directory / "switch_fixture_connectors.py").write_text(FIXTURE_MODULE, encoding="utf-8")
    (directory / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    return directory


@pytest.fixture
def child_environment(fixture_dir, monkeypatch):
    """Children see the repo, the fixture connector, and no ambient config."""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(fixture_dir), *REPO_PATHS]))
    monkeypatch.delenv("CONFIG_FILE", raising=False)


@pytest.fixture
async def make_manager(store_root, child_environment, monkeypatch):
    """Managers on the scratch root, whose children are all reaped after."""
    from osprey.mcp_server.control_system import target_state

    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: store_root)
    created = []

    def factory(raw, config_path):
        manager = ConnectorHostManager(
            MCPServerConfig(raw=raw, config_path=config_path),
            drain_timeout_s=1.0,
            probe_timeout_s=10.0,
            spawn_timeout_s=SPAWN_TIMEOUT_S,
            terminate_grace_s=2.0,
        )
        manager.reset_state()
        created.append(manager)
        return manager

    yield factory

    for manager in created:
        with contextlib.suppress(Exception):
            await manager.shutdown()


@pytest.fixture
def mixed_project(tmp_path):
    """A project config arming the simulator and leaving the machine unarmed."""
    return project_config(
        tmp_path,
        {"writes_enabled": False, "connector": {VIRTUAL_ACCELERATOR: {"writes_enabled": True}}},
    )


def _mixed_raw() -> dict[str, Any]:
    """The raw config whose ``va`` target has two DIFFERENT gateway ports.

    That difference is the whole instrument: where both roles name the same
    endpoint — the shipped virtual-accelerator preset — a parent and a child
    that disagree about the role still agree about the host and port, and the
    disagreement never surfaces.
    """
    return gateway_config(writes_enabled=False, va_writes_enabled=True, va_gateways=True)


class TestSwitchingOntoANarrowedTarget:
    """The reviewer's scenario, end to end on real connector-host children."""

    async def test_an_armed_target_lands_on_its_write_gateway(
        self, make_manager, mixed_project, store_root
    ):
        """The control: nothing narrowed, so the write role is the one served."""
        manager = make_manager(_mixed_raw(), mixed_project)
        await manager.ensure_started()

        result = await manager.switch(VA)

        assert result["selected_role"] == "write_access"
        assert result["endpoint"]["port"] == VA_WRITE_GATEWAY_PORT

    async def test_a_narrowed_target_switches_with_both_sides_on_the_read_gateway(
        self, make_manager, mixed_project, store_root
    ):
        """No SwitchError, and the two sides agree on the role and the port.

        Before the parent read the store, this raised a verification
        ``SwitchError``: the parent derived ``write_access`` from config while
        the child, reading the same narrowing the operator set, connected on
        ``read_only``.
        """
        narrow(store_root, VA)
        manager = make_manager(_mixed_raw(), mixed_project)
        await manager.ensure_started()

        result = await manager.switch(VA)

        # The parent's side.
        assert result["selected_role"] == "read_only"
        assert result["endpoint"]["port"] == VA_READ_GATEWAY_PORT
        assert result["endpoint"]["host"] == GATEWAY_HOST
        assert manager.active_target() == VA
        # The child's own post-connect report, which is what was verified.
        assert manager.status()["selected_role"] == "read_only"

    async def test_the_narrowed_session_can_still_read_the_target(
        self, make_manager, mixed_project, store_root
    ):
        """A narrowing takes writes away and leaves the session working."""
        narrow(store_root, VA)
        manager = make_manager(_mixed_raw(), mixed_project)
        await manager.ensure_started()
        await manager.switch(VA)

        value = await manager.active_proxy().read_channel(VA_PROBE, timeout=10.0)

        assert isinstance(value, ChannelValue)

    async def test_a_narrowing_of_the_active_target_survives_a_respawn(
        self, make_manager, mixed_project, store_root
    ):
        """The respawn re-derives, and the fresh child must verify against it.

        This is the path a session already sitting on the target takes when the
        operator narrows it: the parent replaces the child, and a parent still
        deriving the configured posture would refuse its own respawn.
        """
        manager = make_manager(_mixed_raw(), mixed_project)
        await manager.ensure_started()
        armed = await manager.switch(VA)
        assert armed["selected_role"] == "write_access"

        narrow(store_root, VA)
        respawned = await manager.respawn_same_target()

        assert respawned["target"] == VA
        assert respawned["selected_role"] == "read_only"
        assert respawned["endpoint"]["port"] == VA_READ_GATEWAY_PORT
        assert manager.status()["selected_role"] == "read_only"

    async def test_the_unnarrowed_target_is_unaffected_by_its_neighbours_narrowing(
        self, make_manager, mixed_project, store_root
    ):
        """A narrowing is per target: 'live' keeps the role its own block earns."""
        narrow(store_root, VA)
        manager = make_manager(_mixed_raw(), mixed_project)
        await manager.ensure_started()
        await manager.switch(VA)

        result = await manager.switch(LIVE)

        # 'live' inherits the deployment-wide off, so it was already read_only —
        # the point is that it switched at all, and against its own derivation.
        assert result["target"] == LIVE
        assert result["selected_role"] == "read_only"
        assert manager.active_target() == LIVE
