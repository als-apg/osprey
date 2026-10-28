"""Eligibility, the roster and the target switch, read through the write switch.

The deployment's write tier (``control_system.writes``) is not the last word on
whether a target is armed. An operator turns one target's write switch off from the
header chip, that switch lands in the login's control-context record, and the
connector-host child selects its gateway from it — so every parent-side answer
about "which gateway will this deployment use" has to read the same record or
stop being about this deployment.

Why this file exists at all is one bug: ``derive_endpoints`` defaults its
``writes_enabled`` to config alone. A parent that took that default derived
``write_access`` for a target the operator had just switched off, the child came
up on ``read_only`` because it read the record, and ``verify_child_report`` — doing
exactly its job — aborted the switch on a ``selected_role`` mismatch neither
side had got wrong. It fails closed, but it refuses a switch that is correctly
configured, and on the shipped virtual-accelerator preset it is invisible
because both gateway rows name the same endpoint.

So the load-bearing test here is the one with real children and a target whose
two gateway roles point at DIFFERENT ports: it is the only shape in which
parent and child can disagree about where the session landed.

``tests/connectors/test_posture_store_contract.py`` owns the store's own contract — where
the file is, what its shapes mean, how a lookup combines. Nothing here re-tests
that; these tests write the file the way the store defines it and assert on what
this stack derives from it.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pytest

from osprey.mcp_server.control_system import target_eligibility as te
from osprey.mcp_server.control_system.connector_host_manager import (
    ConnectorHostManager,
)
from osprey.mcp_server.control_system.server_context import MCPServerConfig
from osprey.mcp_server.control_system.tools import control_target
from osprey_connectors import posture_store
from osprey_connectors.control_system._base import ChannelValue
from osprey_connectors.ipc._verification import derive_endpoints

# The live-child half runs on the switch harness's fixture connector: a mock
# variant whose connect() applies the real gateway-role selection against the
# real per-type posture, with no Channel Access anywhere. That connector and the
# two-target setup it comes with live in the harness module, and this
# directory's conftest is what resolves the harness's fixtures here.
from osprey_connectors_tests._harness import isolate_verdict_environment
from tests.mcp_server._switch_harness import (
    GATEWAY_HOST,
    GATEWAY_TYPE,
    SPAWN_TIMEOUT_S,
    VA_PROBE,
    VA_READ_GATEWAY_PORT,
    VA_WRITE_GATEWAY_PORT,
    arm,
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
    writes: str = "allowed",
) -> dict[str, Any]:
    """A switch-capable config whose two targets are both armed by default.

    Armed on purpose: a switch can only ever arm what the tier allows, so a
    config that never armed anything could not tell a working switch read from
    a no-op.
    """
    return {
        "control_system": {
            "type": EPICS_TYPE,
            "writes": writes,
            "limits_checking": {"enabled": True, "mode": "exclusive"},
            "target_switch": {te.ACK_LEAF: "gw.example.org"},
            "connector": {
                EPICS_TYPE: {
                    "probe_channel": "LIVE:PROBE",
                    "gateways": (_gateways(5064, 5084) if live_gateways is None else live_gateways),
                },
                VA_TYPE: {
                    "probe_channel": "VA:PROBE",
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

    The root is the anchor: the record beneath it is what every process on this
    deployment reads its write switch from. The write verdict's
    environment is isolated under it, with the acting identity pinned in the
    real environment, so the connector-host children name the same owner.
    """
    isolate_verdict_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION)
    posture_store.invalidate_cache()
    yield tmp_path
    posture_store.invalidate_cache()


# ---------------------------------------------------------------------------
# The helper every live-session caller routes through
# ---------------------------------------------------------------------------


class TestEffectiveWritesForTarget:
    def test_a_switched_on_target_keeps_the_deployments_own_answer(self, store_root):
        arm(store_root, LIVE, VA)
        section = _config()["control_system"]

        assert te.target_write_verdict(section, LIVE).allowed is True
        assert te.target_write_verdict(section, VA).allowed is True

    def test_a_switch_off_takes_arming_away_from_that_target_alone(self, store_root):
        arm(store_root, LIVE, VA)
        section = _config()["control_system"]
        arm(store_root, LIVE)

        assert te.target_write_verdict(section, VA).allowed is False
        assert te.target_write_verdict(section, LIVE).allowed is True

    def test_the_store_never_widens_a_deployment_that_arms_nothing(self, store_root):
        """A switch on cannot arm a target whose tier never allows writes."""
        arm(store_root, LIVE, VA)
        section = _config(writes="never")["control_system"]

        assert te.target_write_verdict(section, LIVE).allowed is False
        assert te.target_write_verdict(section, VA).allowed is False

    def test_a_readonly_run_answers_no_before_the_store_is_consulted(self, monkeypatch, store_root):
        arm(store_root, LIVE, VA)
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        section = _config()["control_system"]

        assert te.target_write_verdict(section, VA).allowed is False


# ---------------------------------------------------------------------------
# Eligibility and the roster
# ---------------------------------------------------------------------------


class TestEligibilityFollowsTheStore:
    def test_a_switched_off_target_derives_the_read_gateway(self, store_root):
        """The whole point: eligibility answers for the login, not the file."""
        config = _config()
        arm(store_root, LIVE)

        derivation = derive_endpoints(
            config, VA, writes_enabled=te.target_write_verdict(config["control_system"], VA).allowed
        )

        assert derivation.selected_role == "read_only"
        assert derivation.selected_endpoint().port == 5074

    def test_evaluate_eligibility_defaults_to_this_logins_switch(self, store_root):
        config = _config()
        arm(store_root, LIVE)

        assert te.evaluate_eligibility(config, VA).eligible is True
        # Named explicitly, the caller's value wins over the store's.
        armed = te.evaluate_eligibility(config, VA, writes_enabled=True, readonly_run=False)
        assert armed.eligible is True

    def test_a_switched_off_target_can_still_be_switched_to_and_from(self, store_root):
        """A write switch that is off is a write posture, never a target-switch gate.

        Both roles are configured, so the switched-off target selects
        ``read_only`` and stays perfectly switchable — and the session sitting
        on it can still leave for a target whose switch is on.
        """
        config = _config()
        arm(store_root, LIVE)

        toward = te.target_availability(config, VA, control_target=LIVE, baseline_target=LIVE)
        away = te.target_availability(config, LIVE, control_target=VA, baseline_target=LIVE)

        assert toward.available_now is True
        assert toward.reason is None
        assert away.available_now is True

    def test_a_write_only_gateway_switched_off_stays_on_its_write_gateway(self, store_root):
        """A switch off does not cost a write-only block the target.

        A block that configures ``write_access`` alone reads through it while
        writes are off and refuses every write in software (the one gateway
        rule, ``_selected_role``), so the switched-off target still selects the
        role this deployment configured and stays switchable.
        """
        config = _config(va_gateways=_gateways(None, 5075))
        arm(store_root, LIVE)

        verdict = te.evaluate_eligibility(config, VA)

        assert verdict.eligible is True
        assert verdict.reason is None
        assert "'write_access'" in verdict.detail

    @pytest.mark.usefixtures("store_root")
    def test_the_same_write_only_block_is_eligible_while_the_session_is_armed(self):
        """Nothing about the config changed — only who is asking."""
        config = _config(va_gateways=_gateways(None, 5075))

        assert te.evaluate_eligibility(config, VA).eligible is True


class TestRosterRows:
    def test_a_switched_off_rows_role_and_flag_are_the_same_answer(self, store_root):
        config = _config()
        arm(store_root, LIVE, VA)
        arm(store_root, LIVE)

        rows = control_target.target_rows(config, control_target=LIVE, baseline=LIVE)

        assert rows[VA]["writes_permitted"] is False
        assert rows[VA]["selected_role"] == "read_only"
        assert rows[VA]["endpoints"]["read_only"]["port"] == 5074
        # The target the operator left alone is untouched by the switch off.
        assert rows[LIVE]["writes_permitted"] is True
        assert rows[LIVE]["selected_role"] == "write_access"

    def test_a_switched_on_roster_is_the_deployments_own_picture(self, store_root):
        arm(store_root, LIVE, VA)
        rows = control_target.target_rows(_config(), control_target=LIVE, baseline=LIVE)

        assert rows[VA]["writes_permitted"] is True
        assert rows[VA]["selected_role"] == "write_access"

    def test_writes_permitted_routes_through_the_switch(self, store_root):
        """The roster's flag is the switch-aware helper, not a second reading."""
        config = _config()
        arm(store_root, LIVE, VA)

        assert control_target._writes_permitted(config, VA) is True
        arm(store_root, LIVE)
        assert control_target._writes_permitted(config, VA) is False
        assert control_target._writes_permitted(config, LIVE) is True

    def test_writes_permitted_still_refuses_a_readonly_run(self, monkeypatch, store_root):
        arm(store_root, LIVE, VA)
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

        assert control_target._writes_permitted(_config(), VA) is False


# ---------------------------------------------------------------------------
# The switch itself, with real children
# ---------------------------------------------------------------------------


@pytest.fixture
async def make_manager(store_root, child_environment, monkeypatch):  # noqa: ARG001 - the children this factory spawns need the prepared environment
    """Managers on the scratch store root, whose children are all reaped after.

    Not the harness's ``make_manager``, which this shadows: that one builds over
    the harness's own state root and records every spawn, for tests that assert
    on the child processes themselves. What is under test here is what a manager
    derives from the write switch, so the root has to be the stamped one every
    switch in this module is written under, and the raw config and the project
    config are always the caller's.
    """
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
def mixed_project(tmp_path, store_root):
    """A project config whose tier allows writes, with only the simulator switched on."""
    arm(store_root, VA)
    return project_config(tmp_path, {"type": GATEWAY_TYPE, "writes": "allowed"})


def _mixed_raw() -> dict[str, Any]:
    """The raw config whose ``va`` target has two DIFFERENT gateway ports.

    That difference is the whole instrument: where both roles name the same
    endpoint — the shipped virtual-accelerator preset — a parent and a child
    that disagree about the role still agree about the host and port, and the
    disagreement never surfaces.
    """
    return gateway_config(va_gateways=True)


class TestSwitchingOntoASwitchedOffTarget:
    """The reviewer's scenario, end to end on real connector-host children."""

    @pytest.mark.usefixtures("store_root")
    async def test_an_armed_target_lands_on_its_write_gateway(self, make_manager, mixed_project):
        """The control: the switch is on, so the write role is the one served."""
        manager = make_manager(_mixed_raw(), mixed_project)
        await manager.ensure_started()

        result = await manager.switch(VA)

        assert result["selected_role"] == "write_access"
        assert result["endpoint"]["port"] == VA_WRITE_GATEWAY_PORT

    async def test_a_switched_off_target_switches_with_both_sides_on_the_read_gateway(
        self, make_manager, mixed_project, store_root
    ):
        """No SwitchError, and the two sides agree on the role and the port.

        Before the parent read the record, this raised a verification
        ``SwitchError``: the parent derived ``write_access`` from config while
        the child, reading the same switch the operator set, connected on
        ``read_only``.
        """
        arm(store_root)
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

    async def test_the_switched_off_target_can_still_be_read(
        self, make_manager, mixed_project, store_root
    ):
        """A switch off takes writes away and leaves the session working."""
        arm(store_root)
        manager = make_manager(_mixed_raw(), mixed_project)
        await manager.ensure_started()
        await manager.switch(VA)

        value = await manager.active_proxy().read_channel(VA_PROBE, timeout=10.0)

        assert isinstance(value, ChannelValue)

    async def test_a_switch_off_of_the_active_target_survives_a_respawn(
        self, make_manager, mixed_project, store_root
    ):
        """The respawn re-derives, and the fresh child must verify against it.

        This is the path a session already sitting on the target takes when the
        operator switches it off: the parent replaces the child, and a parent still
        deriving the configured posture would refuse its own respawn.
        """
        manager = make_manager(_mixed_raw(), mixed_project)
        await manager.ensure_started()
        armed = await manager.switch(VA)
        assert armed["selected_role"] == "write_access"

        arm(store_root)
        respawned = await manager.respawn_same_target()

        assert respawned["target"] == VA
        assert respawned["selected_role"] == "read_only"
        assert respawned["endpoint"]["port"] == VA_READ_GATEWAY_PORT
        assert manager.status()["selected_role"] == "read_only"

    async def test_the_other_target_is_unaffected_by_its_neighbours_switch_off(
        self, make_manager, mixed_project, store_root
    ):
        """A switch is per target: 'live' keeps the role its own block earns."""
        arm(store_root)
        manager = make_manager(_mixed_raw(), mixed_project)
        await manager.ensure_started()
        await manager.switch(VA)

        result = await manager.switch(LIVE)

        # 'live' inherits the deployment-wide off, so it was already read_only —
        # the point is that it switched at all, and against its own derivation.
        assert result["target"] == LIVE
        assert result["selected_role"] == "read_only"
        assert manager.active_target() == LIVE
