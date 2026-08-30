"""``switch_gate``: the three refusals, as a function anything can ask.

The gate used to be inline in ``control_target_set``, which made the tool the
only way to find out whether a switch would be allowed. It is a function now
because a second caller needs the same answer — the session-control reconciler
asks it immediately before ``hosts.switch()``, on a path that has no MCP
request behind it — and a second implementation of "may this session move
there" is exactly the thing that would eventually disagree with the first.

So what is pinned here is not the wording of the three refusals (that is
``test_control_target_set.py``'s job, from the operator's side) but their
*identity*: for each branch, the verdict the gate returns is the refusal the
tool reports, compared against a live tool call rather than a copied string.
A gate that drifted from the tool would be worse than no gate, because the two
surfaces would then refuse the same switch for different stated reasons.
"""

from __future__ import annotations

import os

from osprey.mcp_server.control_system.target_eligibility import (
    REASON_ALREADY_ACTIVE,
    REASON_PROBE_CHANNEL_MISSING,
)
from osprey.mcp_server.control_system.tools import control_target
from tests.mcp_server import test_control_target_set as set_suite
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

allow_every_target = set_suite.allow_every_target
config_with_gateways = set_suite.config_with_gateways
install_context = set_suite.install_context
write_marker = set_suite.write_marker

# The switch-lifecycle fixtures, rebound so pytest collects them here too.
# ``state_root`` and ``child_environment`` are autouse, and are what anchors
# every state file this module writes under tmp_path.
child_environment = set_suite.child_environment
emitted = set_suite.emitted
fixture_dir = set_suite.fixture_dir
make_manager = set_suite.make_manager
state_root = set_suite.state_root

TOOL = get_tool_fn(control_target.control_target_set)


async def refusal_of(target: str) -> dict:
    """The envelope the tool reports for *target*, so it can be compared."""
    with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
        await TOOL(target=target)
    return ctx["envelope"]


def assert_verdict_is_the_refusal(verdict, envelope) -> None:
    """The gate's verdict and the tool's refusal are the same refusal."""
    assert verdict is not None
    assert verdict.detail == envelope["error_message"]
    assert verdict.details == envelope["details"]
    assert verdict.suggestions == envelope["suggestions"]
    assert verdict.reason == envelope["details"]["reason"]


class TestTheGateIsTheToolsOwnRefusals:
    async def test_a_readonly_run(self, make_manager, monkeypatch, emitted):
        manager = make_manager(raw=config_with_gateways())
        context = install_context(manager, monkeypatch)
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

        verdict = await control_target.switch_gate(context, "va")

        assert verdict.reason == control_target.REASON_READONLY_RUN
        assert_verdict_is_the_refusal(verdict, await refusal_of("va"))

    async def test_an_execution_in_flight(self, make_manager, monkeypatch, emitted):
        manager = make_manager(raw=config_with_gateways())
        context = install_context(manager, monkeypatch)
        write_marker("va", pid=os.getpid())

        verdict = await control_target.switch_gate(context, "va")

        assert verdict.reason == control_target.REASON_EXECUTION_IN_FLIGHT
        assert verdict.details["executing_target"] == "va"
        assert_verdict_is_the_refusal(verdict, await refusal_of("va"))

    async def test_an_ineligible_target(self, make_manager, monkeypatch, emitted):
        """Eligibility's own words travel through the gate unchanged."""
        manager = make_manager(raw=config_with_gateways(va_probe=None))
        context = install_context(manager, monkeypatch)

        verdict = await control_target.switch_gate(context, "va")

        assert verdict.reason == REASON_PROBE_CHANNEL_MISSING
        assert_verdict_is_the_refusal(verdict, await refusal_of("va"))

    async def test_the_active_target(self, make_manager, monkeypatch, emitted):
        """``already_active`` arrives inside eligibility, not as a fourth branch."""
        manager = make_manager(raw=config_with_gateways())
        context = install_context(manager, monkeypatch)
        active = manager.active_target()

        verdict = await control_target.switch_gate(context, active)

        assert verdict.reason == REASON_ALREADY_ACTIVE
        assert_verdict_is_the_refusal(verdict, await refusal_of(active))


class TestTheOrderIsTheToolsOrder:
    async def test_a_readonly_run_outranks_everything_later(self, make_manager, monkeypatch):
        """Both later checks would also refuse; the read-only run is still the answer."""
        manager = make_manager(raw=config_with_gateways(va_probe=None))
        context = install_context(manager, monkeypatch)
        write_marker("live", pid=os.getpid())
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

        verdict = await control_target.switch_gate(context, "va")

        assert verdict.reason == control_target.REASON_READONLY_RUN

    async def test_an_execution_in_flight_outranks_an_ineligible_target(
        self, make_manager, monkeypatch
    ):
        manager = make_manager(raw=config_with_gateways(va_probe=None))
        context = install_context(manager, monkeypatch)
        write_marker("live", pid=os.getpid())

        verdict = await control_target.switch_gate(context, "va")

        assert verdict.reason == control_target.REASON_EXECUTION_IN_FLIGHT


class TestNoneMeansProceed:
    async def test_a_switchable_target_returns_no_verdict(self, make_manager, monkeypatch):
        """Nothing to refuse: the caller may go on to ``hosts.switch()``."""
        manager = make_manager(raw=config_with_gateways())
        context = install_context(manager, monkeypatch)

        assert await control_target.switch_gate(context, "va") is None

    async def test_the_gate_reads_the_session_target_from_the_context(
        self, make_manager, monkeypatch
    ):
        """It takes the context and the wanted target, and nothing from a tool call.

        The reconciler has no MCP request to hand it, so everything the gate
        needs — the session's target of record and this deployment's baseline —
        it reads from the manager the context holds.
        """
        manager = make_manager(raw=config_with_gateways())
        context = install_context(manager, monkeypatch)
        allow_every_target(monkeypatch)

        assert await control_target.switch_gate(context, "va") is None
        assert manager.active_target() == "live"
