"""FR5 acceptance: VA inherits the same base-class write-safety wiring as
EPICS/mock -- ``VirtualAcceleratorConnector`` is an unmodified
``EPICSConnector`` subclass, so ``control_system.writes_enabled: false``
must hard-block writes at the connector guard with zero CA I/O, exactly as
``tests/connectors/test_writes_enabled.py`` proves for the base class in
isolation. This test proves the same contract against a real container: the
blocked write never reaches the IOC (read-back over CA confirms no change),
and the *same* channel accepts the write once writes are enabled -- so the
blocking is demonstrably the guard, not something else broken.

The mandatory-approval hook path itself (the agent's PreToolUse hook chain)
is connector-agnostic -- it gates on the detected write pattern, not on
``control_system.type`` -- and is already covered by
``tests/hooks/test_approval_hook.py`` and friends; nothing about VA changes
that layer, so it isn't re-tested here.

Drives the third corrector the served tree binds -- the first two are
exclusively owned by test_orbit_response.py / test_limits_enforcement.py for
the life of the session container. Which channel that is comes from the tree's
own ``va_bindings.json``, never a device name written down here.
"""

from __future__ import annotations

import pytest

from osprey.services.virtual_accelerator.bindings import Binding
from osprey_connectors.control_system import WriteOutcome
from tests.va.e2e import conftest as e2e_conftest

#: Which writable corrector this lane drives, as a slot in the served tree's
#: own kick bindings rather than a device name: slot 0 is
#: ``test_orbit_response.py``'s for the life of the session container and slot
#: 1 is ``test_limits_enforcement.py``'s, and this lane owns slot 2. Nothing
#: about the addresses is spelled here
#: -- the bindings document says which channels kick the beam and where each
#: one reads back.
_CORRECTOR_SLOT = 2

DEMO_CURRENT = 10.0


@pytest.fixture(scope="module")
def corrector() -> Binding:
    """The kick binding this lane drives, read when the lane runs.

    A fixture rather than a module constant: which channel this is, is a
    question about the served tree, and a tree that cannot answer it belongs in
    this lane's own failure rather than in the collection of every lane beside
    it.
    """
    return e2e_conftest.kick_binding(_CORRECTOR_SLOT)


@pytest.fixture(scope="module")
def corrector_sp(corrector: Binding) -> str:
    """The address this lane writes a demand to."""
    return corrector.setpoint_address


@pytest.fixture(scope="module")
def corrector_rb(corrector: Binding) -> str:
    """The address the same magnet reads its own field back on."""
    assert corrector.readback_address is not None
    return corrector.readback_address


class TestApprovalSmoke:
    @pytest.mark.asyncio
    async def test_write_blocked_when_writes_disabled_no_ca_io(
        self, va_container, corrector_sp, corrector_rb
    ):
        with e2e_conftest.patched_config(**{"control_system.writes_enabled": False}):
            connector = await e2e_conftest.connect_va()

            sp_before = (await connector.read_channel(corrector_sp)).value
            rb_before = (await connector.read_channel(corrector_rb)).value

            result = await connector.write_channel(corrector_sp, DEMO_CURRENT)

        assert result.outcome is WriteOutcome.REFUSED
        assert "writes are disabled" in result.error_message
        assert "control_system.writes_enabled" in result.error_message

        # Read back with writes still disabled (reads are never gated) to
        # confirm the blocked write never reached the IOC over CA.
        with e2e_conftest.patched_config(**{"control_system.writes_enabled": False}):
            connector = await e2e_conftest.connect_va()
            sp_after = (await connector.read_channel(corrector_sp)).value
            rb_after = (await connector.read_channel(corrector_rb)).value

        assert sp_after == pytest.approx(sp_before), (
            f"{corrector_sp} changed from {sp_before} to {sp_after} despite the write "
            "being blocked at the connector guard"
        )
        assert rb_after == pytest.approx(rb_before), (
            f"{corrector_rb} changed from {rb_before} to {rb_after} despite the write "
            "being blocked at the connector guard"
        )

    @pytest.mark.asyncio
    async def test_same_write_succeeds_once_writes_enabled(self, va_container, corrector_sp):
        with e2e_conftest.patched_config(**{"control_system.writes_enabled": True}):
            connector = await e2e_conftest.connect_va()

            result = await connector.write_channel(corrector_sp, DEMO_CURRENT)
            assert result.outcome is WriteOutcome.CONFIRMED, (
                f"write unexpectedly {result.outcome}: {result.error_message or result.notes}"
            )

            sp_after = (await connector.read_channel(corrector_sp)).value
            assert sp_after == pytest.approx(DEMO_CURRENT)

            # Leave this lane's corrector at a known, in-limits state (no limits
            # validator configured in this test, so nothing enforces this -- tidy
            # anyway).
            reset = await connector.write_channel(corrector_sp, 0.0)
            assert reset.outcome is WriteOutcome.CONFIRMED
