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

Uses device 03 (``SR:MAG:HCM:03``) -- devices 01/02 are exclusively owned by
test_orbit_response.py / test_limits_enforcement.py for the life of the
session container.
"""

from __future__ import annotations

import pytest

from tests.va.e2e import conftest as e2e_conftest

CORRECTOR_SP = "SR:MAG:HCM:03:CURRENT:SP"
CORRECTOR_RB = "SR:MAG:HCM:03:CURRENT:RB"
DEMO_CURRENT = 10.0


class TestApprovalSmoke:
    @pytest.mark.asyncio
    async def test_write_blocked_when_writes_disabled_no_ca_io(self, va_container):
        with e2e_conftest.patched_config(**{"control_system.writes_enabled": False}):
            connector = await e2e_conftest.connect_va()

            sp_before = (await connector.read_channel(CORRECTOR_SP)).value
            rb_before = (await connector.read_channel(CORRECTOR_RB)).value

            result = await connector.write_channel(CORRECTOR_SP, DEMO_CURRENT)

        assert result.success is False
        assert "writes are disabled" in result.error_message
        assert "control_system.writes_enabled" in result.error_message

        # Read back with writes still disabled (reads are never gated) to
        # confirm the blocked write never reached the IOC over CA.
        with e2e_conftest.patched_config(**{"control_system.writes_enabled": False}):
            connector = await e2e_conftest.connect_va()
            sp_after = (await connector.read_channel(CORRECTOR_SP)).value
            rb_after = (await connector.read_channel(CORRECTOR_RB)).value

        assert sp_after == pytest.approx(sp_before), (
            f"{CORRECTOR_SP} changed from {sp_before} to {sp_after} despite the write "
            "being blocked at the connector guard"
        )
        assert rb_after == pytest.approx(rb_before), (
            f"{CORRECTOR_RB} changed from {rb_before} to {rb_after} despite the write "
            "being blocked at the connector guard"
        )

    @pytest.mark.asyncio
    async def test_same_write_succeeds_once_writes_enabled(self, va_container):
        with e2e_conftest.patched_config(**{"control_system.writes_enabled": True}):
            connector = await e2e_conftest.connect_va()

            result = await connector.write_channel(CORRECTOR_SP, DEMO_CURRENT)
            assert result.success, f"write unexpectedly blocked: {result.error_message}"

            sp_after = (await connector.read_channel(CORRECTOR_SP)).value
            assert sp_after == pytest.approx(DEMO_CURRENT)

            # Leave device 03 at a known, in-limits state (no limits validator
            # configured in this test, so nothing enforces this -- tidy anyway).
            reset = await connector.write_channel(CORRECTOR_SP, 0.0)
            assert reset.success
