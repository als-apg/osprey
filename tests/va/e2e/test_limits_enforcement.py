"""SC9 acceptance: channel_limits.json enforcement against the real container.

Drives the second corrector the served tree binds -- the first is exclusively
owned by test_orbit_response.py for the life of the session container. Which
channel that is comes from the tree's own ``va_bindings.json``, never a device
name written down here.

Per findings recorded during this run: the shipped corrector entries in
channel_limits.json carry a +-12A min/max window and omit
max_step entirely (max_step enforcement is connector-blind -- see
``.claude/.logs/FRAMEWORK_GAP_max_step_connector_blind.md`` -- so this suite
only exercises min/max, matching what's actually shipped). The out-of-limits
write must be rejected *before* it ever reaches the IOC: LimitsValidator.validate()
raises synchronously, before ``EPICSConnector.write_channel`` issues any
``epics.caput`` at all, so a rejected write can never move the record over CA.
"""

from __future__ import annotations

import pytest

from osprey.errors import ChannelLimitsViolationError
from osprey.services.virtual_accelerator.bindings import Binding
from osprey_connectors.control_system import WriteOutcome
from tests.va.e2e import conftest as e2e_conftest

#: Which writable corrector this lane drives, as a slot in the served tree's
#: own kick bindings rather than a device name: slot 0 is
#: ``test_orbit_response.py``'s for the life of the session container, and
#: this lane owns slot 1. Nothing about the addresses is spelled here
#: -- the bindings document says which channels kick the beam and where each
#: one reads back.
_CORRECTOR_SLOT = 1

IN_LIMITS_CURRENT = 10.0
OUT_OF_LIMITS_HIGH = 15.0  # shipped window is +-12A
OUT_OF_LIMITS_LOW = -14.0

LIMITS_OVERRIDES = {
    "control_system.writes_enabled": True,
    "control_system.limits_checking.enabled": True,
    "control_system.limits_checking.database_path": str(e2e_conftest.LIMITS_DB_PATH),
    "control_system.limits_checking.allow_unlisted_channels": True,
}


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


class TestLimitsEnforcement:
    @pytest.mark.asyncio
    async def test_out_of_limits_write_rejected_before_reaching_ioc(
        self, va_container, corrector_sp, corrector_rb
    ):
        with e2e_conftest.patched_config(**LIMITS_OVERRIDES):
            connector = await e2e_conftest.connect_va()

            sp_before = (await connector.read_channel(corrector_sp)).value
            rb_before = (await connector.read_channel(corrector_rb)).value

            with pytest.raises(ChannelLimitsViolationError) as exc_info:
                await connector.write_channel(corrector_sp, OUT_OF_LIMITS_HIGH)
            assert exc_info.value.violation_type == "MAX_EXCEEDED"

            with pytest.raises(ChannelLimitsViolationError) as exc_info:
                await connector.write_channel(corrector_sp, OUT_OF_LIMITS_LOW)
            assert exc_info.value.violation_type == "MIN_EXCEEDED"

            sp_after = (await connector.read_channel(corrector_sp)).value
            rb_after = (await connector.read_channel(corrector_rb)).value
            assert sp_after == pytest.approx(sp_before), (
                f"{corrector_sp} changed from {sp_before} to {sp_after} despite a "
                "rejected out-of-limits write"
            )
            assert rb_after == pytest.approx(rb_before), (
                f"{corrector_rb} changed from {rb_before} to {rb_after} despite a "
                "rejected out-of-limits write"
            )

    @pytest.mark.asyncio
    async def test_in_limits_write_succeeds(self, va_container, corrector_sp):
        with e2e_conftest.patched_config(**LIMITS_OVERRIDES):
            connector = await e2e_conftest.connect_va()

            result = await connector.write_channel(corrector_sp, IN_LIMITS_CURRENT)
            assert result.outcome is WriteOutcome.CONFIRMED, (
                f"in-limits write {result.outcome}: {result.error_message or result.notes}"
            )

            sp_after = (await connector.read_channel(corrector_sp)).value
            assert sp_after == pytest.approx(IN_LIMITS_CURRENT)

            # Leave this lane's corrector at a known, in-limits state.
            reset = await connector.write_channel(corrector_sp, 0.0)
            assert reset.outcome is WriteOutcome.CONFIRMED
