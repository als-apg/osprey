"""SC9 acceptance: channel_limits.json enforcement against the real container.

Uses device 02 (``SR:MAG:HCM:02``) -- device 01 is exclusively owned by
test_orbit_response.py for the life of the session container.

Per findings recorded during this run: the shipped SR HCM/VCM corrector
entries in channel_limits.json carry a +-12A min/max window and omit
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
from tests.va.e2e import conftest as e2e_conftest

CORRECTOR_SP = "SR:MAG:HCM:02:CURRENT:SP"
CORRECTOR_RB = "SR:MAG:HCM:02:CURRENT:RB"

IN_LIMITS_CURRENT = 10.0
OUT_OF_LIMITS_HIGH = 15.0  # shipped window is +-12A
OUT_OF_LIMITS_LOW = -14.0

LIMITS_OVERRIDES = {
    "control_system.writes_enabled": True,
    "control_system.limits_checking.enabled": True,
    "control_system.limits_checking.database_path": str(e2e_conftest.LIMITS_DB_PATH),
    "control_system.limits_checking.allow_unlisted_channels": True,
}


class TestLimitsEnforcement:
    @pytest.mark.asyncio
    async def test_out_of_limits_write_rejected_before_reaching_ioc(self, va_container):
        with e2e_conftest.patched_config(**LIMITS_OVERRIDES):
            connector = await e2e_conftest.connect_va()

            sp_before = (await connector.read_channel(CORRECTOR_SP)).value
            rb_before = (await connector.read_channel(CORRECTOR_RB)).value

            with pytest.raises(ChannelLimitsViolationError) as exc_info:
                await connector.write_channel(CORRECTOR_SP, OUT_OF_LIMITS_HIGH)
            assert exc_info.value.violation_type == "MAX_EXCEEDED"

            with pytest.raises(ChannelLimitsViolationError) as exc_info:
                await connector.write_channel(CORRECTOR_SP, OUT_OF_LIMITS_LOW)
            assert exc_info.value.violation_type == "MIN_EXCEEDED"

            sp_after = (await connector.read_channel(CORRECTOR_SP)).value
            rb_after = (await connector.read_channel(CORRECTOR_RB)).value
            assert sp_after == pytest.approx(sp_before), (
                f"{CORRECTOR_SP} changed from {sp_before} to {sp_after} despite a "
                "rejected out-of-limits write"
            )
            assert rb_after == pytest.approx(rb_before), (
                f"{CORRECTOR_RB} changed from {rb_before} to {rb_after} despite a "
                "rejected out-of-limits write"
            )

    @pytest.mark.asyncio
    async def test_in_limits_write_succeeds(self, va_container):
        with e2e_conftest.patched_config(**LIMITS_OVERRIDES):
            connector = await e2e_conftest.connect_va()

            result = await connector.write_channel(CORRECTOR_SP, IN_LIMITS_CURRENT)
            assert result.success, f"in-limits write rejected: {result.error_message}"

            sp_after = (await connector.read_channel(CORRECTOR_SP)).value
            assert sp_after == pytest.approx(IN_LIMITS_CURRENT)

            # Leave device 02 at a known, in-limits state.
            reset = await connector.write_channel(CORRECTOR_SP, 0.0)
            assert reset.success
