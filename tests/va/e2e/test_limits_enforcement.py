"""SC9 acceptance: limits enforcement against the real container.

The limits database is the view a build renders from the preset's
``data/facility/limits.yaml``, loaded in the ``optional`` mode. Two correctors
are driven, each named by its slot in the served tree's own
``va_bindings.json``, never by a device name written down here:

- slot 0 carries a +-12 A record. The out-of-limits writes go to it, and they
  are rejected *before* they reach the IOC: ``LimitsValidator.validate()``
  raises synchronously, before ``EPICSConnector.write_channel`` issues any
  ``epics.caput`` at all, so a rejected write can never move the record over
  CA. That is also why these legs leave ``test_orbit_response.py``'s ownership
  of slot 0 undisturbed: nothing is ever written there.
- slot 1 carries no record. Under the ``optional`` mode a channel with no
  record is written as is, so the in-limits leg writes it and reads it back.

Only min/max is exercised: the record carries no ``max_step``.
"""

from __future__ import annotations

import pytest

from osprey.errors import ChannelLimitsViolationError
from osprey.services.virtual_accelerator.bindings import Binding
from osprey_connectors.control_system import WriteOutcome
from tests.va.e2e import conftest as e2e_conftest

#: The kick slot whose setpoint carries a limits record, and the slot whose
#: setpoint carries none. Slot 0 is ``test_orbit_response.py``'s for the life
#: of the session container; this lane only ever sends it writes the limits
#: check refuses before any caput.
_BANDED_SLOT = 0
_UNLIMITED_SLOT = 1

IN_LIMITS_CURRENT = 10.0
OUT_OF_LIMITS_HIGH = 15.0  # the record's window is +-12A
OUT_OF_LIMITS_LOW = -14.0

LIMITS_OVERRIDES = {
    "control_system.writes_enabled": True,
    "control_system.limits_checking.enabled": True,
    "control_system.limits_checking.database_path": str(e2e_conftest.LIMITS_DB_PATH),
    "control_system.limits_checking.mode": "optional",
}

#: Floor for this module's own test count -- a guard against a refactor that
#: leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 3


@pytest.fixture(scope="module")
def banded() -> Binding:
    """The kick binding whose setpoint carries a limits record.

    A fixture rather than a module constant: which channel this is, is a
    question about the served tree, and a tree that cannot answer it belongs in
    this lane's own failure rather than in the collection of every lane beside
    it.
    """
    return e2e_conftest.kick_binding(_BANDED_SLOT)


@pytest.fixture(scope="module")
def banded_sp(banded: Binding) -> str:
    """The banded address this lane sends out-of-limits demands to."""
    return banded.setpoint_address


@pytest.fixture(scope="module")
def banded_rb(banded: Binding) -> str:
    """The address the banded magnet reads its own field back on."""
    assert banded.readback_address is not None
    return banded.readback_address


@pytest.fixture(scope="module")
def unlimited_sp() -> str:
    """The address with no limits record this lane writes a demand to."""
    return e2e_conftest.kick_binding(_UNLIMITED_SLOT).setpoint_address


class TestLimitsEnforcement:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("va_container")
    async def test_out_of_limits_write_rejected_before_reaching_ioc(self, banded_sp, banded_rb):
        with e2e_conftest.patched_config(**LIMITS_OVERRIDES):
            connector = await e2e_conftest.connect_va()

            sp_before = (await connector.read_channel(banded_sp)).value
            rb_before = (await connector.read_channel(banded_rb)).value

            with pytest.raises(ChannelLimitsViolationError) as exc_info:
                await connector.write_channel(banded_sp, OUT_OF_LIMITS_HIGH)
            assert exc_info.value.violation_type == "MAX_EXCEEDED"

            with pytest.raises(ChannelLimitsViolationError) as exc_info:
                await connector.write_channel(banded_sp, OUT_OF_LIMITS_LOW)
            assert exc_info.value.violation_type == "MIN_EXCEEDED"

            sp_after = (await connector.read_channel(banded_sp)).value
            rb_after = (await connector.read_channel(banded_rb)).value
            assert sp_after == pytest.approx(sp_before), (
                f"{banded_sp} changed from {sp_before} to {sp_after} despite a "
                "rejected out-of-limits write"
            )
            assert rb_after == pytest.approx(rb_before), (
                f"{banded_rb} changed from {rb_before} to {rb_after} despite a "
                "rejected out-of-limits write"
            )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("va_container")
    async def test_unlimited_write_succeeds(self, unlimited_sp):
        with e2e_conftest.patched_config(**LIMITS_OVERRIDES):
            connector = await e2e_conftest.connect_va()

            result = await connector.write_channel(unlimited_sp, IN_LIMITS_CURRENT)
            assert result.outcome is WriteOutcome.CONFIRMED, (
                f"write {result.outcome}: {result.error_message or result.notes}"
            )

            sp_after = (await connector.read_channel(unlimited_sp)).value
            assert sp_after == pytest.approx(IN_LIMITS_CURRENT)

            # Leave this lane's corrector at a known state.
            reset = await connector.write_channel(unlimited_sp, 0.0)
            assert reset.outcome is WriteOutcome.CONFIRMED


# ---------------------------------------------------------------------------


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_limits_enforcement.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
