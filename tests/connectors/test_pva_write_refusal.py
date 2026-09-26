"""PVA-routed addresses are read-only: ``write_channel`` refuses them outright.

OSPREY's PVAccess support is a read path. An address matching one of the
``control_system.connector.epics.pva_channels`` globs must never reach a write
primitive — not p4p's ``put``, and not pyepics' ``caput`` either (the CA client
would happily write a same-named record over a different transport).

The refusal is the FIRST statement of ``write_channel`` for a concrete reason,
and that ordering is what most of this file protects: the very next step is
limits validation, which reads the channel's current value with a blocking
``caget`` to check ``max_step``. A refusal placed after it would put the CA
client on an address routed over PVAccess — the one thing this refusal exists
to prevent — and would do it while comparing an image array against a scalar
limit. The array cases below are the regression, not a formality.

Convention (matching ``test_epics_connector.py``): the base class wraps
``write_channel`` with a ``_writes_enabled`` pre-check that is False in a
config-less test environment, so the ``writes_enabled`` fixture patches it as a
property to let the real body run. Fakes are injected instead of connecting.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from osprey.connectors.control_system.base import WriteOutcome
from tests.connectors._epics_fakes import (
    ca_connector,
    fake_p4p_module,
    writes_enabled,  # noqa: F401 - fixture, used by name
)

PVA_GLOB = "SR:CAM*:IMAGE"
PVA_ADDRESS = "SR:CAM1:IMAGE"
CA_ADDRESS = "SR:BEAM:CURRENT"


class _RecordingValidator:
    """A limits validator that records every call and lets it through.

    It must not raise: ``write_channel`` turns any exception from ``validate``
    into the same REFUSED outcome these tests expect, so a raising fake would
    hide the very ordering bug it is meant to catch. The tests read the call
    lists instead.
    """

    def __init__(self):
        self.validate_calls = []
        self.resolve_confirm_calls = []

    def validate(self, channel_address, value, read_current=None):
        self.validate_calls.append((channel_address, value, read_current))

    def resolve_confirm(self, channel_address):
        self.resolve_confirm_calls.append(channel_address)
        return True


def _connector(*, globs=(PVA_GLOB,)):
    """A connector wired for both transports without touching ``connect()``."""
    connector = ca_connector(
        epics=MagicMock(name="epics"), limits_validator=_RecordingValidator(), timeout=3.0
    )
    connector._epics.caput.return_value = True
    connector._pva_channel_globs = list(globs)
    # A p4p stub carrying only ``Context``, so any use of it would be visible.
    connector._p4p = fake_p4p_module(errors=False)
    connector._pva_context = MagicMock(name="pva_context")
    return connector


def _assert_nothing_downstream_ran(connector):
    """Neither transport, nor the limits validator, may have been touched at all.

    The validator's ``max_step`` check does a blocking ``caget`` on the address,
    and a camera image has no scalar limit to be compared against. Both are
    reached only by a write that was not refused first.
    """
    validator = connector._limits_validator
    assert validator.validate_calls == [], (
        f"limits validation ran for a refused write: {validator.validate_calls}"
    )
    assert validator.resolve_confirm_calls == [], (
        f"confirm policy resolved for a refused write: {validator.resolve_confirm_calls}"
    )
    assert connector._pva_context.mock_calls == [], (
        f"PVA context was used on a refused write: {connector._pva_context.mock_calls}"
    )
    assert connector._epics.mock_calls == [], (
        f"CA client was used on a refused write: {connector._epics.mock_calls}"
    )


def _assert_refusal(result, address, value):
    assert result.outcome is WriteOutcome.REFUSED
    assert result.refusal_reason == "VALIDATION_ERROR"
    assert result.channel_address == address
    assert result.observed_value is None
    message = result.error_message
    assert message is not None
    assert address in message
    assert "PVAccess writes are not supported" in message
    assert "pva_channels" in message
    assert "No write was attempted." in message
    # The refused value is echoed back untouched, arrays included.
    assert result.value_written is value


# ---------------------------------------------------------------------------
# Refusal
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("writes_enabled")
class TestPvaWriteRefusal:
    @pytest.mark.asyncio
    async def test_scalar_write_to_a_pva_address_is_refused(self):
        connector = _connector()

        result = await connector.write_channel(PVA_ADDRESS, 1.5)

        _assert_refusal(result, PVA_ADDRESS, 1.5)
        _assert_nothing_downstream_ran(connector)

    @pytest.mark.asyncio
    async def test_array_write_to_a_pva_address_is_refused(self):
        """The array case is the one the ordering exists for."""
        connector = _connector()
        value = np.zeros((48, 64), dtype=np.uint8)

        result = await connector.write_channel(PVA_ADDRESS, value)

        _assert_refusal(result, PVA_ADDRESS, value)
        _assert_nothing_downstream_ran(connector)

    @pytest.mark.asyncio
    async def test_explicit_confirm_does_not_bypass_the_refusal(self):
        """A caller-supplied confirm must not open a side door around the check."""
        connector = _connector()

        result = await connector.write_channel(PVA_ADDRESS, 1.0, confirm=False)

        _assert_refusal(result, PVA_ADDRESS, 1.0)
        _assert_nothing_downstream_ran(connector)


# ---------------------------------------------------------------------------
# The CA path is untouched
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("writes_enabled")
class TestChannelAccessWritesUnchanged:
    @pytest.mark.asyncio
    async def test_non_matching_address_still_writes_over_ca(self):
        connector = _connector()

        result = await connector.write_channel(CA_ADDRESS, 4.2, confirm=False)

        assert result.outcome is WriteOutcome.UNREQUESTED
        assert result.refusal_reason is None
        connector._epics.caput.assert_called_once()
        assert connector._epics.caput.call_args.args[0] == CA_ADDRESS
        assert connector._pva_context.mock_calls == []

    @pytest.mark.asyncio
    async def test_connector_without_pva_globs_writes_everything_over_ca(self):
        """The default (no pva_channels configured) is zero behavior change."""
        connector = _connector(globs=())

        result = await connector.write_channel(PVA_ADDRESS, 4.2, confirm=False)

        assert result.outcome is WriteOutcome.UNREQUESTED
        connector._epics.caput.assert_called_once()
