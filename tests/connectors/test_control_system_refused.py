"""A control-system denial of a write is a structured refusal, not an internal error.

When an IOC's access security refuses a put, the client library raises
``epics.ca.CASeverityException`` out of ``caput``. That exception used to escape
``write_channel`` and land in the MCP catch-all as ``internal_error`` — an
answer that told the operator nothing and misattributed the denial.

The invariant this file pins: such a denial comes back as
``ChannelWriteResult(success=False, blocked=True,
refusal_reason="CONTROL_SYSTEM_REFUSED")``, its message names the CONTROL
SYSTEM rather than OSPREY's reference monitor, and every rendering path that
sees it says the same thing. The narrowing matters as much as the catch: any
other exception raised by ``caput`` (a dead gateway, a timeout) is still a
genuine failure and still propagates untouched.

``epics`` is never imported here — not at module scope, not inside a test. The
connector resolves the exception class off the module it connected with, so a
locally defined stand-in attached to the mock is a faithful stand-in.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from osprey.connectors.control_system.base import (
    ChannelWriteResult,
    raise_for_write_result,
)
from osprey.connectors.control_system.epics_connector import EPICSConnector
from osprey.errors import ChannelWriteBlockedError
from osprey.mcp_server.control_system.error_handling import (
    ToolError,
    connector_error_handler,
)

CHANNEL = "TEST:MAG:PS:SP"


class CASeverityException(Exception):
    """Stand-in for ``epics.ca.CASeverityException``.

    pyepics is not installed in this environment and must not be imported by
    the connector module in any environment, so the production code resolves
    the class off the connected module object. That makes a locally defined
    subclass — attached where the real one lives — an exact stand-in.
    """

    def __init__(self, fcn="put", msg="Write access denied"):
        self.fcn = fcn
        self.msg = msg
        super().__init__(f" {fcn} returned '{msg}'")


def _writes_enabled_config(key, default=None):
    """Config stub: writes enabled so the base wrapper reaches write_channel."""
    if key == "control_system.writes_enabled":
        return True
    return default


def _make_connector(caput_side_effect=None, expose_exception_class=True):
    """Build an EPICSConnector wired with a mock epics module.

    Bypasses connect() (which imports pyepics) by setting the attributes the
    write path depends on directly. ``expose_exception_class`` controls whether
    the mock module carries a real exception class at ``ca.CASeverityException``,
    which is what production resolution keys on.
    """
    connector = EPICSConnector()
    connector._epics = MagicMock()
    connector._epics.caput = MagicMock(side_effect=caput_side_effect, return_value=True)
    if expose_exception_class:
        connector._epics.ca.CASeverityException = CASeverityException
    connector._limits_validator = MagicMock()
    connector._limits_validator.validate = MagicMock()
    connector._timeout = 5.0
    connector._connected = True
    return connector


async def _write(connector, value=42.0, level="none"):
    with patch("osprey.utils.config.get_config_value", side_effect=_writes_enabled_config):
        return await connector.write_channel(CHANNEL, value, verification_level=level)


class TestConnectorRefusal:
    @pytest.mark.asyncio
    async def test_access_denied_becomes_a_structured_refusal(self):
        """A denied caput comes back blocked, not raised and not a bare failure."""
        connector = _make_connector(caput_side_effect=CASeverityException())

        result = await _write(connector)

        assert isinstance(result, ChannelWriteResult)
        assert result.success is False
        assert result.blocked is True
        assert result.refusal_reason == "CONTROL_SYSTEM_REFUSED"
        # The caput WAS attempted — that is what distinguishes this refusal.
        connector._epics.caput.assert_called_once()

    @pytest.mark.asyncio
    async def test_message_names_the_control_system_and_the_channel(self):
        """The operator is told who refused, which channel, and that nothing moved."""
        connector = _make_connector(caput_side_effect=CASeverityException())

        result = await _write(connector)

        assert (
            f"Write to '{CHANNEL}' refused by the control system (access security); "
            "no value was written" in result.error_message
        )
        # The control system's own words survive into the message.
        assert "Write access denied" in result.error_message
        assert "reference monitor" not in result.error_message

    @pytest.mark.asyncio
    async def test_refusal_reason_stays_inside_the_shared_vocabulary(self):
        """The new code is part of the blocked-error vocabulary, not a local string."""
        connector = _make_connector(caput_side_effect=CASeverityException())

        result = await _write(connector)

        assert result.refusal_reason in ChannelWriteBlockedError._VALID_REASONS
        assert "CONTROL_SYSTEM_REFUSED" in ChannelWriteBlockedError._VALID_REASONS

    @pytest.mark.asyncio
    async def test_other_caput_errors_still_propagate_untouched(self):
        """The catch is narrow: a dead gateway is a failure, never a refusal.

        Reclassifying every caput exception as a refusal would tell a caller
        that nothing was written when in truth nobody knows — the opposite of
        the safety claim a refusal makes.
        """
        connector = _make_connector(caput_side_effect=ConnectionError("gateway down"))

        with pytest.raises(ConnectionError):
            await _write(connector)

    @pytest.mark.asyncio
    async def test_refusal_survives_a_verifying_write_level(self):
        """The denial is answered before verification, at every level."""
        connector = _make_connector(caput_side_effect=CASeverityException())

        result = await _write(connector, level="callback")

        assert result.blocked is True
        assert result.refusal_reason == "CONTROL_SYSTEM_REFUSED"

    @pytest.mark.asyncio
    async def test_a_mock_attribute_is_not_mistaken_for_the_exception_class(self):
        """The class guard rejects a MagicMock auto-attribute.

        ``self._epics`` is a MagicMock in every test that does not connect, so
        ``_epics.ca.CASeverityException`` always *exists* — it is a Mock, not a
        class. A truthiness check would build an ``except`` clause out of it and
        change behaviour for every mocked connector in the suite, which is why
        the guard is isinstance/issubclass. Without a real class attached,
        nothing is caught and the error propagates exactly as before.
        """
        connector = _make_connector(
            caput_side_effect=CASeverityException(), expose_exception_class=False
        )

        with pytest.raises(CASeverityException):
            await _write(connector)


class TestDenialContract:
    @pytest.mark.asyncio
    async def test_raise_for_write_result_raises_blocked_with_the_new_reason(self):
        """The denial contract routes the new code to the refusal exception."""
        connector = _make_connector(caput_side_effect=CASeverityException())

        result = await _write(connector)

        with pytest.raises(ChannelWriteBlockedError) as excinfo:
            raise_for_write_result(result)

        assert excinfo.value.reason == "CONTROL_SYSTEM_REFUSED"
        assert excinfo.value.channel_address == CHANNEL

    def test_bare_construction_does_not_misattribute_the_refusal(self):
        """With no message passed, the default text still names the right refuser."""
        err = ChannelWriteBlockedError(CHANNEL, "CONTROL_SYSTEM_REFUSED")

        assert str(err) == (
            f"Write to '{CHANNEL}' refused by the control system (CONTROL_SYSTEM_REFUSED)"
        )
        # Policy reasons keep their existing default verbatim.
        assert str(ChannelWriteBlockedError(CHANNEL, "LIMITS")) == (
            f"Write to '{CHANNEL}' refused by reference monitor (LIMITS)"
        )


async def _render(exc: Exception) -> dict:
    """Run one exception through the MCP tool error handler; return its envelope."""
    with pytest.raises(ToolError) as excinfo:
        async with connector_error_handler("channel_write"):
            raise exc
    return json.loads(str(excinfo.value))


class TestEnvelopeRendering:
    @pytest.mark.asyncio
    async def test_envelope_attributes_the_refusal_to_the_control_system(self):
        envelope = await _render(
            ChannelWriteBlockedError(
                CHANNEL,
                "CONTROL_SYSTEM_REFUSED",
                message=(
                    f"Write to '{CHANNEL}' refused by the control system "
                    "(access security); no value was written"
                ),
            )
        )

        assert envelope["error_type"] == "write_refused"
        assert "refused by the control system" in envelope["error_message"]
        assert "reference monitor" not in envelope["error_message"]
        rendered = " ".join(envelope["suggestions"])
        assert "reference monitor" not in rendered
        # The old guidance was flatly wrong here: the write WAS sent.
        assert "never sent to the control system" not in rendered
        assert "no value was written" in rendered
        assert envelope["details"] == {
            "channel": CHANNEL,
            "reason": "CONTROL_SYSTEM_REFUSED",
        }

    @pytest.mark.asyncio
    async def test_policy_refusals_render_exactly_as_before(self):
        """Other reasons keep their wording byte-for-byte."""
        envelope = await _render(ChannelWriteBlockedError(CHANNEL, "WRITES_DISABLED"))

        assert envelope["error_message"] == (
            "Write refused by the reference monitor during channel_write: "
            f"Write to '{CHANNEL}' refused by reference monitor (WRITES_DISABLED)"
        )
        assert envelope["suggestions"] == [
            "This write was refused on policy grounds; it was never sent to the control system.",
            "Do NOT attempt to work around the refusal.",
        ]


def _write_result_stub(channel, reason):
    """A minimal connector result the channel_write tool serialises unchanged."""
    stub = MagicMock()
    stub.channel_address = channel
    stub.value_written = 1.0
    stub.success = False
    stub.error_message = f"Write to '{channel}' refused"
    stub.blocked = True
    stub.refusal_reason = reason
    stub.verification = None
    return stub


async def _run_all_blocked_batch(tmp_path, monkeypatch, reason):
    """Drive the channel_write tool with a batch every op of which was refused."""
    from unittest.mock import AsyncMock

    from osprey.mcp_server.control_system.server_context import initialize_server_context
    from osprey.mcp_server.control_system.tools.channel_write import channel_write

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

    connector = AsyncMock()
    connector.write_multiple_channels.return_value = [
        _write_result_stub("PV:A", reason),
        _write_result_stub("PV:B", reason),
    ]

    fn = channel_write.fn if hasattr(channel_write, "fn") else channel_write
    with (
        patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ),
        patch(
            "osprey.connectors.control_system.limits_validator.LimitsValidator.from_config",
            return_value=None,
        ),
        pytest.raises(ToolError) as excinfo,
    ):
        await fn(
            operations=[
                {"channel": "PV:A", "value": 1.0},
                {"channel": "PV:B", "value": 1.0},
            ]
        )
    return json.loads(str(excinfo.value))


class TestAllBlockedBatchAttribution:
    """The batch escalation names the same refuser the single-write path does."""

    @pytest.mark.asyncio
    async def test_control_system_refusals_name_the_control_system(self, tmp_path, monkeypatch):
        envelope = await _run_all_blocked_batch(tmp_path, monkeypatch, "CONTROL_SYSTEM_REFUSED")

        assert envelope["error_type"] == "write_refused"
        assert (
            "All 2 write(s) refused by the control system: PV:A, PV:B"
            in (envelope["error_message"])
        )

    @pytest.mark.asyncio
    async def test_policy_refusals_still_name_the_reference_monitor(self, tmp_path, monkeypatch):
        envelope = await _run_all_blocked_batch(tmp_path, monkeypatch, "WRITES_DISABLED")

        assert (
            "All 2 write(s) refused by the reference monitor: PV:A, PV:B"
            in (envelope["error_message"])
        )
