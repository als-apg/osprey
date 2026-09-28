"""A control-system denial of a write is a structured refusal, not an internal error.

When an IOC's access security refuses a put, pvapy raises its one exception
type, ``pvaccess.PvaException``, with a message ending ``Write access denied``.
Such an exception escaping ``write_channel`` would land in the MCP catch-all as
``internal_error`` — an answer that told the operator nothing and misattributed
the denial.

The invariant this file pins: such a denial comes back as
``ChannelWriteResult(outcome=WriteOutcome.REFUSED,
refusal_reason="CONTROL_SYSTEM_REFUSED")``, its message names the CONTROL
SYSTEM rather than OSPREY's reference monitor, and every rendering path that
sees it says the same thing. The narrowing matters as much as the catch: any
other exception raised by the put (a dead gateway, an unrecognized pvapy
failure) is still a genuine failure and still propagates (pinned in
test_write_fail_closed.py).

The real ``pvaccess`` is never used here. The connector resolves the exception
class off the module it connected with, so the fake module's
``PvaException`` is a faithful stand-in; the message text it carries is the
one ``tests/connectors/test_epics_soft_ioc.py`` pins against a real IOC.
"""

import json
from unittest.mock import patch

import pytest

from osprey.connectors.control_system.base import (
    ChannelWriteResult,
    WriteOutcome,
    raise_for_write_result,
)
from osprey.errors import ChannelWriteBlockedError
from osprey.mcp_server.control_system.error_handling import (
    ToolError,
    connector_error_handler,
)
from tests.connectors._epics_fakes import access_denied
from tests.connectors._write_fakes import make_mock_epics_connector
from tests.connectors._write_fakes import writes_enabled_config as _writes_enabled_config

CHANNEL = "TEST:MAG:PS:SP"


def _make_connector(put_error=None):
    """A connector over a fake ``pvaccess`` whose put of CHANNEL raises ``put_error``."""
    return make_mock_epics_connector(
        channel=CHANNEL,
        put_error=put_error if put_error is not None else access_denied(CHANNEL),
    )


async def _write(connector, value=42.0, confirm=False):
    with patch("osprey.utils.config.get_config_value", side_effect=_writes_enabled_config):
        return await connector.write_channel(CHANNEL, value, confirm=confirm)


class TestConnectorRefusal:
    @pytest.mark.asyncio
    async def test_access_denied_becomes_a_structured_refusal(self):
        """A denied put comes back blocked, not raised and not a bare failure."""
        connector = _make_connector()

        result = await _write(connector)

        assert isinstance(result, ChannelWriteResult)
        assert result.outcome is WriteOutcome.REFUSED
        assert result.refusal_reason == "CONTROL_SYSTEM_REFUSED"
        # The put WAS attempted — that is what distinguishes this refusal.
        assert len(connector._pvaccess.calls("put")) == 1

    @pytest.mark.asyncio
    async def test_message_names_the_control_system_and_the_channel(self):
        """The operator is told who refused, which channel, and that nothing moved."""
        connector = _make_connector()

        result = await _write(connector)

        assert (
            f"Write to '{CHANNEL}' refused by the control system (access security); "
            "no value was written" in result.error_message
        )
        # The control system's own words survive into the message.
        assert "Write access denied" in result.error_message
        assert "reference monitor" not in result.error_message

    @pytest.mark.asyncio
    async def test_refusal_survives_a_confirming_write(self):
        """The denial of a confirming (put-callback) put is the same refusal."""
        connector = _make_connector()

        result = await _write(connector, confirm=True)

        assert result.outcome is WriteOutcome.REFUSED
        assert result.refusal_reason == "CONTROL_SYSTEM_REFUSED"
        assert connector._pvaccess.calls("put")[0]["request"] == "record[block=true]field(value)"

    @pytest.mark.asyncio
    async def test_the_denial_text_on_another_exception_type_is_not_a_refusal(self):
        """Classification keys on pvapy's exception TYPE before its text.

        A failure of any other type is never classified, whatever it says — a
        refusal claims nothing was written, and only pvapy's own exception
        can vouch for that. It propagates exactly as raised.
        """
        error = RuntimeError(f"channel {CHANNEL} PvaClientPut::put Write access denied")
        connector = _make_connector(put_error=error)

        with pytest.raises(RuntimeError) as raised:
            await _write(connector)

        assert raised.value is error


class TestDenialContract:
    @pytest.mark.asyncio
    async def test_raise_for_write_result_raises_blocked_with_the_new_reason(self):
        """The denial contract routes the new code to the refusal exception."""
        connector = _make_connector()

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
    return ChannelWriteResult(
        channel_address=channel,
        value_written=1.0,
        outcome=WriteOutcome.REFUSED,
        refusal_reason=reason,
        error_message=f"Write to '{channel}' refused",
    )


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
