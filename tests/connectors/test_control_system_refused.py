"""A control-system denial of a write is a structured refusal, not an internal error.

When an IOC's access security refuses a put, Channel Access answers it with
``ECA_NOWTACCESS`` ("Write access denied"). That denial once escaped
``write_channel`` as an exception and landed in the MCP catch-all as
``internal_error`` — an answer that told the operator nothing and
misattributed the denial.

The invariant this file pins: such a denial comes back as
``ChannelWriteResult(outcome=WriteOutcome.REFUSED,
refusal_reason="CONTROL_SYSTEM_REFUSED")``, its message names the CONTROL
SYSTEM rather than OSPREY's reference monitor, and every rendering path that
sees it says the same thing. The narrowing matters as much as the catch: any
other put failure (a lost circuit) is still a genuine failure and still
propagates (pinned in
test_write_fail_closed.py::test_caput_connection_error_propagates_not_refused).
"""

import json
from unittest.mock import patch

import pytest
import yaml

from osprey.mcp_server.control_system.error_handling import (
    ToolError,
    connector_error_handler,
)
from osprey_connectors.control_system.base import (
    ChannelWriteResult,
    WriteOutcome,
    raise_for_write_result,
)
from osprey_connectors.errors import ChannelWriteBlockedError
from tests.connectors._epics_fakes import ECA_NOWTACCESS
from tests.connectors._write_fakes import make_mock_epics_connector
from tests.connectors._write_fakes import writes_enabled_config as _writes_enabled_config
from tests.facility.served_tree import in_process_config, served_tree

CHANNEL = "TEST:MAG:PS:SP"


def _make_connector():
    """A connector whose every put is denied by the IOC's access security."""
    return make_mock_epics_connector(put=ECA_NOWTACCESS)


async def _write(connector, value=42.0, confirm=False):
    with patch("osprey_connectors.config.get_config_value", side_effect=_writes_enabled_config):
        return await connector.write_channel(CHANNEL, value, confirm=confirm)


class TestConnectorRefusal:
    @pytest.mark.asyncio
    async def test_access_denied_becomes_a_structured_refusal(self):
        """A denied caput comes back blocked, not raised and not a bare failure."""
        connector = _make_connector()

        result = await _write(connector)

        assert isinstance(result, ChannelWriteResult)
        assert result.outcome is WriteOutcome.REFUSED
        assert result.refusal_reason == "CONTROL_SYSTEM_REFUSED"
        # The caput WAS attempted — that is what distinguishes this refusal.
        assert [name for name, _value, _wait in connector._aioca.puts] == [CHANNEL]

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
    async def test_refusal_survives_a_verifying_write_level(self):
        """The denial is answered before verification, at every level."""
        connector = _make_connector()

        result = await _write(connector, confirm=True)

        assert result.outcome is WriteOutcome.REFUSED
        assert result.refusal_reason == "CONTROL_SYSTEM_REFUSED"


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

    def test_a_refusal_without_a_reason_keeps_no_reason(self):
        """A refused result that names no reason is not read as writes-disabled."""
        result = ChannelWriteResult(
            channel_address=CHANNEL,
            value_written=1.0,
            outcome=WriteOutcome.REFUSED,
            refusal_reason=None,
        )

        with pytest.raises(ChannelWriteBlockedError) as excinfo:
            raise_for_write_result(result)

        assert excinfo.value.reason is None
        assert str(excinfo.value) == f"Write to '{CHANNEL}' refused"

    def test_bare_construction_does_not_misattribute_the_refusal(self):
        """With no message passed, the default text still names the right refuser."""
        err = ChannelWriteBlockedError(CHANNEL, "CONTROL_SYSTEM_REFUSED")

        assert str(err) == (
            f"Write to '{CHANNEL}' refused by the control system (CONTROL_SYSTEM_REFUSED)"
        )
        # Policy reasons keep their existing default verbatim.
        assert str(ChannelWriteBlockedError(CHANNEL, "LIMITS")) == (
            f"Write to '{CHANNEL}' refused by the reference monitor (LIMITS)"
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
            f"Write to '{CHANNEL}' refused by the reference monitor (WRITES_DISABLED)"
        )
        assert envelope["suggestions"] == [
            "This write was refused on policy grounds; it was never sent to the control system.",
            "Do NOT attempt to work around the refusal.",
        ]

    @pytest.mark.asyncio
    async def test_unknown_reason_envelope_names_no_refuser(self):
        envelope = await _render(ChannelWriteBlockedError(CHANNEL, "SOMETHING_ELSE"))

        assert envelope["error_type"] == "write_refused"
        assert envelope["error_message"] == (
            f"Write refused during channel_write: Write to '{CHANNEL}' refused (SOMETHING_ELSE)"
        )
        assert envelope["suggestions"] == ["Do NOT attempt to work around the refusal."]
        assert envelope["details"] == {"channel": CHANNEL, "reason": "SOMETHING_ELSE"}


def _write_result_stub(channel, reason):
    """A minimal connector result the channel_write tool serialises unchanged."""
    return ChannelWriteResult(
        channel_address=channel,
        value_written=1.0,
        outcome=WriteOutcome.REFUSED,
        refusal_reason=reason,
        error_message=f"Write to '{channel}' refused",
    )


async def _run_all_blocked_batch(tmp_path, monkeypatch, *reasons):
    """Drive the channel_write tool with a batch every op of which was refused.

    One reason refuses both writes alike; two give each write its own reason,
    in operation order.
    """
    first_reason, second_reason = reasons if len(reasons) == 2 else reasons * 2
    from unittest.mock import AsyncMock

    from osprey.mcp_server.control_system.server_context import initialize_server_context
    from osprey.mcp_server.control_system.tools.channel_write import channel_write

    monkeypatch.chdir(tmp_path)
    view = served_tree(tmp_path / "served", ["PV:A", "PV:B"])
    control_system = {
        "type": "virtual_accelerator",
        "connector": {"virtual_accelerator": in_process_config(view)},
    }
    (tmp_path / "config.yml").write_text(yaml.safe_dump({"control_system": control_system}))
    initialize_server_context()

    connector = AsyncMock()
    connector.write_multiple_channels.return_value = [
        _write_result_stub("PV:A", first_reason),
        _write_result_stub("PV:B", second_reason),
    ]

    fn = channel_write.fn if hasattr(channel_write, "fn") else channel_write
    with (
        patch(
            "osprey_connectors.factory.ConnectorFactory.create_control_system_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ),
        patch(
            "osprey_connectors.control_system.limits_validator.LimitsValidator.from_config",
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

    @pytest.mark.asyncio
    async def test_mixed_refusals_name_each_refuser(self, tmp_path, monkeypatch):
        envelope = await _run_all_blocked_batch(
            tmp_path, monkeypatch, "CONTROL_SYSTEM_REFUSED", "LIMITS"
        )

        assert envelope["error_type"] == "write_refused"
        assert (
            "All 2 write(s) refused — by the control system: PV:A; "
            "by the reference monitor: PV:B" in envelope["error_message"]
        )

    @pytest.mark.asyncio
    async def test_unknown_address_first_does_not_hide_a_limits_refusal(
        self, tmp_path, monkeypatch
    ):
        envelope = await _run_all_blocked_batch(
            tmp_path, monkeypatch, "CHANNEL_NOT_FOUND", "LIMITS"
        )

        assert envelope["error_type"] == "write_refused"
        assert envelope["details"]["reason"] == "LIMITS"
        assert envelope["details"]["channel"] == "PV:B"
        assert (
            "All 2 write(s) refused — by the reference monitor: PV:B; "
            "PV:A is not a channel of this facility" in envelope["error_message"]
        )

    @pytest.mark.asyncio
    async def test_a_refusal_without_a_reason_names_no_refuser(self, tmp_path, monkeypatch):
        envelope = await _run_all_blocked_batch(tmp_path, monkeypatch, None)

        assert envelope["error_type"] == "write_refused"
        assert envelope["details"]["reason"] is None
        assert "All 2 write(s) refused — refused: PV:A, PV:B" in envelope["error_message"]
        assert "reference monitor" not in envelope["error_message"]
