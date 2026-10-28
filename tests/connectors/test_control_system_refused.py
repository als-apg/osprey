"""A control-system denial of a write is a structured refusal, not an internal error.

When an IOC's access security refuses a put, Channel Access answers it with
``ECA_NOWTACCESS`` ("Write access denied"). That denial once escaped
``write_channel`` as an exception and landed in the MCP catch-all as
``internal_error`` — an answer that told the operator nothing and
misattributed the denial.

The invariant this file pins: such a denial comes back as
``ChannelWriteResult(outcome=WriteOutcome.REFUSED,
refusal_reason="CONTROL_SYSTEM_REFUSED", error_message=...)``, its message names the CONTROL
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
from osprey_connectors.control_system._base import (
    ChannelWriteResult,
    WriteOutcome,
    raise_for_write_result,
)
from osprey_connectors.errors import ChannelWriteBlockedError, RefusalReason
from osprey_connectors_tests._epics_fakes import ECA_NOWTACCESS
from osprey_connectors_tests._write_fakes import arm_switch, make_mock_epics_connector
from osprey_connectors_tests._write_fakes import writes_enabled_config as _writes_enabled_config
from tests.facility.served_tree import in_process_config, served_tree

CHANNEL = "TEST:MAG:PS:SP"

#: The outcomes whose result must carry a message, as ChannelWriteResult documents.
_MESSAGE_OUTCOMES = {WriteOutcome.REFUSED, WriteOutcome.FAILED, WriteOutcome.UNCONFIRMED}


@pytest.fixture(autouse=True)
def _switch_on(monkeypatch, tmp_path):
    """The login's switch on for the simulator target, connectors stamped with it.

    With :func:`writes_enabled_config` serving a write tier that allows, that is
    everything the reference monitor's verdict asks before a write reaches the
    transport; a test that disarms serves a tier that does not.
    """
    return arm_switch(monkeypatch, tmp_path / "armed-data")


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


class TestResultInvariants:
    """A write result cannot be built in a shape its own docs forbid."""

    @pytest.mark.parametrize("reason", [None, "SOMETHING_ELSE"])
    def test_a_refused_result_without_a_known_reason_cannot_be_built(self, reason):
        with pytest.raises(ValueError) as excinfo:
            ChannelWriteResult(
                channel_address=CHANNEL,
                value_written=1.0,
                outcome=WriteOutcome.REFUSED,
                refusal_reason=reason,
                error_message="refused",
            )

        assert CHANNEL in str(excinfo.value)
        assert "refusal_reason" in str(excinfo.value)
        assert repr(reason) in str(excinfo.value)

    @pytest.mark.parametrize("reason", list(RefusalReason))
    def test_a_known_reason_string_is_coerced_to_its_member(self, reason):
        result = ChannelWriteResult(
            channel_address=CHANNEL,
            value_written=1.0,
            outcome="refused",
            refusal_reason=reason.value,
            error_message="refused",
        )

        assert result.refusal_reason is reason

    @pytest.mark.parametrize("outcome", [o for o in WriteOutcome if o is not WriteOutcome.REFUSED])
    def test_a_reason_on_a_result_that_is_not_refused_cannot_be_built(self, outcome):
        with pytest.raises(ValueError) as excinfo:
            ChannelWriteResult(
                channel_address=CHANNEL,
                value_written=1.0,
                outcome=outcome,
                refusal_reason=RefusalReason.LIMITS,
                error_message="message" if outcome in _MESSAGE_OUTCOMES else None,
            )

        assert CHANNEL in str(excinfo.value)
        assert str(outcome) in str(excinfo.value)
        assert "refusal_reason" in str(excinfo.value)

    @pytest.mark.parametrize("outcome", list(WriteOutcome))
    def test_a_message_is_set_exactly_on_refused_failed_and_unconfirmed(self, outcome):
        reason = RefusalReason.LIMITS if outcome is WriteOutcome.REFUSED else None
        needs_message = outcome in _MESSAGE_OUTCOMES

        built = ChannelWriteResult(
            channel_address=CHANNEL,
            value_written=1.0,
            outcome=outcome,
            refusal_reason=reason,
            error_message="message" if needs_message else None,
        )
        assert bool(built.error_message) is needs_message

        for wrong in (None, "") if needs_message else ("message",):
            with pytest.raises(ValueError) as excinfo:
                ChannelWriteResult(
                    channel_address=CHANNEL,
                    value_written=1.0,
                    outcome=outcome,
                    refusal_reason=reason,
                    error_message=wrong,
                )
            assert CHANNEL in str(excinfo.value)
            assert str(outcome) in str(excinfo.value)
            assert "error_message" in str(excinfo.value)


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
        """A refused result that names no reason cannot be built at all."""
        with pytest.raises(ValueError, match="refusal_reason"):
            ChannelWriteResult(
                channel_address=CHANNEL,
                value_written=1.0,
                outcome=WriteOutcome.REFUSED,
                refusal_reason=None,
                error_message=f"Write to '{CHANNEL}' refused",
            )

    def test_a_bare_refusal_exception_names_no_reason_and_no_refuser(self):
        """The exception itself still takes no reason, and then guesses none."""
        err = ChannelWriteBlockedError(CHANNEL, None)

        assert err.reason is None
        assert str(err) == f"Write to '{CHANNEL}' refused"

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

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reason", [r for r in RefusalReason if r is not RefusalReason.CHANNEL_NOT_FOUND]
    )
    async def test_suggestions_are_the_reasons_remedy(self, reason):
        """The envelope suggests what the reason's remedy says, then the one constant."""
        envelope = await _render(ChannelWriteBlockedError(CHANNEL, reason.value))

        assert envelope["suggestions"] == [
            *reason.remedy,
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


async def _run_all_blocked_batch(tmp_path, monkeypatch, *reasons, connector=None):
    """Drive the channel_write tool with a batch every op of which was refused.

    One reason refuses both writes alike; two give each write its own reason,
    in operation order. A ``connector`` given in place of the reasons answers
    the writes itself. Returns the envelope and the connector's results.
    """
    from unittest.mock import AsyncMock

    from osprey.mcp_server.control_system.server_context import initialize_server_context
    from osprey.mcp_server.control_system.tools.channel_write import channel_write

    monkeypatch.chdir(tmp_path)
    view = served_tree(tmp_path / "served", ["PV:A", "PV:B"])
    # A write tier that allows, and approval switched off: the autouse switch is
    # on for the simulator, so the tool's stage 1 lets the batch reach the
    # connector, whose refusals are the subject here.
    control_system = {
        "type": "virtual_accelerator",
        "writes": "allowed",
        "connector": {"virtual_accelerator": in_process_config(view)},
    }
    (tmp_path / "config.yml").write_text(
        yaml.safe_dump({"control_system": control_system, "approval": {"enabled": False}})
    )
    initialize_server_context()

    answered: list[ChannelWriteResult] = []
    if connector is None:
        first_reason, second_reason = reasons if len(reasons) == 2 else reasons * 2
        connector = AsyncMock()
        connector.write_multiple_channels.return_value = [
            _write_result_stub("PV:A", first_reason),
            _write_result_stub("PV:B", second_reason),
        ]
    write_multiple = connector.write_multiple_channels

    async def _recording(*args, **kwargs):
        answered.extend(await write_multiple(*args, **kwargs))
        return answered

    connector.write_multiple_channels = _recording

    fn = channel_write.fn if hasattr(channel_write, "fn") else channel_write
    with (
        patch(
            "osprey_connectors.factory.ConnectorFactory.create_control_system_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ),
        patch(
            "osprey_connectors.control_system._limits_validator.LimitsValidator.from_config",
            return_value=None,
        ),
        patch("osprey_connectors.config.get_config_value", side_effect=_writes_enabled_config),
        pytest.raises(ToolError) as excinfo,
    ):
        await fn(
            operations=[
                {"channel": "PV:A", "value": 1.0},
                {"channel": "PV:B", "value": 1.0},
            ]
        )
    return json.loads(str(excinfo.value)), answered


def _row_facts(rows):
    return [(row["channel"], row["outcome"], row["refusal_reason"], row["error"]) for row in rows]


def _result_facts(results):
    return [(r.channel_address, str(r.outcome), r.refusal_reason, r.error_message) for r in results]


class TestAllBlockedBatchAttribution:
    """The batch escalation names the same refuser the single-write path does."""

    @pytest.mark.asyncio
    async def test_control_system_refusals_name_the_control_system(self, tmp_path, monkeypatch):
        envelope, _ = await _run_all_blocked_batch(tmp_path, monkeypatch, "CONTROL_SYSTEM_REFUSED")

        assert envelope["error_type"] == "write_refused"
        assert (
            "All 2 write(s) refused by the control system: PV:A, PV:B"
            in (envelope["error_message"])
        )

    @pytest.mark.asyncio
    async def test_policy_refusals_still_name_the_reference_monitor(self, tmp_path, monkeypatch):
        envelope, _ = await _run_all_blocked_batch(tmp_path, monkeypatch, "WRITES_DISABLED")

        assert (
            "All 2 write(s) refused by the reference monitor: PV:A, PV:B"
            in (envelope["error_message"])
        )

    @pytest.mark.asyncio
    async def test_mixed_refusals_name_each_refuser(self, tmp_path, monkeypatch):
        envelope, answered = await _run_all_blocked_batch(
            tmp_path, monkeypatch, "CONTROL_SYSTEM_REFUSED", "LIMITS"
        )

        assert envelope["error_type"] == "write_refused"
        assert _row_facts(envelope["details"]["results"]) == _result_facts(answered)
        assert (
            "All 2 write(s) refused — by the control system: PV:A; "
            "by the reference monitor: PV:B" in envelope["error_message"]
        )
        assert envelope["suggestions"] == [
            *RefusalReason.CONTROL_SYSTEM_REFUSED.remedy,
            *RefusalReason.LIMITS.remedy,
            "Do NOT attempt to work around the refusal.",
        ]

    @pytest.mark.asyncio
    async def test_a_remedy_two_reasons_share_is_suggested_once(self, tmp_path, monkeypatch):
        envelope, _ = await _run_all_blocked_batch(
            tmp_path, monkeypatch, "WRITES_DISABLED", "LIMITS"
        )

        assert RefusalReason.WRITES_DISABLED.remedy == RefusalReason.LIMITS.remedy
        assert envelope["suggestions"] == [
            *RefusalReason.LIMITS.remedy,
            "Do NOT attempt to work around the refusal.",
        ]

    @pytest.mark.asyncio
    async def test_unknown_address_first_does_not_hide_a_limits_refusal(
        self, tmp_path, monkeypatch
    ):
        envelope, _ = await _run_all_blocked_batch(
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
    async def test_the_refusal_carries_every_row(self, tmp_path, monkeypatch):
        envelope, answered = await _run_all_blocked_batch(tmp_path, monkeypatch, "WRITES_DISABLED")

        assert _row_facts(envelope["details"]["results"]) == _result_facts(answered)
        assert "missing" not in envelope["details"]

    @pytest.mark.asyncio
    async def test_an_access_denied_batch_carries_the_connector_rows(self, tmp_path, monkeypatch):
        """Rows a real connector built from the IOC's denial reach the agent unchanged."""
        envelope, answered = await _run_all_blocked_batch(
            tmp_path, monkeypatch, connector=_make_connector()
        )

        assert envelope["error_type"] == "write_refused"
        assert envelope["details"]["reason"] == "CONTROL_SYSTEM_REFUSED"
        assert [r.outcome for r in answered] == [WriteOutcome.REFUSED] * 2
        assert _row_facts(envelope["details"]["results"]) == _result_facts(answered)
