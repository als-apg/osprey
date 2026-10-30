"""Unit tests for typed exceptions in osprey_connectors.errors."""

import re
from pathlib import Path

import pytest

import osprey_connectors.errors as errors_module
from osprey_connectors.control_system import WriteOutcome
from osprey_connectors.errors import (
    ChannelError,
    ChannelLimitsViolationError,
    ChannelNotFoundError,
    ChannelUnreachableError,
    ChannelWriteBlockedError,
    ChannelWriteFailedError,
    ConnectorInternalError,
    ConnectorUnavailableError,
    RefusalReason,
)


class TestChannelWriteBlockedError:
    def test_default_message(self):
        err = ChannelWriteBlockedError("RING:MAG:PS:SP", "WRITES_DISABLED")
        assert err.channel_address == "RING:MAG:PS:SP"
        assert err.reason == "WRITES_DISABLED"
        assert str(err) == (
            "Write to 'RING:MAG:PS:SP' refused by the reference monitor (WRITES_DISABLED)"
        )

    def test_custom_message(self):
        err = ChannelWriteBlockedError("RING:MAG:PS:SP", "LIMITS", "value out of range")
        assert err.channel_address == "RING:MAG:PS:SP"
        assert err.reason == "LIMITS"
        assert str(err) == "value out of range"

    def test_is_exception(self):
        with pytest.raises(ChannelWriteBlockedError):
            raise ChannelWriteBlockedError("chan", "VALIDATION_ERROR")

    def test_refusal_reason_values(self):
        assert [reason.value for reason in RefusalReason] == [
            "WRITES_DISABLED",
            "LIMITS",
            "VALIDATION_ERROR",
            "CONTROL_SYSTEM_REFUSED",
            "RAW_CLIENT_WRITE",
            "CHANNEL_NOT_FOUND",
        ]

    def test_unknown_reason_permitted(self):
        # The exception keeps an unknown reason as given (permissive by design).
        err = ChannelWriteBlockedError("chan", "SOMETHING_ELSE")
        assert err.reason == "SOMETHING_ELSE"
        assert str(err) == "Write to 'chan' refused (SOMETHING_ELSE)"

    def test_a_known_reason_is_stored_as_its_member(self):
        err = ChannelWriteBlockedError("chan", "LIMITS")
        assert err.reason is RefusalReason.LIMITS

    def test_every_reason_has_a_refuser_and_a_remedy_entry(self):
        for reason in RefusalReason:
            assert ChannelWriteBlockedError.refuser(reason) == reason.refuser
            assert ChannelWriteBlockedError.remedy(reason) == reason.remedy
            assert ChannelWriteBlockedError.remedy(reason.value) == reason.remedy
            # A reason that denies a write names its refuser and says what to do;
            # only an address outside the facility names neither.
            if reason is RefusalReason.CHANNEL_NOT_FOUND:
                assert (reason.refuser, reason.remedy) == (None, ())
            else:
                assert reason.refuser
                assert reason.remedy
                assert all(isinstance(sentence, str) and sentence for sentence in reason.remedy)

    def test_the_control_system_remedy_names_the_control_system(self):
        remedy = RefusalReason.CONTROL_SYSTEM_REFUSED.remedy
        assert RefusalReason.CONTROL_SYSTEM_REFUSED.refuser == "the control system"
        assert "no value was written" in remedy[0]
        assert "never sent" not in " ".join(remedy)

    def test_unknown_reason_names_no_refuser(self):
        assert ChannelWriteBlockedError.refuser("SOMETHING_ELSE") is None
        assert ChannelWriteBlockedError.refuser(None) is None

    def test_unknown_reason_has_no_remedy(self):
        assert ChannelWriteBlockedError.remedy("SOMETHING_ELSE") == ()
        assert ChannelWriteBlockedError.remedy(None) == ()


class TestChannelNotFoundError:
    def test_message_names_the_address(self):
        err = ChannelNotFoundError("NO:SUCH:CHANNEL")
        assert str(err) == "NO:SUCH:CHANNEL is not a channel of this facility"

    def test_carries_the_address(self):
        assert ChannelNotFoundError("NO:SUCH:CHANNEL").channel_address == "NO:SUCH:CHANNEL"

    def test_reason(self):
        assert ChannelNotFoundError.reason is RefusalReason.CHANNEL_NOT_FOUND

    def test_is_a_lookup_error(self):
        assert isinstance(ChannelNotFoundError("X"), LookupError)

    def test_a_bare_blocked_write_names_no_refuser(self):
        err = ChannelWriteBlockedError("X", "CHANNEL_NOT_FOUND")
        assert str(err) == "X is not a channel of this facility"


#: The roots that partition what a first-party connector raises.
FAILURE_ROOTS = (ChannelError, ConnectorUnavailableError, ConnectorInternalError)


class TestTheFailureRoots:
    def test_an_unreachable_channel_is_a_channel_error_and_no_connection_error(self):
        failure = ChannelUnreachableError("SR:DEAD:IOC")
        assert isinstance(failure, ChannelError)
        assert not isinstance(failure, (ConnectionError, TimeoutError, OSError))

    def test_a_lost_connector_is_a_connection_error(self):
        assert isinstance(ConnectorUnavailableError("the transport is gone"), ConnectionError)

    def test_an_unreachable_channel_names_its_address(self):
        failure = ChannelUnreachableError("SR:DEAD:IOC")
        assert failure.channel_address == "SR:DEAD:IOC"
        assert str(failure) == "SR:DEAD:IOC did not answer"

    def test_an_unreachable_channel_keeps_the_text_it_was_given(self):
        failure = ChannelUnreachableError("SR:DEAD:IOC", message="SR:DEAD:IOC timed out")
        assert failure.channel_address == "SR:DEAD:IOC"
        assert str(failure) == "SR:DEAD:IOC timed out"

    def test_a_lost_connector_carries_no_recovery_until_its_supervisor_sets_one(self):
        failure = ConnectorUnavailableError("the transport is gone")
        assert failure.recovery == ()
        assert str(failure) == "the transport is gone"

    def test_a_lost_connector_takes_the_supervisors_recovery(self):
        record = {"action": "connector_dropped", "text": "dropped"}
        failure = ConnectorUnavailableError("gone", recovery=(record,))
        assert failure.recovery == (record,)

    def test_the_roots_are_disjoint(self):
        for index, root in enumerate(FAILURE_ROOTS):
            for other in FAILURE_ROOTS[index + 1 :]:
                assert not issubclass(root, other) and not issubclass(other, root)

    def test_no_public_class_lies_under_two_roots(self):
        public = [getattr(errors_module, name) for name in errors_module.__all__]
        for cls in (item for item in public if isinstance(item, type)):
            roots = [root for root in FAILURE_ROOTS if issubclass(cls, root)]
            assert len(roots) <= 1, f"{cls.__name__} lies under {roots}"

    def test_the_roots_are_public(self):
        for root in (*FAILURE_ROOTS, ChannelUnreachableError):
            assert root.__name__ in errors_module.__all__


class TestChannelWriteFailedError:
    def test_default_message(self):
        err = ChannelWriteFailedError("RING:MAG:PS:SP", "FAILED")
        assert err.channel_address == "RING:MAG:PS:SP"
        assert err.reason == "FAILED"
        assert str(err) == "Write to 'RING:MAG:PS:SP' failed (FAILED)"

    def test_custom_message(self):
        err = ChannelWriteFailedError("RING:MAG:PS:SP", "UNCONFIRMED", "re-read raised")
        assert err.channel_address == "RING:MAG:PS:SP"
        assert err.reason == "UNCONFIRMED"
        assert str(err) == "re-read raised"

    def test_is_exception(self):
        with pytest.raises(ChannelWriteFailedError):
            raise ChannelWriteFailedError("chan", "UNCONFIRMED")

    def test_valid_reasons_reference(self):
        assert ChannelWriteFailedError._VALID_REASONS == (
            "FAILED",
            "MISMATCH",
            "UNCONFIRMED",
        )

    def test_unknown_reason_permitted(self):
        err = ChannelWriteFailedError("chan", "WHATEVER")
        assert err.reason == "WHATEVER"

    def test_write_detail_attributes_default_to_none(self):
        """Positional construction stays valid; the new detail is optional."""
        err = ChannelWriteFailedError("chan", "FAILED")
        assert err.outcome is None
        assert err.value_written is None
        assert err.observed_value is None

    def test_mismatch_carries_outcome_and_both_values(self):
        """A mismatch must say what was sent and what the channel holds."""
        err = ChannelWriteFailedError(
            "RING:MAG:PS:SP",
            "MISMATCH",
            outcome=WriteOutcome.MISMATCH,
            value_written=1.7,
            observed_value=1.5,
        )
        assert err.outcome == WriteOutcome.MISMATCH
        assert err.value_written == 1.7
        assert err.observed_value == 1.5
        assert "1.7" in str(err)
        assert "1.5" in str(err)
        assert "RING:MAG:PS:SP" in str(err)

    def test_mismatch_message_can_still_be_overridden(self):
        err = ChannelWriteFailedError(
            "chan", "MISMATCH", "clamped by the power supply", value_written=1.7, observed_value=1.5
        )
        assert str(err) == "clamped by the power supply"

    def test_mismatch_without_detail_does_not_report_none_as_a_reading(self):
        """A detail-less mismatch falls back rather than saying "sent None"."""
        err = ChannelWriteFailedError("chan", "MISMATCH")
        assert str(err) == "Write to 'chan' failed (MISMATCH)"

    def test_non_mismatch_reason_keeps_the_generic_message(self):
        err = ChannelWriteFailedError("chan", "UNCONFIRMED", value_written=1.7)
        assert str(err) == "Write to 'chan' failed (UNCONFIRMED)"


#: One instance of each single-channel failure, by class, all about ``"SR:Q"``.
SINGLE_CHANNEL_FAILURES = {
    ChannelLimitsViolationError: lambda: ChannelLimitsViolationError(
        "SR:Q", 9.0, "range", "Value 9.0 above maximum 5.0", min_value=0.0, max_value=5.0
    ),
    ChannelWriteBlockedError: lambda: ChannelWriteBlockedError("SR:Q", "WRITES_DISABLED"),
    ChannelWriteFailedError: lambda: ChannelWriteFailedError("SR:Q", "FAILED"),
    ChannelNotFoundError: lambda: ChannelNotFoundError("SR:Q"),
    ChannelUnreachableError: lambda: ChannelUnreachableError("SR:Q"),
}


@pytest.mark.parametrize("cls", list(SINGLE_CHANNEL_FAILURES), ids=lambda cls: cls.__name__)
def test_every_single_channel_failure_is_a_channel_error(cls):
    failure = SINGLE_CHANNEL_FAILURES[cls]()
    assert isinstance(failure, ChannelError)
    assert failure.channel_address == "SR:Q"


@pytest.mark.parametrize(
    ("cls", "text"),
    [
        (
            ChannelLimitsViolationError,
            "\n".join(
                [
                    "\n" + "=" * 70,
                    "CHANNEL LIMITS VIOLATION DETECTED",
                    "=" * 70,
                    "Channel Address: SR:Q",
                    "Attempted Value: 9.0",
                    "Violation: Value 9.0 above maximum 5.0",
                    "Allowed Range: [0.0, 5.0]",
                    "=" * 70,
                    "\u26a0\ufe0f  Write operation BLOCKED for safety",
                    "=" * 70,
                ]
            ),
        ),
        (
            ChannelWriteBlockedError,
            "Write to 'SR:Q' refused by the reference monitor (WRITES_DISABLED)",
        ),
        (ChannelWriteFailedError, "Write to 'SR:Q' failed (FAILED)"),
        (ChannelNotFoundError, "SR:Q is not a channel of this facility"),
        (ChannelUnreachableError, "SR:Q did not answer"),
    ],
    ids=lambda item: item.__name__ if isinstance(item, type) else "text",
)
def test_each_channel_errors_text(cls, text):
    failure = SINGLE_CHANNEL_FAILURES[cls]()
    assert str(failure) == text
    assert failure.args == (text,)


def test_the_readme_lists_every_refusal_reason():
    """The package README names the refusal reasons the enum declares, in its order."""
    readme = (
        Path(__file__).resolve().parents[1] / "packages" / "osprey-connectors" / "README.md"
    ).read_text()
    paragraph = readme.split("- `ChannelWriteBlockedError`", 1)[1].split("\n- ", 1)[0]
    listed = paragraph.split("one of", 1)[1].split(";", 1)[0]
    assert re.findall(r"`([A-Z_]+)`", listed) == [reason.value for reason in RefusalReason]
