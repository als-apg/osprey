"""Typed-exception contract for the connector-host IPC channel.

The child and the proxy both encode against this spec, and the registry is
derived from the public failure vocabulary, so the parity tests here read the
vocabulary and the registry rather than a copied list: a public failure with no
wire detail, or no sample to round-trip, fails in this file in the commit that
adds it. Assertions are on the concrete reconstructed exception — every field
the parent's error rendering reads — never merely that a round trip "didn't
raise".
"""

from collections.abc import Callable

import pytest

import osprey_connectors.errors as errors_module
import osprey_connectors.ipc
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
)
from osprey_connectors.ipc import _exceptions as exceptions
from osprey_connectors.ipc import _frames as frames


def _round_trip(exc: BaseException) -> BaseException:
    """Encode ``exc`` to a frame body and rebuild it the way the parent would."""
    body = exceptions.encode_exception(exc)
    assert set(body) == {"class_tag", "message", "fields"}
    return exceptions.decode_exception(body)


# ---------------------------------------------------------------- protocol pin


def _vocabulary() -> list[type[BaseException]]:
    """Every class the errors module publishes under a contract root."""
    published = (getattr(errors_module, name) for name in errors_module.__all__)
    return [
        item
        for item in published
        if isinstance(item, type) and issubclass(item, exceptions._CONTRACT_ROOTS)
    ]


#: One sample per registered class, by tag. A class with no entry fails the
#: round-trip test by name.
_SAMPLES: dict[str, Callable[[], BaseException]] = {
    "ConnectionError": lambda: ConnectionError("gateway unreachable"),
    "TimeoutError": lambda: TimeoutError("read timed out after 2.0s"),
    "ChannelError": lambda: ChannelError("SR:Q", "SR:Q failed its own way"),
    "ChannelLimitsViolationError": lambda: ChannelLimitsViolationError(
        channel_address="SR:Q",
        value=999.0,
        violation_type="range",
        violation_reason="Value 999.0 above maximum 500.0",
        min_value=0.0,
        max_value=500.0,
        max_step=10.0,
        current_value=120.0,
    ),
    "ChannelNotFoundError": lambda: ChannelNotFoundError("SR:Q"),
    "ChannelUnreachableError": lambda: ChannelUnreachableError("SR:Q", "SR:Q timed out"),
    "ChannelWriteBlockedError": lambda: ChannelWriteBlockedError(
        "SR:Q", "WRITES_DISABLED", "Writes are disabled for this deployment"
    ),
    "ChannelWriteFailedError": lambda: ChannelWriteFailedError(
        "SR:Q",
        "MISMATCH",
        outcome=WriteOutcome.MISMATCH,
        value_written=1.7,
        observed_value=1.5,
    ),
    "ConnectorUnavailableError": lambda: ConnectorUnavailableError("the transport is gone"),
    "ConnectorInternalError": lambda: ConnectorInternalError(
        "connector raised KeyError('x')", class_tag="KeyError"
    ),
}


def test_the_registry_is_the_builtins_plus_the_vocabulary():
    expected = {"ConnectionError", "TimeoutError"} | {cls.__name__ for cls in _vocabulary()}
    assert set(exceptions.EXCEPTION_SPECS) == expected
    for tag, spec in exceptions.EXCEPTION_SPECS.items():
        assert spec.cls.__name__ == tag


def test_every_vocabulary_class_has_its_wire_detail():
    missing = [cls.__name__ for cls in _vocabulary() if cls not in exceptions._WIRE_DETAIL]
    assert missing == []


def test_every_channel_error_carries_its_channel_first():
    for spec in exceptions.EXCEPTION_SPECS.values():
        if issubclass(spec.cls, ChannelError):
            assert spec.fields[0] == "channel_address", spec.cls.__name__
            assert spec.fields.count("channel_address") == 1, spec.cls.__name__


@pytest.mark.parametrize("tag", list(exceptions.EXCEPTION_SPECS))
def test_every_vocabulary_class_round_trips(tag):
    sample = _SAMPLES.get(tag)
    if sample is None:
        pytest.fail(f"{tag} has no round-trip sample")
    original = sample()
    spec = exceptions.EXCEPTION_SPECS[tag]
    assert type(original) is spec.cls

    in_process = exceptions.decode_exception(exceptions.encode_exception(original))
    frame = frames.decode_frame(frames.encode_error("req-p", original))
    assert isinstance(frame, frames.ErrorFrame)

    for rebuilt in (in_process, frame.exception):
        assert type(rebuilt) is spec.cls
        assert str(rebuilt) == str(original)
        for name in spec.fields:
            assert getattr(rebuilt, name) == getattr(original, name), name
            assert type(getattr(rebuilt, name)) is type(getattr(original, name)), name


def test_the_contract_fault_is_no_connection_error():
    # No ``except ConnectionError`` or ``except OSError`` may read a connector
    # bug as an unreachable control system.
    assert not issubclass(ConnectorInternalError, OSError)
    assert issubclass(ConnectorInternalError, RuntimeError)


# ---------------------------------------------------------------- round trips


def test_limits_violation_round_trips_every_spec_field():
    original = ChannelLimitsViolationError(
        channel_address="SR:BEND:1:CUR",
        value=999.0,
        violation_type="range",
        violation_reason="Value 999.0 above maximum 500.0",
        min_value=0.0,
        max_value=500.0,
        max_step=10.0,
        current_value=120.0,
    )
    body = exceptions.encode_exception(original)

    assert body["class_tag"] == "ChannelLimitsViolationError"
    # The wire spells the attribute name, not the constructor keyword.
    assert set(body["fields"]) == set(
        exceptions.EXCEPTION_SPECS["ChannelLimitsViolationError"].fields
    )
    assert body["fields"]["attempted_value"] == 999.0

    rebuilt = exceptions.decode_exception(body)

    assert isinstance(rebuilt, ChannelLimitsViolationError)
    assert rebuilt.channel_address == "SR:BEND:1:CUR"
    assert rebuilt.attempted_value == 999.0
    assert rebuilt.violation_type == "range"
    assert rebuilt.violation_reason == "Value 999.0 above maximum 500.0"
    assert rebuilt.min_value == 0.0
    assert rebuilt.max_value == 500.0
    assert rebuilt.max_step == 10.0
    assert rebuilt.current_value == 120.0
    # The envelope the operator sees is re-rendered from those fields, intact.
    assert str(rebuilt) == str(original)


def test_limits_violation_keeps_unset_bounds_as_none():
    original = ChannelLimitsViolationError(
        channel_address="SR:X",
        value="on",
        violation_type="unlisted",
        violation_reason="Channel not in the limits registry",
    )
    rebuilt = _round_trip(original)

    assert isinstance(rebuilt, ChannelLimitsViolationError)
    assert rebuilt.attempted_value == "on"
    assert rebuilt.min_value is None
    assert rebuilt.max_value is None
    assert rebuilt.max_step is None
    assert rebuilt.current_value is None


def test_write_blocked_round_trips_reason_and_custom_message():
    original = ChannelWriteBlockedError(
        channel_address="SR:BEND:1:CUR",
        reason="WRITES_DISABLED",
        message="Writes are disabled for this deployment",
    )
    body = exceptions.encode_exception(original)

    assert body["class_tag"] == "ChannelWriteBlockedError"
    assert body["fields"] == {"channel_address": "SR:BEND:1:CUR", "reason": "WRITES_DISABLED"}

    rebuilt = exceptions.decode_exception(body)

    assert isinstance(rebuilt, ChannelWriteBlockedError)
    assert rebuilt.channel_address == "SR:BEND:1:CUR"
    assert rebuilt.reason == "WRITES_DISABLED"
    assert str(rebuilt) == "Writes are disabled for this deployment"


def test_write_blocked_default_message_survives():
    original = ChannelWriteBlockedError(channel_address="SR:Y", reason="LIMITS")
    rebuilt = _round_trip(original)

    assert isinstance(rebuilt, ChannelWriteBlockedError)
    assert rebuilt.channel_address == "SR:Y"
    assert rebuilt.reason == "LIMITS"
    assert str(rebuilt) == str(original)


def test_an_unknown_channel_round_trips_as_itself():
    original = ChannelNotFoundError("X")
    rebuilt = _round_trip(original)

    assert type(rebuilt) is ChannelNotFoundError
    assert rebuilt.channel_address == "X"
    assert str(rebuilt) == str(original)


def test_an_unreachable_channel_round_trips_as_itself():
    original = ChannelUnreachableError("SR:DEAD:IOC", message="SR:DEAD:IOC timed out")
    rebuilt = _round_trip(original)

    assert type(rebuilt) is ChannelUnreachableError
    assert rebuilt.channel_address == "SR:DEAD:IOC"
    assert str(rebuilt) == "SR:DEAD:IOC timed out"


def test_connector_unavailable_round_trips():
    # The recovery is not a wire field: the parent's supervisor decides it.
    original = ConnectorUnavailableError(
        "PVA client context is gone", recovery=({"action": "child_replaced", "text": "x"},)
    )
    rebuilt = _round_trip(original)

    assert type(rebuilt) is ConnectorUnavailableError
    assert str(rebuilt) == "PVA client context is gone"
    assert rebuilt.recovery == ()


def test_connection_error_round_trips():
    rebuilt = _round_trip(ConnectionError("gateway unreachable"))

    assert type(rebuilt) is ConnectionError
    assert str(rebuilt) == "gateway unreachable"


def test_timeout_error_round_trips():
    rebuilt = _round_trip(TimeoutError("read timed out after 2.0s"))

    assert type(rebuilt) is TimeoutError
    assert str(rebuilt) == "read timed out after 2.0s"


# ---------------------------------------------------------------- field reading


def test_fields_are_read_by_spec_name_not_from_the_instance_dict():
    original = ChannelWriteBlockedError(channel_address="SR:Z", reason="VALIDATION_ERROR")
    original.internal_handle = "a child-only attribute"  # type: ignore[attr-defined]

    body = exceptions.encode_exception(original)

    assert "internal_handle" not in body["fields"]
    assert set(body["fields"]) == {"channel_address", "reason"}


def test_missing_attribute_encodes_as_none_without_raising():
    original = ChannelLimitsViolationError(
        channel_address="SR:W",
        value=5.0,
        violation_type="step",
        violation_reason="Step 5.0 above maximum step 1.0",
        max_step=1.0,
    )
    # An instance from an older or newer build of the child may simply not
    # carry a field this spec enumerates.
    del original.max_step

    body = exceptions.encode_exception(original)

    assert body["fields"]["max_step"] is None
    assert body["fields"]["channel_address"] == "SR:W"

    rebuilt = exceptions.decode_exception(body)

    assert isinstance(rebuilt, ChannelLimitsViolationError)
    assert rebuilt.max_step is None
    assert rebuilt.attempted_value == 5.0


def test_unregistered_exception_has_no_fields():
    class SomeChildOnlyError(Exception):
        pass

    assert exceptions.exception_fields(SomeChildOnlyError("boom")) == {}


# ---------------------------------------------------------------- failing closed


def test_unknown_exception_type_encodes_fail_closed_to_the_contract_fault():
    class SomeChildOnlyError(Exception):
        pass

    original = SomeChildOnlyError("libca segfaulted")
    body = exceptions.encode_exception(original)

    assert body["class_tag"] == "ConnectorInternalError"
    assert body["fields"] == {"class_tag": "SomeChildOnlyError"}
    # Fail closed, but never silently: the original repr survives.
    assert repr(original) in body["message"]
    assert "libca segfaulted" in body["message"]

    rebuilt = exceptions.decode_exception(body)

    assert type(rebuilt) is ConnectorInternalError
    assert rebuilt.class_tag == "SomeChildOnlyError"
    assert repr(original) in str(rebuilt)


def test_a_subclass_crosses_as_its_nearest_vocabulary_class():
    class _Mine(ChannelUnreachableError):
        pass

    class SlowLinkError(TimeoutError):
        pass

    mine = _round_trip(_Mine("SR:Q", "SR:Q went quiet"))
    refused = _round_trip(ConnectionRefusedError("SR:Q refused the connection"))
    slow = _round_trip(SlowLinkError("SR:Q did not answer"))

    assert type(mine) is ChannelUnreachableError
    assert mine.channel_address == "SR:Q"
    assert str(mine) == "SR:Q went quiet"
    assert type(refused) is ConnectionError
    assert str(refused) == "SR:Q refused the connection"
    assert type(slow) is TimeoutError
    assert str(slow) == "SR:Q did not answer"


def test_unknown_class_tag_decodes_fail_closed_to_the_contract_fault():
    rebuilt = exceptions.decode_exception(
        {"class_tag": "WeirdChildError", "message": "original text", "fields": {}}
    )

    assert type(rebuilt) is ConnectorInternalError
    assert rebuilt.class_tag == "WeirdChildError"
    assert "WeirdChildError" in str(rebuilt)
    assert "original text" in str(rebuilt)


def test_unusable_field_set_decodes_fail_closed_to_the_contract_fault():
    rebuilt = exceptions.decode_exception(
        {
            "class_tag": "ChannelWriteBlockedError",
            "message": "refused",
            "fields": {"channel_address": "SR:Q", "reason": "LIMITS", "bogus": 1},
        }
    )

    assert type(rebuilt) is ConnectorInternalError
    assert rebuilt.class_tag == "ChannelWriteBlockedError"
    assert "ChannelWriteBlockedError" in str(rebuilt)
    assert "refused" in str(rebuilt)


def test_an_outcome_word_the_parent_lacks_decodes_fail_closed_to_the_contract_fault():
    rebuilt = exceptions.decode_exception(
        {
            "class_tag": "ChannelWriteFailedError",
            "message": "write not confirmed",
            "fields": {
                "channel_address": "SR:Q",
                "reason": "FAILED",
                "outcome": "no-such-outcome",
                "value_written": 1.0,
                "observed_value": None,
            },
        }
    )

    assert type(rebuilt) is ConnectorInternalError
    assert rebuilt.class_tag == "ChannelWriteFailedError"
    assert "write not confirmed" in str(rebuilt)


@pytest.mark.parametrize(
    ("body", "expected"),
    [({}, ConnectorInternalError), ({"class_tag": "ConnectionError"}, ConnectionError)],
)
def test_incomplete_frame_body_still_yields_an_exception(body, expected):
    rebuilt = exceptions.decode_exception(body)

    assert type(rebuilt) is expected


def test_the_registry_is_internal_to_the_package():
    # Both sides of the pipe are this package's own code, so the registry is
    # not API: the namespace lists none of it.
    assert not set(exceptions.__all__) & set(osprey_connectors.ipc.__all__)
