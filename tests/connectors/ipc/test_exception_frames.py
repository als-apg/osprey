"""Typed-exception contract for the connector-host IPC channel.

The child and the proxy both encode against this spec, so the field lists are
pinned here by exact contents: a class or field added on one side of the
boundary without the other has to fail in this file first. Assertions are on
the concrete reconstructed exception — every field the parent's error rendering
reads — never merely that a round trip "didn't raise".
"""

import pytest

from osprey_connectors.errors import ChannelLimitsViolationError, ChannelWriteBlockedError
from osprey_connectors.ipc import exceptions


def _round_trip(exc: BaseException) -> BaseException:
    """Encode ``exc`` to a frame body and rebuild it the way the parent would."""
    body = exceptions.encode_exception(exc)
    assert set(body) == {"class_tag", "message", "fields"}
    return exceptions.decode_exception(body)


# ---------------------------------------------------------------- protocol pin


def test_registry_carries_exactly_the_four_agreed_classes():
    assert set(exceptions.EXCEPTION_SPECS) == {
        "ConnectionError",
        "TimeoutError",
        "ChannelWriteBlockedError",
        "ChannelLimitsViolationError",
    }
    registry = exceptions.EXCEPTION_SPECS
    assert registry["ConnectionError"].cls is ConnectionError
    assert registry["TimeoutError"].cls is TimeoutError
    assert registry["ChannelWriteBlockedError"].cls is ChannelWriteBlockedError
    assert registry["ChannelLimitsViolationError"].cls is ChannelLimitsViolationError


def test_limits_violation_field_list_is_the_published_spec():
    assert exceptions.CHANNEL_LIMITS_VIOLATION_FIELDS == (
        "channel_address",
        "attempted_value",
        "violation_type",
        "violation_reason",
        "min_value",
        "max_value",
        "max_step",
        "current_value",
    )
    spec = exceptions.EXCEPTION_SPECS["ChannelLimitsViolationError"]
    assert spec.fields == exceptions.CHANNEL_LIMITS_VIOLATION_FIELDS
    # The spec spells the attribute; the constructor spells it `value`.
    assert spec.ctor_kwargs == {"attempted_value": "value"}


def test_write_blocked_field_list_is_the_published_spec():
    assert exceptions.CHANNEL_WRITE_BLOCKED_FIELDS == ("channel_address", "reason")
    spec = exceptions.EXCEPTION_SPECS["ChannelWriteBlockedError"]
    assert spec.fields == exceptions.CHANNEL_WRITE_BLOCKED_FIELDS
    # Its text is not a field: the frame's own message feeds the ctor keyword.
    assert spec.message_kwarg == "message"


def test_builtin_specs_carry_no_fields():
    assert exceptions.EXCEPTION_SPECS["ConnectionError"].fields == ()
    assert exceptions.EXCEPTION_SPECS["TimeoutError"].fields == ()


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
    assert set(body["fields"]) == set(exceptions.CHANNEL_LIMITS_VIOLATION_FIELDS)
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
    assert set(body["fields"]) == set(exceptions.CHANNEL_WRITE_BLOCKED_FIELDS)


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


def test_unknown_exception_type_encodes_fail_closed():
    class SomeChildOnlyError(Exception):
        pass

    original = SomeChildOnlyError("libca segfaulted")
    body = exceptions.encode_exception(original)

    assert body["class_tag"] == "ConnectionError"
    assert body["fields"] == {}
    # Fail closed, but never silently: the original repr survives.
    assert repr(original) in body["message"]
    assert "SomeChildOnlyError" in body["message"]
    assert "libca segfaulted" in body["message"]

    rebuilt = exceptions.decode_exception(body)

    assert type(rebuilt) is ConnectionError
    assert "SomeChildOnlyError" in str(rebuilt)
    assert "libca segfaulted" in str(rebuilt)


def test_unknown_class_tag_decodes_fail_closed_to_connection_error():
    rebuilt = exceptions.decode_exception(
        {"class_tag": "WeirdChildError", "message": "original text", "fields": {}}
    )

    assert type(rebuilt) is ConnectionError
    assert "WeirdChildError" in str(rebuilt)
    assert "original text" in str(rebuilt)


def test_unusable_field_set_decodes_fail_closed_to_connection_error():
    rebuilt = exceptions.decode_exception(
        {
            "class_tag": "ChannelWriteBlockedError",
            "message": "refused",
            "fields": {"channel_address": "SR:Q", "reason": "LIMITS", "bogus": 1},
        }
    )

    assert type(rebuilt) is ConnectionError
    assert "ChannelWriteBlockedError" in str(rebuilt)
    assert "refused" in str(rebuilt)


@pytest.mark.parametrize("body", [{}, {"class_tag": "ConnectionError"}])
def test_incomplete_frame_body_still_yields_an_exception(body):
    rebuilt = exceptions.decode_exception(body)

    assert isinstance(rebuilt, ConnectionError)
