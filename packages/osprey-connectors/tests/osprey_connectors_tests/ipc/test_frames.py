"""Wire-format contract for the connector-host IPC codec.

These tests are the shared contract: the child (which serves requests) and the
proxy (which issues them) both encode against this format, so a change here is
a change to both sides at once. Assertions are on concrete payloads — the exact
value that came back out, the exact fields on a reconstructed exception —
never merely that a round trip "didn't raise".
"""

import json
import re
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel

from osprey_connectors.control_system._base import (
    Alarm,
    ChannelMetadata,
    ChannelReadBatch,
    ChannelValue,
    ChannelWriteResult,
    WriteOutcome,
)
from osprey_connectors.errors import (
    ChannelLimitsViolationError,
    ChannelNotFoundError,
    ChannelUnreachableError,
    ChannelWriteBlockedError,
    ConnectorInternalError,
    ConnectorUnavailableError,
    LimitsViolation,
)
from osprey_connectors.gateways import Endpoint
from osprey_connectors.ipc import _exceptions as exceptions
from osprey_connectors.ipc import _frames as frames
from osprey_connectors.refusals import RefusalReason

TS = datetime(2026, 8, 22, 14, 30, 5, 123456, tzinfo=UTC)


def _round_trip(payload: bytes):
    """Decode a frame the way the far side would: through the stream reader."""
    reader = frames.FrameReader()
    decoded = reader.feed(payload)
    assert len(decoded) == 1
    assert len(reader) == 0
    return decoded[0]


def _raised(frame) -> BaseException:
    """The exception the parent raises for an error frame."""
    assert isinstance(frame, frames.ErrorFrame)
    return exceptions.decode_exception(frame.error)


def _error(request_id: str, exc: BaseException) -> bytes:
    """An error frame for *exc*, as the child writes one."""
    return frames.encode_error(request_id, exceptions.encode_exception(exc))


# ---------------------------------------------------------------- requests


def test_request_frame_round_trips_method_and_kwargs():
    payload = frames.encode_request(
        "req-1",
        "write_channel",
        {
            "address": "SR:BEND:1:CUR",
            "value": 1.5,
            "timeout_s": None,
            "verify": True,
            "retries": 3,
            "history": [1, 2.5, "three"],
            "options": {"tolerance": 0.01, "level": "readback"},
            "at": TS,
        },
    )
    frame = _round_trip(payload)

    assert isinstance(frame, frames.RequestFrame)
    assert frame.request_id == "req-1"
    assert frame.method == "write_channel"
    assert frame.kwargs == {
        "address": "SR:BEND:1:CUR",
        "value": 1.5,
        "timeout_s": None,
        "verify": True,
        "retries": 3,
        "history": [1, 2.5, "three"],
        "options": {"tolerance": 0.01, "level": "readback"},
        "at": TS,
    }
    assert frame.kwargs["verify"] is True
    assert frame.kwargs["timeout_s"] is None


def test_batched_multi_channel_read_request_round_trips():
    payload = frames.encode_request(
        "req-batch",
        "read_multiple_channels",
        {"addresses": ["SR:A", "SR:B", "SR:C"], "timeout_s": 2.0},
    )
    frame = _round_trip(payload)

    assert frame.method == "read_multiple_channels"
    assert frame.kwargs["addresses"] == ["SR:A", "SR:B", "SR:C"]
    assert frame.kwargs["timeout_s"] == 2.0


def test_codec_does_not_validate_method_names():
    """The codec is method-agnostic: an unknown method is carried, not refused."""
    frame = _round_trip(frames.encode_request("req-x", "not_a_real_method", {}))

    assert frame.method == "not_a_real_method"
    assert frame.kwargs == {}


def test_new_request_id_is_unique():
    ids = {frames.new_request_id() for _ in range(100)}

    assert len(ids) == 100


# ---------------------------------------------------------------- results


def test_channel_value_result_round_trips_as_a_real_dataclass():
    value = ChannelValue(
        address="SR:DCCT",
        value=1.234,
        sampled_at=TS,
        alarm=Alarm(status="NO_ALARM", severity=0),
        metadata=ChannelMetadata(
            units="mA",
            precision=3,
            description="beam current",
            display_low=0.0,
            display_high=500.0,
            raw_metadata={"egu": "mA", "count": 1},
        ),
    )
    frame = _round_trip(frames.encode_result("req-r", value))

    assert isinstance(frame.value, ChannelValue)
    assert frame.value.address == "SR:DCCT"
    assert frame.value.value == 1.234
    assert frame.value.sampled_at == TS
    assert frame.value.alarm == Alarm(status="NO_ALARM", severity=0)
    assert isinstance(frame.value.metadata, ChannelMetadata)
    assert frame.value.metadata.units == "mA"
    assert frame.value.metadata.precision == 3
    assert frame.value.metadata.display_high == 500.0
    assert frame.value.metadata.raw_metadata == {"egu": "mA", "count": 1}
    # Not an enum channel: both enum fields survive as None rather than as
    # something the far side has to interpret.
    assert frame.value.metadata.enum_labels is None
    assert frame.value.enum_label is None


def test_enum_metadata_crosses_the_seam_as_the_tuple_of_labels_sent():
    """An mbbi reading keeps both halves across the connector-host boundary.

    The labels cross as a JSON array, which decodes as a list; the field turns
    the list back into the tuple the connector built, so what a proxy hands
    back compares equal to it. A list given on the sending side is held as a
    tuple there too.
    """
    value = ChannelValue(
        address="SR:MODE",
        value=2,
        sampled_at=TS,
        alarm=Alarm(status="NO_ALARM"),
        enum_label="ACQUIRING",
        metadata=ChannelMetadata(enum_labels=["OFFLINE", "STANDBY", "ACQUIRING", "FAULT"]),
    )
    frame = _round_trip(frames.encode_result("req-enum", value))

    assert frame.value.value == 2
    assert frame.value.enum_label == "ACQUIRING"
    assert frame.value.metadata.enum_labels == ("OFFLINE", "STANDBY", "ACQUIRING", "FAULT")
    assert frame.value == value


def test_batched_read_result_dict_round_trips():
    reading = {
        "SR:A": ChannelValue(
            address="SR:A", value=1.0, sampled_at=TS, metadata=ChannelMetadata(units="mA")
        ),
        "SR:B": ChannelValue(address="SR:B", value="Off", sampled_at=TS),
        "SR:C": ChannelValue(address="SR:C", value=None, sampled_at=TS),
    }
    frame = _round_trip(frames.encode_result("req-batch", reading))

    assert sorted(frame.value) == ["SR:A", "SR:B", "SR:C"]
    assert frame.value["SR:A"].value == 1.0
    assert frame.value["SR:A"].metadata.units == "mA"
    assert frame.value["SR:B"].value == "Off"
    assert frame.value["SR:C"].value is None
    assert all(isinstance(v, ChannelValue) for v in frame.value.values())


def test_batch_read_round_trips_its_values_and_each_channel_s_own_error():
    batch = ChannelReadBatch(
        values={"SR:A": ChannelValue(address="SR:A", value=1.0, sampled_at=TS)},
        failures={
            "SR:B": ChannelUnreachableError("SR:B", "SR:B did not answer within 2s"),
            "SR:C": ChannelNotFoundError("SR:C"),
        },
    )

    frame = _round_trip(frames.encode_result("req-batch", batch))

    assert type(frame.value) is ChannelReadBatch
    assert frame.value.values == batch.values
    assert list(frame.value.failures) == ["SR:B", "SR:C"]
    for address, sent in batch.failures.items():
        got = frame.value.failures[address]
        assert type(got) is type(sent)
        assert got.address == address
        assert str(got) == str(sent)


def test_a_batch_failure_crosses_as_an_error_node_with_its_class_and_fields():
    batch = ChannelReadBatch(failures={"SR:B": ChannelUnreachableError("SR:B", "no answer")})

    failures = _header_of(frames.encode_result("req-batch", batch))["value"]["fields"]["failures"]

    [[address, node]] = failures["items"]
    assert address == "SR:B"
    assert node[_TAG] == "error"
    assert node["class_tag"] == "ChannelUnreachableError"
    assert node["message"] == "no answer"
    assert node["fields"] == {_TAG: "dict", "items": [["address", "SR:B"]]}


def test_an_exception_that_is_no_channel_s_cannot_cross_as_a_value():
    with pytest.raises(frames.FrameEncodeError):
        frames.encode_result("req-x", ConnectorUnavailableError("the gateway is gone"))


def test_write_result_round_trips_the_verdict_reason():
    result = ChannelWriteResult(
        address="SR:BEND:1:CUR",
        value_written=2.5,
        outcome=WriteOutcome.REFUSED,
        refusal_reason="WRITES_DISABLED",
        error_message="Write to 'SR:BEND:1:CUR' blocked",
        verdict_reason="switching",
    )
    frame = _round_trip(frames.encode_result("req-w", result))

    assert frame.value.verdict_reason == "switching"


def test_write_result_round_trips_with_outcome_and_alarm():
    """A confirmed result crosses with its re-read as a nested reading, alarm and all."""
    observed = ChannelValue(
        address="SR:BEND:1:CUR",
        value=2.5,
        sampled_at=TS,
        alarm=Alarm(status="NO_ALARM", severity=0),
        metadata=ChannelMetadata(units="A"),
    )
    result = ChannelWriteResult.confirmed("SR:BEND:1:CUR", 2.5, observed)
    payload = frames.encode_result("req-w", result)

    node = _header_of(payload)["value"]
    assert node[_TAG] == "ChannelWriteResult"
    assert node["fields"]["observed"][_TAG] == "ChannelValue"
    assert node["fields"]["observed"]["fields"]["alarm"][_TAG] == "Alarm"
    frame = _round_trip(payload)

    assert isinstance(frame.value, ChannelWriteResult)
    assert frame.value == result
    assert frame.value.outcome is WriteOutcome.CONFIRMED
    assert frame.value.refusal_reason is None
    assert frame.value.error_message is None
    assert type(frame.value.observed) is ChannelValue
    assert frame.value.observed == observed
    assert frame.value.observed.alarm == Alarm(status="NO_ALARM", severity=0)
    assert frame.value.observed_number == 2.5


def test_write_result_outcome_crosses_as_a_string_and_returns_an_enum():
    """The outcome is a plain JSON string on the wire, an enum on both sides.

    ``WriteOutcome`` is a ``StrEnum``, so it encodes as the bare word; the
    decoder rebuilds the dataclass from the decoded fields and the dataclass's
    ``__post_init__`` is what turns that word back into the member. Without
    that coercion a parent-side ``outcome == WriteOutcome.MISMATCH`` would
    still hold but ``outcome is WriteOutcome.MISMATCH`` would not, and the
    enum would stop being the single owned verdict across the boundary.
    """
    payload = frames.encode_result(
        "req-wm1",
        ChannelWriteResult(
            address="SR:BEND:1:CUR",
            value_written=2.5,
            outcome=WriteOutcome.MISMATCH,
            observed=ChannelValue(address="SR:BEND:1:CUR", value=2.0, sampled_at=TS),
            notes="observed 2.0, sent 2.5",
        ),
    )
    assert b'"outcome": "mismatch"' in payload

    frame = _round_trip(payload)

    assert type(frame.value.outcome) is WriteOutcome
    assert frame.value.outcome is WriteOutcome.MISMATCH
    assert frame.value.value_written == 2.5
    assert frame.value.observed.value == 2.0
    assert frame.value.error_message is None
    assert frame.value.notes == "observed 2.0, sent 2.5"


def test_write_result_carries_an_array_observed_value():
    result = ChannelWriteResult(
        address="SR:WF",
        value_written=[1.0, 2.0, 3.0],
        outcome=WriteOutcome.CONFIRMED,
        observed=ChannelValue(
            address="SR:WF", value=np.array([1.0, 2.0, 3.0], dtype=np.float32), sampled_at=TS
        ),
    )
    frame = _round_trip(frames.encode_result("req-wa", result))

    assert frame.value.outcome is WriteOutcome.CONFIRMED
    assert frame.value.value_written == [1.0, 2.0, 3.0]
    assert isinstance(frame.value.observed.value, np.ndarray)
    assert frame.value.observed.value.dtype == np.float32
    assert frame.value.observed.value.tolist() == [1.0, 2.0, 3.0]


#: What the limits said of a refused value, as a limits refusal carries it.
_VIOLATION = LimitsViolation(
    kind="MAX_EXCEEDED", reason="Value 2 above maximum 1.0", min_value=0.0, max_value=1.0
)


@pytest.mark.parametrize("reason", list(RefusalReason))
def test_a_refusal_reason_crosses_ipc_as_its_member(reason):
    """The parent rebuilds a refused result with the reason's member, not a bare string."""
    result = ChannelWriteResult(
        address="SR:R",
        value_written=1,
        outcome=WriteOutcome.REFUSED,
        refusal_reason=reason,
        error_message="refused",
        limits_violation=_VIOLATION if reason is RefusalReason.LIMITS else None,
    )
    frame = _round_trip(frames.encode_result("req-rr", result))

    assert frame.value.refusal_reason is reason
    assert frame.value.refusal_reason.refuser == reason.refuser


def test_write_result_list_round_trips():
    results = [
        ChannelWriteResult(
            address="SR:A",
            value_written=1,
            outcome=WriteOutcome.CONFIRMED,
            observed=ChannelValue(address="SR:A", value=1, sampled_at=TS),
        ),
        ChannelWriteResult(
            address="SR:B",
            value_written=2,
            outcome=WriteOutcome.REFUSED,
            refusal_reason="LIMITS",
            error_message="Write to 'SR:B' blocked: outside the configured limits.",
            limits_violation=_VIOLATION,
        ),
        ChannelWriteResult(
            address="SR:C",
            value_written=3,
            outcome=WriteOutcome.UNREQUESTED,
        ),
    ]
    frame = _round_trip(frames.encode_result("req-wm", results))

    assert all(isinstance(r, ChannelWriteResult) for r in frame.value)
    assert [r.address for r in frame.value] == ["SR:A", "SR:B", "SR:C"]
    assert [r.outcome for r in frame.value] == [
        WriteOutcome.CONFIRMED,
        WriteOutcome.REFUSED,
        WriteOutcome.UNREQUESTED,
    ]
    assert frame.value[0].observed.value == 1
    assert frame.value[1].refusal_reason == "LIMITS"
    assert frame.value[1].limits_violation == _VIOLATION
    assert frame.value[1].error_message.startswith("Write to 'SR:B' blocked")
    assert frame.value[2].refusal_reason is None
    assert frame.value[2].observed is None


def test_disconnect_result_carries_none():
    frame = _round_trip(frames.encode_result("req-d", None))

    assert isinstance(frame, frames.ResultFrame)
    assert frame.value is None


def test_child_report_dict_round_trips():
    report = {
        "selected_role": "read_only",
        "mode": "name_server",
        "host": "localhost",
        "port": 5074,
        "_epics_configured": True,
    }
    frame = _round_trip(frames.encode_result("req-probe", report))

    assert frame.value == report
    assert frame.value["_epics_configured"] is True


# ---------------------------------------------------------------- arrays


def test_float64_2d_array_round_trips_with_dtype_and_shape():
    array = np.linspace(0.0, 1.0, 12).reshape(3, 4)
    frame = _round_trip(frames.encode_result("req-a", array))

    assert isinstance(frame.value, np.ndarray)
    assert frame.value.dtype == np.dtype("float64")
    assert frame.value.shape == (3, 4)
    assert np.array_equal(frame.value, array)


def test_a_blob_that_is_not_a_single_npy_array_is_refused():
    buffer = BytesIO()
    np.savez(buffer, a=np.arange(3))

    with pytest.raises(frames.FrameDecodeError, match="not a single .npy array"):
        frames._npy_load(buffer.getvalue())


def test_int_array_round_trips_with_dtype_preserved():
    array = np.array([1, 2, 3, 4], dtype=np.int32)
    frame = _round_trip(frames.encode_result("req-i", array))

    assert frame.value.dtype == np.dtype("int32")
    assert frame.value.tolist() == [1, 2, 3, 4]


def test_array_inside_a_channel_value_round_trips():
    waveform = np.arange(6, dtype=np.float64) * 0.5
    frame = _round_trip(
        frames.encode_result("req-wf", ChannelValue(address="SR:WF", value=waveform, sampled_at=TS))
    )

    assert isinstance(frame.value, ChannelValue)
    assert np.array_equal(frame.value.value, waveform)
    assert frame.value.value.dtype == np.dtype("float64")


def test_multiple_arrays_in_one_frame_keep_their_order():
    payload = frames.encode_result(
        "req-multi",
        {
            "a": np.array([1, 2], dtype=np.int16),
            "b": np.array([[3.0], [4.0]]),
        },
    )
    frame = _round_trip(payload)

    assert frame.value["a"].tolist() == [1, 2]
    assert frame.value["a"].dtype == np.dtype("int16")
    assert frame.value["b"].shape == (2, 1)
    assert frame.value["b"].tolist() == [[3.0], [4.0]]


def test_numpy_scalar_round_trips_with_dtype():
    frame = _round_trip(frames.encode_result("req-s", np.float32(2.5)))

    assert isinstance(frame.value, np.floating)
    assert frame.value.dtype == np.dtype("float32")
    assert float(frame.value) == 2.5


def test_unsupported_value_is_refused_at_encode_time():
    with pytest.raises(frames.FrameEncodeError, match="object"):
        frames.encode_result("req-bad", object())


# ---------------------------------------------------------------- records as models


def _header_of(payload: bytes) -> dict:
    body = payload[8:]
    (header_len,) = frames._HEADER_STRUCT.unpack_from(body, 0)
    return json.loads(body[4 : 4 + header_len])


def test_a_record_crosses_as_its_tag_and_its_fields_in_order():
    metadata = ChannelMetadata(units="mA", precision=3)

    payload = frames.encode_result("req-m", metadata)

    node = _header_of(payload)["value"]
    assert node[_TAG] == "ChannelMetadata"
    assert list(node["fields"]) == list(ChannelMetadata.model_fields)
    frame = _round_trip(payload)
    assert type(frame.value) is ChannelMetadata
    assert frame.value == metadata


def test_a_model_defined_outside_the_package_is_refused_at_encode_time():
    class Outsider(BaseModel):
        name: str

    with pytest.raises(frames.FrameEncodeError, match="'Outsider'"):
        frames.encode_result("req-o", Outsider(name="x"))


def test_another_package_model_is_refused_at_encode_time():
    """Only a connector's three result records cross, not every model the package defines."""
    with pytest.raises(frames.FrameEncodeError, match="'Endpoint'"):
        frames.encode_result("req-e", Endpoint(host="h", port=1, mode="addr_list"))


def test_a_frame_naming_another_package_model_is_refused():
    """A frame cannot make the far side build a record no call returns."""
    reader = frames.FrameReader()
    naming = _doctored(
        _result(),
        _set("value", {_TAG: "Endpoint", "fields": {"host": "h", "port": 1, "mode": "addr_list"}}),
    )

    with pytest.raises(frames.FrameDecodeError, match="unknown value tag 'Endpoint'"):
        reader.feed(naming)


def test_a_subclass_of_a_record_is_refused_at_encode_time():
    """Only the class the far side rebuilds crosses."""

    class MyMetadata(ChannelMetadata):
        pass

    with pytest.raises(frames.FrameEncodeError, match="'MyMetadata'"):
        frames.encode_result("req-s", MyMetadata(units="mA"))


def test_a_model_reply_its_class_refuses_fails_only_that_call():
    reader = frames.FrameReader()
    refused = _doctored(_result(), _set("value", {_TAG: "ChannelMetadata", "fields": {"units": 7}}))
    follow_up = frames.encode_result("req-next", 2)

    frame, next_frame = reader.feed(refused + follow_up)

    assert isinstance(frame, frames.ErrorFrame)
    assert frame.request_id == "req-bad"
    assert frame.error.class_tag == "ConnectorInternalError"
    assert frame.error.fields == {"class_tag": "ChannelMetadata"}
    assert type(_raised(frame)) is ConnectorInternalError
    assert next_frame == frames.ResultFrame(request_id="req-next", value=2)


def test_a_reading_stamped_with_no_zone_fails_only_that_call():
    """A naive ``sampled_at`` names no instant; the parent refuses it as a connector bug."""
    reader = frames.FrameReader()
    naive = {_TAG: "datetime", "iso": TS.replace(tzinfo=None).isoformat()}
    refused = _doctored(
        _result(),
        _set("value", {_TAG: "ChannelValue", "fields": {"value": 1.0, "sampled_at": naive}}),
    )

    (frame,) = reader.feed(refused)

    assert isinstance(frame, frames.ErrorFrame)
    assert frame.error.fields == {"class_tag": "ChannelValue"}
    assert type(_raised(frame)) is ConnectorInternalError


def test_the_request_an_undecodable_frame_names_is_read_back():
    unknown = _doctored(
        frames.encode_request("req-u", "read_channel", {"address": "SR:Q"}),
        _set("kwargs", {_TAG: "Bogus", "fields": {}}),
    )

    with pytest.raises(frames.FrameDecodeError):
        frames.decode_frame(unknown)
    assert frames.request_id_of(unknown) == "req-u"
    assert frames.request_id_of(_repack(b"{not json")) is None
    assert frames.request_id_of(_doctored(_result(), _set("request_id", 7))) is None


# ---------------------------------------------------------------- exceptions


# The field-by-field spec for each class is pinned in test_exception_frames.py;
# here the point is that every spec field survives the JSON frame codec.
TYPED_EXCEPTIONS = [
    pytest.param(
        ChannelLimitsViolationError(
            address="SR:BEND:1:CUR",
            value=999.0,
            violation_type="range",
            violation_reason="Value 999.0 above maximum 500.0",
            min_value=0.0,
            max_value=500.0,
            max_step=10.0,
            current_value=120.0,
        ),
        id="limits-full",
    ),
    pytest.param(
        ChannelLimitsViolationError(
            address="SR:X",
            value="on",
            violation_type="unlisted",
            violation_reason="Channel not in the limits registry",
        ),
        id="limits-unset-bounds",
    ),
    pytest.param(
        ChannelWriteBlockedError(
            address="SR:BEND:1:CUR",
            reason="WRITES_DISABLED",
            message="Writes are disabled for this deployment",
        ),
        id="blocked-custom-message",
    ),
    pytest.param(
        ChannelWriteBlockedError(address="SR:Y", reason="LIMITS"),
        id="blocked-default-message",
    ),
    pytest.param(ConnectionError("gateway unreachable"), id="connection"),
    pytest.param(TimeoutError("read timed out after 2.0s"), id="timeout"),
]


@pytest.mark.parametrize("original", TYPED_EXCEPTIONS)
def test_a_typed_exception_round_trips_with_every_spec_field(original):
    frame = _round_trip(_error("req-e", original))

    assert isinstance(frame, frames.ErrorFrame)
    assert frame.request_id == "req-e"
    assert frame.error.class_tag == type(original).__name__
    rebuilt = _raised(frame)
    assert type(rebuilt) is type(original)
    # The operator-facing message is re-rendered (or carried) intact.
    assert str(rebuilt) == str(original)
    assert exceptions.encode_exception(rebuilt) == exceptions.encode_exception(original)


def test_a_vocabulary_subclass_is_tagged_as_its_registered_ancestor():
    # No host in between: the one encoder names the class the far side rebuilds.
    class _Mine(ChannelUnreachableError):
        pass

    frame = _round_trip(_error("req-s", _Mine("SR:Q", "SR:Q went quiet")))
    rebuilt = _raised(frame)

    assert frame.error.class_tag == "ChannelUnreachableError"
    assert type(rebuilt) is ChannelUnreachableError
    assert rebuilt.address == "SR:Q"
    assert str(rebuilt) == "SR:Q went quiet"


def test_unknown_class_tag_fails_closed_to_the_contract_fault():
    # A tag this build does not register, as a child of another build could send.
    body = exceptions.ExceptionFrameBody(class_tag="SomeChildOnlyError", message="libca segfaulted")

    frame = _round_trip(frames.encode_error("req-u", body))
    rebuilt = _raised(frame)

    assert frame.error == body
    assert type(rebuilt) is ConnectorInternalError
    assert rebuilt.class_tag == "SomeChildOnlyError"
    # Fail closed, but never silently: the original tag and message both survive.
    assert "SomeChildOnlyError" in str(rebuilt)
    assert "libca segfaulted" in str(rebuilt)


def test_an_error_frame_holds_the_body_it_carried_and_no_live_exception():
    assert set(frames.ErrorFrame.model_fields) == {"request_id", "error"}
    assert not frames.ErrorFrame.model_config.get("arbitrary_types_allowed")


# ---------------------------------------------------------------- stream reader


def test_reader_reassembles_a_frame_split_one_byte_at_a_time():
    payload = frames.encode_result(
        "req-split",
        {"wave": np.arange(32, dtype=np.float64), "label": "split frame"},
    )
    reader = frames.FrameReader()

    decoded = []
    for index in range(len(payload)):
        decoded.extend(reader.feed(payload[index : index + 1]))
        if index < len(payload) - 1:
            assert decoded == [], f"frame emitted early after {index + 1} bytes"

    assert len(decoded) == 1
    assert decoded[0].request_id == "req-split"
    assert decoded[0].value["label"] == "split frame"
    assert np.array_equal(decoded[0].value["wave"], np.arange(32, dtype=np.float64))
    assert len(reader) == 0


@pytest.mark.parametrize("cut", [1, 4, 7, 8, 9, 15, 40])
def test_reader_reassembles_a_frame_split_at_an_arbitrary_boundary(cut):
    payload = frames.encode_request("req-cut", "read_channel", {"address": "SR:A"})
    reader = frames.FrameReader()

    assert reader.feed(payload[:cut]) == []
    tail = reader.feed(payload[cut:])

    assert len(tail) == 1
    assert tail[0].kwargs == {"address": "SR:A"}


def test_reader_emits_several_coalesced_frames_and_buffers_the_partial_tail():
    one = frames.encode_result("req-1", 1)
    two = frames.encode_result("req-2", 2)
    three = frames.encode_result("req-3", 3)
    reader = frames.FrameReader()

    decoded = reader.feed(one + two + three[:5])

    assert [f.request_id for f in decoded] == ["req-1", "req-2"]
    assert [f.value for f in decoded] == [1, 2]
    assert len(reader) == 5
    assert [f.request_id for f in reader.feed(three[5:])] == ["req-3"]


def test_reader_rejects_a_desynchronised_stream():
    reader = frames.FrameReader()

    with pytest.raises(frames.FrameDecodeError, match="out of sync"):
        reader.feed(b"garbage-not-a-frame-at-all")


def test_reader_refuses_an_absurd_declared_length():
    reader = frames.FrameReader(max_frame_bytes=1024)
    payload = frames.encode_result("req-big", "x" * 4096)

    with pytest.raises(frames.FrameDecodeError, match="above the 1024 limit"):
        reader.feed(payload)


# ---------------------------------------------------------------- no pickle


def test_module_source_never_imports_pickle():
    source = Path(frames.__file__).read_text()

    assert not re.search(r"^\s*import pickle\b", source, re.MULTILINE)
    assert not re.search(r"^\s*from pickle\b", source, re.MULTILINE)
    assert not re.search(r"\bpickle\.(loads|load|dumps|dump)\b", source)


def test_every_npy_call_in_the_module_disables_pickle():
    source = Path(frames.__file__).read_text()
    calls = re.findall(r"np\.(?:save|load)\([^)]*\)", source)

    assert len(calls) == 2, f"expected exactly the two _npy helpers, found {calls}"
    for call in calls:
        assert "allow_pickle=False" in call, call


def test_np_save_is_invoked_with_allow_pickle_false(monkeypatch):
    seen = []
    real_save = frames.np.save

    def spy(file, arr, **kwargs):
        seen.append(kwargs)
        return real_save(file, arr, **kwargs)

    monkeypatch.setattr(frames.np, "save", spy)
    frames.encode_result("req-npy", np.arange(4))

    assert seen == [{"allow_pickle": False}]


def test_np_load_is_invoked_with_allow_pickle_false(monkeypatch):
    payload = frames.encode_result("req-npy", np.arange(4))
    seen = []
    real_load = frames.np.load

    def spy(file, **kwargs):
        seen.append(kwargs)
        return real_load(file, **kwargs)

    monkeypatch.setattr(frames.np, "load", spy)
    frames.decode_frame(payload)

    assert seen == [{"allow_pickle": False}]


def test_object_array_cannot_be_encoded():
    """An object array is exactly what pickle would be needed for — so it is refused."""
    with pytest.raises(ValueError, match="pickle"):
        frames.encode_result("req-obj", np.array([{"a": 1}], dtype=object))


# ---------------------------------------------------------------- malformed frames


def _split(frame: bytes) -> tuple[dict, bytes]:
    """Take a well-formed frame apart into its JSON header and its blob bytes."""
    body = frame[8:]
    (header_len,) = frames._HEADER_STRUCT.unpack_from(body, 0)
    return json.loads(body[4 : 4 + header_len]), body[4 + header_len :]


def _repack(header: bytes, blob_bytes: bytes = b"") -> bytes:
    """Frame raw header bytes the way ``_pack`` does, with consistent outer lengths."""
    body = frames._HEADER_STRUCT.pack(len(header)) + header + blob_bytes
    return frames.MAGIC + frames._HEADER_STRUCT.pack(len(body)) + body


def _doctored(frame: bytes, edit) -> bytes:
    """``frame`` with its header passed through ``edit``, everything else intact."""
    header, blob_bytes = _split(frame)
    edit(header)
    return _repack(json.dumps(header).encode(), blob_bytes)


def _result(value=1) -> bytes:
    return frames.encode_result("req-bad", value)


def _array_result() -> bytes:
    return frames.encode_result("req-bad", np.arange(3))


def _two_array_result() -> bytes:
    return frames.encode_result("req-bad", [np.arange(3), np.arange(2)])


def _set(key, value):
    return lambda header: header.__setitem__(key, value)


def _drop(key):
    return lambda header: header.pop(key)


_TAG = frames._TAG

MALFORMED_FRAMES = [
    pytest.param(
        lambda: _doctored(_result(), _set("value", {"a": 1})),
        "untagged object in frame payload",
        id="untagged-object",
    ),
    pytest.param(
        lambda: _doctored(_result(), _set("value", {_TAG: "Bogus", "fields": {}})),
        "unknown value tag 'Bogus' in frame payload",
        id="unknown-value-tag",
    ),
    pytest.param(
        lambda: _doctored(_array_result(), _set("value", {_TAG: "ndarray", "blob": 5})),
        "blob index 5 outside the frame's 1 segments",
        id="blob-index-out-of-range",
    ),
    pytest.param(
        # True == 1 would index the second of two segments if it were let through.
        lambda: _doctored(_two_array_result(), _set("value", {_TAG: "ndarray", "blob": True})),
        "blob index True outside the frame's 2 segments",
        id="blob-index-is-a-bool",
    ),
    pytest.param(
        lambda: _result()[:-2],
        "body bytes but carries",
        id="truncated-body",
    ),
    pytest.param(
        lambda: b"XXXX" + _result()[4:],
        "does not start with the expected magic prefix",
        id="bad-magic",
    ),
    pytest.param(
        lambda: _result()[:8] + frames._HEADER_STRUCT.pack(10_000) + _result()[12:],
        "frame header length runs past the end of the frame",
        id="header-length-past-the-end",
    ),
    pytest.param(
        lambda: _repack(b"{not json"),
        "frame header is not valid JSON",
        id="non-json-header",
    ),
    pytest.param(
        lambda: _repack(b"\xff\xfe"),
        "frame header is not valid JSON",
        id="non-utf8-header",
    ),
    pytest.param(
        lambda: _repack(b"[1, 2]"),
        "frame header is not a JSON object",
        id="header-not-an-object",
    ),
    pytest.param(
        lambda: _doctored(_result(), _set("v", 99)),
        f"frame protocol version 99 != {frames.PROTOCOL_VERSION}",
        id="wrong-protocol-version",
    ),
    pytest.param(
        lambda: _doctored(_result(), _set("v", 1)),
        f"frame protocol version 1 != {frames.PROTOCOL_VERSION}",
        id="version-1-peer-without-verdict-reasons",
    ),
    pytest.param(
        lambda: _doctored(_result(), _set("v", 2)),
        f"frame protocol version 2 != {frames.PROTOCOL_VERSION}",
        id="version-2-peer-raising-limits-refusals",
    ),
    pytest.param(
        lambda: _doctored(_array_result(), lambda h: h.__setitem__("blobs", [h["blobs"][0] + 7])),
        "frame blob segment is truncated",
        id="truncated-blob-segment",
    ),
    pytest.param(
        lambda: _doctored(_result(), _drop("request_id")),
        "frame is missing its request id",
        id="missing-request-id",
    ),
    pytest.param(
        lambda: _doctored(_result(), _set("request_id", 7)),
        "frame is missing its request id",
        id="non-string-request-id",
    ),
    pytest.param(
        lambda: _doctored(_result(), _set("kind", "gossip")),
        "unknown frame kind 'gossip'",
        id="unknown-frame-kind",
    ),
    pytest.param(
        lambda: _doctored(_result(), _set("value", {_TAG: "error", "class_tag": 7})),
        "error node does not name its class and message as strings",
        id="error-node-without-a-class",
    ),
]


def _batch_failing_with(**node: object) -> dict:
    """A ``ChannelReadBatch`` node whose one failure's error node is edited by *node*."""
    error = {
        _TAG: "error",
        "class_tag": "ChannelUnreachableError",
        "message": "no answer",
        "fields": {_TAG: "dict", "items": [["address", "SR:Q"]]},
        **node,
    }
    return {
        _TAG: "ChannelReadBatch",
        "fields": {
            "values": {_TAG: "dict", "items": []},
            "failures": {_TAG: "dict", "items": [["SR:Q", error]]},
        },
    }


UNREBUILDABLE_REPLIES = [
    pytest.param(
        {
            _TAG: "ChannelWriteResult",
            "fields": {"address": "SR:Q", "value_written": 1.0, "outcome": "nope"},
        },
        "ChannelWriteResult",
        id="value-refused-by-its-constructor",
    ),
    pytest.param(
        {_TAG: "dict", "items": [["SR:Q", {_TAG: "ChannelValue", "fields": {"bogus": 1}}]]},
        "ChannelValue",
        id="nested-value-with-a-field-the-class-lacks",
    ),
    pytest.param(
        _batch_failing_with(class_tag="SomeChildOnlyError"),
        "ChannelReadBatch",
        id="batch-failure-of-a-class-the-parent-does-not-know",
    ),
    pytest.param(
        _batch_failing_with(fields={_TAG: "dict", "items": [["no_such_field", 1]]}),
        "ChannelReadBatch",
        id="batch-failure-whose-fields-its-class-refuses",
    ),
    pytest.param(
        _batch_failing_with(class_tag="ConnectorUnavailableError"),
        "ChannelReadBatch",
        id="batch-failure-that-is-no-channel-error",
    ),
]


@pytest.mark.parametrize(("value", "class_tag"), UNREBUILDABLE_REPLIES)
def test_a_reply_the_parent_cannot_rebuild_fails_only_that_call(value, class_tag):
    reader = frames.FrameReader()
    unrebuildable = _doctored(_result(), _set("value", value))
    follow_up = frames.encode_result("req-next", 2)

    frame, next_frame = reader.feed(unrebuildable + follow_up)

    assert isinstance(frame, frames.ErrorFrame)
    assert frame.request_id == "req-bad"
    assert frame.error.class_tag == "ConnectorInternalError"
    assert frame.error.fields == {"class_tag": class_tag}
    rebuilt = _raised(frame)
    assert type(rebuilt) is ConnectorInternalError
    assert rebuilt.class_tag == class_tag
    assert class_tag in str(rebuilt)
    # The stream is intact: the next reply decodes as sent.
    assert next_frame == frames.ResultFrame(request_id="req-next", value=2)


@pytest.mark.parametrize(("build", "message"), MALFORMED_FRAMES)
def test_decode_frame_refuses_a_malformed_frame_with_a_specific_reason(build, message):
    payload = build()

    with pytest.raises(frames.FrameDecodeError) as caught:
        frames.decode_frame(payload)

    assert message in str(caught.value)


def test_the_malformed_frame_builders_start_from_a_frame_that_decodes():
    """The table above mutates real frames; the unmutated ones must be valid, or
    every row could be failing for the same unrelated reason."""
    assert frames.decode_frame(_doctored(_result(), lambda header: None)).value == 1
    decoded = frames.decode_frame(_doctored(_array_result(), lambda header: None)).value
    assert decoded.tolist() == [0, 1, 2]


def test_a_json_node_no_json_parser_produces_is_refused():
    """``json.loads`` never yields a tuple or a set, so this arm cannot be reached
    through a frame; it stops a caller-built node tree from decoding as nothing."""
    with pytest.raises(frames.FrameDecodeError, match="unsupported JSON node of type 'set'"):
        frames._decode_value({1, 2}, [])


def test_encoding_a_frame_over_the_size_limit_is_refused(monkeypatch):
    monkeypatch.setattr(frames, "MAX_FRAME_BYTES", 64)

    with pytest.raises(frames.FrameEncodeError, match=r"frame of \d+ bytes exceeds the 64 limit"):
        frames.encode_result("req-big", "x" * 100)


def test_an_error_body_whose_fields_are_not_keyed_by_names_is_a_malformed_frame():
    """A body's fields are attribute values by attribute name; no sender writes another key."""
    error = _error("req-k", ChannelNotFoundError("SR:Q"))
    keyed_by_number = _doctored(error, _set("fields", {_TAG: "dict", "items": [[7, "SR:Q"]]}))

    with pytest.raises(frames.FrameDecodeError, match="error frame fields are not a mapping"):
        frames.decode_frame(keyed_by_number)
