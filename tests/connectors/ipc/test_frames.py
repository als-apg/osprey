"""Wire-format contract for the connector-host IPC codec.

These tests are the shared contract: the child (which serves requests) and the
proxy (which issues them) both encode against this format, so a change here is
a change to both sides at once. Assertions are on concrete payloads — the exact
value that came back out, the exact fields on a reconstructed exception —
never merely that a round trip "didn't raise".
"""

import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from osprey_connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    WriteVerification,
)
from osprey_connectors.errors import ChannelLimitsViolationError, ChannelWriteBlockedError
from osprey_connectors.ipc import frames

TS = datetime(2026, 8, 22, 14, 30, 5, 123456, tzinfo=UTC)


def _round_trip(payload: bytes):
    """Decode a frame the way the far side would: through the stream reader."""
    reader = frames.FrameReader()
    decoded = reader.feed(payload)
    assert len(decoded) == 1
    assert len(reader) == 0
    return decoded[0]


# ---------------------------------------------------------------- requests


def test_request_frame_round_trips_method_and_kwargs():
    payload = frames.encode_request(
        "req-1",
        "write_channel",
        {
            "address": "SR:BEND:1:CUR",
            "value": 1.5,
            "timeout": None,
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
        "timeout": None,
        "verify": True,
        "retries": 3,
        "history": [1, 2.5, "three"],
        "options": {"tolerance": 0.01, "level": "readback"},
        "at": TS,
    }
    assert frame.kwargs["verify"] is True
    assert frame.kwargs["timeout"] is None


def test_batched_multi_channel_read_request_round_trips():
    payload = frames.encode_request(
        "req-batch",
        "read_multiple_channels",
        {"addresses": ["SR:A", "SR:B", "SR:C"], "timeout": 2.0},
    )
    frame = _round_trip(payload)

    assert frame.method == "read_multiple_channels"
    assert frame.kwargs["addresses"] == ["SR:A", "SR:B", "SR:C"]
    assert frame.kwargs["timeout"] == 2.0


def test_codec_does_not_validate_method_names():
    """The codec is method-agnostic: an unknown method is carried, not refused."""
    frame = _round_trip(frames.encode_request("req-x", "not_a_real_method", {}))

    assert frame.method == "not_a_real_method"
    assert frame.kwargs == {}


def test_request_ids_match_replies_out_of_order():
    first = frames.encode_result("req-1", 11.0)
    second = frames.encode_result("req-2", 22.0)

    reader = frames.FrameReader()
    decoded = reader.feed(second + first)

    assert [f.request_id for f in decoded] == ["req-2", "req-1"]
    assert [f.value for f in decoded] == [22.0, 11.0]


def test_new_request_id_is_unique():
    ids = {frames.new_request_id() for _ in range(100)}

    assert len(ids) == 100


# ---------------------------------------------------------------- results


def test_channel_value_result_round_trips_as_a_real_dataclass():
    value = ChannelValue(
        value=1.234,
        timestamp=TS,
        metadata=ChannelMetadata(
            units="mA",
            precision=3,
            alarm_status="NO_ALARM",
            timestamp=TS,
            description="beam current",
            display_low=0.0,
            display_high=500.0,
            raw_metadata={"egu": "mA", "count": 1},
        ),
    )
    frame = _round_trip(frames.encode_result("req-r", value))

    assert isinstance(frame.value, ChannelValue)
    assert frame.value.value == 1.234
    assert frame.value.timestamp == TS
    assert isinstance(frame.value.metadata, ChannelMetadata)
    assert frame.value.metadata.units == "mA"
    assert frame.value.metadata.precision == 3
    assert frame.value.metadata.display_high == 500.0
    assert frame.value.metadata.raw_metadata == {"egu": "mA", "count": 1}


def test_batched_read_result_dict_round_trips():
    reading = {
        "SR:A": ChannelValue(value=1.0, timestamp=TS, metadata=ChannelMetadata(units="mA")),
        "SR:B": ChannelValue(value="Off", timestamp=TS),
        "SR:C": ChannelValue(value=None, timestamp=TS),
    }
    frame = _round_trip(frames.encode_result("req-batch", reading))

    assert sorted(frame.value) == ["SR:A", "SR:B", "SR:C"]
    assert frame.value["SR:A"].value == 1.0
    assert frame.value["SR:A"].metadata.units == "mA"
    assert frame.value["SR:B"].value == "Off"
    assert frame.value["SR:C"].value is None
    assert all(isinstance(v, ChannelValue) for v in frame.value.values())


def test_write_result_round_trips_with_nested_verification():
    result = ChannelWriteResult(
        channel_address="SR:BEND:1:CUR",
        value_written=2.5,
        success=True,
        verification=WriteVerification(
            level="readback",
            verified=True,
            readback_value=2.4999,
            tolerance_used=0.001,
            notes="within tolerance",
            readback_alarm_status="NO_ALARM",
            readback_alarm_severity=0,
            failure_kind=None,
        ),
        error_message=None,
        blocked=False,
        refusal_reason=None,
    )
    frame = _round_trip(frames.encode_result("req-w", result))

    assert isinstance(frame.value, ChannelWriteResult)
    assert frame.value.channel_address == "SR:BEND:1:CUR"
    assert frame.value.value_written == 2.5
    assert frame.value.success is True
    assert frame.value.blocked is False
    assert frame.value.refusal_reason is None
    assert isinstance(frame.value.verification, WriteVerification)
    assert frame.value.verification.level == "readback"
    assert frame.value.verification.readback_value == 2.4999
    assert frame.value.verification.readback_alarm_severity == 0
    assert frame.value.verification.failure_kind is None


def test_write_result_list_round_trips():
    results = [
        ChannelWriteResult(channel_address="SR:A", value_written=1, success=True),
        ChannelWriteResult(
            channel_address="SR:B",
            value_written=2,
            success=False,
            blocked=True,
            refusal_reason="LIMITS",
        ),
    ]
    frame = _round_trip(frames.encode_result("req-wm", results))

    assert [r.channel_address for r in frame.value] == ["SR:A", "SR:B"]
    assert frame.value[1].blocked is True
    assert frame.value[1].refusal_reason == "LIMITS"


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


def test_int_array_round_trips_with_dtype_preserved():
    array = np.array([1, 2, 3, 4], dtype=np.int32)
    frame = _round_trip(frames.encode_result("req-i", array))

    assert frame.value.dtype == np.dtype("int32")
    assert frame.value.tolist() == [1, 2, 3, 4]


def test_array_inside_a_channel_value_round_trips():
    waveform = np.arange(6, dtype=np.float64) * 0.5
    frame = _round_trip(frames.encode_result("req-wf", ChannelValue(value=waveform, timestamp=TS)))

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


# ---------------------------------------------------------------- exceptions


def test_channel_limits_violation_round_trips_every_field():
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
    frame = _round_trip(frames.encode_error("req-e", original))

    assert isinstance(frame, frames.ErrorFrame)
    assert frame.request_id == "req-e"
    assert frame.class_tag == "ChannelLimitsViolationError"
    rebuilt = frame.exception
    assert isinstance(rebuilt, ChannelLimitsViolationError)
    assert rebuilt.channel_address == "SR:BEND:1:CUR"
    assert rebuilt.attempted_value == 999.0
    assert rebuilt.violation_type == "range"
    assert rebuilt.violation_reason == "Value 999.0 above maximum 500.0"
    assert rebuilt.min_value == 0.0
    assert rebuilt.max_value == 500.0
    assert rebuilt.max_step == 10.0
    assert rebuilt.current_value == 120.0
    # The constructor re-renders the operator-facing message from those fields.
    assert str(rebuilt) == str(original)


def test_channel_limits_violation_keeps_unset_bounds_as_none():
    original = ChannelLimitsViolationError(
        channel_address="SR:X",
        value="on",
        violation_type="unlisted",
        violation_reason="Channel not in the limits registry",
    )
    rebuilt = _round_trip(frames.encode_error("req-e2", original)).exception

    assert rebuilt.attempted_value == "on"
    assert rebuilt.min_value is None
    assert rebuilt.max_value is None
    assert rebuilt.max_step is None
    assert rebuilt.current_value is None


def test_channel_write_blocked_round_trips_reason_and_message():
    original = ChannelWriteBlockedError(
        channel_address="SR:BEND:1:CUR",
        reason="WRITES_DISABLED",
        message="Writes are disabled for this deployment",
    )
    frame = _round_trip(frames.encode_error("req-b", original))
    rebuilt = frame.exception

    assert frame.class_tag == "ChannelWriteBlockedError"
    assert isinstance(rebuilt, ChannelWriteBlockedError)
    assert rebuilt.channel_address == "SR:BEND:1:CUR"
    assert rebuilt.reason == "WRITES_DISABLED"
    assert str(rebuilt) == "Writes are disabled for this deployment"


def test_channel_write_blocked_default_message_survives():
    original = ChannelWriteBlockedError(channel_address="SR:Y", reason="LIMITS")
    rebuilt = _round_trip(frames.encode_error("req-b2", original)).exception

    assert rebuilt.reason == "LIMITS"
    assert str(rebuilt) == str(original)


def test_connection_error_round_trips():
    frame = _round_trip(frames.encode_error("req-c", ConnectionError("gateway unreachable")))

    assert frame.class_tag == "ConnectionError"
    assert type(frame.exception) is ConnectionError
    assert str(frame.exception) == "gateway unreachable"


def test_timeout_error_round_trips():
    frame = _round_trip(frames.encode_error("req-t", TimeoutError("read timed out after 2.0s")))

    assert frame.class_tag == "TimeoutError"
    assert type(frame.exception) is TimeoutError
    assert str(frame.exception) == "read timed out after 2.0s"


def test_unknown_class_tag_fails_closed_to_connection_error():
    class SomeChildOnlyError(Exception):
        pass

    frame = _round_trip(frames.encode_error("req-u", SomeChildOnlyError("libca segfaulted")))

    assert frame.class_tag == "SomeChildOnlyError"
    assert type(frame.exception) is ConnectionError
    # Fail closed, but never silently: the original tag and message both survive.
    assert "SomeChildOnlyError" in str(frame.exception)
    assert "libca segfaulted" in str(frame.exception)


def test_unknown_class_tag_on_the_wire_fails_closed():
    """A tag this codec has never heard of still decodes, as a ConnectionError."""
    payload = frames.encode_error("req-u2", ConnectionError("original text"))
    # Same byte length, so the frame's declared header length stays valid.
    tampered = payload.replace(b"ConnectionError", b"WeirdChildError")
    assert len(tampered) == len(payload)

    frame = _round_trip(tampered)

    assert frame.class_tag == "WeirdChildError"
    assert type(frame.exception) is ConnectionError
    assert "WeirdChildError" in str(frame.exception)
    assert "original text" in str(frame.exception)


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


def test_decode_frame_rejects_a_truncated_frame():
    payload = frames.encode_result("req-short", 1)

    with pytest.raises(frames.FrameDecodeError):
        frames.decode_frame(payload[:-2])


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
