"""PVA reads: routing onto a pvapy channel and mapping NT structures to ChannelValue.

Routing decided WHICH transport an address uses; this file covers what happens
once PVAccess is chosen — the ``_read_channel_pva`` worker call, the
normative-type mapping (:func:`_channel_value`, shared with Channel Access),
and the error classification that keeps a PVA outage looking exactly like a CA
one to everything upstream.

Two kinds of stand-in are used deliberately:

* the **client** is always the fake ``pvaccess`` module
  (``tests/connectors/_epics_fakes.py``), so the routing, timeout and failure
  assertions run on every machine and open no socket;
* the **values** the mapping is asserted against are REAL pvapy ``PvObject``
  structures (``pvaccess.NtScalar``, ``NtEnum``, ``NtNdArray``) wherever the
  assertion is about their shape. A hand-rolled dict would let the mapping
  agree with a fiction of the NT layout; an authentic NTNDArray is what proves
  the union member, the dimension order and the carried dtype are read the way
  pvapy actually hands them over. Building one opens no channel, so the real
  module is safe to import here; those tests are ``importorskip``-guarded for
  a host without a pvapy wheel.
"""

from datetime import datetime

import numpy as np
import pytest

from osprey.connectors.control_system.epics_connector import (
    _ACCESS_DENIED,
    _UNREACHABLE,
    _channel_value,
    _classify_client_error,
)
from tests.connectors._epics_fakes import (
    CA_READ_REQUEST,
    PVA_READ_REQUEST,
    FakePvaccess,
    FakePvaException,
    FakePvObject,
    Unconvertible,
    access_denied,
    ndarray_record,
    pva_connector,
    record,
    timed_out,
)

PVA_ADDRESS = "SR:CAM1:IMAGE"
CA_ADDRESS = "SR:BEAM:CURRENT"


def _pva():
    """The real pvapy module, for building authentic PvObjects — or skip."""
    return pytest.importorskip("pvaccess")


def _map(obj, address: str = PVA_ADDRESS):
    return _channel_value(address, obj, "pva")


def _frame(frame: np.ndarray, member: str, *, color_mode: int = 0):
    """A real NTNDArray carrying ``frame``, dimensions innermost first."""
    pva = _pva()
    value = pva.NtNdArray()
    value["value"] = {member: frame.flatten()}
    value["dimension"] = [
        pva.PvDimension(size, 0, size, 1, False) for size in reversed(frame.shape)
    ]
    value["attribute"] = [pva.NtAttribute("ColorMode", pva.PvInt(color_mode))]
    return value


# ---------------------------------------------------------------------------
# NTScalar / NTScalarArray
# ---------------------------------------------------------------------------


class TestNtScalarMapping:
    def test_float_scalar_carries_value_units_alarm_and_timestamp(self):
        pva = _pva()
        value = pva.NtScalar(pva.DOUBLE, 1.5)
        value["alarm"] = {"severity": 1, "status": 3, "message": "HIGH"}
        value["timeStamp"] = {
            "secondsPastEpoch": 1_700_000_000,
            "nanoseconds": 250_000_000,
            "userTag": 0,
        }
        value["display"] = {
            "units": "mA",
            "limitLow": 0.0,
            "limitHigh": 10.0,
            "description": "beam current",
            "format": "F6.2",
        }

        result = _map(value, CA_ADDRESS)

        assert result.value == 1.5
        assert result.metadata.units == "mA"
        assert result.metadata.precision == 2  # from display.format
        assert result.metadata.description == "beam current"
        assert result.metadata.display_low == 0.0
        assert result.metadata.display_high == 10.0
        assert result.metadata.alarm_status == "HIGH"
        assert result.metadata.alarm_severity == 1
        assert result.metadata.raw_metadata["nt_type"] is None  # a plain scalar
        assert result.metadata.raw_metadata["provider"] == "pva"
        assert result.metadata.raw_metadata["severity"] == 1
        assert result.metadata.raw_metadata["status"] == 3
        assert result.timestamp.timestamp() == pytest.approx(1_700_000_000.25)
        assert result.metadata.timestamp == result.timestamp

    def test_integer_scalar_stays_an_int(self):
        pva = _pva()

        result = _map(pva.NtScalar(pva.INT, 42))

        assert result.value == 42
        assert isinstance(result.value, int)

    def test_string_scalar_maps_through_unchanged(self):
        pva = _pva()

        result = _map(pva.NtScalar(pva.STRING, "Injecting"))

        assert result.value == "Injecting"
        assert result.metadata.raw_metadata["dtype"] == "str"

    def test_scalar_array_is_a_numpy_array_with_its_dtype(self):
        pva = _pva()
        value = pva.PvObject({"value": [pva.DOUBLE]}, {"value": [1.0, 2.0, 3.0]})

        result = _map(value)

        assert isinstance(result.value, np.ndarray)
        assert result.value.tolist() == [1.0, 2.0, 3.0]
        assert result.metadata.raw_metadata["dtype"] == "float64"
        assert result.metadata.raw_metadata["nt_type"] is None

    def test_missing_timestamp_falls_back_to_now(self):
        """An unset timeStamp must not read as 1970 — the CA path behaves the same."""
        pva = _pva()

        before = datetime.now().timestamp()
        result = _map(pva.NtScalar(pva.DOUBLE, 0.25))

        assert result.timestamp.timestamp() >= before

    def test_precision_is_read_when_the_server_publishes_it_as_a_number(self):
        """QSRV2 adds display.precision; pvapy's own NtScalar carries only ``format``."""
        value = FakePvObject(
            {
                "value": 3.25,
                "display": {"units": "mm", "precision": 4, "format": "F9.2"},
                "alarm": {"severity": 0, "status": 0, "message": ""},
            }
        )

        result = _map(value)

        assert result.value == 3.25
        assert result.metadata.precision == 4  # the number wins over the format
        assert result.metadata.units == "mm"
        # A reported healthy severity stays 0 — distinct from "not reported".
        assert result.metadata.alarm_severity == 0
        # A PVA message is free text, not a status name: empty stays "not reported".
        assert result.metadata.alarm_status is None

    def test_absent_display_leaves_metadata_empty_rather_than_failing(self):
        result = _map(FakePvObject({"value": 7.0}))

        assert result.value == 7.0
        assert result.metadata.units == ""
        assert result.metadata.precision is None
        assert result.metadata.alarm_status is None
        assert result.metadata.alarm_severity is None

    def test_one_unconvertible_field_does_not_cost_the_reading(self):
        """Fields are converted one by one: a field pvapy cannot hand over is skipped."""
        value = FakePvObject(
            {"value": 2.0, "display": {"units": "V"}, "valueAlarm": Unconvertible()}
        )

        result = _map(value)

        assert result.value == 2.0
        assert result.metadata.units == "V"

    def test_failing_introspection_maps_as_a_scalar(self):
        """``getStructureDict`` is an optimization, never a precondition."""
        value = FakePvObject({"value": 5.0}, structure_error=RuntimeError("no introspection"))

        result = _map(value)

        assert result.value == 5.0
        assert result.metadata.raw_metadata["nt_type"] is None


# ---------------------------------------------------------------------------
# NTEnum
# ---------------------------------------------------------------------------


class TestNtEnumMapping:
    """An enum reading carries both halves: the index as the value, the label beside it.

    The index is what ``value`` holds, so a state channel has the same
    machine-readable type whether it was routed over PVAccess or Channel
    Access. What the operator reads -- "On" rather than "2" -- is the label,
    first-class metadata rather than something to be reconstructed from a
    choices list a later read may not repeat.
    """

    def test_enum_reads_as_the_index_with_its_label_alongside(self):
        pva = _pva()

        result = _map(pva.NtEnum(["Off", "Standby", "On"], 2), "SR:RF:STATE")

        assert result.value == 2
        assert not isinstance(result.value, str)
        assert result.metadata.enum_label == "On"
        assert result.metadata.enum_labels == ["Off", "Standby", "On"]
        assert result.metadata.raw_metadata["enum_index"] == 2
        assert result.metadata.raw_metadata["enum_choices"] == ["Off", "Standby", "On"]
        assert result.metadata.raw_metadata["nt_type"] == "NTEnum"

    def test_enum_index_zero_is_a_choice_not_a_falsy_miss(self):
        pva = _pva()

        result = _map(pva.NtEnum(["Off", "On"], 0), "SR:RF:STATE")

        assert result.value == 0
        assert result.metadata.enum_label == "Off"
        assert result.metadata.raw_metadata["enum_index"] == 0

    def test_enum_without_choices_still_reports_the_index(self):
        """Servers send the choices list only on change; a later read may omit it."""
        pva = _pva()
        value = pva.NtEnum(["placeholder"], 0)
        value["value"] = {"index": 1, "choices": []}

        result = _map(value, "SR:RF:STATE")

        assert result.value == 1
        assert result.metadata.enum_label is None
        assert result.metadata.enum_labels is None
        assert result.metadata.raw_metadata["enum_choices"] == []

    def test_a_scalar_channel_reports_no_enum_fields_at_all(self):
        """The two fields are how a consumer tells an enum from anything else."""
        result = _map(FakePvObject({"value": 7.0}))

        assert result.metadata.enum_label is None
        assert result.metadata.enum_labels is None


# ---------------------------------------------------------------------------
# NTNDArray
# ---------------------------------------------------------------------------


class TestNtNdArrayMapping:
    def test_uint16_frame_keeps_its_unsigned_dtype(self):
        """The carried dtype is preserved: bright pixels must not read as negative."""
        frame = np.array([[0, 40000, 65535], [12000, 33000, 60000]], dtype=np.uint16)

        result = _map(_frame(frame, "ushortValue"))

        assert isinstance(result.value, np.ndarray)
        assert result.value.dtype == np.uint16
        assert result.value.shape == (2, 3)
        assert result.value.tolist() == frame.tolist()
        assert result.value.min() >= 0

    def test_dimensions_are_reversed_into_numpy_order(self):
        """NT dimensions run innermost-first (width, height); numpy shape is reversed."""
        frame = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64)

        result = _map(_frame(frame, "ushortValue"))

        assert result.metadata.raw_metadata["dimensions"] == [64, 48]
        assert result.value.shape == (48, 64)
        assert result.metadata.raw_metadata["shape"] == [48, 64]

    def test_raw_metadata_records_dtype_colormode_and_codec(self):
        raw = _map(_frame(np.zeros((4, 5), dtype=np.uint16), "ushortValue")).metadata.raw_metadata

        assert raw["nt_type"] == "NTNDArray"
        assert raw["dtype"] == "uint16"
        assert raw["color_mode"] == 0  # Mono
        assert raw["codec"] == ""

    def test_rgb_frame_keeps_its_colour_axis(self):
        frame = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)

        result = _map(_frame(frame, "ubyteValue", color_mode=2))

        assert result.value.shape == (2, 4, 3)
        assert result.value.dtype == np.uint8
        assert result.metadata.raw_metadata["color_mode"] == 2  # RGB1
        assert result.metadata.raw_metadata["dimensions"] == [3, 4, 2]

    def test_signed_frame_stays_signed(self):
        frame = np.array([[-3, 4], [5, -6]], dtype=np.int32)

        result = _map(_frame(frame, "intValue"))

        assert result.value.dtype == np.int32
        assert result.value.tolist() == [[-3, 4], [5, -6]]

    def test_dimension_mismatch_names_the_channel_and_the_dims(self):
        pva = _pva()
        value = _frame(np.zeros(6, dtype=np.uint16), "ushortValue")
        value["dimension"] = [pva.PvDimension(4, 0, 4, 1, False)] * 2

        with pytest.raises(ValueError) as excinfo:
            _map(value)

        message = str(excinfo.value)
        assert PVA_ADDRESS in message
        assert "[4, 4]" in message


# ---------------------------------------------------------------------------
# Codec guard
# ---------------------------------------------------------------------------


class TestCodecGuard:
    def test_compressed_frame_is_refused_with_codec_and_dims(self):
        value = _frame(np.zeros((48, 64), dtype=np.uint16), "ushortValue")
        value["codec"] = {"name": "blosc"}

        with pytest.raises(ValueError) as excinfo:
            _map(value)

        message = str(excinfo.value)
        assert "compressed NTNDArray unsupported" in message
        assert "disable ADPva compression" in message
        assert "blosc" in message
        assert "[64, 48]" in message
        assert PVA_ADDRESS in message

    def test_codec_guard_fires_before_any_reshape(self):
        """A compressed payload has nothing to do with the advertised dims.

        Reshaping it either explodes with an unrelated numpy message or — worse —
        succeeds and yields a plausible image full of meaningless statistics.
        The guard must be the thing that raises.
        """
        pva = _pva()
        value = _frame(np.zeros(7, dtype=np.uint8), "ubyteValue")  # compressed blob
        value["codec"] = {"name": "lz4"}
        value["dimension"] = [
            pva.PvDimension(64, 0, 64, 1, False),
            pva.PvDimension(48, 0, 48, 1, False),
        ]

        with pytest.raises(ValueError) as excinfo:
            _map(value)

        message = str(excinfo.value)
        assert "compressed NTNDArray unsupported" in message
        assert "lz4" in message
        assert "reshape" not in message

    @pytest.mark.asyncio
    async def test_codec_guard_reaches_read_channel(self):
        pvaccess = FakePvaccess()
        pvaccess.serve(
            PVA_ADDRESS, ndarray_record(np.zeros(16, dtype=np.uint16), [4, 4], codec="jpeg")
        )

        with pytest.raises(ValueError, match="compressed NTNDArray unsupported"):
            await pva_connector(pvaccess).read_channel(PVA_ADDRESS)


# ---------------------------------------------------------------------------
# read_channel dispatch
# ---------------------------------------------------------------------------


class TestReadChannelDispatch:
    @pytest.mark.asyncio
    async def test_globbed_address_reads_the_whole_structure_over_pva(self):
        pvaccess = FakePvaccess()
        frame = np.full(4, 50000, dtype=np.uint16)
        pvaccess.serve(PVA_ADDRESS, ndarray_record(frame, [2, 2]))

        result = await pva_connector(pvaccess).read_channel(PVA_ADDRESS, timeout=1.5)

        assert [(e["op"], e["provider"], e["request"], e["timeout"]) for e in pvaccess.log] == [
            ("get", "PVA", PVA_READ_REQUEST, 1.5)
        ]
        assert result.value.dtype == np.uint16
        assert result.value.tolist() == [[50000, 50000], [50000, 50000]]

    @pytest.mark.asyncio
    async def test_non_globbed_address_stays_on_channel_access(self):
        pvaccess = FakePvaccess()
        pvaccess.serve(CA_ADDRESS, record(0.5))

        result = await pva_connector(pvaccess).read_channel(CA_ADDRESS)

        assert result.value == 0.5
        assert result.metadata.raw_metadata["provider"] == "ca"
        assert pvaccess.calls(provider="PVA") == []
        assert pvaccess.calls("get", request=CA_READ_REQUEST)[0]["provider"] == "CA"

    @pytest.mark.asyncio
    async def test_default_timeout_is_used_when_none_is_given(self):
        pvaccess = FakePvaccess()
        pvaccess.serve(PVA_ADDRESS, {"value": 1.0})
        connector = pva_connector(pvaccess, timeout=7.0)

        await connector.read_channel(PVA_ADDRESS)

        assert pvaccess.log[0]["timeout"] == 7.0

    @pytest.mark.asyncio
    async def test_both_providers_keep_separate_channels_for_one_address(self):
        """The channel cache is keyed by address AND provider."""
        pvaccess = FakePvaccess()
        pvaccess.serve(PVA_ADDRESS, record(1.0))
        connector = pva_connector(pvaccess)

        await connector.read_channel(PVA_ADDRESS)
        connector._pva_channel_globs = []
        await connector.read_channel(PVA_ADDRESS)

        assert sorted(channel.provider for channel in pvaccess.channels) == ["CA", "PVA"]


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class TestPvaErrorTranslation:
    @pytest.mark.asyncio
    async def test_an_unreachable_channel_becomes_a_connection_error(self):
        """ConnectionError is what the MCP error envelope and invalidation key on."""
        connector = pva_connector(FakePvaccess())

        with pytest.raises(ConnectionError) as excinfo:
            await connector.read_channel(PVA_ADDRESS, timeout=2.0)

        message = str(excinfo.value)
        assert f"PVA channel '{PVA_ADDRESS}'" in message
        assert "timeout after 2.0s" in message
        assert f"Channel {PVA_ADDRESS} timed out." in message  # pvapy's own words kept

    @pytest.mark.asyncio
    async def test_read_without_a_client_is_a_connection_error(self):
        """A PVA-routed address on a connector that never loaded pvapy."""
        connector = pva_connector()
        connector._pvaccess = None

        with pytest.raises(ConnectionError) as excinfo:
            await connector.read_channel(PVA_ADDRESS)

        assert PVA_ADDRESS in str(excinfo.value)

    def test_unexpected_client_errors_are_not_swallowed(self):
        """Only the classified texts are translated; anything else propagates as raised."""
        pvaccess = FakePvaccess()
        error = FakePvaException("Invalid field name: nosuch")

        def refuse(_request):
            raise error

        pvaccess.get_hooks[PVA_ADDRESS] = refuse
        connector = pva_connector(pvaccess)

        with pytest.raises(FakePvaException) as excinfo:
            connector._read_channel_pva(PVA_ADDRESS, 1.0)

        assert excinfo.value is error


class TestClassifyClientError:
    """The one place a pvapy failure is given a meaning — by type, then by text."""

    def test_the_timed_out_text_is_unreachable(self):
        assert _classify_client_error(timed_out("SR:X"), FakePvaException) == _UNREACHABLE

    def test_the_access_security_text_is_a_denial(self):
        assert _classify_client_error(access_denied("SR:X"), FakePvaException) == _ACCESS_DENIED

    @pytest.mark.parametrize(
        "message",
        [
            "Channel SR:X timed out. Retrying",  # anchored at both ends
            "channel SR:X timed out.",  # pvapy capitalizes it
            "Invalid pvRequest",
            "",
        ],
    )
    def test_any_other_text_is_unrecognized(self, message):
        assert _classify_client_error(FakePvaException(message), FakePvaException) is None

    def test_the_text_on_another_exception_type_is_unrecognized(self):
        """A Boost ArgumentError that happens to say "timed out" is not an outage."""
        assert (
            _classify_client_error(RuntimeError("Channel SR:X timed out."), FakePvaException)
            is None
        )

    def test_a_missing_or_non_class_error_type_classifies_nothing(self):
        """A module without ``PvaException`` (or a mock in its place) must not match."""
        assert _classify_client_error(timed_out("SR:X"), None) is None
        assert _classify_client_error(timed_out("SR:X"), object()) is None
