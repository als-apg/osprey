"""PVA reads: routing into the p4p client and mapping NT structures to ChannelValue.

Task 1.1 decided WHICH transport an address uses; this file covers what happens
once PVAccess is chosen — the ``_read_channel_pva`` thread call, the normative-
type mapping, and the error translation that keeps a PVA outage looking exactly
like a CA one to everything upstream.

Two kinds of fake are used deliberately:

* the **p4p module** is always faked (a client context object with a ``get``
  and the exception classes the connector catches), so the routing, timeout and
  disconnect assertions run on every machine;
* the **values** those fakes return are REAL ``p4p.Value`` structures built with
  ``p4p.nt`` wherever the assertion is about mapping. A hand-rolled dict would
  let the mapping agree with a fiction of the NT layout; an authentic NTNDArray
  is what proves the union member, the dimension order and the carried dtype are
  read the way an areaDetector IOC actually sends them. Those tests are
  ``importorskip``-guarded (p4p has no wheel on every platform).
"""

import types
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from osprey.connectors.control_system.epics_connector import EPICSConnector

PVA_GLOB = "SR:CAM*:IMAGE"
PVA_ADDRESS = "SR:CAM1:IMAGE"
CA_ADDRESS = "SR:BEAM:CURRENT"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeDisconnected(RuntimeError):
    """Stands in for p4p's Disconnected — a RuntimeError, NOT a ConnectionError."""


class FakeRemoteError(RuntimeError):
    """Stands in for p4p's RemoteError."""


class FakeCancelled(RuntimeError):
    """Stands in for p4p's Cancelled."""


def _fake_p4p_module() -> types.ModuleType:
    """A ``p4p`` module exposing exactly what the read path looks up on it."""
    thread_mod = types.ModuleType("p4p.client.thread")
    thread_mod.Context = MagicMock(name="Context")
    thread_mod.Disconnected = FakeDisconnected
    thread_mod.RemoteError = FakeRemoteError
    thread_mod.Cancelled = FakeCancelled
    thread_mod.TimeoutError = TimeoutError
    client_mod = types.ModuleType("p4p.client")
    client_mod.thread = thread_mod
    p4p_mod = types.ModuleType("p4p")
    p4p_mod.client = client_mod
    return p4p_mod


class FakeValue:
    """Minimal p4p ``Value`` stand-in: a struct id plus (possibly nested) fields."""

    def __init__(self, type_id: str, fields: dict):
        self._type_id = type_id
        self._fields = {
            key: FakeValue("", value) if isinstance(value, dict) else value
            for key, value in fields.items()
        }

    def getID(self) -> str:  # noqa: N802 - p4p's spelling
        return self._type_id

    def get(self, name, default=None):
        return self._fields.get(name, default)

    def __contains__(self, name) -> bool:
        return name in self._fields


def _pva_connector(get=None, globs=(PVA_GLOB,), context=True) -> EPICSConnector:
    """A connector wired for PVA reads without touching connect()."""
    connector = EPICSConnector()
    connector._pva_channel_globs = list(globs)
    connector._p4p = _fake_p4p_module()
    connector._pva_context = SimpleNamespace(get=get) if context else None
    connector._timeout = 3.0
    connector._epics = MagicMock()
    return connector


def _serving(value):
    """A context ``get`` that records its call and returns ``value``."""
    calls = []

    def get(name, timeout=None, **kwargs):
        calls.append({"name": name, "timeout": timeout, **kwargs})
        return value

    get.calls = calls
    return get


def _nt():
    """The real p4p normative-type helpers, or skip."""
    return pytest.importorskip("p4p.nt")


# ---------------------------------------------------------------------------
# NTScalar / NTScalarArray
# ---------------------------------------------------------------------------


class TestNtScalarMapping:
    def test_float_scalar_carries_value_units_alarm_and_timestamp(self):
        nt = _nt()
        value = nt.NTScalar("d", display=True).wrap(
            {
                "value": 1.5,
                "alarm": {"severity": 1, "status": 3, "message": "HIGH"},
                "timeStamp": {"secondsPastEpoch": 1_700_000_000, "nanoseconds": 250_000_000},
                "display": {
                    "units": "mA",
                    "limitLow": 0.0,
                    "limitHigh": 10.0,
                    "description": "beam current",
                },
            }
        )

        result = _pva_connector()._pva_channel_value(CA_ADDRESS, value)

        assert result.value == 1.5
        assert result.metadata.units == "mA"
        assert result.metadata.description == "beam current"
        assert result.metadata.display_low == 0.0
        assert result.metadata.display_high == 10.0
        assert result.metadata.alarm_status == "HIGH"
        assert result.metadata.raw_metadata["nt_id"] == "epics:nt/NTScalar:1.0"
        assert result.metadata.raw_metadata["severity"] == 1
        assert result.metadata.raw_metadata["status"] == 3
        assert result.timestamp.timestamp() == pytest.approx(1_700_000_000.25)
        assert result.metadata.timestamp == result.timestamp

    def test_integer_scalar_stays_an_int(self):
        nt = _nt()
        value = nt.NTScalar("i").wrap(42)

        result = _pva_connector()._pva_channel_value(CA_ADDRESS, value)

        assert result.value == 42
        assert isinstance(result.value, int)

    def test_string_scalar_maps_through_unchanged(self):
        nt = _nt()
        value = nt.NTScalar("s").wrap("Injecting")

        result = _pva_connector()._pva_channel_value(CA_ADDRESS, value)

        assert result.value == "Injecting"
        assert result.metadata.raw_metadata["dtype"] == "str"

    def test_scalar_array_is_a_numpy_array_with_its_dtype(self):
        nt = _nt()
        value = nt.NTScalar("ad").wrap(np.array([1.0, 2.0, 3.0]))

        result = _pva_connector()._pva_channel_value(CA_ADDRESS, value)

        assert isinstance(result.value, np.ndarray)
        assert result.value.tolist() == [1.0, 2.0, 3.0]
        assert result.metadata.raw_metadata["dtype"] == "float64"
        assert result.metadata.raw_metadata["nt_id"] == "epics:nt/NTScalarArray:1.0"

    def test_missing_timestamp_falls_back_to_now(self):
        """An unset timeStamp must not read as 1970 — the CA path behaves the same."""
        nt = _nt()
        value = nt.NTScalar("d").wrap(0.25)

        before = datetime.now().timestamp()
        result = _pva_connector()._pva_channel_value(CA_ADDRESS, value)

        assert result.timestamp.timestamp() >= before

    def test_precision_is_read_when_the_server_publishes_it(self):
        """QSRV2 adds display.precision; p4p's own NTScalar type does not carry it."""
        value = FakeValue(
            "epics:nt/NTScalar:1.0",
            {
                "value": 3.25,
                "display": {"units": "mm", "precision": 4},
                "alarm": {"severity": 0, "status": 0, "message": ""},
            },
        )

        result = _pva_connector()._pva_channel_value(CA_ADDRESS, value)

        assert result.value == 3.25
        assert result.metadata.precision == 4
        assert result.metadata.units == "mm"

    def test_absent_display_leaves_metadata_empty_rather_than_failing(self):
        value = FakeValue("epics:nt/NTScalar:1.0", {"value": 7.0})

        result = _pva_connector()._pva_channel_value(CA_ADDRESS, value)

        assert result.value == 7.0
        assert result.metadata.units == ""
        assert result.metadata.precision is None
        assert result.metadata.alarm_status is None


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
        nt = _nt()
        value = nt.NTEnum().wrap({"index": 2, "choices": ["Off", "Standby", "On"]})

        result = _pva_connector()._pva_channel_value("SR:RF:STATE", value)

        assert result.value == 2
        assert not isinstance(result.value, str)
        assert result.metadata.enum_label == "On"
        assert result.metadata.enum_labels == ["Off", "Standby", "On"]
        assert result.metadata.raw_metadata["enum_index"] == 2
        assert result.metadata.raw_metadata["enum_choices"] == ["Off", "Standby", "On"]
        assert result.metadata.raw_metadata["nt_id"] == "epics:nt/NTEnum:1.0"

    def test_enum_index_zero_is_a_choice_not_a_falsy_miss(self):
        nt = _nt()
        value = nt.NTEnum().wrap({"index": 0, "choices": ["Off", "On"]})

        result = _pva_connector()._pva_channel_value("SR:RF:STATE", value)

        assert result.value == 0
        assert result.metadata.enum_label == "Off"
        assert result.metadata.raw_metadata["enum_index"] == 0

    def test_enum_without_choices_still_reports_the_index(self):
        """Servers send the choices list only on change; a later read may omit it."""
        nt = _nt()
        value = nt.NTEnum().wrap({"index": 1, "choices": []})

        result = _pva_connector()._pva_channel_value("SR:RF:STATE", value)

        assert result.value == 1
        assert result.metadata.enum_label is None
        assert result.metadata.enum_labels is None
        assert result.metadata.raw_metadata["enum_choices"] == []

    def test_a_scalar_channel_reports_no_enum_fields_at_all(self):
        """The two fields are how a consumer tells an enum from anything else."""
        value = FakeValue("epics:nt/NTScalar:1.0", {"value": 7.0})

        result = _pva_connector()._pva_channel_value(CA_ADDRESS, value)

        assert result.metadata.enum_label is None
        assert result.metadata.enum_labels is None


# ---------------------------------------------------------------------------
# NTNDArray
# ---------------------------------------------------------------------------


class TestNtNdArrayMapping:
    def test_uint16_frame_keeps_its_unsigned_dtype(self):
        """The carried dtype is preserved: bright pixels must not read as negative."""
        nt = _nt()
        frame = np.array([[0, 40000, 65535], [12000, 33000, 60000]], dtype=np.uint16)
        value = nt.NTNDArray().wrap(frame)

        result = _pva_connector()._pva_channel_value(PVA_ADDRESS, value)

        assert isinstance(result.value, np.ndarray)
        assert result.value.dtype == np.uint16
        assert result.value.shape == (2, 3)
        assert result.value.tolist() == frame.tolist()
        assert result.value.min() >= 0

    def test_dimensions_are_reversed_into_numpy_order(self):
        """NT dimensions run innermost-first (width, height); numpy shape is reversed."""
        nt = _nt()
        frame = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64)
        value = nt.NTNDArray().wrap(frame)

        result = _pva_connector()._pva_channel_value(PVA_ADDRESS, value)

        assert result.metadata.raw_metadata["dimensions"] == [64, 48]
        assert result.value.shape == (48, 64)
        assert result.metadata.raw_metadata["shape"] == [48, 64]

    def test_raw_metadata_records_dtype_colormode_and_codec(self):
        nt = _nt()
        value = nt.NTNDArray().wrap(np.zeros((4, 5), dtype=np.uint16))

        raw = _pva_connector()._pva_channel_value(PVA_ADDRESS, value).metadata.raw_metadata

        assert raw["nt_id"] == "epics:nt/NTNDArray:1.0"
        assert raw["dtype"] == "uint16"
        assert raw["color_mode"] == 0  # Mono
        assert raw["codec"] == ""

    def test_rgb_frame_keeps_its_colour_axis(self):
        nt = _nt()
        frame = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
        value = nt.NTNDArray().wrap(frame)

        result = _pva_connector()._pva_channel_value(PVA_ADDRESS, value)

        assert result.value.shape == (2, 4, 3)
        assert result.value.dtype == np.uint8
        assert result.metadata.raw_metadata["color_mode"] == 2  # RGB1
        assert result.metadata.raw_metadata["dimensions"] == [3, 4, 2]

    def test_signed_frame_stays_signed(self):
        nt = _nt()
        frame = np.array([[-3, 4], [5, -6]], dtype=np.int32)
        value = nt.NTNDArray().wrap(frame)

        result = _pva_connector()._pva_channel_value(PVA_ADDRESS, value)

        assert result.value.dtype == np.int32
        assert result.value.tolist() == [[-3, 4], [5, -6]]

    def test_dimension_mismatch_names_the_channel_and_the_dims(self):
        nt = _nt()
        value = nt.NTNDArray().wrap(np.zeros(6, dtype=np.uint16))
        value["dimension"] = [{"size": 4}, {"size": 4}]

        with pytest.raises(ValueError) as excinfo:
            _pva_connector()._pva_channel_value(PVA_ADDRESS, value)

        message = str(excinfo.value)
        assert PVA_ADDRESS in message
        assert "[4, 4]" in message


# ---------------------------------------------------------------------------
# Codec guard
# ---------------------------------------------------------------------------


class TestCodecGuard:
    def test_compressed_frame_is_refused_with_codec_and_dims(self):
        nt = _nt()
        value = nt.NTNDArray().wrap(np.zeros((48, 64), dtype=np.uint16))
        value["codec.name"] = "blosc"

        with pytest.raises(ValueError) as excinfo:
            _pva_connector()._pva_channel_value(PVA_ADDRESS, value)

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
        nt = _nt()
        value = nt.NTNDArray().wrap(np.zeros(7, dtype=np.uint8))  # compressed blob
        value["codec.name"] = "lz4"
        value["dimension"] = [{"size": 64}, {"size": 48}]

        with pytest.raises(ValueError) as excinfo:
            _pva_connector()._pva_channel_value(PVA_ADDRESS, value)

        message = str(excinfo.value)
        assert "compressed NTNDArray unsupported" in message
        assert "lz4" in message
        assert "reshape" not in message

    @pytest.mark.asyncio
    async def test_codec_guard_reaches_read_channel(self):
        nt = _nt()
        value = nt.NTNDArray().wrap(np.zeros((4, 4), dtype=np.uint16))
        value["codec.name"] = "jpeg"
        connector = _pva_connector(get=_serving(value))

        with pytest.raises(ValueError, match="compressed NTNDArray unsupported"):
            await connector.read_channel(PVA_ADDRESS)


# ---------------------------------------------------------------------------
# read_channel dispatch
# ---------------------------------------------------------------------------


class TestReadChannelDispatch:
    @pytest.mark.asyncio
    async def test_globbed_address_reads_through_the_pva_context(self, monkeypatch):
        nt = _nt()
        value = nt.NTNDArray().wrap(np.full((2, 2), 50000, dtype=np.uint16))
        get = _serving(value)
        connector = _pva_connector(get=get)
        monkeypatch.setattr(
            connector,
            "_read_channel_sync",
            lambda *a, **k: pytest.fail("PVA address must not touch Channel Access"),
        )

        result = await connector.read_channel(PVA_ADDRESS, timeout=1.5)

        assert get.calls == [{"name": PVA_ADDRESS, "timeout": 1.5}]
        assert result.value.dtype == np.uint16
        assert result.value.tolist() == [[50000, 50000], [50000, 50000]]

    @pytest.mark.asyncio
    async def test_non_globbed_address_stays_on_channel_access(self, monkeypatch):
        get = _serving(FakeValue("epics:nt/NTScalar:1.0", {"value": 0.0}))
        connector = _pva_connector(get=get)
        sentinel = object()
        monkeypatch.setattr(connector, "_read_channel_sync", lambda *a, **k: sentinel)

        result = await connector.read_channel(CA_ADDRESS)

        assert result is sentinel
        assert get.calls == []

    @pytest.mark.asyncio
    async def test_default_timeout_is_used_when_none_is_given(self):
        get = _serving(FakeValue("epics:nt/NTScalar:1.0", {"value": 1.0}))
        connector = _pva_connector(get=get)
        connector._timeout = 7.0

        await connector.read_channel(PVA_ADDRESS)

        assert get.calls[0]["timeout"] == 7.0

    @pytest.mark.asyncio
    async def test_unwrapped_result_is_mapped_through_its_raw_value(self):
        """A default p4p Context returns ntfloat/ntndarray wrappers, not Values.

        The mapping reads ``.raw`` so it sees the full NT structure (codec,
        dimensions, alarm) rather than the augmented python object.
        """
        nt = _nt()
        raw_value = nt.NTScalar("d", display=True).wrap({"value": 2.5, "display": {"units": "kV"}})
        unwrapped = SimpleNamespace(raw=raw_value)
        connector = _pva_connector(get=_serving(unwrapped))

        result = await connector.read_channel(PVA_ADDRESS)

        assert result.value == 2.5
        assert result.metadata.units == "kV"


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------


class TestPvaErrorTranslation:
    @pytest.mark.asyncio
    async def test_disconnected_becomes_a_connection_error(self):
        """p4p's Disconnected is a RuntimeError — without this it escapes unclassified.

        ConnectionError is what the MCP error envelope and the connector
        invalidation logic key on.
        """

        def get(name, timeout=None):
            raise FakeDisconnected("channel disconnected")

        connector = _pva_connector(get=get)

        with pytest.raises(ConnectionError) as excinfo:
            await connector.read_channel(PVA_ADDRESS)

        assert PVA_ADDRESS in str(excinfo.value)
        assert "channel disconnected" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_remote_error_becomes_a_connection_error(self):
        def get(name, timeout=None):
            raise FakeRemoteError("server said no")

        connector = _pva_connector(get=get)

        with pytest.raises(ConnectionError):
            await connector.read_channel(PVA_ADDRESS)

    @pytest.mark.asyncio
    async def test_timeout_is_reported_as_timeout_error_with_the_budget(self):
        def get(name, timeout=None):
            raise TimeoutError()

        connector = _pva_connector(get=get)

        with pytest.raises(TimeoutError) as excinfo:
            await connector.read_channel(PVA_ADDRESS, timeout=2.0)

        message = str(excinfo.value)
        assert PVA_ADDRESS in message
        assert "2.0" in message
        assert not isinstance(excinfo.value, ConnectionError)

    @pytest.mark.asyncio
    async def test_read_without_a_context_is_a_connection_error(self):
        """A PVA-routed address on a connector that never built a context."""
        connector = _pva_connector(context=False)

        with pytest.raises(ConnectionError) as excinfo:
            await connector.read_channel(PVA_ADDRESS)

        assert PVA_ADDRESS in str(excinfo.value)

    def test_unexpected_p4p_errors_are_not_swallowed(self):
        """Only the reachability exceptions are translated; anything else propagates."""

        def get(name, timeout=None):
            raise KeyError("no such field")

        connector = _pva_connector(get=get)

        with pytest.raises(KeyError):
            connector._read_channel_pva(PVA_ADDRESS, 1.0)

    def test_error_types_survive_a_p4p_without_those_names(self):
        """A p4p that renamed an exception must degrade, not TypeError in `except`."""
        connector = _pva_connector()
        del connector._p4p.client.thread.Cancelled

        assert FakeDisconnected in connector._pva_connection_error_types()
        assert all(
            isinstance(t, type) and issubclass(t, BaseException)
            for t in connector._pva_connection_error_types()
        )
