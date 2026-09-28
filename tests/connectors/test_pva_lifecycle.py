"""PVA lifecycle completion: disconnect teardown, metadata and validation.

The PVA read, write-refusal and monitor paths are covered in their own files.
What is left is everything that happens around them:

* ``disconnect()`` tears down both transports, and the order is load-bearing:
  every monitor is stopped (on its own Channel) before the cached read
  Channels are dropped. pvapy Channels have no close of their own — the client
  releases one when its last reference goes — so forgetting the cache is the
  teardown, and doing it is what keeps connector invalidation (a production
  path: a ``ConnectionError`` invalidates and rebuilds the connector) from
  leaking channels on every retry.
* ``get_metadata`` and ``validate_channel`` are the two entry points that want
  a channel's *description*, not its payload. Over PVA those are asked for with
  a field-limited pvRequest, so a camera channel costs a handful of fields
  instead of a full frame per call — and, as a direct consequence, a frame in a
  codec the connector cannot decode does not make a reachable channel report
  itself as invalid.

Conventions follow the sibling PVA files: a fake ``pvaccess`` module rather
than the real one, and assertions on the concrete payload — the request
string, the mapped units, the call order — never merely that a call "didn't
raise".
"""

import numpy as np
import pytest

from tests.connectors._epics_fakes import (
    CA_READ_REQUEST,
    PVA_METADATA_REQUEST,
    FakePvaccess,
    FakePvaException,
    ndarray_record,
    pva_connector,
    record,
)

PVA_ADDRESS = "SR:CAM1:IMAGE"
CA_ADDRESS = "SR:BEAM:CURRENT"


def _metadata_fields() -> dict:
    """A PVA NTScalar carrying the fields a metadata-only get asks for, and a value."""
    fields = record(
        42.0,
        units="mA",
        fmt="F8.3",
        description="Storage ring current",
        severity=1,
        status=3,
        message="HIGH_ALARM",
        seconds=1_700_000_000,
        nanoseconds=500_000_000,
        limit_low=0.0,
        limit_high=500.0,
    )
    fields["display"]["precision"] = 3  # QSRV2 publishes it as a number, too
    return fields


def _connector(fields: dict | None = None, *, address: str = PVA_ADDRESS):
    pvaccess = FakePvaccess()
    if fields is not None:
        pvaccess.serve(address, fields)
    return pva_connector(pvaccess)


# ---------------------------------------------------------------------------
# disconnect(): two transports, one order
# ---------------------------------------------------------------------------


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_monitors_stop_before_the_channels_are_dropped(self):
        """Every subscription is torn down, PVA and CA alike, then the cache goes."""
        pvaccess = FakePvaccess()
        pvaccess.serve(PVA_ADDRESS, _metadata_fields())
        pvaccess.serve(CA_ADDRESS, record(1.0))
        connector = pva_connector(pvaccess)
        await connector.read_channel(PVA_ADDRESS)
        await connector.read_channel(CA_ADDRESS)
        pva_sub = await connector.subscribe(PVA_ADDRESS, lambda v: None)
        ca_sub = await connector.subscribe(CA_ADDRESS, lambda v: None)
        names = {sub_id: connector._subscriptions[sub_id].name for sub_id in (pva_sub, ca_sub)}
        assert len(connector._channels) == 2

        await connector.disconnect()

        teardown = [
            (entry["op"], entry["address"], entry["request"]) for entry in pvaccess.log[-4:]
        ]
        assert teardown == [
            ("stopMonitor", PVA_ADDRESS, None),
            ("unsubscribe", PVA_ADDRESS, names[pva_sub]),
            ("stopMonitor", CA_ADDRESS, None),
            ("unsubscribe", CA_ADDRESS, names[ca_sub]),
        ]
        assert connector._subscriptions == {}
        assert connector._channels == {}
        assert connector._connected is False

    @pytest.mark.asyncio
    async def test_second_disconnect_is_a_no_op(self):
        connector = _connector(_metadata_fields())
        await connector.subscribe(PVA_ADDRESS, lambda v: None)
        await connector.disconnect()
        logged = len(connector._pvaccess.log)

        await connector.disconnect()

        assert len(connector._pvaccess.log) == logged

    @pytest.mark.asyncio
    async def test_a_failing_monitor_stop_is_swallowed_and_teardown_continues(self):
        """Teardown is best effort: one dead handle cannot strand the rest."""
        connector = _connector(_metadata_fields())
        await connector.read_channel(PVA_ADDRESS)
        sub_id = await connector.subscribe(PVA_ADDRESS, lambda v: None)
        channel = connector._subscriptions[sub_id].channel

        def dead():
            raise FakePvaException("monitor already gone")

        channel.stopMonitor = dead

        await connector.disconnect()

        assert channel.subscribers == {}  # the subscriber was still removed
        assert connector._channels == {}
        assert connector._connected is False

    @pytest.mark.asyncio
    async def test_a_read_after_disconnect_opens_a_fresh_channel(self):
        """Nothing cached survives: a rebuilt connection starts from the wire."""
        connector = _connector(_metadata_fields())
        await connector.read_channel(PVA_ADDRESS)
        await connector.disconnect()

        await connector.read_channel(PVA_ADDRESS)

        assert len(connector._pvaccess.channels_for(PVA_ADDRESS)) == 2


# ---------------------------------------------------------------------------
# get_metadata()
# ---------------------------------------------------------------------------


class TestGetMetadataOverPva:
    @pytest.mark.asyncio
    async def test_display_and_alarm_fields_are_mapped(self):
        connector = _connector(_metadata_fields())

        metadata = await connector.get_metadata(PVA_ADDRESS)

        assert metadata.units == "mA"
        assert metadata.precision == 3
        assert metadata.description == "Storage ring current"
        assert metadata.display_low == 0.0
        assert metadata.display_high == 500.0
        assert metadata.alarm_status == "HIGH_ALARM"
        assert metadata.alarm_severity == 1
        assert metadata.raw_metadata["severity"] == 1
        assert metadata.raw_metadata["status"] == 3
        assert metadata.raw_metadata["provider"] == "pva"

    @pytest.mark.asyncio
    async def test_timestamp_comes_from_the_nt_timestamp_field(self):
        connector = _connector(_metadata_fields())

        metadata = await connector.get_metadata(PVA_ADDRESS)

        assert metadata.timestamp is not None
        assert metadata.timestamp.timestamp() == pytest.approx(1_700_000_000.5)

    @pytest.mark.asyncio
    async def test_the_get_is_field_limited_and_uses_the_connector_timeout(self):
        """Asking for the whole structure would pull a full frame per lookup."""
        connector = _connector(_metadata_fields())

        await connector.get_metadata(PVA_ADDRESS)

        assert [
            (entry["op"], entry["provider"], entry["request"], entry["timeout"])
            for entry in connector._pvaccess.log
        ] == [("get", "PVA", PVA_METADATA_REQUEST, 3.0)]

    @pytest.mark.asyncio
    async def test_ndarray_metadata_never_touches_the_payload(self):
        """A camera channel answers out of its header: no dtype, dims or shape."""
        frame = ndarray_record(np.zeros(12, dtype=np.uint16), [4, 3], units="counts")
        connector = _connector(frame)

        metadata = await connector.get_metadata(PVA_ADDRESS)

        assert metadata.units == "counts"
        assert "dimensions" not in metadata.raw_metadata
        assert "shape" not in metadata.raw_metadata
        assert "dtype" not in metadata.raw_metadata

    @pytest.mark.asyncio
    async def test_missing_display_structure_degrades_to_empty_metadata(self):
        """Servers that publish no display block still produce a ChannelMetadata."""
        connector = _connector({"value": 1.0})

        metadata = await connector.get_metadata(PVA_ADDRESS)

        assert metadata.units == ""
        assert metadata.precision is None
        assert metadata.alarm_status is None
        assert metadata.alarm_severity is None
        assert metadata.timestamp is not None  # falls back to now()

    @pytest.mark.asyncio
    async def test_ca_address_on_a_pva_capable_connector_reads_over_ca(self):
        connector = _connector(record(42.0, units="mA", fmt="F8.2"), address=CA_ADDRESS)

        metadata = await connector.get_metadata(CA_ADDRESS)

        assert connector._pvaccess.calls(provider="PVA") == []  # PVA untouched
        assert len(connector._pvaccess.calls("get", request=CA_READ_REQUEST)) == 1
        assert metadata.units == "mA"
        assert metadata.precision == 2

    @pytest.mark.asyncio
    async def test_without_a_client_it_is_a_connection_error(self):
        connector = _connector()
        connector._pvaccess = None

        with pytest.raises(ConnectionError, match="not connected"):
            await connector.get_metadata(PVA_ADDRESS)

    @pytest.mark.asyncio
    async def test_an_unreachable_channel_is_a_connection_error(self):
        connector = _connector()

        with pytest.raises(ConnectionError, match=f"PVA channel '{PVA_ADDRESS}'"):
            await connector.get_metadata(PVA_ADDRESS)


# ---------------------------------------------------------------------------
# validate_channel()
# ---------------------------------------------------------------------------


class TestValidateChannelOverPva:
    @pytest.mark.asyncio
    async def test_reachable_channel_validates_true_with_the_configured_timeout(self):
        """The probe is bounded by the connector's own ``timeout``, not a literal."""
        connector = _connector(_metadata_fields())

        assert await connector.validate_channel(PVA_ADDRESS) is True
        (probe,) = connector._pvaccess.log
        assert (probe["request"], probe["timeout"]) == (PVA_METADATA_REQUEST, 3.0)

    @pytest.mark.asyncio
    async def test_unreachable_validates_false(self):
        assert await _connector().validate_channel(PVA_ADDRESS) is False

    @pytest.mark.asyncio
    async def test_missing_client_validates_false(self):
        connector = _connector()
        connector._pvaccess = None

        assert await connector.validate_channel(PVA_ADDRESS) is False

    @pytest.mark.asyncio
    async def test_undecodable_frame_does_not_make_a_reachable_channel_invalid(self):
        """Reachability is a property of the channel, not of its payload codec.

        A compressed NTNDArray makes the *read* path raise; validation asks only
        for the metadata fields, so the channel reports itself as reachable.
        """
        frame = ndarray_record(np.zeros(7, dtype=np.uint8), [64, 48], codec="blosc")
        connector = _connector(frame)

        with pytest.raises(ValueError, match="compressed NTNDArray unsupported"):
            await connector.read_channel(PVA_ADDRESS)
        assert await connector.validate_channel(PVA_ADDRESS) is True

    @pytest.mark.asyncio
    async def test_ca_address_validates_over_channel_access(self):
        connector = _connector(record(1.0), address=CA_ADDRESS)

        assert await connector.validate_channel(CA_ADDRESS) is True
        assert connector._pvaccess.calls(provider="PVA") == []
        assert connector._pvaccess.calls("get", request=CA_READ_REQUEST)[0]["provider"] == "CA"

    @pytest.mark.asyncio
    async def test_unreachable_ca_address_validates_false(self):
        assert await _connector().validate_channel(CA_ADDRESS) is False
