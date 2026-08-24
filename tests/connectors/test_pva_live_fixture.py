"""What a client sees when it reads a real PVAccess server over the wire.

Everything here runs in **one process**: a real ``p4p`` server hosting real
normative-type structures, the real EPICS connector routing addresses onto its
PVA client, and — for the batch and artifact assertions — the real
``channel_read`` tool body deciding what the agent gets to see. No fake p4p
module, no hand-rolled ``Value``. ``tests/connectors/test_pva_read_mapping.py``
already proves the mapping in process against an injected client; this file is
the check that the wire agrees with it, and it is where success criteria 1-4 of
the PVA feature are asserted concretely.

Three things about the venue are worth stating plainly, because a green run
means nothing without them.

**p4p is the venue's hard requirement.** The module is ``importorskip``-gated,
and a skipped suite proves nothing at all — which is why
``tests/connectors/test_pva_venue_guard.py`` fails the lane on any platform
where p4p must be importable.

**The server is isolated, and the client is pinned to it.** ``Server(...,
isolate=True)`` binds to 127.0.0.1 on a randomly chosen port and does not
beacon off the loopback interface, so concurrent runs of this file — and the
operator's own IOCs — can never answer for each other. The connector is pointed
at that port through ``pva_gateway.use_name_server``, which sets
``EPICS_PVA_NAME_SERVERS``: the client then opens a TCP connection straight to
the server instead of UDP-searching for it, so no search packet leaves the
host. The suite-wide ``restore_environ`` fixture puts every ``EPICS_PVA_*``
variable the connector wrote back after each test.

**The Channel Access half of the mixed batch is unreachable on purpose.** Its
address matches no ``pva_channels`` glob, so it takes the CA path, where no
server answers it — that is the point, and the connector's timeout is kept
short so the failure costs a second rather than a suite.
"""

import gc
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

pytest.importorskip("p4p", reason="p4p is required to serve a live PVAccess fixture")

from p4p.nt import NTEnum, NTNDArray, NTScalar  # noqa: E402
from p4p.server import Server  # noqa: E402
from p4p.server.thread import SharedPV  # noqa: E402

from osprey.connectors.control_system.epics_connector import EPICSConnector  # noqa: E402
from tests.mcp_server.conftest import extract_response_dict, get_tool_fn  # noqa: E402

# The served namespace, kept to one prefix so a single glob routes all of it
# over PVAccess and nothing else in the suite can collide with it.
PREFIX = "OSPREY:TEST:"
PVA_GLOB = f"{PREFIX}*"

SCALAR = f"{PREFIX}CURRENT"
ENUM = f"{PREFIX}SHUTTER"
IMAGE = f"{PREFIX}CAM:IMAGE"
CODEC = f"{PREFIX}CAM:COMPRESSED"
UNSERVED = f"{PREFIX}NOT:SERVED"

# Routed over Channel Access (it matches no PVA glob) and answered by nobody:
# the failure half of the mixed batch.
CA_UNREACHABLE = "OSPREY:NOSERVER:CURRENT"

SCALAR_VALUE = 12.75
SCALAR_UNITS = "mA"
SCALAR_PRECISION = 3
SCALAR_DESCRIPTION = "Stored beam current"
SCALAR_LIMITS = (0.0, 500.0)

ENUM_CHOICES = ["Closed", "Open", "Moving"]
ENUM_INDEX = 1

FRAME_HEIGHT = 48
FRAME_WIDTH = 64
CODEC_NAME = "blosc"

# Generous for a loopback read, short enough that a wedged server fails the
# test instead of the suite.
READ_TIMEOUT_S = 5.0
# The mixed batch pays this once for the address nobody serves.
BATCH_TIMEOUT_S = 2.0

# Everything ``connect()`` writes when a PVA gateway is configured, restored
# when the module's connectors are torn down.
_PVA_ENV_KEYS = ("EPICS_PVA_NAME_SERVERS", "EPICS_PVA_ADDR_LIST", "EPICS_PVA_AUTO_ADDR_LIST")


def _gaussian_spot(height: int = FRAME_HEIGHT, width: int = FRAME_WIDTH) -> np.ndarray:
    """A 48x64 uint16 frame with a Gaussian spot — a synthetic camera image."""
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    radius2 = (rows - height / 2) ** 2 + (cols - width / 2) ** 2
    return (50000 * np.exp(-radius2 / (2 * 8.0**2))).astype(np.uint16)


FRAME = _gaussian_spot()


def _scalar_pv() -> SharedPV:
    """An NTScalar carrying the display fields a metadata-only get asks for.

    ``form=True`` is what puts ``precision`` in the display structure — p4p's
    plain ``display=True`` serves ``format`` instead, and the connector reads
    ``display.precision``.
    """
    nt = NTScalar("d", display=True, form=True)
    return SharedPV(
        nt=nt,
        initial=nt.wrap(
            {
                "value": SCALAR_VALUE,
                "display": {
                    "units": SCALAR_UNITS,
                    "precision": SCALAR_PRECISION,
                    "description": SCALAR_DESCRIPTION,
                    "limitLow": SCALAR_LIMITS[0],
                    "limitHigh": SCALAR_LIMITS[1],
                },
            }
        ),
    )


def _codec_pv() -> SharedPV:
    """An NTNDArray tagged with a codec, served as a raw ``Value``.

    p4p's own ``NTNDArray`` wrapping ignores ``codec`` entirely, so a PV opened
    with ``nt=NTNDArray()`` would drop the tag on every post and serve a frame
    that looks uncompressed. Opening the PV on an already-built ``Value`` keeps
    the tag on the wire, which is what an ADPva IOC with compression enabled
    actually sends.
    """
    value = NTNDArray().wrap(FRAME)
    value["codec.name"] = CODEC_NAME
    return SharedPV(initial=value)


@pytest.fixture(scope="module")
def pva_server():
    """A localhost-only PVAccess server on a random port, serving four channels.

    ``isolate=True`` picks the port and the interface; ``conf()`` reports which,
    so the connector can be pointed straight at it. Stopped unconditionally, so
    a failing assertion never leaves a listening socket behind.
    """
    server = Server(
        providers=[
            {
                SCALAR: _scalar_pv(),
                ENUM: SharedPV(nt=NTEnum(), initial={"index": ENUM_INDEX, "choices": ENUM_CHOICES}),
                IMAGE: SharedPV(nt=NTNDArray(), initial=FRAME),
                CODEC: _codec_pv(),
            }
        ],
        isolate=True,
    )
    try:
        yield server.conf()
    finally:
        server.stop()


@pytest.fixture(scope="module")
def connector_factory(pva_server):
    """Builds connectors pointed at the live server, and tears every one down.

    Module-scoped on purpose. Every ``connect()`` builds a p4p client context,
    and a context binds UDP sockets of its own even when it is reaching the
    server by name server; building and tearing one down per test was observed
    to fail intermittently with "Unable to bind random UDP6 port" in a full
    directory run. Two contexts for the module is both faster and steadier, and
    a connector carries no per-test state — reads do not mutate it.

    Driven with ``asyncio.run`` rather than as an async fixture so the setup and
    the teardown do not need a module-scoped event loop of their own; the
    connector's own reads are plain ``asyncio.to_thread`` calls that work in
    whichever loop a test brings.

    The teardown is not bookkeeping: the PVA client context owns worker threads,
    and the CA half of the mixed batch leaves a pyepics PV that must be
    disconnected deterministically rather than collected at some later moment.
    The ``EPICS_PVA_*`` variables ``connect()`` writes are put back at the same
    time, so nothing downstream of this module inherits a pinned name server.
    """
    import asyncio
    import os

    saved_env = {key: os.environ.get(key) for key in _PVA_ENV_KEYS}
    built: list[EPICSConnector] = []

    def build(timeout: float = READ_TIMEOUT_S) -> EPICSConnector:
        connector = EPICSConnector()
        asyncio.run(
            connector.connect(
                {
                    "timeout": timeout,
                    "pva_channels": [PVA_GLOB],
                    "pva_gateway": {
                        "address": "127.0.0.1",
                        # The TCP port the isolated server listens on. Reached by
                        # name-server lookup, so the client never broadcasts.
                        "port": int(pva_server["EPICS_PVAS_SERVER_PORT"]),
                        "use_name_server": True,
                    },
                }
            )
        )
        built.append(connector)
        return connector

    try:
        yield build
    finally:
        for connector in built:
            asyncio.run(connector.disconnect())
        gc.collect()
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(scope="module")
def connector(connector_factory):
    """One connected connector, shared by every read this module makes."""
    return connector_factory()


@pytest.fixture(scope="module")
def batch_connector(connector_factory):
    """A second connector on a short timeout, for the address nobody answers."""
    return connector_factory(timeout=BATCH_TIMEOUT_S)


# ---------------------------------------------------------------------------
# Criterion 1 (reads) and criterion 2 (frames): the four served channels
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLiveReads:
    """Every normative type the connector claims to map, read off the wire."""

    async def test_ntscalar_returns_its_value(self, connector):
        """A PVA-only read answers with the number, and does not time out."""
        reading = await connector.read_channel(SCALAR)

        assert reading.value == pytest.approx(SCALAR_VALUE)
        assert reading.metadata.raw_metadata["nt_id"] == "epics:nt/NTScalar:1.0"
        assert reading.metadata.units == SCALAR_UNITS
        assert reading.metadata.precision == SCALAR_PRECISION

    async def test_ntenum_reads_as_the_index_with_its_label(self, connector):
        """An operator reading a state channel gets 1 *and* "Open"."""
        reading = await connector.read_channel(ENUM)

        assert reading.value == ENUM_INDEX
        assert not isinstance(reading.value, str)
        assert reading.metadata.enum_label == ENUM_CHOICES[ENUM_INDEX]
        assert reading.metadata.enum_labels == ENUM_CHOICES
        raw = reading.metadata.raw_metadata
        assert raw["nt_id"] == "epics:nt/NTEnum:1.0"
        assert raw["enum_index"] == ENUM_INDEX
        assert raw["enum_choices"] == ENUM_CHOICES

    async def test_ntndarray_returns_the_served_frame(self, connector):
        """The frame arrives unsigned, reshaped rows-by-columns, pixel for pixel.

        The shape assertion is the row/column order: the server sent dimensions
        ``[64, 48]`` (innermost first), and a connector that reshaped by that
        order rather than reversing it would hand back a (64, 48) array of the
        same 3072 pixels — a plausible image with its rows and columns swapped.
        """
        reading = await connector.read_channel(IMAGE)

        assert isinstance(reading.value, np.ndarray)
        assert reading.value.dtype == np.uint16
        assert reading.value.shape == (FRAME_HEIGHT, FRAME_WIDTH)
        np.testing.assert_array_equal(reading.value, FRAME)

    async def test_ntndarray_raw_metadata_reports_the_wire_layout(self, connector):
        """NT dimensions stay innermost-first; the numpy shape is the reverse."""
        reading = await connector.read_channel(IMAGE)

        raw = reading.metadata.raw_metadata
        assert raw["nt_id"] == "epics:nt/NTNDArray:1.0"
        assert raw["dtype"] == "uint16"
        assert raw["dimensions"] == [FRAME_WIDTH, FRAME_HEIGHT]
        assert raw["shape"] == [FRAME_HEIGHT, FRAME_WIDTH]
        assert raw["codec"] == ""

    async def test_compressed_frame_is_refused_with_the_remedy(self, connector):
        """A codec-tagged frame raises rather than reshaping a compressed blob."""
        with pytest.raises(ValueError) as excinfo:
            await connector.read_channel(CODEC)

        message = str(excinfo.value)
        assert "compressed NTNDArray unsupported" in message
        assert "disable ADPva compression" in message
        assert CODEC_NAME in message
        assert CODEC in message


# ---------------------------------------------------------------------------
# Metadata and reachability, against the same live server
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetadataAndValidation:
    """What the connector learns without reading a payload."""

    async def test_get_metadata_maps_the_display_structure(self, connector):
        """The field-limited get comes back with the display fields intact.

        The connector asks for ``field(alarm,timeStamp,display)`` and never
        falls back to a full get, so everything asserted here travelled in a
        reply that carried no value — which is the point on a camera channel.
        """
        metadata = await connector.get_metadata(SCALAR)

        assert metadata.units == SCALAR_UNITS
        assert metadata.precision == SCALAR_PRECISION
        assert metadata.description == SCALAR_DESCRIPTION
        assert metadata.display_low == pytest.approx(SCALAR_LIMITS[0])
        assert metadata.display_high == pytest.approx(SCALAR_LIMITS[1])

    async def test_metadata_of_a_frame_channel_costs_no_frame(self, connector):
        """A camera channel answers a metadata lookup out of its header alone."""
        metadata = await connector.get_metadata(IMAGE)

        assert metadata.raw_metadata["nt_id"] == "epics:nt/NTNDArray:1.0"
        # The payload branches never ran: no dtype, no dimensions, no shape.
        assert "dimensions" not in metadata.raw_metadata
        assert "shape" not in metadata.raw_metadata

    async def test_validate_channel_is_true_for_a_served_address(self, connector):
        assert await connector.validate_channel(SCALAR) is True

    async def test_validate_channel_is_false_for_an_unserved_address(self, connector):
        """A PVA-globbed address nobody serves is not reachable."""
        assert await connector.validate_channel(UNSERVED) is False

    async def test_a_compressed_channel_still_validates(self, connector):
        """Reachability is a property of the channel, not of its payload.

        Reading this address raises; validating it must not, or an operator
        would be told a camera that is plainly on the network does not exist.
        """
        assert await connector.validate_channel(CODEC) is True


# ---------------------------------------------------------------------------
# Criterion 4: PVA writes are refused, for scalars and for arrays
# ---------------------------------------------------------------------------


@pytest.fixture
def writes_enabled(monkeypatch):
    """Enable writes at the deployment gate, so the PVA refusal is what answers.

    Without this the connector's ``writes_enabled`` pre-check refuses first and
    the PVA-specific refusal would never be reached — a green test that proved
    only that the test environment has no config file.
    """
    monkeypatch.setattr(EPICSConnector, "_writes_enabled", property(lambda self: True))


@pytest.mark.unit
@pytest.mark.usefixtures("writes_enabled")
class TestWriteRefusal:
    """A PVA-routed address is read-only, whatever the value's shape."""

    @pytest.mark.parametrize(
        "value",
        [pytest.param(99.0, id="scalar"), pytest.param(np.zeros(4, dtype=np.uint16), id="array")],
    )
    async def test_write_is_refused_before_the_network(self, connector, value):
        result = await connector.write_channel(SCALAR, value)

        assert result.success is False
        assert result.blocked is True
        assert result.refusal_reason == "VALIDATION_ERROR"
        assert "PVAccess writes are not supported" in result.error_message
        assert "No write was attempted." in result.error_message

    async def test_the_served_value_is_untouched_by_a_refused_write(self, connector):
        """The refusal's claim that no write was attempted, checked at the server."""
        await connector.write_channel(SCALAR, 99.0)

        reading = await connector.read_channel(SCALAR)
        assert reading.value == pytest.approx(SCALAR_VALUE)


# ---------------------------------------------------------------------------
# Criteria 1 and 2 through the tool body: batch failures and the artifact path
# ---------------------------------------------------------------------------


@pytest.fixture
def tool_singletons():
    """Reset the process-wide MCP singletons around a tool-body test.

    The server context, the artifact store and the config caches all outlive a
    test. Resetting on both sides keeps a store rooted at a deleted ``tmp_path``
    from leaking into whatever runs next in this directory.
    """
    import osprey.utils.config as config_module
    from osprey.mcp_server.control_system.server_context import reset_server_context
    from osprey.stores.artifact_store import reset_artifact_store
    from osprey.utils.workspace import reset_config_cache

    def reset_all():
        reset_server_context()
        reset_artifact_store()
        reset_config_cache()
        config_module._config_cache.clear()

    reset_all()
    yield
    reset_all()


async def _read_through_the_tool(tmp_path, monkeypatch, connector, channels: list[str]) -> dict:
    """Drive the real ``channel_read`` tool body against the live connector."""
    from osprey.mcp_server.control_system.server_context import initialize_server_context
    from osprey.mcp_server.control_system.tools.channel_read import channel_read
    from osprey.stores.artifact_store import initialize_artifact_store

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: epics\n")
    initialize_server_context()
    initialize_artifact_store(workspace_root=tmp_path / "agent_data")

    with (
        patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ),
        patch("osprey.infrastructure.server_launcher.ensure_artifact_server", lambda: None),
    ):
        result = await get_tool_fn(channel_read)(channels=channels)
    return extract_response_dict(result)


@pytest.mark.unit
@pytest.mark.usefixtures("tool_singletons")
class TestThroughTheToolBody:
    """What the agent is handed when the reads come off a real PVA server."""

    async def test_mixed_batch_reports_the_unreachable_address(
        self, tmp_path, monkeypatch, batch_connector
    ):
        """One dead CA address does not cost the batch its PVA readings."""
        data = await _read_through_the_tool(
            tmp_path, monkeypatch, batch_connector, [SCALAR, CA_UNREACHABLE, ENUM]
        )

        assert data["status"] == "success"
        readings = data["summary"]["readings"]
        assert readings[SCALAR]["value"] == pytest.approx(SCALAR_VALUE)
        assert readings[ENUM]["value"] == ENUM_INDEX
        assert readings[ENUM]["enum_label"] == ENUM_CHOICES[ENUM_INDEX]
        assert readings[ENUM]["enum_labels"] == ENUM_CHOICES
        # The scalar is not an enum, so it grows neither key.
        assert "enum_label" not in readings[SCALAR]
        assert CA_UNREACHABLE not in readings

        failures = data["summary"]["channels_failed"]
        assert CA_UNREACHABLE in failures
        assert CA_UNREACHABLE in failures[CA_UNREACHABLE]
        assert failures[CA_UNREACHABLE].startswith("ConnectionError:")

    async def test_the_frame_takes_the_artifact_path(self, tmp_path, monkeypatch, connector):
        """3072 pixels are over the inline budget, so the value is withheld."""
        data = await _read_through_the_tool(tmp_path, monkeypatch, connector, [IMAGE])

        entry = data["summary"]["readings"][IMAGE]

        assert "value" not in entry
        assert entry["value_withheld"] is True
        assert entry["artifact_reason"] == "per_value_threshold"
        assert entry["shape"] == [FRAME_HEIGHT, FRAME_WIDTH]
        assert entry["dtype"] == "uint16"
        assert entry["element_count"] == FRAME_HEIGHT * FRAME_WIDTH
        assert entry["artifact_id"]
        assert entry["data_file"].endswith(".npy")

    async def test_the_persisted_frame_round_trips_exactly(self, tmp_path, monkeypatch, connector):
        """``np.load(data_file)`` gives back the machine's own pixels, unsigned."""
        from pathlib import Path

        from osprey.stores.artifact_store import get_artifact_store

        data = await _read_through_the_tool(tmp_path, monkeypatch, connector, [IMAGE])
        entry = data["summary"]["readings"][IMAGE]

        path = Path(get_artifact_store().repo_root) / entry["data_file"]
        assert path.exists(), entry["data_file"]
        loaded = np.load(path)

        assert loaded.dtype == np.uint16
        assert loaded.shape == (FRAME_HEIGHT, FRAME_WIDTH)
        np.testing.assert_array_equal(loaded, FRAME)
