"""What a client sees when it reads a real PVAccess server over the wire.

A real pvapy ``PvaServer`` hosts real normative-type structures, the real EPICS
connector reads them through pvapy's PVA client, and — for the batch and
artifact assertions — the real ``channel_read`` tool body decides what the
agent gets to see. No fake ``pvaccess`` module, no hand-built ``PvObject``.
``tests/connectors/test_pva_read_mapping.py`` already proves the mapping in
process against an injected client; this file is the check that the wire
agrees with it, and it is where success criteria 1-4 of the PVA feature are
asserted concretely.

Four things about the venue are worth stating plainly, because a green run
means nothing without them.

**pvapy is the venue's hard requirement.** The module is ``importorskip``-gated,
and a skipped suite proves nothing at all — which is why
``tests/connectors/test_pva_venue_guard.py`` fails the lane on any platform
where pvapy must be importable.

**Nothing here talks PVAccess from the pytest process.** pvapy reads
``EPICS_PVA_*`` when the first PVA channel of a process is created and never
again, and an xdist worker is shared with every other test it runs; and on
macOS a thread that used pvapy is suspended forever when it exits. So, as in
``test_epics_soft_ioc.py``, the server is its own process, and every client
scenario runs this same file as a script (``python -m
tests.connectors.test_pva_live_fixture <scenario> <port> <workdir>``) in a
fresh interpreter, printing its observations as one JSON line. Each scenario
runs once per module; the tests below assert on what it saw.

**The server is isolated, and the client is pinned to it.** The server binds
127.0.0.1 on a randomly chosen port (``EPICS_PVAS_INTF_ADDR_LIST``,
``EPICS_PVAS_SERVER_PORT``) and beacons only to the loopback, so concurrent
runs of this file — and the operator's own IOCs — can never answer for each
other. The connector is pointed at that port through
``pva_gateway.use_name_server``, which sets ``EPICS_PVA_NAME_SERVERS``: the
client opens a TCP connection straight to the server instead of UDP-searching
for it, so no search packet leaves the host.

**The Channel Access half of the mixed batch is unreachable on purpose.** Its
address matches no ``pva_channels`` glob, so it takes the CA path, where no
server answers it — that is the point. The scenario's ``EPICS_CA_ADDR_LIST``
names a loopback port nobody listens on, with auto-addressing off, so that
half sends no Channel Access search off the host either.
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("pvaccess", reason="pvapy is required to serve a live PVAccess fixture")

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHONPATH = os.pathsep.join(
    [
        str(REPO_ROOT),
        str(REPO_ROOT / "src"),
        str(REPO_ROOT / "packages" / "osprey-connectors" / "src"),
    ]
)

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
# The mixed batch and the unserved-address probe each pay this once for the
# address nobody serves.
BATCH_TIMEOUT_S = 2.0
SERVER_READY_TIMEOUT_S = 30.0
SCENARIO_TIMEOUT_S = 90.0


def _gaussian_spot(height: int = FRAME_HEIGHT, width: int = FRAME_WIDTH) -> np.ndarray:
    """A 48x64 uint16 frame with a Gaussian spot — a synthetic camera image."""
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    radius2 = (rows - height / 2) ** 2 + (cols - width / 2) ** 2
    return (50000 * np.exp(-radius2 / (2 * 8.0**2))).astype(np.uint16)


FRAME = _gaussian_spot()


def _free_port(kind: int = socket.SOCK_STREAM) -> int:
    with socket.socket(socket.AF_INET, kind) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _scrubbed_env() -> dict[str, str]:
    """This process's environment with every EPICS client/server variable removed."""
    return {k: v for k, v in os.environ.items() if not k.startswith("EPICS_")}


# ---------------------------------------------------------------------------
# The server: its own process
# ---------------------------------------------------------------------------


def _ndarray(frame: np.ndarray, *, codec: str = "") -> Any:
    """An NTNDArray carrying ``frame``, optionally tagged with a codec.

    A codec tag is exactly what an ADPva IOC with compression enabled sends;
    here the payload stays the raw frame, which is all the connector needs to
    see to refuse it.
    """
    import pvaccess

    value = pvaccess.NtNdArray()
    value["value"] = {"ushortValue": frame.flatten()}
    value["dimension"] = [
        pvaccess.PvDimension(size, 0, size, 1, False) for size in reversed(frame.shape)
    ]
    value["attribute"] = [pvaccess.NtAttribute("ColorMode", pvaccess.PvInt(0))]
    if codec:
        value["codec"] = {"name": codec}
    return value


def serve() -> None:
    """Serve the four channels until killed or orphaned (run in the server process).

    The wait is a sleep loop, not a blocking read of stdin: a server whose
    main thread sat in ``sys.stdin.read()`` was observed never to answer a
    single get. Should the test process die without stopping it, the server
    notices it has been reparented and exits on its own.

    The scalar carries its precision inside ``display.format`` (``"F8.3"``),
    the way pvapy's own NT servers — and its Channel Access provider — spell it.
    """
    import pvaccess

    scalar = pvaccess.NtScalar(pvaccess.DOUBLE, SCALAR_VALUE)
    scalar["display"] = {
        "units": SCALAR_UNITS,
        "format": f"F8.{SCALAR_PRECISION}",
        "description": SCALAR_DESCRIPTION,
        "limitLow": SCALAR_LIMITS[0],
        "limitHigh": SCALAR_LIMITS[1],
    }
    server = pvaccess.PvaServer()
    server.addRecord(SCALAR, scalar)
    server.addRecord(ENUM, pvaccess.NtEnum(ENUM_CHOICES, ENUM_INDEX))
    server.addRecord(IMAGE, _ndarray(FRAME))
    server.addRecord(CODEC, _ndarray(FRAME, codec=CODEC_NAME))
    print("serving", flush=True)
    parent = os.getppid()
    while os.getppid() == parent:
        time.sleep(0.5)
    server.stop()


class PvaServerProcess:
    """A pvapy ``PvaServer`` on its own loopback port, in its own process."""

    def __init__(self, log: Path) -> None:
        self.port = _free_port()
        env = _scrubbed_env()
        env.update(
            {
                "PYTHONPATH": PYTHONPATH,
                "EPICS_PVAS_INTF_ADDR_LIST": "127.0.0.1",
                "EPICS_PVAS_SERVER_PORT": str(self.port),
                "EPICS_PVAS_BROADCAST_PORT": str(_free_port(socket.SOCK_DGRAM)),
                "EPICS_PVAS_AUTO_BEACON_ADDR_LIST": "NO",
                "EPICS_PVAS_BEACON_ADDR_LIST": "127.0.0.1",
            }
        )
        self.log = log
        with log.open("wb") as output:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "tests.connectors.test_pva_live_fixture", "serve"],
                cwd=REPO_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        self._wait_until_serving()

    def _wait_until_serving(self) -> None:
        """Wait for the server's own "serving" line.

        Not a TCP connect to the port, which is how ``test_epics_soft_ioc.py``
        waits for an IOC: a bare connect-and-close was observed to leave the
        pvapy server answering no get at all afterwards.
        """
        deadline = time.monotonic() + SERVER_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"PVA server exited with {self.process.returncode}: "
                    f"{self.log.read_text(errors='replace')}"
                )
            if "serving" in self.log.read_text(errors="replace"):
                return
            time.sleep(0.1)
        self.stop()
        raise RuntimeError(f"PVA server never came up on 127.0.0.1:{self.port}")

    def stop(self) -> None:
        self.process.kill()
        self.process.wait(timeout=10)


@pytest.fixture(scope="module")
def pva_server(tmp_path_factory):
    """The server, stopped unconditionally so no listening socket outlives the module."""
    server = PvaServerProcess(tmp_path_factory.mktemp("pva_server") / "server.log")
    try:
        yield server
    finally:
        server.stop()


def _drive(scenario: str, port: int, workdir: Path) -> dict[str, Any]:
    """Run one client scenario in a fresh interpreter pointed at ``port`` alone."""
    workdir.mkdir(parents=True, exist_ok=True)
    config = workdir / "config.yml"
    config.write_text(
        "control_system:\n"
        "  type: epics\n"
        "  writes_enabled: true\n"
        "  limits_checking:\n"
        "    enabled: false\n"
    )
    env = _scrubbed_env()
    env.update(
        {
            "PYTHONPATH": PYTHONPATH,
            "CONFIG_FILE": str(config),
            # Channel Access searches go to a loopback port nobody answers on.
            "EPICS_CA_ADDR_LIST": f"127.0.0.1:{_free_port(socket.SOCK_DGRAM)}",
            "EPICS_CA_AUTO_ADDR_LIST": "NO",
        }
    )
    env.pop("OSPREY_EXECUTION_MODE", None)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.connectors.test_pva_live_fixture",
                scenario,
                str(port),
                str(workdir),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=SCENARIO_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"scenario {scenario} never exited; it printed: {exc.stdout!r}\n{exc.stderr!r}")
    assert result.returncode == 0, f"scenario {scenario} failed:\n{result.stderr}"
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    assert lines, f"scenario {scenario} printed no result:\n{result.stdout}\n{result.stderr}"
    return json.loads(lines[-1])


@pytest.fixture(scope="module")
def wire(pva_server, tmp_path_factory) -> dict[str, Any]:
    """Everything the connector saw reading, probing and writing the live server."""
    return _drive("wire", pva_server.port, tmp_path_factory.mktemp("wire"))


@pytest.fixture(scope="module")
def tool(pva_server, tmp_path_factory) -> dict[str, Any]:
    """What the ``channel_read`` tool body handed back for the live server."""
    return _drive("tool", pva_server.port, tmp_path_factory.mktemp("tool"))


# ---------------------------------------------------------------------------
# Scenarios: run in the child interpreter, never in the pytest process.
# ---------------------------------------------------------------------------


async def _connector(port: int, timeout: float = READ_TIMEOUT_S):
    from osprey_connectors.control_system.epics_connector import EPICSConnector

    connector = EPICSConnector()
    await connector.connect(
        {
            "timeout": timeout,
            "pva_channels": [PVA_GLOB],
            "pva_gateway": {
                "address": "127.0.0.1",
                # The TCP port the server listens on. Reached by name-server
                # lookup, so the client never broadcasts.
                "port": port,
                "use_name_server": True,
            },
        }
    )
    return connector


async def _raised(call) -> dict[str, Any]:
    try:
        await call
    except Exception as exc:
        return {"type": type(exc).__name__, "message": str(exc)}
    return {"type": None, "message": None}


def _write(result) -> dict[str, Any]:
    return {
        "outcome": str(result.outcome.value),
        "refusal_reason": result.refusal_reason,
        "error_message": result.error_message,
    }


async def scenario_wire(port: int, _workdir: Path) -> dict[str, Any]:
    connector = await _connector(port)
    short = await _connector(port, timeout=BATCH_TIMEOUT_S)
    try:
        scalar = await connector.read_channel(SCALAR)
        enum = await connector.read_channel(ENUM)
        image = await connector.read_channel(IMAGE)
        scalar_meta = await connector.get_metadata(SCALAR)
        image_meta = await connector.get_metadata(IMAGE)
        return {
            "scalar": {
                "value": scalar.value,
                "units": scalar.metadata.units,
                "precision": scalar.metadata.precision,
                "raw": scalar.metadata.raw_metadata,
            },
            "enum": {
                "value": enum.value,
                "value_is_str": isinstance(enum.value, str),
                "label": enum.metadata.enum_label,
                "labels": enum.metadata.enum_labels,
                "raw": enum.metadata.raw_metadata,
            },
            "image": {
                "is_ndarray": isinstance(image.value, np.ndarray),
                "dtype": str(image.value.dtype),
                "shape": list(image.value.shape),
                "equals_frame": bool(np.array_equal(image.value, FRAME)),
                "raw": image.metadata.raw_metadata,
            },
            "codec": await _raised(connector.read_channel(CODEC)),
            "scalar_meta": {
                "units": scalar_meta.units,
                "precision": scalar_meta.precision,
                "description": scalar_meta.description,
                "display_low": scalar_meta.display_low,
                "display_high": scalar_meta.display_high,
            },
            "image_meta_raw": image_meta.raw_metadata,
            "valid": {
                "scalar": await connector.validate_channel(SCALAR),
                "codec": await connector.validate_channel(CODEC),
                "unserved": await short.validate_channel(UNSERVED),
            },
            "writes": {
                "scalar": _write(await connector.write_channel(SCALAR, 99.0)),
                "array": _write(
                    await connector.write_channel(SCALAR, np.zeros(4, dtype=np.uint16))
                ),
            },
            "scalar_after_writes": (await connector.read_channel(SCALAR)).value,
        }
    finally:
        await connector.disconnect()
        await short.disconnect()


async def scenario_tool(port: int, workdir: Path) -> dict[str, Any]:
    """Drive the real ``channel_read`` tool body against live connectors."""
    from unittest.mock import AsyncMock, patch

    from osprey.mcp_server.control_system.server_context import initialize_server_context
    from osprey.mcp_server.control_system.tools.channel_read import channel_read
    from osprey.stores.artifact_store import get_artifact_store, initialize_artifact_store
    from tests.mcp_server.conftest import extract_response_dict, get_tool_fn

    os.chdir(workdir)
    initialize_server_context()
    initialize_artifact_store(workspace_root=workdir / "agent_data")
    connector = await _connector(port)
    short = await _connector(port, timeout=BATCH_TIMEOUT_S)

    async def read(target, channels: list[str]) -> dict[str, Any]:
        with (
            patch(
                "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
                new_callable=AsyncMock,
                return_value=target,
            ),
            patch("osprey.infrastructure.server_launcher.ensure_artifact_server", lambda: None),
        ):
            result = await get_tool_fn(channel_read)(channels=channels)
        return extract_response_dict(result)

    try:
        batch = await read(short, [SCALAR, CA_UNREACHABLE, ENUM])
        frame = await read(connector, [IMAGE])
        entry = frame["summary"]["readings"][IMAGE]
        path = Path(get_artifact_store().repo_root) / entry.get("data_file", "missing")
        persisted: dict[str, Any] = {"exists": path.exists()}
        if path.exists():
            loaded = np.load(path)
            persisted.update(
                dtype=str(loaded.dtype),
                shape=list(loaded.shape),
                equals_frame=bool(np.array_equal(loaded, FRAME)),
            )
        return {"batch": batch, "frame_entry": entry, "persisted": persisted}
    finally:
        await connector.disconnect()
        await short.disconnect()


# ---------------------------------------------------------------------------
# Criterion 1 (reads) and criterion 2 (frames): the four served channels
# ---------------------------------------------------------------------------


class TestLiveReads:
    """Every normative type the connector claims to map, read off the wire."""

    def test_ntscalar_returns_its_value(self, wire):
        """A PVA-only read answers with the number, its units and its precision."""
        scalar = wire["scalar"]

        assert scalar["value"] == pytest.approx(SCALAR_VALUE)
        assert scalar["raw"]["provider"] == "pva"
        assert scalar["raw"]["nt_type"] is None  # a scalar, mapped as one
        assert scalar["units"] == SCALAR_UNITS
        assert scalar["precision"] == SCALAR_PRECISION  # from display.format

    def test_ntenum_reads_as_the_index_with_its_label(self, wire):
        """An operator reading a state channel gets 1 *and* "Open"."""
        enum = wire["enum"]

        assert enum["value"] == ENUM_INDEX
        assert enum["value_is_str"] is False
        assert enum["label"] == ENUM_CHOICES[ENUM_INDEX]
        assert enum["labels"] == ENUM_CHOICES
        assert enum["raw"]["nt_type"] == "NTEnum"
        assert enum["raw"]["enum_index"] == ENUM_INDEX
        assert enum["raw"]["enum_choices"] == ENUM_CHOICES

    def test_ntndarray_returns_the_served_frame(self, wire):
        """The frame arrives unsigned, reshaped rows-by-columns, pixel for pixel.

        The shape assertion is the row/column order: the server sent dimensions
        ``[64, 48]`` (innermost first), and a connector that reshaped by that
        order rather than reversing it would hand back a (64, 48) array of the
        same 3072 pixels — a plausible image with its rows and columns swapped.
        """
        image = wire["image"]

        assert image["is_ndarray"] is True
        assert image["dtype"] == "uint16"
        assert image["shape"] == [FRAME_HEIGHT, FRAME_WIDTH]
        assert image["equals_frame"] is True

    def test_ntndarray_raw_metadata_reports_the_wire_layout(self, wire):
        """NT dimensions stay innermost-first; the numpy shape is the reverse."""
        raw = wire["image"]["raw"]

        assert raw["nt_type"] == "NTNDArray"
        assert raw["dtype"] == "uint16"
        assert raw["dimensions"] == [FRAME_WIDTH, FRAME_HEIGHT]
        assert raw["shape"] == [FRAME_HEIGHT, FRAME_WIDTH]
        assert raw["codec"] == ""
        assert raw["color_mode"] == 0

    def test_compressed_frame_is_refused_with_the_remedy(self, wire):
        """A codec-tagged frame raises rather than reshaping a compressed blob."""
        raised = wire["codec"]

        assert raised["type"] == "ValueError"
        assert "compressed NTNDArray unsupported" in raised["message"]
        assert "disable ADPva compression" in raised["message"]
        assert CODEC_NAME in raised["message"]
        assert CODEC in raised["message"]


# ---------------------------------------------------------------------------
# Metadata and reachability, against the same live server
# ---------------------------------------------------------------------------


class TestMetadataAndValidation:
    """What the connector learns without reading a payload."""

    def test_get_metadata_maps_the_display_structure(self, wire):
        """The field-limited get comes back with the display fields intact.

        The connector asks for ``field(alarm,timeStamp,display)`` and never
        falls back to a full get, so everything asserted here travelled in a
        reply that carried no value — which is the point on a camera channel.
        """
        meta = wire["scalar_meta"]

        assert meta["units"] == SCALAR_UNITS
        assert meta["precision"] == SCALAR_PRECISION
        assert meta["description"] == SCALAR_DESCRIPTION
        assert meta["display_low"] == pytest.approx(SCALAR_LIMITS[0])
        assert meta["display_high"] == pytest.approx(SCALAR_LIMITS[1])

    def test_metadata_of_a_frame_channel_costs_no_frame(self, wire):
        """A camera channel answers a metadata lookup out of its header alone."""
        raw = wire["image_meta_raw"]

        assert raw["provider"] == "pva"
        # The payload branches never ran: no dtype, no dimensions, no shape.
        assert "dimensions" not in raw
        assert "shape" not in raw
        assert "dtype" not in raw

    def test_validate_channel_is_true_for_a_served_address(self, wire):
        assert wire["valid"]["scalar"] is True

    def test_validate_channel_is_false_for_an_unserved_address(self, wire):
        """A PVA-globbed address nobody serves is not reachable.

        The probe waits out the connector's whole timeout, so it runs on the
        short-timeout connector.
        """
        assert wire["valid"]["unserved"] is False

    def test_a_compressed_channel_still_validates(self, wire):
        """Reachability is a property of the channel, not of its payload.

        Reading this address raises; validating it must not, or an operator
        would be told a camera that is plainly on the network does not exist.
        """
        assert wire["valid"]["codec"] is True


# ---------------------------------------------------------------------------
# Criterion 4: PVA writes are refused, for scalars and for arrays
# ---------------------------------------------------------------------------


class TestWriteRefusal:
    """A PVA-routed address is read-only, whatever the value's shape."""

    @pytest.mark.parametrize("kind", ["scalar", "array"])
    def test_write_is_refused_before_the_network(self, wire, kind):
        result = wire["writes"][kind]

        assert result["outcome"] == "refused"
        assert result["refusal_reason"] == "VALIDATION_ERROR"
        assert "PVAccess writes are not supported" in result["error_message"]
        assert "No write was attempted." in result["error_message"]

    def test_the_served_value_is_untouched_by_a_refused_write(self, wire):
        """The refusal's claim that no write was attempted, checked at the server."""
        assert wire["scalar_after_writes"] == pytest.approx(SCALAR_VALUE)


# ---------------------------------------------------------------------------
# Criteria 1 and 2 through the tool body: batch failures and the artifact path
# ---------------------------------------------------------------------------


class TestThroughTheToolBody:
    """What the agent is handed when the reads come off a real PVA server."""

    def test_mixed_batch_reports_the_unreachable_address(self, tool):
        """One dead CA address does not cost the batch its PVA readings."""
        data = tool["batch"]

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

    def test_the_frame_takes_the_artifact_path(self, tool):
        """3072 pixels are over the inline budget, so the value is withheld."""
        entry = tool["frame_entry"]

        assert "value" not in entry
        assert entry["value_withheld"] is True
        assert entry["artifact_reason"] == "per_value_threshold"
        assert entry["shape"] == [FRAME_HEIGHT, FRAME_WIDTH]
        assert entry["dtype"] == "uint16"
        assert entry["element_count"] == FRAME_HEIGHT * FRAME_WIDTH
        assert entry["artifact_id"]
        assert entry["data_file"].endswith(".npy")

    def test_the_persisted_frame_round_trips_exactly(self, tool):
        """``np.load(data_file)`` gives back the machine's own pixels, unsigned."""
        persisted = tool["persisted"]

        assert persisted["exists"] is True, tool["frame_entry"]
        assert persisted["dtype"] == "uint16"
        assert persisted["shape"] == [FRAME_HEIGHT, FRAME_WIDTH]
        assert persisted["equals_frame"] is True


async def _main(scenario: str, port: int, workdir: Path) -> None:
    """Run one scenario and print its result before the loop and process wind down."""
    result = await globals()[f"scenario_{scenario}"](port, workdir)
    print(json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        serve()
    else:
        asyncio.run(_main(sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])))
