"""Tests for the 2-D+ artifact path of ``channel_read``.

An oversized multi-dimensional reading - a camera frame, an RGB image, a stack
no renderer can draw - is persisted as an artifact instead of being flooded into
the response. These tests drive the real tool body against a mock connector and
then check what actually landed: the response envelope the agent reads, the PNG
the gallery serves over its own route, and the ``.npy`` an ``execute`` script
loads back.
"""

import struct
from contextlib import nullcontext
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from osprey.connectors.control_system.base import ChannelMetadata, ChannelValue
from osprey.interfaces.artifacts.app import create_app
from osprey.mcp_server.control_system.server_context import initialize_server_context
from osprey.stores.artifact_store import get_artifact_store, initialize_artifact_store
from tests.mcp_server.conftest import extract_response_dict, get_tool_fn

ADDRESS = "CAM:1:IMAGE"

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _channel_value(value, color_mode=None):
    """A real ChannelValue, so the tool sees the dataclass it sees in production."""
    raw = {"color_mode": color_mode} if color_mode is not None else {}
    return ChannelValue(
        value=value,
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        metadata=ChannelMetadata(
            units="counts",
            precision=0,
            alarm_status="NO_ALARM",
            description="Camera frame",
            raw_metadata=raw,
        ),
    )


class _MockConnector:
    """Connector stub feeding synthetic frames through the real tool body."""

    def __init__(self, values: dict, color_modes: dict | None = None):
        self._values = values
        self._color_modes = color_modes or {}

    async def read_channel(self, address: str):
        return _channel_value(self._values[address], self._color_modes.get(address))


def _config_patch(overrides: dict):
    """Patch only the named config keys, delegating everything else."""
    if not overrides:
        return nullcontext()

    from osprey.utils import config as config_module

    real = config_module.get_config_value

    def fake(path, default=None, *args, **kwargs):
        if path in overrides:
            return overrides[path]
        return real(path, default, *args, **kwargs)

    return patch("osprey.utils.config.get_config_value", side_effect=fake)


async def _read(tmp_path, monkeypatch, values, *, color_modes=None, overrides=None, **kwargs):
    """Run the real channel_read tool body against a mock connector."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()
    initialize_artifact_store(workspace_root=tmp_path / "agent_data")

    from osprey.mcp_server.control_system.tools.channel_read import channel_read

    connector = _MockConnector(values, color_modes)
    with (
        patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ),
        patch("osprey.infrastructure.server_launcher.ensure_artifact_server", lambda: None),
        _config_patch(overrides or {}),
    ):
        result = await get_tool_fn(channel_read)(channels=list(values), **kwargs)
    return extract_response_dict(result)


def _gaussian_spot(height: int = 48, width: int = 64) -> np.ndarray:
    """A 48x64 uint16 frame with a Gaussian spot - a synthetic camera image."""
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    radius2 = (rows - height / 2) ** 2 + (cols - width / 2) ** 2
    return (50000 * np.exp(-radius2 / (2 * 8.0**2))).astype(np.uint16)


def _resolve(entry_fields: dict) -> "object":
    """Absolute path of the entry's data_file, the way an agent would resolve it."""
    from pathlib import Path

    store = get_artifact_store()
    candidate = Path(store.repo_root) / entry_fields["data_file"]
    assert candidate.exists(), f"data_file does not resolve: {entry_fields['data_file']}"
    return candidate


def _png_size(png_bytes: bytes) -> tuple[int, int]:
    """(width, height) parsed straight out of the PNG's IHDR chunk."""
    assert png_bytes[:8] == PNG_MAGIC, "not a PNG"
    return struct.unpack(">II", png_bytes[16:24])


@pytest.mark.unit
async def test_frame_response_carries_summary_and_handle(tmp_path, monkeypatch):
    """A camera frame comes back as a summary plus a working artifact handle."""
    frame = _gaussian_spot()
    data = await _read(tmp_path, monkeypatch, {ADDRESS: frame})

    assert data["status"] == "success"
    entry = data["summary"]["readings"][ADDRESS]

    assert "value" not in entry
    assert entry["value_withheld"] is True
    assert entry["artifact_reason"] == "per_value_threshold"
    assert entry["shape"] == [48, 64]
    assert entry["dtype"] == "uint16"
    assert entry["element_count"] == 48 * 64
    assert entry["timestamp"] == "2024-01-15 10:30:00"
    assert entry["min"] == float(frame.min())
    assert entry["max"] == float(frame.max())
    assert entry["mean"] == pytest.approx(float(frame.mean()))
    assert entry["artifact_id"]
    assert entry["data_file"].endswith(".npy")
    assert entry["artifact_note"]
    assert "artifact_error" not in entry
    assert "artifact_status" not in entry


@pytest.mark.unit
async def test_view_hint_names_the_png_by_path(tmp_path, monkeypatch):
    """The image is reachable the established way: Read the PNG at this path."""
    data = await _read(tmp_path, monkeypatch, {ADDRESS: _gaussian_spot()})

    entry = data["summary"]["readings"][ADDRESS]
    hint = entry["view_hint"]

    assert hint.startswith("Use the Read tool on ")
    png_relative = hint.split("Use the Read tool on ", 1)[1].split(" to view")[0]
    assert png_relative.endswith(".png")

    from pathlib import Path

    store = get_artifact_store()
    assert (Path(store.repo_root) / png_relative).exists(), png_relative


@pytest.mark.unit
async def test_npy_round_trip_preserves_the_frame(tmp_path, monkeypatch):
    """numpy.load(data_file) gives back exactly what the machine reported."""
    frame = _gaussian_spot()
    data = await _read(tmp_path, monkeypatch, {ADDRESS: frame})

    loaded = np.load(_resolve(data["summary"]["readings"][ADDRESS]))

    assert loaded.dtype == frame.dtype
    assert loaded.shape == frame.shape
    np.testing.assert_array_equal(loaded, frame)


@pytest.mark.unit
async def test_gallery_serves_the_rendered_png(tmp_path, monkeypatch):
    """The entry's primary file is a PNG the gallery's own route serves."""
    frame = _gaussian_spot()
    data = await _read(tmp_path, monkeypatch, {ADDRESS: frame})

    artifact_id = data["summary"]["readings"][ADDRESS]["artifact_id"]
    stored = get_artifact_store().get_entry(artifact_id)
    assert stored is not None
    assert stored.artifact_type == "image"
    assert stored.mime_type == "image/png"

    client = TestClient(create_app(workspace_root=tmp_path / "agent_data"))
    response = client.get(f"/files/{artifact_id}/{stored.filename}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    # Rendered at the frame's native resolution: width x height, not transposed.
    assert _png_size(response.content) == (64, 48)


@pytest.mark.unit
async def test_artifact_metadata_and_category_feed_retention(tmp_path, monkeypatch):
    """channel + channel_values are the keys the retention sweep selects on."""
    data = await _read(tmp_path, monkeypatch, {ADDRESS: _gaussian_spot()})

    stored = get_artifact_store().get_entry(data["summary"]["readings"][ADDRESS]["artifact_id"])

    assert stored is not None
    assert stored.category == "channel_values"
    assert stored.tool_source == "channel_read"
    assert stored.metadata["channel"] == ADDRESS
    assert stored.metadata["shape"] == [48, 64]
    assert stored.metadata["dtype"] == "uint16"
    # NOT timeseries: that key picks the gallery's interactive chart viewport,
    # which cannot render a frame.
    assert stored.metadata["data_type"] != "timeseries"
    assert stored.summary["channel"] == ADDRESS
    assert stored.summary["element_count"] == 48 * 64
    assert "value_withheld" not in stored.summary
    assert stored.access_details["view_hint"]


@pytest.mark.unit
async def test_rgb_frame_renders_from_the_trailing_colour_axis(tmp_path, monkeypatch):
    """With no colorMode reported, a trailing 3-sized axis is the colour axis."""
    frame = np.zeros((32, 40, 3), dtype=np.uint8)
    frame[..., 0] = 200
    data = await _read(tmp_path, monkeypatch, {ADDRESS: frame})

    entry = data["summary"]["readings"][ADDRESS]
    assert entry["shape"] == [32, 40, 3]
    assert "no_preview_reason" not in entry

    stored = get_artifact_store().get_entry(entry["artifact_id"])
    assert stored is not None and stored.mime_type == "image/png"

    png = (get_artifact_store().get_file_path(entry["artifact_id"])).read_bytes()
    assert _png_size(png) == (40, 32)
    np.testing.assert_array_equal(np.load(_resolve(entry)), frame)


@pytest.mark.unit
async def test_reported_color_mode_places_the_colour_axis_first(tmp_path, monkeypatch):
    """RGB1 means the colour components lead: (3, H, W), not (H, W, 3)."""
    frame = np.zeros((3, 32, 40), dtype=np.uint8)
    frame[1] = 255
    data = await _read(tmp_path, monkeypatch, {ADDRESS: frame}, color_modes={ADDRESS: "RGB1"})

    entry = data["summary"]["readings"][ADDRESS]
    assert "no_preview_reason" not in entry

    png = (get_artifact_store().get_file_path(entry["artifact_id"])).read_bytes()
    # Drawn as a 40x32 picture, not as a 32x40 one: the leading axis was colour.
    assert _png_size(png) == (40, 32)


@pytest.mark.unit
async def test_mono_color_mode_is_not_drawn_as_rgb(tmp_path, monkeypatch):
    """A frame the control system calls Mono is not RGB, however its dims line up."""
    frame = np.arange(3 * 32 * 40, dtype=np.uint16).reshape(3, 32, 40)
    data = await _read(tmp_path, monkeypatch, {ADDRESS: frame}, color_modes={ADDRESS: "Mono"})

    entry = data["summary"]["readings"][ADDRESS]
    assert entry["no_preview_reason"]
    stored = get_artifact_store().get_entry(entry["artifact_id"])
    assert stored is not None and stored.mime_type != "image/png"
    np.testing.assert_array_equal(np.load(_resolve(entry)), frame)


@pytest.mark.unit
async def test_integer_color_mode_rgb1_places_the_colour_axis_first(tmp_path, monkeypatch):
    """areaDetector sends colorMode as an enum: 2 is RGB1, so (3, H, W)."""
    # Both the leading and the trailing axis are 3, so inference alone would take
    # the trailing one - only the reported code can pick the leading one.
    frame = np.zeros((3, 800, 3), dtype=np.uint8)
    frame[1] = 255
    data = await _read(tmp_path, monkeypatch, {ADDRESS: frame}, color_modes={ADDRESS: 2})

    entry = data["summary"]["readings"][ADDRESS]
    assert "no_preview_reason" not in entry

    png = (get_artifact_store().get_file_path(entry["artifact_id"])).read_bytes()
    # 3 wide by 800 high: the leading axis was colour, not the trailing one.
    assert _png_size(png) == (3, 800)


@pytest.mark.unit
async def test_integer_color_mode_rgb2_places_the_colour_axis_in_the_middle(tmp_path, monkeypatch):
    """Enum 3 is RGB2: the colour components are the row axis, (H, 3, W)."""
    frame = np.zeros((32, 3, 40), dtype=np.uint8)
    frame[:, 2, :] = 255
    data = await _read(tmp_path, monkeypatch, {ADDRESS: frame}, color_modes={ADDRESS: 3})

    entry = data["summary"]["readings"][ADDRESS]
    assert "no_preview_reason" not in entry

    png = (get_artifact_store().get_file_path(entry["artifact_id"])).read_bytes()
    assert _png_size(png) == (40, 32)


@pytest.mark.unit
async def test_integer_color_mode_rgb3_places_the_colour_axis_last(tmp_path, monkeypatch):
    """Enum 4 is RGB3: the same frame as the RGB1 case, drawn the other way."""
    frame = np.zeros((3, 800, 3), dtype=np.uint8)
    frame[1] = 255
    data = await _read(tmp_path, monkeypatch, {ADDRESS: frame}, color_modes={ADDRESS: 4})

    entry = data["summary"]["readings"][ADDRESS]
    assert "no_preview_reason" not in entry

    png = (get_artifact_store().get_file_path(entry["artifact_id"])).read_bytes()
    # 800 wide by 3 high - the trailing axis was colour this time.
    assert _png_size(png) == (800, 3)


@pytest.mark.unit
async def test_integer_mono_color_mode_is_not_drawn_as_rgb(tmp_path, monkeypatch):
    """Enum 0 is Mono: a 3-plane stack, not a colour frame, whatever its dims."""
    frame = np.arange(3 * 32 * 40, dtype=np.uint16).reshape(3, 32, 40)
    data = await _read(tmp_path, monkeypatch, {ADDRESS: frame}, color_modes={ADDRESS: 0})

    entry = data["summary"]["readings"][ADDRESS]
    assert entry["no_preview_reason"]
    stored = get_artifact_store().get_entry(entry["artifact_id"])
    assert stored is not None and stored.mime_type != "image/png"
    np.testing.assert_array_equal(np.load(_resolve(entry)), frame)


@pytest.mark.unit
async def test_unknown_integer_color_mode_falls_back_to_inference(tmp_path, monkeypatch):
    """A code outside the enum reads as nothing reported, so the shape decides."""
    frame = np.zeros((32, 40, 3), dtype=np.uint8)
    frame[..., 2] = 200
    data = await _read(tmp_path, monkeypatch, {ADDRESS: frame}, color_modes={ADDRESS: 77})

    entry = data["summary"]["readings"][ADDRESS]
    assert "no_preview_reason" not in entry

    png = (get_artifact_store().get_file_path(entry["artifact_id"])).read_bytes()
    assert _png_size(png) == (40, 32)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0, "Mono"),
        (1, "Bayer"),
        (2, "RGB1"),
        (3, "RGB2"),
        (4, "RGB3"),
        (77, None),
        (True, None),
        ("RGB2", "RGB2"),
        ("", None),
        ("   ", None),
        (None, None),
        (2.0, None),
    ],
)
def test_color_mode_accepts_only_enum_codes_and_named_modes(raw, expected):
    """The reported mode is read as a plain int code or a non-empty name, nothing else."""
    from osprey.mcp_server.control_system.tools.channel_read import _color_mode

    assert _color_mode(_channel_value(np.zeros((2, 2)), raw)) == expected


@pytest.mark.unit
def test_color_mode_ignores_mock_metadata():
    """A mock's attribute is not a reported mode - the shape still decides."""
    from osprey.mcp_server.control_system.tools.channel_read import _color_mode

    assert _color_mode(MagicMock()) is None


@pytest.mark.unit
async def test_constant_frame_renders_without_dividing_by_zero(tmp_path, monkeypatch):
    """A frame with no contrast still renders - autoscaling must not blow up."""
    frame = np.full((48, 64), 7, dtype=np.uint16)
    data = await _read(tmp_path, monkeypatch, {ADDRESS: frame})

    entry = data["summary"]["readings"][ADDRESS]
    assert "artifact_error" not in entry
    assert entry["artifact_id"]

    png = (get_artifact_store().get_file_path(entry["artifact_id"])).read_bytes()
    assert _png_size(png) == (64, 48)
    np.testing.assert_array_equal(np.load(_resolve(entry)), frame)


@pytest.mark.unit
async def test_all_nan_frame_renders_without_blowing_up(tmp_path, monkeypatch):
    """Every pixel NaN leaves no scale to compute; the frame still stores."""
    frame = np.full((48, 64), np.nan, dtype=np.float64)
    data = await _read(tmp_path, monkeypatch, {ADDRESS: frame})

    entry = data["summary"]["readings"][ADDRESS]
    assert "artifact_error" not in entry
    png = (get_artifact_store().get_file_path(entry["artifact_id"])).read_bytes()
    assert _png_size(png) == (64, 48)


@pytest.mark.unit
async def test_four_dimensional_value_takes_the_data_only_path(tmp_path, monkeypatch):
    """No honest rendering exists for a 4-D stack; the data is kept anyway."""
    stack = np.arange(2 * 4 * 16 * 40, dtype=np.int32).reshape(2, 4, 16, 40)
    data = await _read(tmp_path, monkeypatch, {ADDRESS: stack})

    assert data["status"] == "success"
    entry = data["summary"]["readings"][ADDRESS]
    assert entry["artifact_id"]
    assert entry["data_file"].endswith(".npy")
    assert entry["no_preview_reason"]
    assert "artifact_status" not in entry

    stored = get_artifact_store().get_entry(entry["artifact_id"])
    assert stored is not None
    assert stored.mime_type != "image/png"
    assert stored.category == "channel_values"
    assert stored.metadata["channel"] == ADDRESS

    loaded = np.load(_resolve(entry))
    assert loaded.dtype == stack.dtype
    np.testing.assert_array_equal(loaded, stack)


@pytest.mark.unit
async def test_render_failure_degrades_to_data_only(tmp_path, monkeypatch):
    """A renderable-looking frame whose render blows up still keeps its data."""
    monkeypatch.setattr(
        "osprey.mcp_server.control_system.tools.channel_read._render_png",
        MagicMock(side_effect=RuntimeError("encoder exploded")),
    )
    frame = _gaussian_spot()
    data = await _read(tmp_path, monkeypatch, {ADDRESS: frame})

    assert data["status"] == "success"
    entry = data["summary"]["readings"][ADDRESS]
    assert "RuntimeError" in entry["no_preview_reason"]
    assert entry["artifact_id"]
    np.testing.assert_array_equal(np.load(_resolve(entry)), frame)


@pytest.mark.unit
async def test_retention_fires_after_an_image_save(tmp_path, monkeypatch):
    """The rolling window is swept on the image path too, with the config window."""
    calls: list[tuple] = []

    def fake_prune(self, address, keep):
        calls.append((address, keep))
        return []

    monkeypatch.setattr(
        "osprey.stores.artifact_store.ArtifactStore.prune_channel_readings",
        fake_prune,
    )

    await _read(
        tmp_path,
        monkeypatch,
        {ADDRESS: _gaussian_spot()},
        overrides={"control_system.channel_read_artifact_retention": 3},
    )

    assert calls == [(ADDRESS, 3)]


@pytest.mark.unit
async def test_retention_keeps_only_the_configured_window(tmp_path, monkeypatch):
    """Repeated frames of one channel do not grow the gallery without bound."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()
    initialize_artifact_store(workspace_root=tmp_path / "agent_data")

    from osprey.mcp_server.control_system.tools.channel_read import channel_read

    connector = _MockConnector({ADDRESS: _gaussian_spot()})
    with (
        patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ),
        patch("osprey.infrastructure.server_launcher.ensure_artifact_server", lambda: None),
        _config_patch({"control_system.channel_read_artifact_retention": 2}),
    ):
        for _ in range(4):
            await get_tool_fn(channel_read)(channels=[ADDRESS])

    entries = get_artifact_store().list_entries(category_filter="channel_values")
    assert len(entries) == 2, [e.id for e in entries]

    # Both halves of every pruned entry are gone, not just the thumbnail.
    store_dir = get_artifact_store().artifact_dir
    assert len(list(store_dir.glob("*.png"))) == 2
    assert len(list(store_dir.glob("*.npy"))) == 2


@pytest.mark.unit
async def test_store_failure_degrades_but_read_stays_successful(tmp_path, monkeypatch):
    """A frame the machine did deliver is never reported as a failed read."""
    monkeypatch.setattr(
        "osprey.stores.artifact_store.ArtifactStore.save_channel_reading",
        MagicMock(side_effect=OSError("disk full")),
    )
    frame = _gaussian_spot()
    data = await _read(tmp_path, monkeypatch, {ADDRESS: frame})

    assert data["status"] == "success"
    entry = data["summary"]["readings"][ADDRESS]
    assert entry["value_withheld"] is True
    assert entry["shape"] == [48, 64]
    assert entry["element_count"] == frame.size
    assert entry["artifact_error"] == "OSError: disk full"
    assert "artifact_id" not in entry
    assert "data_file" not in entry


@pytest.mark.unit
async def test_frames_and_waveforms_coexist_in_one_call(tmp_path, monkeypatch):
    """A mixed batch keeps one entry per address, each with its own artifact shape."""
    frame = _gaussian_spot()
    waveform = np.linspace(0.0, 1.0, 5000, dtype=np.float64)
    data = await _read(
        tmp_path,
        monkeypatch,
        {ADDRESS: frame, "SR:BPM:WAVEFORM": waveform},
    )

    readings = data["summary"]["readings"]
    assert set(readings) == {ADDRESS, "SR:BPM:WAVEFORM"}

    store = get_artifact_store()
    image_entry = store.get_entry(readings[ADDRESS]["artifact_id"])
    series_entry = store.get_entry(readings["SR:BPM:WAVEFORM"]["artifact_id"])
    assert image_entry is not None and image_entry.mime_type == "image/png"
    assert series_entry is not None and series_entry.metadata["data_type"] == "timeseries"

    assert data["access_details"]["values_withheld"]["channels"] == [
        ADDRESS,
        "SR:BPM:WAVEFORM",
    ]
