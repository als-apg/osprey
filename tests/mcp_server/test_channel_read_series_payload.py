"""Tests for the 1-D artifact path of ``channel_read``.

An oversized 1-D reading is persisted as a channel-series JSON artifact in the
same shape the archiver artifacts use, so the gallery's existing timeseries
viewport renders it as an interactive chart with no front-end change. These
tests drive the real tool body against a mock connector and then feed the file
it wrote through the real :func:`extract_channel_series`, which is what the
chart endpoint does - the contract is asserted end to end, not restated.
"""

import json
from contextlib import nullcontext
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from osprey.mcp_server.control_system.server_context import initialize_server_context
from osprey.stores.artifact_store import get_artifact_store, initialize_artifact_store
from osprey.utils.timeseries import extract_channel_series
from tests.mcp_server.conftest import extract_response_dict, get_tool_fn

ADDRESS = "SR:BPM:WAVEFORM"


def _make_channel_value(value, timestamp="2024-01-15T10:30:00"):
    cv = MagicMock()
    cv.value = value
    cv.timestamp = timestamp
    cv.metadata = MagicMock()
    cv.metadata.units = "counts"
    cv.metadata.precision = 3
    cv.metadata.alarm_status = "NO_ALARM"
    cv.metadata.description = "Test channel"
    # Not an enum channel: a bare MagicMock attribute would be truthy, and the
    # read tool ships the enum keys only when they are not None.
    cv.metadata.enum_label = None
    cv.metadata.enum_labels = None
    return cv


class _MockConnector:
    """Connector stub feeding synthetic values through the real tool body."""

    def __init__(self, values: dict):
        self._values = values

    async def read_channel(self, address: str):
        return _make_channel_value(self._values[address])


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


async def _read(tmp_path, monkeypatch, values, *, overrides=None, **kwargs):
    """Run the real channel_read tool body against a mock connector."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()
    initialize_artifact_store(workspace_root=tmp_path / "agent_data")

    from osprey.mcp_server.control_system.tools.channel_read import channel_read

    connector = _MockConnector(values)
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


def _oversized(n: int = 5000) -> np.ndarray:
    """A 1-D array comfortably over the default 2000-element inline budget."""
    return np.linspace(0.0, 1.0, n, dtype=np.float64)


def _stored_payload(entry_fields: dict) -> dict:
    """Parsed JSON of the artifact the reading entry points at."""
    store = get_artifact_store()
    filepath = store.get_file_path(entry_fields["artifact_id"])
    assert filepath is not None and filepath.exists(), "artifact file missing on disk"
    return json.loads(filepath.read_text())


@pytest.mark.unit
async def test_series_payload_survives_extract_channel_series(tmp_path, monkeypatch):
    """The saved file feeds the chart endpoint's own extractor unchanged."""
    array = _oversized()
    data = await _read(tmp_path, monkeypatch, {ADDRESS: array})

    entry = data["summary"]["readings"][ADDRESS]
    payload = _stored_payload(entry)

    series, query = extract_channel_series(payload)

    assert list(series) == [ADDRESS]
    channel = series[ADDRESS]
    # x axis is the synthesized sample index, NOT wall-clock time.
    assert channel["timestamps"] == list(range(len(array)))
    assert channel["values"] == array.tolist()
    assert len(channel["values"]) == len(channel["timestamps"]) == len(array)
    assert query["x_axis"] == "sample_index"
    assert query["channels"] == [ADDRESS]


@pytest.mark.unit
async def test_reading_entry_carries_the_artifact_handle(tmp_path, monkeypatch):
    """The withheld entry keeps its summary and gains artifact_id/data_file/note."""
    data = await _read(tmp_path, monkeypatch, {ADDRESS: _oversized()})

    entry = data["summary"]["readings"][ADDRESS]
    assert data["status"] == "success"
    assert "value" not in entry
    assert entry["value_withheld"] is True
    assert entry["artifact_reason"] == "per_value_threshold"
    assert entry["artifact_id"]
    assert entry["data_file"]
    assert entry["artifact_note"]
    assert "artifact_error" not in entry
    assert "artifact_status" not in entry


@pytest.mark.unit
async def test_data_file_is_openable_and_json_loads(tmp_path, monkeypatch):
    """data_file is the agent-facing pointer: it resolves and parses as JSON."""
    data = await _read(tmp_path, monkeypatch, {ADDRESS: _oversized()})

    entry = data["summary"]["readings"][ADDRESS]
    store = get_artifact_store()
    # The pointer is repo-root relative (the agent's cwd); fall back to the store
    # directory only when the workspace layout gave no clean relative path.
    candidate = (store.repo_root / entry["data_file"]).resolve()
    if not candidate.exists():
        candidate = store.get_file_path(entry["artifact_id"])
    assert candidate is not None and candidate.exists(), entry["data_file"]

    payload = json.loads(candidate.read_text())
    assert set(payload) == {"query", "series"}
    assert payload["series"][ADDRESS]["values"][0] == 0.0


@pytest.mark.unit
async def test_artifact_metadata_and_category_drive_the_gallery(tmp_path, monkeypatch):
    """data_type=timeseries picks the chart viewport; channel+category feed retention."""
    data = await _read(tmp_path, monkeypatch, {ADDRESS: _oversized()})

    artifact_id = data["summary"]["readings"][ADDRESS]["artifact_id"]
    stored = get_artifact_store().get_entry(artifact_id)

    assert stored is not None
    assert stored.metadata["data_type"] == "timeseries"
    assert stored.metadata["channel"] == ADDRESS
    assert stored.category == "channel_values"
    assert stored.tool_source == "channel_read"
    assert stored.summary["channel"] == ADDRESS
    assert stored.summary["sample_count"] == 5000
    assert stored.summary["element_count"] == 5000


@pytest.mark.unit
async def test_non_finite_floats_become_json_null_gaps(tmp_path, monkeypatch):
    """NaN/inf would break the chart endpoint's serializer; they land as null."""
    array = _oversized()
    array[3] = np.nan
    array[7] = np.inf
    data = await _read(tmp_path, monkeypatch, {ADDRESS: array})

    payload = _stored_payload(data["summary"]["readings"][ADDRESS])
    values = payload["series"][ADDRESS]["values"]

    assert values[3] is None
    assert values[7] is None
    assert values[0] == 0.0
    assert len(values) == len(array)
    # Re-serialisable the way the chart endpoint serialises it.
    json.dumps(payload, allow_nan=False)


@pytest.mark.unit
async def test_retention_prunes_with_the_configured_window(tmp_path, monkeypatch):
    """A successful save is followed by the per-channel retention sweep."""
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
        {ADDRESS: _oversized()},
        overrides={"control_system.channel_read_artifact_retention": 7},
    )

    assert calls == [(ADDRESS, 7)]


@pytest.mark.unit
async def test_retention_failure_does_not_cost_the_handle(tmp_path, monkeypatch):
    """Housekeeping trouble must not strip a working handle off a saved artifact."""
    monkeypatch.setattr(
        "osprey.stores.artifact_store.ArtifactStore.prune_channel_readings",
        MagicMock(side_effect=RuntimeError("index locked")),
    )

    data = await _read(tmp_path, monkeypatch, {ADDRESS: _oversized()})

    entry = data["summary"]["readings"][ADDRESS]
    assert entry["artifact_id"]
    assert "artifact_error" not in entry


@pytest.mark.unit
async def test_store_failure_degrades_but_read_stays_successful(tmp_path, monkeypatch):
    """A read the machine answered is never reported as failed."""
    monkeypatch.setattr(
        "osprey.stores.artifact_store.ArtifactStore.save_data",
        MagicMock(side_effect=OSError("disk full")),
    )

    data = await _read(tmp_path, monkeypatch, {ADDRESS: _oversized()})

    assert data["status"] == "success"
    entry = data["summary"]["readings"][ADDRESS]
    assert entry["value_withheld"] is True
    assert entry["element_count"] == 5000
    assert "artifact_id" not in entry
    assert "data_file" not in entry
    assert entry["artifact_error"] == "OSError: disk full"


@pytest.mark.unit
async def test_multi_dimensional_value_takes_no_series_path(tmp_path, monkeypatch):
    """2-D+ goes down the image path instead - never a series the chart would render."""
    frame = np.arange(64 * 48, dtype=np.uint16).reshape(48, 64)
    data = await _read(tmp_path, monkeypatch, {"CAM:IMAGE": frame})

    entry = data["summary"]["readings"]["CAM:IMAGE"]
    stored = get_artifact_store().get_entry(entry["artifact_id"])
    assert stored is not None
    assert stored.metadata["data_type"] != "timeseries"
    assert not entry["data_file"].endswith(".json")
