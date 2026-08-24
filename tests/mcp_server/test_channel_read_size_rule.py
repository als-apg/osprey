"""Tests for the channel_read size rule.

Covers the protocol-independent rule that decides which read values are inlined
and which are reported as a summary: element counts come from ``value.size`` on
numpy arrays only, str/bytes/scalars always inline, arrays over the per-value
threshold or over the call's aggregate budget are withheld with an
``artifact_reason``, statistics are numeric-dtype gated, and every address that
failed to read is reported with its error text.
"""

from contextlib import nullcontext
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from osprey.mcp_server.control_system.server_context import initialize_server_context
from tests.mcp_server.conftest import extract_response_dict, get_tool_fn


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


def _get_channel_read():
    from osprey.mcp_server.control_system.tools.channel_read import channel_read

    return get_tool_fn(channel_read)


class _MockConnector:
    """Connector stub feeding synthetic values through the real tool body.

    Values are keyed by address; an ``Exception`` instance is raised instead of
    returned, so one batch can mix successes and failures.
    """

    def __init__(self, values: dict):
        self._values = values
        self.calls: list[str] = []

    async def read_channel(self, address: str):
        self.calls.append(address)
        value = self._values[address]
        if isinstance(value, Exception):
            raise value
        return _make_channel_value(value)


def _threshold_patch(inline_max: int | None):
    """Patch only the inline-threshold config key, delegating everything else."""
    if inline_max is None:
        return nullcontext()

    from osprey.utils import config as config_module

    real = config_module.get_config_value

    def fake(path, default=None, *args, **kwargs):
        if path == "control_system.read_inline_max_elements":
            return inline_max
        return real(path, default, *args, **kwargs)

    return patch("osprey.utils.config.get_config_value", side_effect=fake)


async def _read(tmp_path, monkeypatch, values, channels=None, inline_max=None, **kwargs):
    """Run the real channel_read tool body against a mock connector."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("control_system:\n  type: mock\n")
    initialize_server_context()

    connector = _MockConnector(values)
    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector",
        new_callable=AsyncMock,
        return_value=connector,
    ):
        fn = _get_channel_read()
        with _threshold_patch(inline_max):
            result = await fn(channels=channels or list(values), **kwargs)
    return extract_response_dict(result), connector


@pytest.mark.unit
async def test_scalars_and_text_always_inline(tmp_path, monkeypatch):
    """Scalars, a 10k-character string and bytes inline regardless of length."""
    long_text = "x" * 10_000
    values = {
        "SR:CURRENT:RB": 500.2,
        "SR:MODE:RB": long_text,
        "SR:BLOB:RB": b"\x00" * 5000,
    }
    data, _ = await _read(tmp_path, monkeypatch, values)

    readings = data["summary"]["readings"]
    assert readings["SR:CURRENT:RB"]["value"] == 500.2
    assert readings["SR:MODE:RB"]["value"] == long_text
    assert "value" in readings["SR:BLOB:RB"]
    for entry in readings.values():
        assert "artifact_reason" not in entry
        assert "value_withheld" not in entry


@pytest.mark.unit
async def test_small_array_inlines_as_json_list(tmp_path, monkeypatch):
    """A small array comes back as an actual JSON list, never stringified."""
    array = np.arange(10, dtype=np.float64)
    data, _ = await _read(tmp_path, monkeypatch, {"SR:BPM:X": array})

    value = data["summary"]["readings"]["SR:BPM:X"]["value"]
    assert isinstance(value, list), f"array was not inlined as a list: {value!r}"
    assert value == [float(x) for x in range(10)]


@pytest.mark.unit
async def test_oversized_array_is_withheld_with_summary(tmp_path, monkeypatch):
    """An array over the per-value threshold gets a summary, not raw numbers."""
    frame = np.arange(64 * 48, dtype=np.uint16).reshape(48, 64)
    data, _ = await _read(tmp_path, monkeypatch, {"CAM:IMAGE": frame})

    entry = data["summary"]["readings"]["CAM:IMAGE"]
    assert "value" not in entry
    assert entry["value_withheld"] is True
    assert entry["artifact_reason"] == "per_value_threshold"
    assert entry["shape"] == [48, 64]
    assert entry["dtype"] == "uint16"
    assert entry["element_count"] == 64 * 48
    assert entry["min"] == 0
    assert entry["max"] == 64 * 48 - 1
    assert entry["mean"] == pytest.approx(float(frame.mean()))
    # Metadata still travels with a withheld value.
    assert entry["units"] == "counts"
    # The withheld value is retrievable, never fabricated: a handle, not numbers.
    assert entry["artifact_id"]
    assert entry["artifact_note"]

    withheld = data["access_details"]["values_withheld"]
    assert withheld["channels"] == ["CAM:IMAGE"]
    assert withheld["inline_max_elements"] == 2000
    assert withheld["aggregate_max_elements"] == 8000
    assert "per_value_threshold" in withheld["artifact_reason"]


@pytest.mark.unit
async def test_aggregate_budget_overflow_is_request_ordered(tmp_path, monkeypatch):
    """Under-threshold arrays that together bust the 4x budget are withheld in order."""
    # Threshold 250 -> aggregate budget 1000. Each array is 240 elements, well
    # under the per-value threshold; four of them fit (960), the fifth does not.
    channels = [f"SR:W{i}" for i in range(5)]
    values = {addr: np.arange(240, dtype=np.float64) for addr in channels}

    data, _ = await _read(tmp_path, monkeypatch, values, channels=channels, inline_max=250)

    readings = data["summary"]["readings"]
    for addr in channels[:4]:
        assert isinstance(readings[addr]["value"], list), f"{addr} should have inlined"
    last = readings[channels[4]]
    assert "value" not in last
    assert last["artifact_reason"] == "aggregate_budget"
    assert last["element_count"] == 240
    assert last["shape"] == [240]
    assert data["access_details"]["values_withheld"]["channels"] == [channels[4]]


@pytest.mark.unit
async def test_aggregate_budget_reason_is_distinct_from_per_value(tmp_path, monkeypatch):
    """A batch can carry both reasons, and each entry names its own."""
    channels = ["SR:BIG", "SR:S1", "SR:S2"]
    values = {
        "SR:BIG": np.zeros(900, dtype=np.float64),  # over threshold 250
        "SR:S1": np.zeros(240, dtype=np.float64),
        "SR:S2": np.zeros(240, dtype=np.float64),
    }

    data, _ = await _read(tmp_path, monkeypatch, values, channels=channels, inline_max=250)

    readings = data["summary"]["readings"]
    assert readings["SR:BIG"]["artifact_reason"] == "per_value_threshold"
    # An over-threshold value does not eat the shared budget, so both small
    # arrays still inline.
    assert isinstance(readings["SR:S1"]["value"], list)
    assert isinstance(readings["SR:S2"]["value"], list)


@pytest.mark.unit
async def test_non_numeric_oversized_array_has_no_stats(tmp_path, monkeypatch):
    """A withheld string array reports shape/dtype/count but no invented extremes."""
    array = np.array(["a"] * 3000)
    data, _ = await _read(tmp_path, monkeypatch, {"SR:LABELS": array})

    entry = data["summary"]["readings"]["SR:LABELS"]
    assert entry["value_withheld"] is True
    assert entry["element_count"] == 3000
    assert entry["shape"] == [3000]
    assert entry["dtype"].startswith("<U")
    for stat in ("min", "max", "mean"):
        assert stat not in entry


@pytest.mark.unit
async def test_failed_address_reported_with_error_text(tmp_path, monkeypatch):
    """One bad address in a batch surfaces in channels_failed; the rest succeed."""
    values = {
        "SR:GOOD:RB": 1.5,
        "SR:BAD:RB": TimeoutError("PVA timeout after 5s"),
        "SR:ALSO_GOOD:RB": 2.5,
    }
    data, connector = await _read(tmp_path, monkeypatch, values, channels=list(values))

    assert data["status"] == "success"
    assert data["summary"]["channels_read"] == 2
    assert set(data["summary"]["readings"]) == {"SR:GOOD:RB", "SR:ALSO_GOOD:RB"}
    failed = data["summary"]["channels_failed"]
    assert set(failed) == {"SR:BAD:RB"}
    assert "PVA timeout after 5s" in failed["SR:BAD:RB"]
    assert "TimeoutError" in failed["SR:BAD:RB"]
    assert "1 failed" in data["description"]
    # One individual read per requested address.
    assert connector.calls == list(values)


@pytest.mark.unit
async def test_no_failures_means_no_channels_failed_key(tmp_path, monkeypatch):
    """The scalar envelope is unchanged when every channel answers."""
    data, _ = await _read(tmp_path, monkeypatch, {"SR:CURRENT:RB": 500.2})

    assert "channels_failed" not in data["summary"]
    assert "values_withheld" not in data["access_details"]


@pytest.mark.unit
async def test_configured_threshold_is_honored(tmp_path, monkeypatch):
    """A configured inline_max_elements decides what inlines."""
    array = np.arange(50, dtype=np.float64)

    data, _ = await _read(tmp_path, monkeypatch, {"SR:BPM:X": array}, inline_max=10)

    entry = data["summary"]["readings"]["SR:BPM:X"]
    assert "value" not in entry
    assert entry["artifact_reason"] == "per_value_threshold"
    assert data["access_details"]["values_withheld"]["inline_max_elements"] == 10

    # Same array, default threshold: inlined.
    data, _ = await _read(tmp_path, monkeypatch, {"SR:BPM:X": array})
    assert isinstance(data["summary"]["readings"]["SR:BPM:X"]["value"], list)
