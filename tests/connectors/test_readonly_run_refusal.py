"""A readonly sandbox run is refused at the connector, whatever the deployment posture.

``control_system.writes_enabled`` is the launch-time deployment posture. The
python executor additionally exports ``OSPREY_EXECUTION_MODE`` into its
sandbox subprocess; when that says ``readonly`` the connector base class
refuses every write before any connector-specific code runs — so
``osprey.runtime.write_channel`` and direct ``connector.write_channel`` calls
alike are blocked in a readonly run even on a write-enabled deployment.
"""

from typing import Any

import pytest

from osprey.connectors.control_system.base import (
    ChannelWriteResult,
    ControlSystemConnector,
)


class _WriteEnabledConnector(ControlSystemConnector):
    """Concrete connector whose deployment posture says writes are on."""

    def __init__(self):
        self.writes: list[tuple[str, Any]] = []

    async def connect(self, config: dict[str, Any]) -> None: ...
    async def disconnect(self) -> None: ...

    async def read_channel(self, channel_address: str, timeout: float | None = None):
        raise NotImplementedError

    async def read_multiple_channels(self, channel_addresses, timeout=None):
        raise NotImplementedError

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,
        verification_level: str = "callback",
        tolerance: float | None = None,
    ) -> ChannelWriteResult:
        self.writes.append((channel_address, value))
        return ChannelWriteResult(
            channel_address=channel_address, value_written=value, success=True
        )

    async def write_multiple_channels(self, operations, timeout=None):
        return [await self.write_channel(addr, val) for addr, val in operations]

    async def subscribe(self, channel_address, callback):
        raise NotImplementedError

    async def unsubscribe(self, channel_address):
        raise NotImplementedError

    async def get_metadata(self, channel_address):
        raise NotImplementedError

    async def validate_channel(self, channel_address) -> bool:
        return True


@pytest.fixture
def writes_enabled_deployment(monkeypatch):
    monkeypatch.setattr(
        "osprey_connectors.config.get_config_value",
        lambda key, default=None: True if key == "control_system.writes_enabled" else default,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_readonly_run_refuses_write(monkeypatch, writes_enabled_deployment):
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    connector = _WriteEnabledConnector()

    result = await connector.write_channel("SR:MAG:QF:01:CURRENT:SP", 150.0)

    assert result.blocked is True
    assert result.success is False
    assert result.refusal_reason == "WRITES_DISABLED"
    assert "readonly" in result.error_message
    assert connector.writes == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_readonly_run_refuses_multi_write(monkeypatch, writes_enabled_deployment):
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    connector = _WriteEnabledConnector()

    results = await connector.write_multiple_channels([("A:SP", 1.0), ("B:SP", 2.0)])

    assert all(r.blocked for r in results)
    assert connector.writes == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_readonly_refusal_message_does_not_blame_deployment(
    monkeypatch, writes_enabled_deployment
):
    """The operator-facing text must not send anyone to flip writes_enabled —
    the deployment allows writes; this *run* was declared readonly."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    connector = _WriteEnabledConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert "writes_enabled" not in result.error_message
    assert "readwrite" in result.error_message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_readwrite_run_passes_through(monkeypatch, writes_enabled_deployment):
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readwrite")
    connector = _WriteEnabledConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert result.success is True
    assert connector.writes == [("A:SP", 1.0)]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_mode_var_means_not_a_sandbox_run(monkeypatch, writes_enabled_deployment):
    """Outside the sandbox (e.g. the controls MCP server) the variable is unset
    and the deployment posture alone decides."""
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    connector = _WriteEnabledConnector()

    result = await connector.write_channel("A:SP", 1.0)

    assert result.success is True
