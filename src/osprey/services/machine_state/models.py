"""Data models for machine state snapshots."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ChannelDefinition:
    """A channel parsed from the machine state channels JSON file."""

    address: str
    label: str = ""
    group: str = ""


@dataclass
class ChannelResult:
    """Result of reading a single channel."""

    address: str
    ok: bool
    value: Any | None
    units: str = ""
    timestamp: datetime | None = None
    label: str = ""
    group: str = ""
    error: str | None = None


@dataclass
class MachineStateSnapshot:
    """Complete snapshot of the machine state."""

    snapshot_time: datetime
    channels: dict[str, ChannelResult] = field(default_factory=dict)
    channels_read: int = 0
    channels_failed: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Fraction of channels successfully read (0.0–1.0)."""
        total = self.channels_read + self.channels_failed
        if total == 0:
            return 1.0
        return self.channels_read / total

    def failed_channels(self) -> list[str]:
        """Return addresses of channels that failed to read."""
        return [addr for addr, result in self.channels.items() if not result.ok]
