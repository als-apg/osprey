"""Machine state reader — bulk-reads a predefined set of control system channels."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osprey.connectors.control_system.base import ControlSystemConnector
from osprey.utils.logger import get_logger

from .models import ChannelDefinition, ChannelResult, MachineStateSnapshot

logger = get_logger("machine_state_reader")

# Reserved metadata fields (underscore prefix) — skipped during parsing
METADATA_PREFIXED = "_"


class MachineStateReader:
    """Reads all channels defined in a JSON file and returns a snapshot.

    Usage::

        reader = MachineStateReader(channels_path, connector)
        snapshot = await reader.read()

    Or from Osprey config::

        reader = MachineStateReader.from_config(connector)
        if reader:
            snapshot = await reader.read()
    """

    def __init__(self, channels_path: str, connector: ControlSystemConnector) -> None:
        self._channels_path = channels_path
        self._connector = connector

    @classmethod
    def from_config(cls, connector: ControlSystemConnector) -> "MachineStateReader | None":
        """Create a reader from Osprey configuration.

        Returns None if the machine_state section is not configured.
        Follows the ``LimitsValidator.from_config()`` pattern.
        """
        try:
            from osprey.utils.config import get_config_value

            channels_file = get_config_value("machine_state.channels_file", None)
            if not channels_file or not isinstance(channels_file, str):
                logger.debug("machine_state.channels_file not configured — reader disabled")
                return None

            # Resolve relative paths against project_root
            path = Path(channels_file)
            if not path.is_absolute():
                project_root = get_config_value("project_root", None)
                if project_root:
                    channels_file = str(Path(project_root) / channels_file)
                    logger.debug(f"Resolved channels file path: {channels_file}")

            return cls(channels_file, connector)
        except (FileNotFoundError, KeyError, RuntimeError) as e:
            logger.debug(f"MachineStateReader not initialized (config unavailable): {e}")
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def read(self) -> MachineStateSnapshot:
        """Read all configured channels and return a snapshot.

        Never raises for partial failures — failed channels are captured
        as ``ok=False`` entries in the snapshot.
        """
        channel_defs, warnings = self._load_channel_defs()

        if not channel_defs:
            return MachineStateSnapshot(
                snapshot_time=datetime.now(tz=UTC),
                warnings=warnings,
            )

        addresses = list(channel_defs.keys())

        try:
            results = await self._connector.read_multiple_channels(addresses)
        except Exception as exc:
            logger.error(f"Bulk channel read failed: {exc}")
            # Every channel is treated as failed
            channels: dict[str, ChannelResult] = {}
            for addr, defn in channel_defs.items():
                channels[addr] = ChannelResult(
                    address=addr,
                    ok=False,
                    value=None,
                    label=defn.label,
                    group=defn.group,
                    error=str(exc),
                )
            return MachineStateSnapshot(
                snapshot_time=datetime.now(tz=UTC),
                channels=channels,
                channels_failed=len(channels),
                warnings=warnings,
            )

        # Build snapshot from connector results
        channels = {}
        ok_count = 0
        fail_count = 0

        for addr, defn in channel_defs.items():
            cv = results.get(addr)
            if cv is not None:
                value = _normalize_value(cv.value)
                channels[addr] = ChannelResult(
                    address=addr,
                    ok=True,
                    value=value,
                    units=cv.metadata.units if cv.metadata else "",
                    timestamp=cv.timestamp,
                    label=defn.label,
                    group=defn.group,
                )
                ok_count += 1
            else:
                channels[addr] = ChannelResult(
                    address=addr,
                    ok=False,
                    value=None,
                    label=defn.label,
                    group=defn.group,
                    error="Channel not returned by connector",
                )
                fail_count += 1

        return MachineStateSnapshot(
            snapshot_time=datetime.now(tz=UTC),
            channels=channels,
            channels_read=ok_count,
            channels_failed=fail_count,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_channel_defs(self) -> tuple[dict[str, ChannelDefinition], list[str]]:
        """Parse the channel definitions JSON file.

        Returns:
            (channel_defs, warnings) — ``channel_defs`` maps address → ChannelDefinition.
        """
        warnings: list[str] = []
        path = Path(self._channels_path)

        if not path.exists():
            raise FileNotFoundError(f"Channel definitions file not found: {self._channels_path}")

        with open(path) as f:
            raw: Any = json.load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Channel definitions must be a JSON object, got {type(raw).__name__}")

        channel_defs: dict[str, ChannelDefinition] = {}
        for key, value in raw.items():
            # Skip metadata keys
            if key.startswith(METADATA_PREFIXED):
                logger.debug(f"Skipping metadata key: {key}")
                continue

            if not isinstance(value, dict):
                msg = (
                    f"Skipping non-dict entry '{key}': expected object, got {type(value).__name__}"
                )
                logger.warning(msg)
                warnings.append(msg)
                continue

            channel_defs[key] = ChannelDefinition(
                address=key,
                label=value.get("label", ""),
                group=value.get("group", ""),
            )

        logger.info(f"Loaded {len(channel_defs)} channel definitions from {self._channels_path}")
        return channel_defs, warnings


def _normalize_value(value: Any) -> Any:
    """Convert numpy arrays to Python lists for JSON-safe snapshots."""
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
