"""Channel Access Tools for State Identification Agent.

Provides tools for reading control system channel values to assess
machine state for optimization readiness.

Uses the existing ConnectorFactory which automatically selects the
appropriate connector based on configuration:
- control_system.type: "mock" -> MockConnector (synthetic data)
- control_system.type: "epics" -> EPICSConnector (real control system)

No additional mocking needed - the MockConnector already provides
realistic synthetic data for any channel name.
"""

from typing import Any

from langchain_core.tools import tool

from osprey.connectors.factory import ConnectorFactory
from osprey.utils.logger import get_logger

logger = get_logger("xopt_optimizer")

# Module-level connector cache for reuse within a session
_connector_cache: dict[str, Any] = {}


async def _get_connector():
    """Get or create control system connector.

    Uses ConnectorFactory which reads from config to determine
    connector type (mock, epics, etc.).

    Returns:
        Connected ControlSystemConnector instance
    """
    if "control_system" not in _connector_cache:
        connector = await ConnectorFactory.create_control_system_connector()
        _connector_cache["control_system"] = connector
    return _connector_cache["control_system"]


def create_read_channels_tool():
    """Create a tool for reading control system channel values.

    Returns:
        LangChain tool function for reading channels
    """

    @tool
    async def read_channel_values(channel_names: str) -> str:
        """Read current values from control system channels.

        Use this tool to check the current state of machine parameters
        that are relevant for determining optimization readiness.

        Args:
            channel_names: Comma-separated list of channel names to read.
                          Example: "BEAM:CURRENT,VACUUM:PRESSURE,SAFETY:INTERLOCK"

        Returns:
            Formatted string with channel values and metadata, or error message
        """
        # Parse channel names
        channels = [name.strip() for name in channel_names.split(",") if name.strip()]

        if not channels:
            return "Error: No channel names provided. Provide comma-separated channel names."

        try:
            connector = await _get_connector()

            results = []
            for channel in channels:
                try:
                    value = await connector.read_channel(channel)
                    # Format the result with relevant metadata
                    result_line = f"{channel}: {value.value}"
                    if value.metadata:
                        if value.metadata.units:
                            result_line += f" {value.metadata.units}"
                        if value.metadata.severity:
                            result_line += f" (severity: {value.metadata.severity})"
                    results.append(result_line)
                except Exception as e:
                    results.append(f"{channel}: ERROR - {e}")

            logger.debug(f"Read {len(channels)} channels for state assessment")
            return "\n".join(results)

        except Exception as e:
            logger.error(f"Failed to read channels: {e}")
            return f"Error connecting to control system: {e}"

    return read_channel_values


def create_channel_access_tools() -> list[Any]:
    """Create all channel access tools.

    Returns:
        List of LangChain tools [read_channel_values]
    """
    return [
        create_read_channels_tool(),
    ]


def clear_connector_cache():
    """Clear the connector cache.

    Useful for testing to ensure fresh connector creation.
    """
    _connector_cache.clear()
