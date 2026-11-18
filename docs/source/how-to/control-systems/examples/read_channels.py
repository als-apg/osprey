"""Read channels from an EPICS control system, with no OSPREY deployment."""

import asyncio

from osprey_connectors.config_sections import ControlSystemSection
from osprey_connectors.factory import ConnectorFactory

CONFIG = {
    "type": "epics",
    "connector": {
        "epics": {
            "gateways": {"read_only": {"address": "cagw.facility.edu", "port": 5064}},
            "timeout_s": 5.0,
        }
    },
}


async def main() -> None:
    connector = await ConnectorFactory.create_control_system_connector(
        ControlSystemSection.read(CONFIG)
    )
    try:
        reading = await connector.read_channel("DEMO:READBACK")
        print(reading.value, reading.metadata.units)

        batch = await connector.read_multiple_channels(["DEMO:SETPOINT", "DEMO:MODE"])
        for address, channel in batch.raise_for_failures().values.items():
            print(address, channel.value, channel.enum_label)
    finally:
        await connector.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
