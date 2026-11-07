"""Read channels from an EPICS control system, with no OSPREY deployment."""

import asyncio

from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors

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
    register_builtin_connectors()  # registers the built-in type names; idempotent
    connector = await ConnectorFactory.create_control_system_connector(CONFIG)
    try:
        reading = await connector.read_channel("DEMO:READBACK")
        print(reading.value, reading.metadata.units)

        values = await connector.read_multiple_channels(["DEMO:SETPOINT", "DEMO:MODE"])
        for address, channel in values.items():
            print(address, channel.value, channel.metadata.enum_label)
    finally:
        await connector.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
