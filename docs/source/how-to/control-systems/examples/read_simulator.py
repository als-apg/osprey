"""Read a channel from the simulator in process, in a built OSPREY deployment."""

import asyncio

from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors

# The simulator in process: serves the addresses of the built facility file.
CONFIG = {
    "type": "virtual_accelerator",
    "connector": {
        "virtual_accelerator": {
            "serving": "in_process",
            "response_delay_ms": 10,
        }
    },
}


async def main() -> None:
    register_builtin_connectors()  # registers the built-in type names; idempotent
    connector = await ConnectorFactory.create_control_system_connector(CONFIG)
    try:
        channel_value = await connector.read_channel("SR:DIAG:DCCT:01:CURRENT:RB")
        print(f"Value: {channel_value.value} {channel_value.metadata.units}")
    finally:
        await connector.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
