"""A write from a script with no OSPREY deployment is refused before anything is sent."""

import asyncio

from osprey_connectors.errors import ChannelWriteBlockedError
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
    register_builtin_connectors()
    connector = await ConnectorFactory.create_control_system_connector(CONFIG)
    try:
        await connector.write_channel_checked("DEMO:SETPOINT", 2.0)
    except ChannelWriteBlockedError as refused:
        print(refused.reason)
        print(refused)
    finally:
        await connector.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
