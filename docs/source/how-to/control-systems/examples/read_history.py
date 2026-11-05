"""Read archived history from the EPICS Archiver Appliance, with no OSPREY deployment."""

import asyncio
from datetime import UTC, datetime, timedelta

from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors

ARCHIVER = {
    "type": "epics_archiver",
    "settings": {"url": "https://archiver.facility.edu:8443", "timeout_s": 30},
}


async def main() -> None:
    register_builtin_connectors()
    archiver = await ConnectorFactory.create_archiver_connector(ARCHIVER)
    try:
        end = datetime.now(UTC)
        frame = await archiver.get_data(
            channels=["DEMO:READBACK", "DEMO:SETPOINT"],
            start_date=end - timedelta(hours=1),
            end_date=end,
        )
        print(frame)
    finally:
        await archiver.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
