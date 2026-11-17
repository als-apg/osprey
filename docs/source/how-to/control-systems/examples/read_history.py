"""Read archived history from the EPICS Archiver Appliance, with no OSPREY deployment."""

import asyncio
from datetime import UTC, datetime, timedelta

from osprey_connectors.factory import ConnectorFactory

ARCHIVER = {
    "type": "epics_archiver",
    "settings": {"url": "https://archiver.facility.edu:8443", "timeout_s": 30},
}


async def main() -> None:
    archiver = await ConnectorFactory.create_archiver_connector(ARCHIVER)
    try:
        end = datetime.now(UTC)
        frame = await archiver.get_data(
            ["DEMO:READBACK", "DEMO:SETPOINT"],
            start_at=end - timedelta(hours=1),
            end_at=end,
        )
        print(frame)
    finally:
        await archiver.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
