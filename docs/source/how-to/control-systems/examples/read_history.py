"""Read archived history from the EPICS Archiver Appliance, with no OSPREY deployment."""

import asyncio
from datetime import UTC, datetime, timedelta

from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.service_answers import BackendSpec

ARCHIVER = {"url": "https://archiver.facility.edu:8443", "timeout_s": 30}


async def main() -> None:
    spec = BackendSpec(
        kind="archive", type="epics_archiver", target="live", settings=ARCHIVER, rehearsal=False
    )
    archive = await ConnectorFactory.create_service(spec)
    try:
        end = datetime.now(UTC)
        frame = await archive.history(
            ["DEMO:READBACK", "DEMO:SETPOINT"],
            start_at=end - timedelta(hours=1),
            end_at=end,
        )
        print(frame)  # or Unavailable(cause=..., reason=...) when the appliance gave no answer
    finally:
        await archive.aclose()


if __name__ == "__main__":
    asyncio.run(main())
