"""Search a ChannelFinder directory and read one channel's record, with no OSPREY deployment."""

import asyncio

from osprey_connectors.directory import DirectoryQuery, DirectoryRecord
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.service_answers import BackendSpec

DIRECTORY = {"url": "https://channelfinder.facility.edu:8443/ChannelFinder", "timeout_s": 10}

#: Seconds each call may spend.
DEADLINE_S = 10.0


async def main() -> None:
    spec = BackendSpec(
        kind="directory", type="channelfinder", target="live", settings=DIRECTORY, rehearsal=False
    )
    directory = await ConnectorFactory.create_service(spec)

    page = await directory.search(DirectoryQuery(words="SR:BPM*"), DEADLINE_S)
    print(f"{page.outcome}: {len(page.rows)} of {page.total}")
    for row in page.rows:
        print(f"  {row.name:<10} {row.channel_type:<3} {row.server:<7} {row.description}")

    record = await directory.describe("SR:BPM1:X", DEADLINE_S)
    if isinstance(record, DirectoryRecord):
        print(f"{record.row.name}: {record.row.status} on {record.row.host}")
    else:
        print(record)  # Outcome.NOT_LISTED, or Unavailable(reason=...)


if __name__ == "__main__":
    asyncio.run(main())
