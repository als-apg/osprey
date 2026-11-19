"""Search a ChannelFinder directory and read one channel's record, with no OSPREY deployment."""

import asyncio

from osprey_connectors.directory import DirectoryDetail, DirectoryQuery
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.service_answers import BackendSpec, Unavailable

DIRECTORY = {"url": "https://channelfinder.facility.edu:8443/ChannelFinder", "timeout_s": 10}

#: Seconds each call may spend.
TIMEOUT_S = 10.0


async def main() -> None:
    spec = BackendSpec(
        kind="directory", type="channelfinder", target="live", settings=DIRECTORY, rehearsal=False
    )
    directory = await ConnectorFactory.create_service(spec)

    page = await directory.search(DirectoryQuery(words="SR:BPM*"), timeout_s=TIMEOUT_S)
    if isinstance(page, Unavailable):
        print(f"{page.cause}: {page.reason}")
        return
    print(f"{page.outcome}: {len(page.rows)} of {page.total}")
    for row in page.rows:
        print(f"  {row.address:<10} {row.channel_type:<3} {row.server:<7} {row.description}")

    record = await directory.describe("SR:BPM1:X", timeout_s=TIMEOUT_S)
    if isinstance(record, DirectoryDetail):
        print(f"{record.entry.address}: {record.entry.status} on {record.entry.host}")
    else:
        print(record)  # NotListed(address=...), or Unavailable(cause=..., reason=...)


if __name__ == "__main__":
    asyncio.run(main())
