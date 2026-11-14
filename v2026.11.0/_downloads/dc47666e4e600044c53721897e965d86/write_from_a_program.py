"""A program writes under a bounded write authorization that a person approves.

Run from the deployment, so ``config.yml`` is the deployment's own. The
use-connectors guide shows this file one step at a time.
"""

import asyncio
import time

from osprey_connectors import write_authorization as wa
from osprey_connectors.config import default_config_path, get_config_value
from osprey_connectors.control_system import WriteOutcome
from osprey_connectors.ipc import ConnectorHostPool


def ask() -> wa.WriteRequest:
    # docs: ask begins
    request = wa.write_request(
        "standin",
        {"SR:CORRECTOR:H1": (-0.5, 0.5), "SR:CORRECTOR:H2": (-0.5, 0.5)},
        steps=41,  # 40 evaluations + apply_best
        expires_at_ns=time.time_ns() + 2 * 3600 * 10**9,
        lease_s=30.0,
        label="injector tuning 7",
        completion="apply_best",
    )
    # docs: ask ends
    return request  # noqa: RET504 - the page shows the assignment


def wait_for_approval(request: wa.WriteRequest) -> wa.Grant:
    # docs: wait begins
    while (grant := wa.approved_grant(request.id)) is None:
        time.sleep(2)
    wa.withdraw_request(request.id)
    wa.start_lease(grant.id)  # once, by the process that renews it
    # docs: wait ends
    return grant


async def write(pool: ConnectorHostPool, grant: wa.Grant) -> None:
    # docs: write begins
    connector = await pool.connector("standin", owner=grant.owner, authorization=grant.id)
    results = await connector.write_multiple_channels(
        [("SR:CORRECTOR:H1", 0.12), ("SR:CORRECTOR:H2", -0.03)], confirm=True
    )
    # docs: write ends
    # docs: outcome begins
    for result in results:
        if result.outcome is WriteOutcome.CONFIRMED:
            print(result.channel_address, "now holds", result.value_written)
        else:
            print(result.channel_address, result.outcome.value, result.verdict_reason)
    # docs: outcome ends
    # docs: renew begins
    if not wa.renew_lease(grant.id):  # False: the lease has lapsed, stop
        return
    # docs: renew ends


async def main() -> None:
    request = ask()
    grant = await asyncio.to_thread(wait_for_approval, request)
    pool = ConnectorHostPool(
        get_config_value("control_system", {}), config_file=default_config_path()
    )
    try:
        await write(pool, grant)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
