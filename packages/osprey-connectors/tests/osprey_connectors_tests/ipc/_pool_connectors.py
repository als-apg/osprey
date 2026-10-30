"""Mock connectors with one misbehaviour each, for the connector-host pool tests.

They are loaded *inside* connector-host children by dotted path, so each one
has to be importable from a fresh interpreter; the pool tests put the repository
root on ``PYTHONPATH`` for exactly that.
"""

import asyncio
import os
import time

from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors.errors import ChannelUnreachableError, ConnectorUnavailableError

#: Channels whose reads and writes take far longer than any test waits.
SLOW_PREFIX = "SLOW:"

#: Channels the connector itself cannot reach: a read raises ChannelUnreachableError.
GONE_PREFIX = "GONE:"

#: Channels whose read finds the connector's own transport gone: a read raises
#: ConnectorUnavailableError.
LOST_PREFIX = "LOST:"

#: Channels whose read hits a bug in the connector: a read raises an exception
#: outside the IPC contract.
BUG_PREFIX = "BUG:"

#: Channels whose read blocks the child's event loop outright, so the child
#: answers nothing at all from then on: a wedged child.
WEDGE_PREFIX = "WEDGE:"

#: When set, every write the connector actually performs appends a line here,
#: so a test can count what reached the "IOC".
WRITE_LOG_ENV = "OSPREY_POOL_TEST_WRITE_LOG"


class SlowInProcessConnector(VAInProcessConnector):
    """The mock, except ``SLOW:`` channels hang, ``GONE:`` channels are
    unreachable (:class:`ChannelUnreachableError`), ``LOST:`` channels find the
    connector unavailable, ``BUG:`` channels raise a connector bug, ``WEDGE:``
    channels wedge the child, and every write is logged."""

    async def read_channel(self, channel_address, timeout=None):
        if channel_address.startswith(WEDGE_PREFIX):
            time.sleep(3600)
        if channel_address.startswith(GONE_PREFIX):
            raise ChannelUnreachableError(
                channel_address, message=f"pool test: {channel_address} is unreachable"
            )
        if channel_address.startswith(LOST_PREFIX):
            raise ConnectorUnavailableError("pool test: the connector's transport is gone")
        if channel_address.startswith(BUG_PREFIX):
            raise KeyError(channel_address)
        if channel_address.startswith(SLOW_PREFIX):
            await asyncio.sleep(3600)
        return await super().read_channel(channel_address, timeout)

    async def write_channel(self, channel_address, value, timeout=None, confirm=None):
        log = os.environ.get(WRITE_LOG_ENV)
        if log:
            with open(log, "a", encoding="utf-8") as handle:
                handle.write(f"{channel_address} {value}\n")
        if channel_address.startswith(SLOW_PREFIX):
            await asyncio.sleep(3600)
        return await super().write_channel(channel_address, value, timeout, confirm)


class FailingConnector(VAInProcessConnector):
    """A connector whose ``connect()`` cannot reach its control system."""

    async def connect(self, config):  # noqa: ARG002 - the base signature
        raise ConnectionError("pool test: the gateway refused the connection")


class HangingConnector(VAInProcessConnector):
    """A connector whose ``connect()`` never returns."""

    async def connect(self, config):  # noqa: ARG002 - the base signature
        await asyncio.sleep(3600)


class SlowStartConnector(VAInProcessConnector):
    """A connector whose ``connect()`` takes a second, then succeeds."""

    async def connect(self, config):
        await asyncio.sleep(1.0)
        await super().connect(config)


class TimingOutConnector(VAInProcessConnector):
    """A connector whose ``connect()`` times out reaching its control system."""

    async def connect(self, config):  # noqa: ARG002 - the base signature
        raise TimeoutError("pool test: the gateway did not answer in time")


class ExitingConnector(VAInProcessConnector):
    """A connector whose ``connect()`` ends the child process outright."""

    async def connect(self, config):  # noqa: ARG002 - the base signature
        os._exit(3)


class ExitingOnReadConnector(VAInProcessConnector):
    """A connector that starts normally and ends the child process on its first read."""

    async def read_channel(self, channel_address, timeout=None):  # noqa: ARG002 - the base signature
        os._exit(4)
