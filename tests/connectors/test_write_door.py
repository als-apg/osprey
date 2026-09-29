"""The write door: a context flag a connector raises around its own put.

The raw-client guard asks :func:`door_is_open` whether a put reached it
through a connector. These tests pin the flag's lifetime: it is closed by
default, opens only inside :func:`open_door`, restores the prior value on
exit (including a nested exit and an exit by exception), stays private to
the thread that opened it, and travels into ``asyncio.to_thread``. The
base connector opens it around every subclass's own write, and only once
the write is allowed to run.

The last section holds the door's contract against the armed raw-put block:
a raw client put passes only inside the door — through a connector's write
and its ``asyncio.to_thread`` hop, or from a callback run inline in that put
(an accepted, documented bypass) — and is refused from the main thread, a
user thread, a fresh thread started during an in-flight connector put, and
a ``run_in_executor`` hop. The client is a fake module registered in
``sys.modules`` under a name no installed library uses; the fixture from
:mod:`tests.runtime._patch_restore` restores whatever the block patched.
"""

import asyncio
import sys
import textwrap
import threading
from types import ModuleType

import pytest

from osprey.runtime import raw_put_block
from osprey_connectors.control_system.base import (
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
)
from osprey_connectors.control_system.write_door import door_is_open, open_door
from osprey_connectors.errors import RAW_CLIENT_WRITE_MARKER, ChannelWriteBlockedError
from tests.runtime._patch_restore import restore_patches  # noqa: F401


def test_door_is_closed_by_default():
    assert door_is_open() is False


def test_open_door_opens_and_closes():
    with open_door():
        assert door_is_open() is True
    assert door_is_open() is False


def test_nested_open_keeps_the_door_open_until_the_outer_exit():
    with open_door():
        with open_door():
            assert door_is_open() is True
        assert door_is_open() is True, "the inner exit closed the outer door"
    assert door_is_open() is False


def test_door_closes_when_the_body_raises():
    with pytest.raises(RuntimeError, match="boom"):
        with open_door():
            raise RuntimeError("boom")
    assert door_is_open() is False


def test_nested_exception_restores_the_outer_open_state():
    with open_door():
        with pytest.raises(ValueError):
            with open_door():
                raise ValueError
        assert door_is_open() is True
    assert door_is_open() is False


def test_door_opened_in_one_thread_is_closed_in_another():
    opened = threading.Event()
    release = threading.Event()
    seen: dict[str, bool] = {}

    def holder():
        with open_door():
            opened.set()
            release.wait(5)

    def observer():
        seen["other"] = door_is_open()

    t_hold = threading.Thread(target=holder)
    t_hold.start()
    assert opened.wait(5)
    t_obs = threading.Thread(target=observer)
    t_obs.start()
    t_obs.join(5)
    main_view = door_is_open()
    release.set()
    t_hold.join(5)

    assert seen["other"] is False
    assert main_view is False


def test_door_propagates_through_to_thread():
    async def main():
        with open_door():
            inside = await asyncio.to_thread(door_is_open)
        outside = await asyncio.to_thread(door_is_open)
        return inside, outside

    assert asyncio.run(main()) == (True, False)


def test_door_opened_inside_to_thread_does_not_leak_back():
    def worker():
        with open_door():
            pass
        cm = open_door()
        cm.__enter__()  # left open on purpose: the copied context is discarded
        return door_is_open()

    async def main():
        result = await asyncio.to_thread(worker)
        return result, door_is_open()

    assert asyncio.run(main()) == (True, False)


# ---------------------------------------------------------------------------
# The base connector opens the door around every subclass's own write
# ---------------------------------------------------------------------------


class _DoorProbe(ControlSystemConnector):
    """A connector whose writes record whether the door was open when they ran."""

    def __init__(self, *, armed: bool, fail: bool = False):
        super().__init__()
        self._armed = armed
        self._fail = fail
        self.seen: list[bool] = []

    @property
    def _writes_enabled(self) -> bool:
        return self._armed

    async def connect(self, config):
        pass

    async def disconnect(self):
        pass

    async def read_channel(self, addr, timeout=None):
        raise NotImplementedError

    async def write_channel(self, channel_address, value, **kwargs):
        self.seen.append(door_is_open())
        if self._fail:
            raise RuntimeError("put failed")
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=WriteOutcome.CONFIRMED,
        )

    async def write_multiple_channels(self, operations, **kwargs):
        self.seen.append(door_is_open())
        # The door reaches a connector's put hopped onto a worker thread.
        self.seen.append(await asyncio.to_thread(door_is_open))
        if self._fail:
            raise RuntimeError("batch failed")
        return [
            ChannelWriteResult(
                channel_address=addr,
                value_written=val,
                outcome=WriteOutcome.CONFIRMED,
            )
            for addr, val in operations
        ]

    async def read_multiple_channels(self, addrs, timeout=None):  # noqa: ARG002 - the control-system connector interface fixes this signature
        return {}

    async def subscribe(self, addr, cb):  # noqa: ARG002 - the control-system connector interface fixes this signature
        return "sub"

    async def unsubscribe(self, sub_id):
        pass

    async def get_metadata(self, addr):
        raise NotImplementedError

    async def validate_channel(self, addr):  # noqa: ARG002 - the control-system connector interface fixes this signature
        return True


def test_write_channel_runs_inside_the_open_door():
    probe = _DoorProbe(armed=True)
    result = asyncio.run(probe.write_channel("SR:PV", 1.0))
    assert result.outcome is WriteOutcome.CONFIRMED
    assert probe.seen == [True]
    assert door_is_open() is False


def test_write_multiple_channels_runs_inside_the_open_door():
    probe = _DoorProbe(armed=True)
    results = asyncio.run(probe.write_multiple_channels([("A", 1), ("B", 2)]))
    assert [r.outcome for r in results] == [WriteOutcome.CONFIRMED] * 2
    assert probe.seen == [True, True]
    assert door_is_open() is False


def test_a_disabled_connector_never_opens_the_door():
    probe = _DoorProbe(armed=False)

    async def main():
        single = await probe.write_channel("SR:PV", 1.0)
        batch = await probe.write_multiple_channels([("A", 1)])
        return single, batch, door_is_open()

    single, batch, after = asyncio.run(main())
    assert single.outcome is not WriteOutcome.CONFIRMED
    assert all(r.outcome is not WriteOutcome.CONFIRMED for r in batch)
    assert probe.seen == [], "the original write ran although writes are disabled"
    assert after is False


def test_the_door_closes_after_a_write_that_raises():
    probe = _DoorProbe(armed=True, fail=True)

    async def main():
        with pytest.raises(RuntimeError, match="put failed"):
            await probe.write_channel("SR:PV", 1.0)
        single_after = door_is_open()
        with pytest.raises(RuntimeError, match="batch failed"):
            await probe.write_multiple_channels([("A", 1)])
        return single_after, door_is_open()

    assert asyncio.run(main()) == (False, False)
    assert probe.seen == [True, True, True]


def test_the_door_closes_after_a_successful_write_in_the_same_task():
    probe = _DoorProbe(armed=True)

    async def main():
        await probe.write_channel("SR:PV", 1.0)
        after_single = door_is_open()
        await probe.write_multiple_channels([("A", 1)])
        return after_single, door_is_open()

    assert asyncio.run(main()) == (False, False)


# ---------------------------------------------------------------------------
# The door against the armed raw-put block
# ---------------------------------------------------------------------------

_FAKE_CLIENT = "osprey_fake_door_client"


@pytest.fixture
def armed_client(monkeypatch, restore_patches):  # noqa: F811
    """A fake client module whose put records the door it saw, block armed on it.

    The put asserts the door is open: the armed block must never let a put
    reach the client with the door closed.
    """
    module = ModuleType(_FAKE_CLIENT)
    module.__file__ = f"<{_FAKE_CLIENT}>"
    source = textwrap.dedent(
        """
        from osprey_connectors.control_system.write_door import door_is_open

        calls = []

        def caput(pvname, value):
            assert door_is_open(), "the armed block let a put through a closed door"
            calls.append((pvname, value))
            return True
        """
    )
    exec(compile(source, module.__file__, "exec"), vars(module))
    monkeypatch.setitem(sys.modules, _FAKE_CLIENT, module)
    restore_patches(module)
    raw_put_block.install(
        "armed",
        blocked_targets=((_FAKE_CLIENT, ("caput",)),),
        rpc_targets=(),
        refuse_rpc=False,
        marker=RAW_CLIENT_WRITE_MARKER,
        rpc_refusals={},
    )
    return module


def _assert_refused(error: BaseException | None, address: str) -> None:
    assert isinstance(error, ChannelWriteBlockedError), f"expected a refusal, got {error!r}"
    assert error.reason == "RAW_CLIENT_WRITE"
    assert error.channel_address == address


def _put_on_fresh_thread(client: ModuleType, address: str) -> BaseException | None:
    """Put from a new ``threading.Thread``, as a libca callback thread would."""
    caught: list[BaseException] = []

    def run():
        try:
            client.caput(address, 2)
        except BaseException as error:
            caught.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(5)
    return caught[0] if caught else None


class _RawPutConnector(_DoorProbe):
    """A connector whose writes put through the fake client on a worker thread."""

    def __init__(self, client: ModuleType, *, callback=None):
        super().__init__(armed=True)
        self._client = client
        self._callback = callback

    def _put(self, address, value):
        result = self._client.caput(address, value)
        if self._callback is not None:
            self._callback()
        return result

    async def write_channel(self, channel_address, value, **kwargs):
        await asyncio.to_thread(self._put, channel_address, value)
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=WriteOutcome.UNREQUESTED,
        )

    async def write_multiple_channels(self, operations, **kwargs):
        results = []
        for address, value in operations:
            await asyncio.to_thread(self._put, address, value)
            results.append(
                ChannelWriteResult(
                    channel_address=address,
                    value_written=value,
                    outcome=WriteOutcome.UNREQUESTED,
                )
            )
        return results


def test_raw_put_inside_the_door_reaches_the_client(armed_client):
    with open_door():
        assert armed_client.caput("SR:PV", 1.0) is True
    assert armed_client.calls == [("SR:PV", 1.0)]


def test_raw_put_on_the_main_thread_is_refused(armed_client):
    with pytest.raises(ChannelWriteBlockedError) as caught:
        armed_client.caput("SR:PV", 1.0)
    _assert_refused(caught.value, "SR:PV")
    assert RAW_CLIENT_WRITE_MARKER in str(caught.value)
    assert armed_client.calls == []


def test_raw_put_on_a_user_thread_is_refused_even_while_the_door_is_open(armed_client):
    with open_door():
        error = _put_on_fresh_thread(armed_client, "SR:USER")
    _assert_refused(error, "SR:USER")
    assert armed_client.calls == []


def test_write_channel_through_to_thread_passes_and_closes_the_door(armed_client):
    connector = _RawPutConnector(armed_client)

    async def main():
        result = await connector.write_channel("SR:PV", 1.0)
        return result, door_is_open()

    result, after = asyncio.run(main())
    assert result.outcome is WriteOutcome.UNREQUESTED
    assert armed_client.calls == [("SR:PV", 1.0)]
    assert after is False
    with pytest.raises(ChannelWriteBlockedError):
        armed_client.caput("SR:PV", 2.0)


def test_write_multiple_channels_through_to_thread_passes_and_closes_the_door(armed_client):
    connector = _RawPutConnector(armed_client)

    async def main():
        results = await connector.write_multiple_channels([("A", 1), ("B", 2)])
        return results, door_is_open()

    results, after = asyncio.run(main())
    assert [r.outcome for r in results] == [WriteOutcome.UNREQUESTED] * 2
    assert armed_client.calls == [("A", 1), ("B", 2)]
    assert after is False


def test_fresh_thread_callback_during_an_in_flight_connector_put_is_refused(armed_client):
    seen: list[BaseException | None] = []
    connector = _RawPutConnector(
        armed_client,
        callback=lambda: seen.append(_put_on_fresh_thread(armed_client, "SR:CB")),
    )

    result = asyncio.run(connector.write_channel("SR:PV", 1.0))

    assert result.outcome is WriteOutcome.UNREQUESTED
    assert len(seen) == 1
    _assert_refused(seen[0], "SR:CB")
    assert armed_client.calls == [("SR:PV", 1.0)], "only the connector's own put reached the client"


def test_inline_callback_during_a_connector_put_passes(armed_client):
    # The documented cooperative bypass: a callback polled inside the
    # connector's own put runs in that put's context, so the door is open.
    connector = _RawPutConnector(armed_client, callback=lambda: armed_client.caput("SR:CB", 2))

    asyncio.run(connector.write_channel("SR:PV", 1.0))

    assert armed_client.calls == [("SR:PV", 1.0), ("SR:CB", 2)]
    assert door_is_open() is False


def test_run_in_executor_hop_from_an_open_door_is_refused(armed_client):
    async def main():
        loop = asyncio.get_running_loop()
        with open_door():
            try:
                await loop.run_in_executor(None, armed_client.caput, "SR:EXEC", 1)
            except ChannelWriteBlockedError as error:
                return error
        return None

    _assert_refused(asyncio.run(main()), "SR:EXEC")
    assert armed_client.calls == []
