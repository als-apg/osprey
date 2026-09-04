"""The runtime's sync-over-async bridge, in both of the contexts it serves.

``_run_async`` is how every synchronous runtime call (``read_channel``,
``write_channel``) reaches its coroutine. It has two callers with two
different loops: the executor subprocess, which has none, and a notebook
kernel, which is already inside one. The one thing a test can pin about the
bridge is that the coroutine's own outcome — its value or its exception —
comes back unchanged from either context. ``ControlTargetChangedError`` is the
case that matters: it is a ``RuntimeError``, and so is the "no running loop"
signal the bridge probes for.
"""

from __future__ import annotations

import asyncio

import pytest

from osprey.runtime import ControlTargetChangedError, _run_async


class SubclassedRuntimeError(RuntimeError):
    """Any ``RuntimeError`` subclass; the refusal class is one of them."""


async def answer(value):
    """A coroutine that returns *value*."""
    return value


async def refuse(error):
    """A coroutine that raises *error*."""
    raise error


def inside_a_running_loop(call):
    """Run *call* synchronously from inside a coroutine on a fresh loop.

    That is a notebook kernel's shape: the cell's synchronous code executes
    while the kernel's own loop is running on the same thread.
    """

    async def driver():
        return call()

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(driver())
    finally:
        loop.close()


ERRORS = {
    "runtime_subclass": lambda: SubclassedRuntimeError("the coroutine's own"),
    "control_target_changed": lambda: ControlTargetChangedError("the target moved"),
    "value_error": lambda: ValueError("not a RuntimeError at all"),
}


class TestWithoutARunningLoop:
    """The executor subprocess: ``asyncio.run`` directly."""

    def test_the_value_comes_back(self):
        assert _run_async(answer(42)) == 42

    @pytest.mark.parametrize("name", sorted(ERRORS))
    def test_the_coroutines_exception_propagates_unchanged(self, name):
        error = ERRORS[name]()

        with pytest.raises(type(error)) as caught:
            _run_async(refuse(error))

        assert caught.value is error


class TestInsideARunningLoop:
    """The notebook kernel: a worker thread with a loop of its own."""

    def test_the_value_comes_back(self):
        assert inside_a_running_loop(lambda: _run_async(answer(42))) == 42

    @pytest.mark.parametrize("name", sorted(ERRORS))
    def test_the_coroutines_exception_propagates_unchanged(self, name):
        """A ``RuntimeError`` from the coroutine is not the "no loop" signal.

        Mistaking it for one would retry ``asyncio.run`` on the thread whose
        loop is running, and the refusal would surface as "asyncio.run()
        cannot be called from a running event loop" instead.
        """
        error = ERRORS[name]()

        with pytest.raises(type(error)) as caught:
            inside_a_running_loop(lambda: _run_async(refuse(error)))

        assert caught.value is error
