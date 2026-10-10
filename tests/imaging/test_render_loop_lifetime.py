"""Every render worker is reaped inside the event loop that spawned it.

A worker whose loop has closed can no longer be waited on: its pipe transports
are collected later and report ``Event loop is closed`` (Python 3.11/3.12) or an
unclosed-transport warning. These cases end the loop with ``asyncio.run`` --
the shape of a CLI backfill -- with a worker cached, or with a render in flight
when the loop's last task is cancelled.
"""

from __future__ import annotations

import asyncio
import gc
import io
import logging
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from osprey.imaging import render

Image = pytest.importorskip("PIL.Image")

pytestmark = pytest.mark.timeout(60)


def _png() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 10, 10)).save(out, "PNG")
    return out.getvalue()


PNG = _png()


def _silent_worker(tmp_path) -> tuple[str, ...]:
    """A worker that hand-shakes and never answers, so a render is always in flight."""
    path = tmp_path / "silent_worker.py"
    path.write_text(
        textwrap.dedent(
            """
            import sys, time
            sys.stdout.buffer.write(b'{"ready": true}\\n')
            sys.stdout.buffer.flush()
            time.sleep(30)
            """
        )
    )
    return (sys.executable, "-I", str(path))


def _stubborn_worker(tmp_path) -> tuple[str, ...]:
    """A worker that answers one render, then ignores end of input until killed."""
    path = tmp_path / "stubborn_worker.py"
    path.write_text(
        textwrap.dedent(
            f"""
            import json, struct, sys, time
            PNG = {PNG!r}
            out, inp = sys.stdout.buffer, sys.stdin.buffer
            out.write(b'{{"ready": true}}\\n'); out.flush()
            (n,) = struct.unpack(">I", inp.read(4)); inp.read(n)
            header = {{"ok": True, "format": "PNG", "w": 8, "h": 8, "mode": "RGB",
                       "mime": "image/png", "reason": None}}
            out.write(json.dumps(header).encode() + b"\\n" + struct.pack(">I", len(PNG)) + PNG)
            out.flush()
            inp.read()
            time.sleep(30)
            """
        )
    )
    return (sys.executable, "-I", str(path))


@pytest.fixture(autouse=True)
def _no_worker(monkeypatch):
    monkeypatch.setattr(render, "RENDER_TASK_TIMEOUT_S", 30.0)
    monkeypatch.setattr(render, "RENDER_READY_TIMEOUT_S", 30.0)
    render._forget_worker()
    # These tests force a collection to make the transports of their own closed
    # loop report. A collection is process-wide: unfrozen, it also walks and
    # finalises whatever the tests before this one left in the process, at their
    # cost and with their reports landing in this test's list. Frozen objects
    # are left out of a collection, so it covers what this test created.
    gc.freeze()
    try:
        yield
    finally:
        gc.unfreeze()
        render._forget_worker()


@contextmanager
def _unraisable() -> Iterator[list[str]]:
    """Collect what the interpreter reports as unraisable (transport finalisers)."""
    seen: list[str] = []
    previous = sys.unraisablehook
    sys.unraisablehook = lambda report: seen.append(repr(report.exc_value))
    try:
        yield seen
    finally:
        sys.unraisablehook = previous


def _cached_process() -> asyncio.subprocess.Process:
    assert render._CLIENT is not None and render._CLIENT.worker is not None
    return render._CLIENT.worker.process


def test_cached_worker_is_reaped_before_asyncio_run_returns():
    seen = []

    async def run() -> None:
        await render.render_isolated(PNG, task_id="a")
        seen.append(_cached_process())

    asyncio.run(run())

    assert seen[0].returncode is not None, "worker outlived its event loop"
    assert render.worker_pid() is None


def test_a_render_cancelled_at_loop_end_is_reaped_inside_the_loop(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(render, "WORKER_ARGV", _silent_worker(tmp_path))
    seen = []

    async def run() -> None:
        task = asyncio.create_task(render.render_isolated(PNG, task_id="x"))
        while render.worker_pid() is None:
            await asyncio.sleep(0.01)
        seen.append(_cached_process())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    with caplog.at_level(logging.WARNING, logger="asyncio"), _unraisable() as unraisable:
        asyncio.run(run())
        time.sleep(0.3)
        gc.collect()

    assert seen[0].returncode is not None, "cancelled worker was not reaped inside its loop"
    assert "is closed" not in caplog.text
    assert unraisable == []


def test_a_render_pending_when_the_loop_shuts_down_is_reaped(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(render, "WORKER_ARGV", _silent_worker(tmp_path))
    seen = []

    async def run() -> None:
        # Left running: asyncio.run cancels it during shutdown.
        asyncio.create_task(render.render_isolated(PNG, task_id="x"))
        while render.worker_pid() is None:
            await asyncio.sleep(0.01)
        seen.append(_cached_process())

    with caplog.at_level(logging.WARNING, logger="asyncio"), _unraisable() as unraisable:
        asyncio.run(run())
        time.sleep(0.3)
        gc.collect()

    assert seen[0].returncode is not None, "worker outlived its event loop"
    assert "is closed" not in caplog.text
    assert unraisable == []


async def test_a_second_cancel_during_the_close_still_reaps_the_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "WORKER_ARGV", _silent_worker(tmp_path))
    task = asyncio.create_task(render.render_isolated(PNG, task_id="x"))
    while render.worker_pid() is None:
        await asyncio.sleep(0.01)
    process = _cached_process()
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.returncode is not None
    await render.close_render_worker()


def test_an_idle_close_cut_short_by_the_loop_shutdown_still_reaps(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "WORKER_ARGV", _stubborn_worker(tmp_path))
    monkeypatch.setattr(render, "RENDER_WORKER_IDLE_S", 0.05)
    seen = []

    async def run() -> None:
        outcome = await render.render_isolated(PNG, task_id="a")
        assert outcome.rendition is not None
        seen.append(_cached_process())
        # The idle close starts and waits out the grace period of a worker that
        # ignores end of input; the loop then shuts down under it.
        await asyncio.sleep(0.3)
        assert render._CLIENT is not None and render._CLIENT.worker is None

    with _unraisable() as unraisable:
        asyncio.run(run())
        gc.collect()

    assert seen[0].returncode is not None, "worker outlived its event loop"
    assert unraisable == []


def test_a_second_loop_finds_no_worker_of_the_first():
    with _unraisable() as unraisable:
        asyncio.run(render.render_isolated(PNG, task_id="a"))
        asyncio.run(render.render_isolated(PNG, task_id="b"))
        render._forget_worker()
        gc.collect()
    assert unraisable == []


def test_a_guard_on_a_loop_in_another_thread_is_cancelled_through_that_loop():
    other = asyncio.new_event_loop()
    thread = threading.Thread(target=other.run_forever, daemon=True)
    thread.start()
    try:

        async def make_client() -> render._Client:
            return render._client()

        client = asyncio.run_coroutine_threadsafe(make_client(), other).result(5)
        scheduled = []
        real = other.call_soon_threadsafe

        def spy(callback, *args, **kwargs):
            scheduled.append(callback)
            return real(callback, *args, **kwargs)

        other.call_soon_threadsafe = spy  # type: ignore[method-assign]

        async def take_over() -> None:
            render._client()

        asyncio.run(take_over())

        assert client._guard.cancel in scheduled
        deadline = time.monotonic() + 5
        while not client._guard.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client._guard.cancelled()
    finally:
        other.call_soon_threadsafe(other.stop)
        thread.join(5)
        other.close()


def test_a_guard_on_a_closed_loop_is_left_alone():
    loop = asyncio.new_event_loop()

    async def make_client() -> render._Client:
        return render._client()

    client = loop.run_until_complete(make_client())
    loop.run_until_complete(asyncio.sleep(0))
    guard = client._guard
    # Closed by hand, without cancelling its tasks: the guard is still pending.
    loop.close()

    async def take_over() -> None:
        render._client()

    asyncio.run(take_over())
    render._forget_worker()

    assert not guard.done()
