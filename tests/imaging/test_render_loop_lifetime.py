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


@pytest.fixture(autouse=True)
def _no_worker(monkeypatch):
    monkeypatch.setattr(render, "RENDER_TASK_TIMEOUT_S", 30.0)
    monkeypatch.setattr(render, "RENDER_READY_TIMEOUT_S", 30.0)
    render._forget_worker()
    yield
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


def test_a_second_loop_finds_no_worker_of_the_first():
    with _unraisable() as unraisable:
        asyncio.run(render.render_isolated(PNG, task_id="a"))
        asyncio.run(render.render_isolated(PNG, task_id="b"))
        render._forget_worker()
        gc.collect()
    assert unraisable == []
