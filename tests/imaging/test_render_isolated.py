"""The parent side of the isolated render worker: :mod:`osprey.imaging.render`.

Failure cases run against stub workers -- small scripts written to ``tmp_path``
that speak the worker's frame protocol and misbehave on chosen payloads. They are
launched exactly like the real worker (``-I``, ``env={}``, ``cwd='/'``), so they
can import nothing from the test environment and carry their reply bytes inline.

Timing constants are scaled (a 1 s task clock against a 3 s sleeper, a 0.5 s
idle close, a 2 s ready wait), so each case keeps its production meaning.
"""

from __future__ import annotations

import asyncio
import gc
import io
import os
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

from osprey.imaging import render
from osprey.imaging.formats import RENDITION_MAX_BYTES

Image = pytest.importorskip("PIL.Image")

pytestmark = pytest.mark.timeout(60)


def _png(size: tuple[int, int] = (8, 8)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, (200, 10, 10)).save(out, "PNG")
    return out.getvalue()


PNG = _png()
SOURCE = _png((32, 24))

# A stub worker: hand-shakes (unless told otherwise), then answers every frame
# from ``behaviour(payload)``, which returns (header dict, body bytes) or acts.
_STUB_TEMPLATE = """
import json, os, struct, sys, time
PNG = {png!r}
{prelude}
out = sys.stdout.buffer
inp = sys.stdin.buffer

def read_exactly(n):
    data = b""
    while len(data) < n:
        chunk = inp.read(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data

def reply(header, body=b""):
    out.write(json.dumps(header).encode() + b"\\n" + struct.pack(">I", len(body)) + body)
    out.flush()

def ok(body=PNG, **over):
    header = {{"ok": True, "format": "PNG", "w": 8, "h": 8, "mode": "RGB",
               "mime": "image/png", "reason": None}}
    header.update(over)
    reply(header, body)

def refuse(reason):
    reply({{"ok": False, "format": None, "w": None, "h": None, "mode": None,
            "mime": None, "reason": reason}})

{handshake}
while True:
    prefix = read_exactly(4)
    if prefix is None:
        break
    payload = read_exactly(struct.unpack(">I", prefix)[0])
    if payload is None:
        break
{behaviour}
"""

_HANDSHAKE = 'out.write(b\'{"ready": true, "pillow": "stub"}\\n\'); out.flush()'


def _stub(
    tmp_path: Path,
    behaviour: str,
    *,
    prelude: str = "",
    handshake: str = _HANDSHAKE,
    name: str = "stub_worker.py",
) -> tuple[str, ...]:
    script = _STUB_TEMPLATE.format(
        png=PNG,
        prelude=prelude,
        handshake=handshake,
        behaviour=textwrap.indent(textwrap.dedent(behaviour).strip(), "    "),
    )
    path = tmp_path / name
    path.write_text(script)
    return (sys.executable, "-I", str(path))


@pytest.fixture(autouse=True)
async def _scaled(monkeypatch):
    """Scaled clocks, and no worker left over between tests.

    A worker the test left running is closed while its loop is still alive;
    one owned by a loop that is gone (``asyncio.run``) is killed by pid.
    """
    monkeypatch.setattr(render, "RENDER_TASK_TIMEOUT_S", 1.0)
    monkeypatch.setattr(render, "RENDER_WORKER_IDLE_S", 30.0)
    monkeypatch.setattr(render, "RENDER_READY_TIMEOUT_S", 2.0)
    render._forget_worker()
    yield
    await render.close_render_worker()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers signal 0; ask the OS for its state.
    try:
        waited, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return False
    return waited == 0


# -- the real worker ----------------------------------------------------------------


async def test_default_argv_env_and_cwd(monkeypatch):
    assert render.WORKER_ARGV == (
        sys.executable,
        "-I",
        "-m",
        "osprey.imaging.render_worker",
    )
    monkeypatch.setattr(render, "RENDER_TASK_TIMEOUT_S", 30.0)
    monkeypatch.setattr(render, "RENDER_READY_TIMEOUT_S", 30.0)
    calls = []
    real = asyncio.create_subprocess_exec

    async def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return await real(*args, **kwargs)

    monkeypatch.setattr(render.asyncio, "create_subprocess_exec", spy)
    outcome = await render.render_isolated(SOURCE)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == render.WORKER_ARGV
    assert kwargs["env"] == {}
    assert kwargs["cwd"] == "/"
    assert kwargs["stdin"] == asyncio.subprocess.PIPE
    assert kwargs["stdout"] == asyncio.subprocess.PIPE
    assert kwargs["stderr"] == asyncio.subprocess.DEVNULL
    assert outcome.reason is None
    assert outcome.rendition is not None
    assert outcome.rendition.mime == "image/png"
    assert (outcome.rendition.width, outcome.rendition.height) == (32, 24)
    assert outcome.rendition.data.startswith(b"\x89PNG\r\n\x1a\n")


async def test_real_worker_content_refusal_is_a_reply_not_a_failure(monkeypatch):
    monkeypatch.setattr(render, "RENDER_TASK_TIMEOUT_S", 30.0)
    monkeypatch.setattr(render, "RENDER_READY_TIMEOUT_S", 30.0)
    outcome = await render.render_isolated(b"<html>login</html>")
    assert outcome.rendition is None
    assert outcome.reason == "not_an_image"


def test_two_consecutive_asyncio_runs_both_succeed(monkeypatch):
    monkeypatch.setattr(render, "RENDER_TASK_TIMEOUT_S", 30.0)
    monkeypatch.setattr(render, "RENDER_READY_TIMEOUT_S", 30.0)
    first = asyncio.run(render.render_isolated(SOURCE))
    first_pid = render.worker_pid()
    second = asyncio.run(render.render_isolated(SOURCE))
    second_pid = render.worker_pid()
    assert first.rendition is not None
    assert second.rendition is not None
    assert first_pid != second_pid
    # The worker of the first loop was killed when the second loop took over.
    for _ in range(50):
        if not _alive(first_pid):
            break
        import time

        time.sleep(0.1)
    assert not _alive(first_pid)
    # The killed worker's transports belong to a closed loop and can only be
    # collected; their unclosed-transport warning is expected here.
    render._forget_worker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        gc.collect()


async def test_probe_reports_a_healthy_worker():
    assert await render.probe_render_worker() is True
    # The probe never leaves a cached worker behind.
    assert render.worker_pid() is None


async def test_probe_reports_a_worker_that_dies_at_import(monkeypatch, tmp_path):
    path = tmp_path / "stub_worker.py"
    path.write_text("import sys\nsys.exit(1)\n")
    monkeypatch.setattr(render, "WORKER_ARGV", (sys.executable, "-I", str(path)))
    assert await render.probe_render_worker() is False


# -- stub workers: failures -----------------------------------------------------------


async def test_sleeper_id_times_out_and_siblings_render(monkeypatch, tmp_path):
    monkeypatch.setattr(
        render,
        "WORKER_ARGV",
        _stub(tmp_path, 'if payload == b"SLEEP":\n    time.sleep(3)\nok()'),
    )
    payloads = [b"a", b"b", b"SLEEP", b"c", b"d"]
    outcomes = await asyncio.gather(
        *(render.render_isolated(p, task_id=p.decode()) for p in payloads)
    )
    by_id = dict(zip(payloads, outcomes, strict=True))
    assert by_id[b"SLEEP"].reason == "decoder_failed"
    assert by_id[b"SLEEP"].rendition is None
    for sibling in (b"a", b"b", b"c", b"d"):
        assert by_id[sibling].reason is None
        assert by_id[sibling].rendition is not None


async def test_the_task_clock_starts_after_the_lock(monkeypatch, tmp_path):
    # Each task takes 0.6 s against a 1 s clock; five queued tasks take 3 s in
    # all, so a clock started at submission would time the later ones out.
    monkeypatch.setattr(render, "WORKER_ARGV", _stub(tmp_path, "time.sleep(0.6)\nok()"))
    outcomes = await asyncio.gather(
        *(render.render_isolated(bytes([i]), task_id=str(i)) for i in range(5))
    )
    assert all(o.rendition is not None for o in outcomes)


async def test_crash_on_one_id_leaves_siblings_and_next_call_rendered(monkeypatch, tmp_path):
    monkeypatch.setattr(
        render,
        "WORKER_ARGV",
        _stub(tmp_path, 'if payload == b"CRASH":\n    os._exit(139)\nok()'),
    )
    first = await render.render_isolated(b"a", task_id="a")
    crashed = await render.render_isolated(b"CRASH", task_id="crash")
    after = await render.render_isolated(b"b", task_id="b")
    assert first.rendition is not None
    assert crashed.reason == "decoder_failed"
    assert after.rendition is not None


async def test_worker_exiting_at_import_is_unavailable(monkeypatch, tmp_path):
    path = tmp_path / "stub_worker.py"
    path.write_text("import sys\nsys.exit(1)\n")
    monkeypatch.setattr(render, "WORKER_ARGV", (sys.executable, "-I", str(path)))
    with pytest.raises(render.RenderUnavailable) as raised:
        await render.render_isolated(PNG, task_id="x")
    assert raised.value.exit_code == 1
    assert "1" in str(raised.value)


async def test_no_ready_within_the_ready_wait_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        render,
        "WORKER_ARGV",
        _stub(tmp_path, "ok()", handshake="time.sleep(10)\n" + _HANDSHAKE),
    )
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(render.RenderUnavailable):
        await render.render_isolated(PNG, task_id="x")
    assert loop.time() - started < 6
    assert render.worker_pid() is None


async def test_garbage_handshake_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        render,
        "WORKER_ARGV",
        _stub(tmp_path, "ok()", handshake="out.write(b'hello\\n'); out.flush()"),
    )
    with pytest.raises(render.RenderUnavailable):
        await render.render_isolated(PNG, task_id="x")


async def test_deaths_on_two_different_ids_are_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "WORKER_ARGV", _stub(tmp_path, "os._exit(139)"))
    first = await render.render_isolated(b"a", task_id="a")
    assert first.reason == "decoder_failed"
    with pytest.raises(render.RenderUnavailable):
        await render.render_isolated(b"b", task_id="b")


async def test_a_good_reply_between_deaths_clears_the_death_count(monkeypatch, tmp_path):
    monkeypatch.setattr(
        render,
        "WORKER_ARGV",
        _stub(tmp_path, 'if payload.startswith(b"CRASH"):\n    os._exit(139)\nok()'),
    )
    assert (await render.render_isolated(b"CRASH1", task_id="1")).reason == "decoder_failed"
    assert (await render.render_isolated(b"ok", task_id="ok")).rendition is not None
    assert (await render.render_isolated(b"CRASH2", task_id="2")).reason == "decoder_failed"


async def test_unknown_reason_is_a_worker_failure_then_decoder_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "WORKER_ARGV", _stub(tmp_path, 'refuse("boom")'))
    pids = []
    real_spawn = render._spawn

    async def counting_spawn(*args, **kwargs):
        worker = await real_spawn(*args, **kwargs)
        pids.append(worker.process.pid)
        return worker

    monkeypatch.setattr(render, "_spawn", counting_spawn)
    outcome = await render.render_isolated(b"x", task_id="x")
    assert outcome.reason == "decoder_failed"
    assert outcome.rendition is None
    # Tried once, the worker was replaced, and the retry ran in a fresh worker.
    assert len(pids) == 2


async def test_a_registered_content_refusal_passes_through(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "WORKER_ARGV", _stub(tmp_path, 'refuse("rendition_too_large")'))
    outcome = await render.render_isolated(b"x", task_id="x")
    assert outcome.reason == "rendition_too_large"
    assert outcome.rendition is None


@pytest.mark.parametrize(
    "behaviour",
    [
        pytest.param('ok(body=b"GIF89a" + b"\\0" * 32)', id="non-png-bytes"),
        pytest.param('ok(mime="image/gif")', id="mime-outside-png-jpeg"),
        pytest.param("ok(w=4096)", id="width-over-1024"),
        pytest.param('ok(mode="CMYK")', id="mode-outside-renditions"),
        pytest.param('ok(format="PDF")', id="format-not-accepted"),
        pytest.param('ok(reason="decoder_failed")', id="ok-with-reason"),
        pytest.param('ok(ok="yes")', id="ok-not-bool"),
        pytest.param(
            'out.write(b"not json\\n" + struct.pack(">I", 0)); out.flush()',
            id="header-not-json",
        ),
        pytest.param(
            'reply({"ok": False, "format": None, "w": None, "h": None, "mode": None,'
            ' "mime": None, "reason": "not_an_image"}, b"extra")',
            id="refusal-with-body",
        ),
    ],
)
async def test_bad_replies_are_worker_failures(monkeypatch, tmp_path, behaviour):
    monkeypatch.setattr(render, "WORKER_ARGV", _stub(tmp_path, behaviour))
    outcome = await render.render_isolated(b"x", task_id="x")
    assert outcome.reason == "decoder_failed"
    assert outcome.rendition is None


@pytest.mark.parametrize(
    ("header", "body"),
    [
        (
            {
                "ok": True,
                "format": "PNG",
                "w": 8,
                "h": 8,
                "mode": "RGB",
                "mime": "image/png",
                "reason": None,
            },
            PNG,
        ),
        (
            {
                "ok": True,
                "format": "JPEG",
                "w": 1024,
                "h": 1,
                "mode": "L",
                "mime": "image/jpeg",
                "reason": None,
            },
            b"\xff\xd8\xff\xe0" + b"\0" * 32,
        ),
        (
            {
                "ok": False,
                "format": None,
                "w": None,
                "h": None,
                "mode": None,
                "mime": None,
                "reason": "format_mismatch",
            },
            b"",
        ),
    ],
)
def test_check_reply_accepts_valid_replies(header, body):
    render._check_reply(header, body)


@pytest.mark.parametrize(
    ("header", "body"),
    [
        ([], b""),
        (
            {
                "ok": True,
                "format": "PNG",
                "w": True,
                "h": 8,
                "mode": "RGB",
                "mime": "image/png",
                "reason": None,
            },
            PNG,
        ),
        (
            {
                "ok": True,
                "format": "PNG",
                "w": 0,
                "h": 8,
                "mode": "RGB",
                "mime": "image/png",
                "reason": None,
            },
            PNG,
        ),
        (
            {
                "ok": True,
                "format": "PNG",
                "w": 8,
                "h": 8,
                "mode": "RGB",
                "mime": "image/png",
                "reason": None,
            },
            b"",
        ),
        (
            {
                "ok": True,
                "format": "PNG",
                "w": 8,
                "h": 8,
                "mode": "RGB",
                "mime": "image/png",
                "reason": None,
            },
            b"\xff\xd8\xff" + b"\0" * 32,
        ),
        (
            {
                "ok": True,
                "format": "PNG",
                "w": 8,
                "h": 8,
                "mode": "RGB",
                "mime": "image/png",
                "reason": None,
            },
            PNG + b"\0" * RENDITION_MAX_BYTES,
        ),
        (
            {
                "ok": False,
                "format": None,
                "w": None,
                "h": None,
                "mode": None,
                "mime": None,
                "reason": None,
            },
            b"",
        ),
    ],
    ids=[
        "not-a-dict",
        "w-bool",
        "w-zero",
        "empty-body",
        "magic-mismatch",
        "oversize",
        "refusal-without-reason",
    ],
)
def test_check_reply_rejects_invalid_replies(header, body):
    with pytest.raises(render._WorkerFailure):
        render._check_reply(header, body)


# -- lifetime ---------------------------------------------------------------------------


async def test_idle_worker_exits(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "RENDER_WORKER_IDLE_S", 0.5)
    monkeypatch.setattr(render, "WORKER_ARGV", _stub(tmp_path, "ok()"))
    await render.render_isolated(b"x", task_id="x")
    pid = render.worker_pid()
    assert pid is not None
    for _ in range(40):
        await asyncio.sleep(0.1)
        if render.worker_pid() is None and not _alive(pid):
            break
    assert render.worker_pid() is None
    assert not _alive(pid)
    # The next call respawns it.
    assert (await render.render_isolated(b"y", task_id="y")).rendition is not None
    assert render.worker_pid() not in (None, pid)


async def test_worker_is_respawned_after_max_tasks(monkeypatch, tmp_path):
    assert render.WORKER_MAX_TASKS == 100
    monkeypatch.setattr(render, "WORKER_MAX_TASKS", 3)
    monkeypatch.setattr(render, "WORKER_ARGV", _stub(tmp_path, "ok()"))
    spawned = []
    real_spawn = render._spawn

    async def counting_spawn(*args, **kwargs):
        worker = await real_spawn(*args, **kwargs)
        spawned.append(worker.process.pid)
        return worker

    monkeypatch.setattr(render, "_spawn", counting_spawn)
    after = []
    for i in range(7):
        await render.render_isolated(bytes([i]), task_id=str(i))
        after.append(render.worker_pid())
    # Three tasks per worker: the third closes it, the next call spawns afresh.
    assert len(spawned) == 3
    assert len(set(spawned)) == 3
    assert after[2] is None and after[5] is None
    assert after[0] == after[1] == spawned[0]
    assert after[3] == after[4] == spawned[1]
    assert after[6] == spawned[2]


async def test_default_task_id_is_content_derived(monkeypatch, tmp_path):
    # The same bytes dying twice in a row are one id, so the second death is a
    # retry of the same picture rather than a death on a different id.
    monkeypatch.setattr(render, "WORKER_ARGV", _stub(tmp_path, "os._exit(139)"))
    outcome = await render.render_isolated(b"same")
    assert outcome.reason == "decoder_failed"


# -- cancellation -----------------------------------------------------------------------


async def test_a_cancelled_render_never_leaks_its_reply_into_the_next_call(monkeypatch, tmp_path):
    # The reply width is the payload length, so each rendition names its picture.
    monkeypatch.setattr(
        render,
        "WORKER_ARGV",
        _stub(tmp_path, 'if payload == b"1":\n    time.sleep(0.8)\nok(w=len(payload))'),
    )
    slow = asyncio.create_task(render.render_isolated(b"1", task_id="one"))
    await asyncio.sleep(0.4)
    first_pid = render.worker_pid()
    assert first_pid is not None
    slow.cancel()
    with pytest.raises(asyncio.CancelledError):
        await slow
    assert render.worker_pid() is None
    outcome = await render.render_isolated(b"22", task_id="two")
    assert outcome.rendition is not None
    assert outcome.rendition.width == 2
    assert render.worker_pid() not in (None, first_pid)
    for _ in range(40):
        if not _alive(first_pid):
            break
        await asyncio.sleep(0.05)
    assert not _alive(first_pid)


async def test_a_cancel_is_not_counted_as_a_worker_death(monkeypatch, tmp_path):
    monkeypatch.setattr(
        render,
        "WORKER_ARGV",
        _stub(
            tmp_path,
            'if payload == b"SLOW":\n    time.sleep(0.8)\n'
            'if payload == b"CRASH":\n    os._exit(139)\nok()',
        ),
    )
    slow = asyncio.create_task(render.render_isolated(b"SLOW", task_id="slow"))
    await asyncio.sleep(0.4)
    slow.cancel()
    with pytest.raises(asyncio.CancelledError):
        await slow
    # A death on another id right after the cancel is that id's first failure.
    crashed = await render.render_isolated(b"CRASH", task_id="crash")
    assert crashed.reason == "decoder_failed"


async def test_a_cancel_during_the_ready_wait_leaves_no_child(monkeypatch, tmp_path):
    monkeypatch.setattr(
        render,
        "WORKER_ARGV",
        _stub(tmp_path, "ok()", handshake="time.sleep(10)\n" + _HANDSHAKE),
    )
    spawned = []
    real_exec = asyncio.create_subprocess_exec

    async def recording_exec(*args, **kwargs):
        process = await real_exec(*args, **kwargs)
        spawned.append(process.pid)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", recording_exec)
    pending = asyncio.create_task(render.render_isolated(PNG, task_id="x"))
    for _ in range(40):
        await asyncio.sleep(0.05)
        if spawned:
            break
    assert len(spawned) == 1
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert render.worker_pid() is None
    for _ in range(40):
        if not _alive(spawned[0]):
            break
        await asyncio.sleep(0.05)
    assert not _alive(spawned[0])


async def test_kill_never_signals_a_reaped_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "WORKER_ARGV", _stub(tmp_path, "ok()"))
    worker = await render._spawn()
    await worker.close(kill=False)
    assert worker.process.returncode is not None
    signalled = []
    monkeypatch.setattr(render.os, "kill", lambda pid, sig: signalled.append(pid))
    worker.kill()
    assert signalled == []
