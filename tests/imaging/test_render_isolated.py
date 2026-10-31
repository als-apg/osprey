"""The parent side of the isolated render worker: :mod:`osprey.imaging.render`.

Failure cases run against stub workers -- small scripts written to ``tmp_path``
that speak the worker's frame protocol and misbehave on chosen payloads. They are
launched exactly like the real worker (``-I``, ``env={}``, ``cwd='/'``), so they
can import nothing from the test environment and carry their reply bytes inline.

Timing constants are scaled (a 1 s task clock against a 3 s sleeper, a 0.5 s
idle close), so each case keeps its production meaning. The ready wait stays
long by default so a loaded host never trips it; a case about the ready wait
sets its own. The cancellation cases hold their worker on a file the test
controls and run on the shipped clocks, so only the cancel ends the render. A
test that waits for a state polls for it with :func:`tests._live.async_wait_until`.
"""

from __future__ import annotations

import asyncio
import io
import sys
import textwrap
from pathlib import Path

import pytest

from osprey.imaging import render
from osprey.imaging.formats import RENDITION_MAX_BYTES
from tests._live import async_wait_until
from tests._processes import running_state

Image = pytest.importorskip("PIL.Image")

pytestmark = pytest.mark.timeout(60)


def _png(size: tuple[int, int] = (8, 8)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, (200, 10, 10)).save(out, "PNG")
    return out.getvalue()


PNG = _png()
SOURCE = _png((32, 24))

# The task clock as shipped, for the cases where only the test may end a render.
_TASK_TIMEOUT_S = render.RENDER_TASK_TIMEOUT_S

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
    monkeypatch.setattr(render, "RENDER_READY_TIMEOUT_S", 30.0)
    render._forget_worker()
    yield
    await render.close_render_worker()


# -- the real worker ----------------------------------------------------------------


async def test_default_argv_env_and_cwd(monkeypatch):
    assert render.WORKER_ARGV == (
        sys.executable,
        "-I",
        "-m",
        "osprey.imaging.render_worker",
    )
    monkeypatch.setattr(render, "RENDER_TASK_TIMEOUT_S", 30.0)
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
    outcome = await render.render_isolated(b"<html>login</html>")
    assert outcome.rendition is None
    assert outcome.reason == "not_an_image"


def test_two_consecutive_asyncio_runs_both_succeed(monkeypatch):
    monkeypatch.setattr(render, "RENDER_TASK_TIMEOUT_S", 30.0)

    async def render_once() -> tuple[render.RenderOutcome, int | None]:
        outcome = await render.render_isolated(SOURCE)
        return outcome, render.worker_pid()

    first, first_pid = asyncio.run(render_once())
    # The first loop's shutdown closed its worker before the loop went away.
    assert render.worker_pid() is None
    assert first_pid is not None and running_state(first_pid) is None
    second, second_pid = asyncio.run(render_once())
    assert first.rendition is not None
    assert second.rendition is not None
    assert first_pid != second_pid


async def test_probe_reports_a_healthy_worker(monkeypatch):
    monkeypatch.setattr(render, "RENDER_READY_TIMEOUT_S", 30.0)
    assert await render.probe_render_worker() is None
    # The probe never leaves a cached worker behind.
    assert render.worker_pid() is None


async def test_probe_reports_a_worker_that_dies_at_import(monkeypatch, tmp_path):
    path = tmp_path / "stub_worker.py"
    path.write_text("import sys\nsys.exit(1)\n")
    monkeypatch.setattr(render, "WORKER_ARGV", (sys.executable, "-I", str(path)))
    reason = await render.probe_render_worker()
    assert reason is not None
    assert "exited before its ready line" in reason


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
    # Each task takes a tenth of the clock and the queue holds one and a half
    # clocks of them, so a clock started at submission would time the later ones
    # out while one started after the lock leaves each task most of its clock.
    clock = render.RENDER_TASK_TIMEOUT_S
    count = 15
    monkeypatch.setattr(render, "WORKER_ARGV", _stub(tmp_path, f"time.sleep({clock / 10})\nok()"))
    loop = asyncio.get_running_loop()
    started = loop.time()
    outcomes = await asyncio.gather(
        *(render.render_isolated(bytes([i]), task_id=str(i)) for i in range(count))
    )
    elapsed = loop.time() - started
    assert all(o.rendition is not None for o in outcomes)
    # The queue really outlasted one clock.
    assert elapsed > clock


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
    monkeypatch.setattr(render, "RENDER_READY_TIMEOUT_S", 2.0)
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
    await async_wait_until(
        lambda: render.worker_pid() is None and running_state(pid) is None,
        timeout=4.0,
        what="the idle worker to exit",
    )
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


def _held(tmp_path: Path, payload: bytes) -> tuple[str, Path, Path]:
    """Stub lines that hold *payload* mid-render: (lines, started, release).

    The worker creates ``started`` once it has read the whole payload, then
    replies only after ``release`` exists.
    """
    started, release = tmp_path / "render-started", tmp_path / "render-release"
    lines = (
        f"if payload == {payload!r}:\n"
        f"    open({str(started)!r}, 'w').close()\n"
        f"    while not os.path.exists({str(release)!r}):\n"
        f"        time.sleep(0.01)\n"
    )
    return lines, started, release


async def _until_held(task: asyncio.Task, started: Path, ceiling: float = 30.0) -> None:
    """Return once the worker holds the picture; fail if the render ends first."""

    def held() -> bool:
        assert not task.done(), f"the render ended before its worker held the picture: {task}"
        return started.exists()

    await async_wait_until(
        held, timeout=ceiling, interval=0.01, what="a worker to hold the picture"
    )


# -- cancellation -----------------------------------------------------------------------


async def test_a_cancelled_render_never_leaks_its_reply_into_the_next_call(monkeypatch, tmp_path):
    held, started, release = _held(tmp_path, b"1")
    # The reply width is the payload length, so each rendition names its picture.
    monkeypatch.setattr(render, "WORKER_ARGV", _stub(tmp_path, held + "ok(w=len(payload))"))
    monkeypatch.setattr(render, "RENDER_TASK_TIMEOUT_S", _TASK_TIMEOUT_S)
    slow = asyncio.create_task(render.render_isolated(b"1", task_id="one"))
    await _until_held(slow, started)
    first_pid = render.worker_pid()
    assert first_pid is not None
    slow.cancel()
    with pytest.raises(asyncio.CancelledError):
        await slow
    assert render.worker_pid() is None
    # A worker that outlived the cancel would now write the first picture's reply.
    release.touch()
    outcome = await render.render_isolated(b"22", task_id="two")
    assert outcome.rendition is not None
    assert outcome.rendition.width == 2
    assert render.worker_pid() not in (None, first_pid)
    await async_wait_until(
        lambda: running_state(first_pid) is None, what="the cancelled worker to exit"
    )


async def test_a_cancel_is_not_counted_as_a_worker_death(monkeypatch, tmp_path):
    held, started, _ = _held(tmp_path, b"SLOW")
    monkeypatch.setattr(
        render,
        "WORKER_ARGV",
        _stub(tmp_path, held + 'if payload == b"CRASH":\n    os._exit(139)\nok()'),
    )
    monkeypatch.setattr(render, "RENDER_TASK_TIMEOUT_S", _TASK_TIMEOUT_S)
    slow = asyncio.create_task(render.render_isolated(b"SLOW", task_id="slow"))
    await _until_held(slow, started)
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
    await async_wait_until(lambda: spawned, what="the worker to spawn")
    assert len(spawned) == 1
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert render.worker_pid() is None
    await async_wait_until(
        lambda: running_state(spawned[0]) is None, what="the cancelled worker to exit"
    )


async def test_kill_never_signals_a_reaped_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "WORKER_ARGV", _stub(tmp_path, "ok()"))
    worker = await render._spawn()
    await worker.close(kill=False)
    assert worker.process.returncode is not None
    signalled = []
    monkeypatch.setattr(render.os, "kill", lambda pid, sig: signalled.append(pid))
    worker.kill()
    assert signalled == []
