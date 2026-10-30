"""Deadline-bounded reads off a ``TestClient`` websocket.

``WebSocketTestSession.receive`` is ``portal.call(self._send_rx.receive)``: a
wait on the stream the handler under test sends into, with no deadline. A
handler that has died or gone quiet therefore parks the test thread in that
wait until the per-test cap ends the run, minutes later, and the failure is a
timeout with no word on which frame never came. The readers here run the same
receive under ``anyio.fail_after`` on the session's portal, so a quiet handler
is an ``AssertionError`` within seconds that names the frame it owed and the
JSON types that did arrive.

The rest is the terminal-socket harness the ``/ws/terminal`` suites share:
:func:`resume_url` and :func:`send_resize` open a resume-mode connect,
:func:`sync_session` round-trips through the handler's main loop,
:func:`seed_session_file` puts a transcript where discovery looks, and
:func:`patch_spawn` swaps the registry's PTY spawn for a ``FakePtySession``.
The app those suites drive and the transcript directory are the ``ws_app`` and
``sessions_dir`` fixtures of this directory's conftest.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import anyio

from tests.interfaces.web_terminal._fakes import FakePtySession

#: How long a read may wait for the frame it is after before the test fails.
#: A handler that answers nothing is a regression, not a stall.
RECV_DEADLINE_S = 10.0


async def _receive_within(rx: Any, seconds: float) -> dict:
    with anyio.fail_after(seconds):
        return await rx.receive()


def _frame_or_none(ws: Any, seconds: float) -> dict | None:
    """One raw frame off *ws*, or ``None`` once *seconds* have passed."""
    try:
        return ws.portal.call(_receive_within, ws._send_rx, seconds)
    except TimeoutError:
        return None


def recv_frame(ws: Any, *, within: float = RECV_DEADLINE_S) -> dict:
    """The next raw frame off *ws* — text, bytes or close — or fail after *within* seconds.

    The drop-in for ``ws.receive()`` where a test reads a close frame itself.
    """
    frame = _frame_or_none(ws, within)
    if frame is None:
        raise AssertionError(f"no frame within {within:g} s")
    return frame


def json_frames_until(
    ws: Any, msg_type: str, *, max_frames: int = 30, within: float = RECV_DEADLINE_S
) -> list[dict]:
    """Every JSON frame up to and including the first whose ``type`` is *msg_type*.

    Binary frames are skipped. Fails, naming *msg_type* and the JSON types seen
    so far, when *within* seconds pass without it, when *max_frames* frames
    arrive without it, or when the server closes the socket first.
    """
    deadline = time.monotonic() + within
    collected: list[dict] = []
    for _ in range(max_frames):
        frame = _frame_or_none(ws, max(deadline - time.monotonic(), 0.0))
        seen = [d.get("type") for d in collected]
        if frame is None:
            raise AssertionError(
                f"no '{msg_type}' frame within {within:g} s; JSON types seen: {seen}"
            )
        if frame["type"] == "websocket.close":
            raise AssertionError(
                f"socket closed ({frame.get('code')}) before '{msg_type}'; JSON types seen: {seen}"
            )
        if "text" in frame:
            data = json.loads(frame["text"])
            collected.append(data)
            if data.get("type") == msg_type:
                return collected
    seen = [d.get("type") for d in collected]
    raise AssertionError(
        f"no '{msg_type}' frame within {max_frames} frames; JSON types seen: {seen}"
    )


def recv_json(
    ws: Any, msg_type: str, *, max_frames: int = 30, within: float = RECV_DEADLINE_S
) -> dict:
    """The first JSON frame whose ``type`` is *msg_type*; see :func:`json_frames_until`."""
    return json_frames_until(ws, msg_type, max_frames=max_frames, within=within)[-1]


def new_session_id() -> str:
    """A fresh session key, as the browser mints one."""
    return str(uuid.uuid4())


def resume_url(session_id: str, *, interrupt: bool = False, end_started: bool = False) -> str:
    """The terminal socket in resume mode for *session_id* (no session-discovery task)."""
    url = f"/ws/terminal?session_id={session_id}&mode=resume"
    if interrupt:
        url += "&interrupt=1"
    if end_started:
        url += "&end_started=1"
    return url


def send_resize(ws: Any, cols: int = 80, rows: int = 24) -> None:
    """Send the initial resize the handler waits for before spawning."""
    ws.send_json({"type": "resize", "cols": cols, "rows": rows})


def sync_session(ws: Any, session_id: str) -> None:
    """Round-trip through the main loop: a switch to the current key is a no-op answer."""
    ws.send_json({"type": "switch_session", "session_id": session_id})
    recv_json(ws, "session_switched")


def seed_session_file(sessions_dir: Path, session_id: str) -> None:
    """Create an (empty) on-disk transcript for *session_id*.

    Seeded before an initial resume connect, it makes ``terminal_ws``'s resume
    confirmation take its trusted synchronous path — the id already exists on
    disk — instead of polling discovery.
    """
    (sessions_dir / f"{session_id}.jsonl").write_text("")


def patch_spawn(app: Any) -> tuple[Any, list[FakePtySession]]:
    """Replace ``_spawn_session`` so no real PTY is created.

    Returns:
        ``(registry, spawned)`` — the app's PTY registry and every
        ``FakePtySession`` it has handed out since, in order.
    """
    registry = app.state.pty_registry
    spawned: list[FakePtySession] = []

    def tracked_spawn(*_args: Any, **_kwargs: Any) -> FakePtySession:
        session = FakePtySession()
        spawned.append(session)
        return session

    registry._spawn_session = tracked_spawn
    return registry, spawned
