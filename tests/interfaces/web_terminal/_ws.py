"""Deadline-bounded reads off a ``TestClient`` websocket.

``WebSocketTestSession.receive`` is ``portal.call(self._send_rx.receive)``: a
wait on the stream the handler under test sends into, with no deadline. A
handler that has died or gone quiet therefore parks the test thread in that
wait until the per-test cap ends the run, minutes later, and the failure is a
timeout with no word on which frame never came. The readers here run the same
receive under ``anyio.fail_after`` on the session's portal, so a quiet handler
is an ``AssertionError`` within seconds that names the frame it owed and the
JSON types that did arrive.
"""

from __future__ import annotations

import json
import time
from typing import Any

import anyio

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
