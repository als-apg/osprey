"""Frame sink for the demo-video recorder's CDP screencast.

Chromium's ``Page.screencastFrame`` event carries one JPEG (base64) and a
``metadata.timestamp`` in epoch seconds. :class:`FrameSink` holds the handler
logic independent of Playwright: the recorder wires :meth:`FrameSink.on_frame`
to the CDP session and passes an ``ack`` callable that sends
``Page.screencastFrameAck``.

Frames are thinned to at least :data:`MIN_INTERVAL_S` apart. The newest
skipped frame is held and written by :meth:`FrameSink.flush`, because the
screencast goes silent while the page is still and the final state before a
still period would otherwise be lost.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

MIN_INTERVAL_S = 0.040


class FrameSink:
    """Write screencast frames to *out_dir* and record their times.

    ``frames`` lists ``(path, t)`` for every written frame, in write order,
    where ``t`` is the frame's epoch time in seconds.
    """

    def __init__(self, out_dir: Path | str) -> None:
        self.out_dir = Path(out_dir)
        self.frames: list[tuple[Path, float]] = []
        self._pending: tuple[bytes, float] | None = None
        self._last_written_t: float | None = None

    def on_frame(self, payload: Mapping[str, Any], ack: Callable[[], Any]) -> None:
        """Handle one ``Page.screencastFrame`` payload.

        The frame is acked before anything else, so a slow or failing write
        never stalls the screencast.
        """
        ack()
        metadata = payload.get("metadata") or {}
        t = metadata.get("timestamp")
        t = float(t) if t is not None else time.time()
        data = base64.b64decode(payload["data"])

        if self._last_written_t is None or t - self._last_written_t >= MIN_INTERVAL_S:
            self._pending = None
            self._write(data, t)
        else:
            self._pending = (data, t)

    def flush(self) -> None:
        """Write the held frame, if any. Call at step exit and when the screencast stops."""
        if self._pending is None:
            return
        data, t = self._pending
        self._pending = None
        self._write(data, t)

    def _write(self, data: bytes, t: float) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"frame_{len(self.frames):06d}.jpg"
        path.write_bytes(data)
        self.frames.append((path, t))
        self._last_written_t = t
