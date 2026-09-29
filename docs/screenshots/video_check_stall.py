"""Tell a working demo agent from a stuck one, within seconds.

A take waits on the agent for minutes at a time. Two things end such a wait
early instead of at the end of its budget:

* a dialog on screen that the take never answers (a one-time notice waiting
  for Enter, a permission prompt nobody clicks);
* no progress for :data:`STALL_S` seconds: neither the terminal text nor the
  session's transcripts (the main one and its subagents') changed.

The spinner's elapsed-time counter ticks while the agent is stuck, so it is
stripped before screens are compared; streamed token counts are kept, since
they move only while the model writes.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path

STALL_S = 45.0

# Dialogs no step of the take answers. The approval prompt the storyboard does
# answer is recognized through the hook log, never through screen text.
NOTICE_DIALOGS = ("Enter to continue",)
PLOT_DIALOGS = (*NOTICE_DIALOGS, "Do you want", "Esc to cancel")

_ELAPSED = re.compile(r"\b\d+(?:\.\d+)?[hms]\b")
_SPINNER = re.compile(r"[·✢✳✶✻✽*]")
_BLANKS = re.compile(r"[ \t]+")


def screen_fingerprint(text: str) -> str:
    """``text`` without the spinner glyph and its elapsed-time counter."""
    return _BLANKS.sub(" ", _SPINNER.sub("", _ELAPSED.sub("", text)))


def dialog_on_screen(text: str, markers: Iterable[str]) -> str | None:
    """The first of ``markers`` shown in ``text``, or None."""
    return next((marker for marker in markers if marker in text), None)


def transcript_bytes(transcript: Path | None) -> int:
    """Bytes in the session transcript plus its subagents' transcripts."""
    if transcript is None:
        return 0
    total = 0
    files = [transcript, *Path(transcript).with_suffix("").rglob("*.jsonl")]
    for path in files:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


class StallWatch:
    """Seconds since the screen or the transcripts last changed."""

    def __init__(self, clock: Callable[[], float], stall_s: float = STALL_S) -> None:
        self.clock = clock
        self.stall_s = stall_s
        self._last: tuple[str, int] | None = None
        self._since = clock()
        self.quiet_s = 0.0

    def observe(self, screen: str, size: int) -> float:
        """Record one sample; return the seconds without progress so far."""
        now = self.clock()
        sample = (screen_fingerprint(screen), size)
        if sample != self._last:
            self._last = sample
            self._since = now
        self.quiet_s = now - self._since
        return self.quiet_s

    @property
    def stalled(self) -> bool:
        return self.quiet_s >= self.stall_s
