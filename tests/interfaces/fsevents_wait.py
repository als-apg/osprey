"""One wait-side mechanism for the tests that drive a real filesystem observer.

A watchdog observer is not delivering the moment ``start()`` returns. The
emitter thread must still create, schedule and start its stream before it
reaches the run loop that dispatches callbacks — on macOS that is
``_fsevents.add_watch`` at ``watchdog/observers/fsevents.py`` line 307, with the
run loop one line further on at 308. Sampled stacks from an 18-worker box catch
emitters in both frames, so the arming window is seconds wide there, and a
stimulus applied inside it is delivered late or not at all.

A fixed pre-write sleep cannot bound that window. Too short and the stimulus
lands before the stream is live; large enough to be safe under load and it is a
tax every green run pays. So nothing here waits longer.

**These helpers re-apply the stimulus.** ``poke`` is a callable that reproduces
the same filesystem change without altering what the test asserts about —
rewriting a control note with the same bytes, re-saving an index through the
store's own atomic write — and it is called once per interval until the expected
delivery is observed. Whichever way the arming window bit (an event lost
outright, or an event merely slow), a poke issued after the stream is live is
delivered, so the wait converges instead of expiring. A poke that changes what
is under test is a bug in the caller, not a licence this module grants.

**A poke must reproduce the event CLASS, not merely the path.** Every handler
here dispatches on kind — ``created``/``modified``/``deleted`` in the workspace
watcher, and ``on_moved`` in the store watcher, which is the *only* event Linux
inotify delivers for a tempfile-plus-``os.replace`` index write. A poke that
touches the right file by the wrong mechanism answers the wait with a frame the
test's own stimulus would never have produced, and a regression in the handling
of the real class then passes unseen. So a creation is re-applied by removing
and recreating, an atomic replace by another atomic replace — and where the
distinction carries the test, :func:`collect_frames` with ``until_type`` makes
the wait itself hold out for the right kind of frame.

Budget exhaustion is reported as a failure naming what never arrived. It is not
absorbed and it is not retried at the test level: an observer that never
delivers in :data:`ARM_BUDGET` seconds of repeated stimulus is broken, not slow.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

#: Overall ceiling on "this observer is delivering". Generous on purpose: it is
#: only ever spent on a run that is already failing, and the poke loop returns
#: the instant a frame lands, so a healthy run never approaches it.
ARM_BUDGET = 60.0

#: Seconds between re-applications of the stimulus. Comfortably above the 0.1 s
#: per-path debounce both handlers apply, so a poke is never swallowed as a
#: duplicate of the poke before it.
POKE_INTERVAL = 1.0

#: How often the wait looks at the queue while it is not poking.
POLL = 0.05

#: Quiet period that ends a drain, and the hard stop on extending it. A frame
#: coalesced by the OS can trail the one that was waited for by a beat; each
#: further frame restarts the floor, and the ceiling stops a stream that is
#: still producing from holding the test open indefinitely.
DRAIN_FLOOR = 0.5
DRAIN_CEILING = 2.0


def poke_until(
    check: Callable[[], bool],
    poke: Callable[[], Any],
    *,
    what: str,
    budget: float = ARM_BUDGET,
) -> None:
    """Re-apply *poke* until *check* passes, then return; fail at *budget*.

    The caller has already applied the stimulus once, so the first poke comes
    one :data:`POKE_INTERVAL` later rather than immediately.

    Args:
        check: True once the delivery under test has been observed. Called
            first, before any poke, so an already-delivered event costs nothing.
        poke: Reproduces the stimulus. Must not change what the test asserts.
        what: Named in the failure message — say what never arrived.
        budget: Seconds of repeated stimulus before giving up.

    Raises:
        AssertionError: *check* never passed within *budget*.
    """
    started = time.monotonic()
    deadline = started + budget
    next_poke = started + POKE_INTERVAL
    pokes = 0
    while True:
        if check():
            return
        now = time.monotonic()
        if now >= deadline:
            raise AssertionError(
                f"{what} was never delivered in {budget:.0f}s of a live observer, "
                f"across {pokes} re-applications of the stimulus — the observer is "
                f"not delivering, rather than delivering late"
            )
        if now >= next_poke:
            poke()
            pokes += 1
            next_poke = now + POKE_INTERVAL
        time.sleep(POLL)


def drain(pump: Callable[[], bool]) -> None:
    """Collect whatever is still in flight behind the frame already waited for.

    Waits :data:`DRAIN_FLOOR` seconds of quiet, restarts that quiet window every
    time *pump* reports another frame, and stops at :data:`DRAIN_CEILING` regardless — a stream still
    producing at the ceiling has said enough, and the caller keeps what it
    collected rather than waiting on a tail that may not end. The ceiling branch
    pumps once on its way out, so nothing already queued is left behind.

    Args:
        pump: Takes everything currently available; True if it took anything.
    """
    started = time.monotonic()
    quiet_until = started + DRAIN_FLOOR
    hard_stop = started + DRAIN_CEILING
    while True:
        now = time.monotonic()
        if now >= hard_stop:
            # Take what is already queued before leaving: the contract is that
            # ceiling exhaustion returns what was *collected*, and a frame
            # sitting in the queue at the instant the ceiling passes has been
            # delivered. Dropping it would make an absence assertion read as
            # clean when the frame it forbids had in fact arrived.
            pump()
            return
        if pump():
            quiet_until = now + DRAIN_FLOOR
        elif now >= quiet_until:
            return
        else:
            time.sleep(POLL)


def collect_frames(
    queue: asyncio.Queue,
    *,
    until: str | None = None,
    until_under: str | None = None,
    poke: Callable[[], Any],
    until_type: str | None = None,
    budget: float = ARM_BUDGET,
) -> list[dict]:
    """Every frame a broadcaster queue delivers up to and shortly after *until*.

    The frame-queue shape of :func:`poke_until` plus :func:`drain`, shared by the
    workspace-watcher tests: wait for the sentinel frame, re-applying the change
    that produces it until it arrives, then take the coalesced tail behind it.

    A sentinel proves the observer is live and delivering. It does not prove
    that every event before it was delivered: a loaded FSEvents daemon hands a
    stream its events in sparse batches, and a later write's frame can land
    while an earlier write's frames are still pending. So a test that requires
    a frame to be *present* waits for that frame — or, with *until_under*, for
    any frame at or below a directory — and pokes with the stimulus that
    produces it, never with a bystander write.

    Frames are returned whole rather than flattened to paths, because a poke
    only proves what it reproduces: a test whose subject is a *creation* has to
    be able to require a ``created`` frame, or a re-applied modification of the
    same path would answer for it. Pass *until_type* to make the wait itself
    require that class, so the poke loop keeps going until the right kind of
    frame lands rather than stopping at the first frame naming the path.

    Args:
        queue: A ``FileEventBroadcaster`` subscription.
        until: The workspace-relative path whose frame ends the wait.
        until_under: A workspace-relative directory instead of a path: the
            first frame at or below it ends the wait. For a store whose write
            lands at a random temp name and a rename, no single path can be
            named in advance, but the store directory can. Exactly one of
            *until* and *until_under* is given.
        poke: Re-applies the change that produces the awaited frame.
        until_type: When given, only a frame of this ``type`` ends the wait.
        budget: Seconds of repeated stimulus before failing.

    Returns:
        Every frame delivered, in arrival order, sentinel included.
    """
    if (until is None) == (until_under is None):
        raise ValueError("pass exactly one of until= and until_under=")
    frames: list[dict] = []

    def pump() -> bool:
        took = False
        while True:
            try:
                frames.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                return took
            took = True

    def at_target(path: str) -> bool:
        if until is not None:
            return path == until
        prefix = until_under.rstrip("/")  # type: ignore[union-attr]
        return path == prefix or path.startswith(prefix + "/")

    def matched(frame: dict) -> bool:
        return at_target(frame["path"]) and (until_type is None or frame["type"] == until_type)

    def arrived() -> bool:
        pump()
        return any(matched(frame) for frame in frames)

    where = f"for {until!r}" if until is not None else f"under {until_under!r}"
    wanted = f"a {until_type!r} frame {where}" if until_type else f"a frame {where}"
    poke_until(arrived, poke, what=wanted, budget=budget)
    drain(pump)
    return frames


def collect_paths(
    queue: asyncio.Queue,
    *,
    until: str | None = None,
    until_under: str | None = None,
    poke: Callable[[], Any],
    budget: float = ARM_BUDGET,
) -> list[str]:
    """The paths of :func:`collect_frames`, for callers that assert on path alone.

    Only for tests whose subject is *which* paths reached the panel, never
    whether a particular kind of change did — those must read the frames.
    """
    frames = collect_frames(queue, until=until, until_under=until_under, poke=poke, budget=budget)
    return [frame["path"] for frame in frames]
