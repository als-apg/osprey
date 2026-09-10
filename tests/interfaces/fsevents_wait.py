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

**A watcher running on a ``PollingObserver`` needs none of that**, and the
wait-side functions for it live here too so a test module still imports one
module for "wait until the watcher has said something". A polling emitter has
no stream to arm and nothing to coalesce: it re-reads the watched directory on
every interval and diffs it against the snapshot it took last, so a change made
after that baseline is reported whether or not anything re-applies it. What a
polling test does have to wait for is the *baseline itself* —
:func:`wait_for_polling_baseline` — because a change already on disk when the
snapshot is taken is inside it and is never a difference.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from watchdog.observers.api import BaseObserver
from watchdog.utils.dirsnapshot import EmptyDirectorySnapshot

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
    poke: Callable[[], Any] | None,
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
            ``None`` for a polling observer, which re-reads the directory itself
            and so has nothing for a poke to fix — see :func:`wait_for`.
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
            if poke is None:
                raise AssertionError(
                    f"{what} was never delivered in {budget:.0f}s of a polling "
                    f"observer — the watcher is not reporting the change, rather "
                    f"than reporting it late"
                )
            raise AssertionError(
                f"{what} was never delivered in {budget:.0f}s of a live observer, "
                f"across {pokes} re-applications of the stimulus — the observer is "
                f"not delivering, rather than delivering late"
            )
        if poke is not None and now >= next_poke:
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


def collect_frames_matching(
    queue: asyncio.Queue,
    *,
    matches: Callable[[dict], bool],
    poke: Callable[[], Any] | None,
    what: str,
    budget: float = ARM_BUDGET,
) -> list[dict]:
    """Every frame delivered up to and shortly after one *matches* accepts.

    The predicate form of :func:`collect_frames`, for a test whose sentinel is a
    *class* of frames rather than one named path — "anything under this
    directory", whose leaf is an atomic write's temp file and therefore has no
    name to spell.

    A test asserting a frame is **present** must wait for that frame here rather
    than for an ordinary note written after it. A note only proves the stream
    reached the note; frames the OS coalesced on the way past are not recovered
    by waiting longer, so a presence assertion hung on someone else's sentinel
    fails for a reason that has nothing to do with its subject. Only a test that
    asserts no *absence* may poke with the very change it is about — pokes here
    reproduce the subject itself, so a caller that also forbids something must
    use :func:`collect_frames` and a poke that cannot manufacture it.

    Args:
        queue: A ``FileEventBroadcaster`` subscription.
        matches: True for a frame that ends the wait.
        poke: Re-applies the change that produces such a frame.
        what: Named in the failure message — say what never arrived.
        budget: Seconds of repeated stimulus before failing.

    Returns:
        Every frame delivered, in arrival order, sentinel included.
    """
    frames: list[dict] = []

    def pump() -> bool:
        took = False
        while True:
            try:
                frames.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                return took
            took = True

    def arrived() -> bool:
        pump()
        return any(matches(frame) for frame in frames)

    poke_until(arrived, poke, what=what, budget=budget)
    drain(pump)
    return frames


def collect_frames(
    queue: asyncio.Queue,
    *,
    until: str,
    poke: Callable[[], Any] | None,
    until_type: str | None = None,
    budget: float = ARM_BUDGET,
) -> list[dict]:
    """Every frame a broadcaster queue delivers up to and shortly after *until*.

    The frame-queue shape of :func:`poke_until` plus :func:`drain`, shared by the
    workspace-watcher tests: wait for the sentinel frame, re-applying the change
    that produces it until it arrives, then take the coalesced tail behind it.

    Frames are returned whole rather than flattened to paths, because a poke
    only proves what it reproduces: a test whose subject is a *creation* has to
    be able to require a ``created`` frame, or a re-applied modification of the
    same path would answer for it. Pass *until_type* to make the wait itself
    require that class, so the poke loop keeps going until the right kind of
    frame lands rather than stopping at the first frame naming the path.

    Args:
        queue: A ``FileEventBroadcaster`` subscription.
        until: The workspace-relative path whose frame ends the wait.
        poke: Re-applies the change that produces the *until* frame.
        until_type: When given, only a frame of this ``type`` ends the wait.
        budget: Seconds of repeated stimulus before failing.

    Returns:
        Every frame delivered, in arrival order, sentinel included.
    """

    def matches(frame: dict) -> bool:
        return frame["path"] == until and (until_type is None or frame["type"] == until_type)

    wanted = f"a {until_type!r} frame for {until!r}" if until_type else f"a frame for {until!r}"
    return collect_frames_matching(queue, matches=matches, poke=poke, what=wanted, budget=budget)


def collect_paths(
    queue: asyncio.Queue,
    *,
    until: str,
    poke: Callable[[], Any],
    budget: float = ARM_BUDGET,
) -> list[str]:
    """The paths of :func:`collect_frames`, for callers that assert on path alone.

    Only for tests whose subject is *which* paths reached the panel, never
    whether a particular kind of change did — those must read the frames.
    """
    return [frame["path"] for frame in collect_frames(queue, until=until, poke=poke, budget=budget)]


# ── polling observers ──────────────────────────────────────────────────────


def wait_for_polling_baseline(observer: BaseObserver, *, budget: float = ARM_BUDGET) -> None:
    """Block until every emitter of a ``PollingObserver`` has snapshotted its watch.

    ``PollingEmitter`` reports what differs from the snapshot it takes in
    ``on_thread_start``, and ``Observer.start()`` returns once that thread
    exists rather than once it has run. A file created in between is inside the
    baseline and is therefore not a difference — the polling counterpart of the
    arming window above, except that it is *observable* rather than only
    provokable, so it is waited for directly instead of poked at.

    The emitter's snapshot attribute is private to watchdog. It is read anyway
    because the alternative is a sleep, and a sleep long enough to be safe on a
    loaded box is a tax every green run pays for a condition that is plainly
    visible.

    Args:
        observer: A started ``PollingObserver``.
        budget: Seconds to wait before failing.

    Raises:
        AssertionError: No emitter ever took a baseline snapshot.
    """
    deadline = time.monotonic() + budget
    while True:
        emitters = list(observer.emitters)
        if emitters and all(
            not isinstance(emitter._snapshot, EmptyDirectorySnapshot) for emitter in emitters
        ):
            return
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"a polling observer took no baseline snapshot in {budget:.0f}s — "
                f"its emitter thread never ran"
            )
        time.sleep(POLL)


def wait_for(check: Callable[[], bool], *, what: str, budget: float = ARM_BUDGET) -> None:
    """:func:`poke_until` for a polling watcher: the same wait, with no stimulus.

    A polling emitter re-reads the directory itself, so there is nothing for a
    poke to fix and a poke would only add a second change to the stream the test
    is reading.
    """
    poke_until(check, None, what=what, budget=budget)


def polled_frames(
    queue: asyncio.Queue,
    *,
    until: str,
    until_type: str | None = None,
    budget: float = ARM_BUDGET,
) -> list[dict]:
    """:func:`collect_frames` for a polling watcher: the same wait, with no stimulus."""
    return collect_frames(queue, until=until, until_type=until_type, poke=None, budget=budget)
