"""Watchdog primitives shared by the interface file watchers.

Both the workspace watcher behind the file panel and the artifacts store-index
watcher run on the same backend and read a coalesced directory frame the same
way, so the observer seam and the listing they diff live here once rather than
once per watcher.

The second trigger lives here for the same reason: a notification stream can go
quiet, and the timer that makes each watcher re-read what it tracks is one
mechanism on one setting rather than one per watcher.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, MutableMapping
from pathlib import Path

from watchdog.observers.api import BaseObserver

logger = logging.getLogger(__name__)

#: Builds the watchdog observer a watcher runs on.
#:
#: The default is the platform's native observer — FSEvents on macOS, inotify on
#: Linux — which is what a deployment runs and the only backend that delivers an
#: atomic file replacement as ``on_moved``. A caller that needs delivery to be
#: decided by the filesystem's contents rather than by a notification stream
#: passes ``watchdog.observers.polling.PollingObserver`` instead: it compares two
#: directory snapshots on a fixed interval, so a change is reported whenever it
#: was made and never coalesced away, at the cost of the event class the native
#: backend would have used.
ObserverFactory = Callable[[], BaseObserver]

#: One directory entry as ``(is_dir, mtime_ns, size)``.
ChangeStamp = tuple[bool, int, int]


def one_level_listing(directory: Path) -> dict[str, ChangeStamp]:
    """One level of *directory* as ``{name: (is_dir, mtime_ns, size)}``.

    The tuple is a change stamp, not metadata anyone reads: two listings of the
    same directory differ at exactly the names that were added, removed, or
    written to. A directory that has gone away between the frame and the scan
    lists as empty rather than raising — the frame is a report about the past.

    One ``scandir`` and the stats it already carries, so the cost of reading a
    frame this way is one syscall plus the entries the directory holds.
    """
    listing: dict[str, ChangeStamp] = {}
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return listing
    for entry in entries:
        try:
            stat = entry.stat()
            listing[entry.name] = (entry.is_dir(), stat.st_mtime_ns, stat.st_size)
        except OSError:  # vanished mid-scan
            continue
    return listing


def entry_stamp(path: Path) -> ChangeStamp | None:
    """*path* as the single entry a listing of its parent would hold for it.

    ``None`` when the path is not there. Otherwise the same
    ``(is_dir, mtime_ns, size)`` tuple :func:`one_level_listing` builds for a
    name in a directory, which is what makes it useful: a stamp taken here and
    a stamp taken by a listing of the parent describe the same file
    identically, so a change recorded through one can be diffed against the
    other without the two disagreeing about what "unchanged" means.

    Both follow symlinks — ``os.scandir``'s ``entry.is_dir()`` and
    ``entry.stat()`` do, and so do :meth:`Path.is_dir` and :meth:`Path.stat`
    here — which is why they agree about a name that is one.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (path.is_dir(), stat.st_mtime_ns, stat.st_size)


#: Seconds between two reconciliation passes when the deployment says nothing.
#:
#: Short enough that a change lost by a notification stream reaches the browser
#: while the operator is still looking at what they did, long enough that a
#: directory listing per tracked directory is not a load anyone measures.
DEFAULT_RECONCILE_SECONDS = 2.0


def reconcile_interval_seconds() -> float:
    """Seconds between the reconciliation passes both interface watchers run.

    One setting for both, read here because this is the one module they already
    share. The config import is local, so importing a watcher does not pull in
    the config machinery, and a read with no config primed falls back to the
    default rather than raising — a watcher must start whether or not a
    deployment config is in front of it.

    A value that is not a positive number is logged and the default kept. No
    value disables the pass: an off switch here is the blind watcher back under
    another name.

    Read on every call — never cached at import — so a test and a re-primed
    process see the value their config declares.
    """
    try:
        from osprey.utils.config import get_config_value

        configured = get_config_value(
            "web.file_watch_reconcile_interval_s", DEFAULT_RECONCILE_SECONDS
        )
    except Exception:
        logger.debug("No config available for web.file_watch_reconcile_interval_s", exc_info=True)
        configured = None

    if isinstance(configured, (int, float)) and not isinstance(configured, bool) and configured > 0:
        return float(configured)
    if configured is not None and configured != DEFAULT_RECONCILE_SECONDS:
        logger.warning(
            "web.file_watch_reconcile_interval_s must be a positive number (got %r); using %s s",
            configured,
            DEFAULT_RECONCILE_SECONDS,
        )
    return DEFAULT_RECONCILE_SECONDS


class Reconciler:
    """The trigger a watcher keeps when its notification stream goes quiet.

    A watcher whose only trigger is the platform's notification stream reports
    nothing for as long as that stream delivers nothing, and never learns what
    it missed: the stream is the fast trigger and the only one that can fall
    silent. This is the second trigger, and it stands on the filesystem rather
    than on a notification — one pass every *interval*, whatever the stream is
    doing — so the watcher's silence is bounded rather than open-ended.

    The loop waits on an event rather than sleeping, so :meth:`stop` returns
    without waiting out an interval. An exception from the pass is logged and
    the loop continues: a watcher must not fall silent because one directory
    listing raised.
    """

    def __init__(self, interval: float, pass_once: Callable[[], None]) -> None:
        """
        Args:
            interval: Seconds between two passes.
            pass_once: Re-reads what the watcher tracks and dispatches whatever
                the stream did not.
        """
        self._interval = interval
        self._pass_once = pass_once
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def interval(self) -> float:
        """Seconds between two passes."""
        return self._interval

    def start(self) -> None:
        """Run the pass on its interval until :meth:`stop`."""
        if self._thread is not None:
            return
        self._stopping.clear()
        self._thread = threading.Thread(target=self._loop, name="osprey-fs-reconcile", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """End the loop and wait briefly for its thread; safe to call twice."""
        self._stopping.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stopping.wait(self._interval):
            try:
                self._pass_once()
            except Exception:
                logger.warning("A filesystem reconciliation pass failed", exc_info=True)


def evict_subtree(listings: MutableMapping[str, dict[str, ChangeStamp]], directory: Path) -> None:
    """Drop the listing for *directory* and for everything that was under it.

    The invariant is that a path no longer in the tree has no listing, and
    neither does anything that was under it: a watcher's listing map is bounded
    by the tree that exists rather than by every tree that ever did, however
    long the process runs.

    The keys are ``str(Path)``, so a descendant is one that begins with
    *directory* followed by the separator. The separator is part of the prefix
    because without it ``/a/bc`` is evicted along with ``/a/b``.
    """
    key = str(directory)
    prefix = key + os.sep
    for listed in [k for k in listings if k == key or k.startswith(prefix)]:
        del listings[listed]
