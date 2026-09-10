"""Watchdog primitives shared by the interface file watchers.

Both the workspace watcher behind the file panel and the artifacts store-index
watcher run on the same backend and read a coalesced directory frame the same
way, so the observer seam and the listing they diff live here once rather than
once per watcher.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from watchdog.observers.api import BaseObserver

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
