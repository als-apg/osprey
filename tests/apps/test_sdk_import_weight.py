"""Importing the application SDK never loads the queue client library.

Tool servers and application containers import ``osprey.apps.sdk`` and may run
without ``bluesky_queueserver_api``; ``queue_is_active`` loads it only when called.
"""

from __future__ import annotations

from tests._fresh_import import non_stdlib_added_by_import


def test_a_fresh_sdk_import_loads_no_queue_client():
    added = non_stdlib_added_by_import("osprey.apps.sdk")
    assert added, "the fresh import reported no module at all"
    assert [name for name in added if name.split(".")[0] == "bluesky_queueserver_api"] == []
