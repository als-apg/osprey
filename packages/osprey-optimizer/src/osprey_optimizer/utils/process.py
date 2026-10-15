"""Process-management helpers shared by the API routers."""

from __future__ import annotations

import multiprocessing


def reap_finished_children() -> None:
    """Reap finished child processes to prevent zombie accumulation."""
    multiprocessing.active_children()
