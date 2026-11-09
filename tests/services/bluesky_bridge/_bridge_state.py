"""The one reset of the Bluesky bridge's process-wide state.

The bridge keeps its process-wide state in module singletons, so every test
that drives the bridge in process resets the same list, here. This is interim:
it goes when the app factory builds the bridge state with one close.
"""

from __future__ import annotations

from osprey.services.bluesky_bridge import (
    analysis,
    document_plane,
    draft,
    figure_cache,
    history_removals,
    live_rows,
    plan_loader,
    queue,
    queue_removals,
)
from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge.session_upload import set_session_uploader


def reset_bridge_state() -> None:
    """Empty every bridge singleton and drop any installed backend or uploader."""
    plan_loader.reset_facility_plans()
    draft._clear()
    queue._clear()
    queue_removals._clear()
    history_removals._clear()
    document_plane._clear()
    live_rows._clear()
    analysis._clear()
    figure_cache._clear()
    # Cleared to None rather than left alone: a test that forgets to install a
    # fake lazily builds a managerless real backend (fail-closed 503s) instead
    # of seeing another test's fake.
    app_module.set_queue_backend(None)
    set_session_uploader(None)
