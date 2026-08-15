"""Shared stub for the web terminal's companion-panel launcher.

Browser and design-system tests need the WORKSPACE tab to look available
without a real artifact-gallery process behind it. They all patch
``web_terminal.app._launch_panel_server``; this is the one side effect they
share, so the stub's semantics — publish the gallery URL, leave every other
companion panel unlaunched — are stated once instead of re-derived per file.
"""

from __future__ import annotations

from typing import Any

from osprey.registry.web import panel_url_state_attr

#: The address the pre-launched/faked gallery is reachable at in these tests.
DEFAULT_ARTIFACT_URL = "http://127.0.0.1:8086"


def publish_artifact_url(url: str | None = DEFAULT_ARTIFACT_URL):
    """Return a ``_launch_panel_server`` side effect that fakes the gallery launch.

    Args:
        url: What ``/api/artifact-server`` should report. ``None`` publishes no
            URL, which is how a test keeps the default panel loaded-but-unhealthy.

    Returns:
        A ``(app, key)`` callable that publishes *url* for the ``artifact``
        server and leaves every other companion panel without a URL — the state
        an unlaunched panel has, so no tab is advertised that nothing serves.
    """

    def _launch(app: Any, key: str) -> None:
        setattr(app.state, panel_url_state_attr(key), url if key == "artifact" else None)

    return _launch
