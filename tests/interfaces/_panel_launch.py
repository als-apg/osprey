"""Shared stub for the web terminal's companion-panel launcher.

Browser and design-system tests need the WORKSPACE tab to look available
without a real artifact-gallery process behind it. They all patch
``web_terminal.app._launch_panel_server``; this is the one side effect they
share, so the stub's semantics — publish the gallery URL, leave every other
companion panel unlaunched — are stated once instead of re-derived per file.
"""

from __future__ import annotations

import socket
from typing import Any

from osprey.registry.web import panel_url_state_attr


def _reserve_unserved_port() -> int:
    """A loopback port that nothing serves on, reserved once per process.

    The published URL must point at NOTHING: the hub dials it server-side with
    the process operator secret the moment a browser opens the WORKSPACE tab,
    so a fixed default-block address (the gallery's own 10200) would, on a dev
    host running a real deployment, proxy the developer's live artifacts into
    the test — a false green — and hand that gallery the test's secret. An
    ephemeral port that was just free reproduces everywhere what CI sees:
    connection refused, panel advertised but unhealthy.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


#: The address the faked gallery launch publishes in these tests — deliberately
#: an address nothing serves on; see :func:`_reserve_unserved_port`.
# import-time required because DEFAULT_ARTIFACT_URL is a default argument of
# publish_artifact_url below, and defaults evaluate when the module imports.
DEFAULT_ARTIFACT_URL = f"http://127.0.0.1:{_reserve_unserved_port()}"


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
