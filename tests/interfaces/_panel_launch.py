"""Shared stub for the web terminal's companion-panel launcher.

Browser and design-system tests need the WORKSPACE tab to look available
without a real artifact-gallery process behind it. They all patch
``web_terminal.app._launch_panel_server``; this is the one side effect they
share, so the stub's semantics — publish the URLs a test hands it, by registry
key, and leave every other companion panel unlaunched — are stated once; the
gallery-only form is the common case.
"""

from __future__ import annotations

import functools
from enum import Enum
from typing import Any

from osprey.registry.web import panel_url_state_attr
from osprey_connectors_tests._harness._ports import hold_ports


class Unserved(Enum):
    """The type of :data:`UNSERVED`."""

    URL = "unserved"


#: An omitted gallery URL: the faked launch publishes :func:`unserved_url`.
UNSERVED = Unserved.URL


@functools.cache
def unserved_url() -> str:
    """A loopback address that nothing serves on, taken once per process on first use.

    The published URL must point at NOTHING: the hub dials it server-side with
    the process operator secret the moment a browser opens the WORKSPACE tab,
    so a fixed default-block address (the gallery's own 10200) would, on a dev
    host running a real deployment, proxy the developer's live artifacts into
    the test, a false green, and hand that gallery the test's secret. A port
    reserved for the process is refused at once on every host: the panel is
    advertised but unhealthy.
    """
    (port,) = hold_ports(1)
    return f"http://127.0.0.1:{port}"


def publish_panel_urls(urls: dict[str, str | None]):
    """Return a ``_launch_panel_server`` side effect publishing *urls* by registry key.

    Args:
        urls: The URL each companion server should be reported at, keyed by its
            registry key (``"artifact"``, ``"okf"``, ...). A key not in the
            mapping is published as ``None``, the state an unlaunched panel has.

    Returns:
        A ``(app, key)`` callable that publishes ``urls.get(key)`` for *key*.
    """

    def _launch(app: Any, key: str) -> None:
        setattr(app.state, panel_url_state_attr(key), urls.get(key))

    return _launch


def publish_artifact_url(url: str | None | Unserved = UNSERVED):
    """Return a ``_launch_panel_server`` side effect that fakes the gallery launch.

    Args:
        url: What ``/api/artifact-server`` should report. Omitted, it is
            :func:`unserved_url`. ``None`` publishes no URL, which is how a test
            keeps the default panel loaded-but-unhealthy.

    Returns:
        A ``(app, key)`` callable that publishes *url* for the ``artifact``
        server and leaves every other companion panel without a URL — the state
        an unlaunched panel has, so no tab is advertised that nothing serves.
    """
    if url is UNSERVED:
        url = unserved_url()
    return publish_panel_urls({"artifact": url})
