"""Dashboard HTML rendering with runtime config injection."""

from __future__ import annotations

import json
from pathlib import Path

_DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"


def render_dashboard_html(
    facility_name: str = "",
    channel_strip_prefix: str = "",
    telemetry_url: str = "",
) -> str:
    """Read dashboard.html and inject runtime config via the OSPREY_CONFIG_PLACEHOLDER sentinel.

    The HTML carries a ``/* OSPREY_CONFIG_PLACEHOLDER */`` comment that this replaces
    with a ``window.__OSPREY_CONFIG__ = {...};`` literal. If the sentinel is absent,
    the HTML is returned unchanged — the replacement is a harmless no-op.

    Args:
        facility_name: Facility display name injected into the dashboard config.
        channel_strip_prefix: Leading prefix trimmed off a channel address before the
            dashboard renders it. Applied to the sources that carry one; a trigger
            whose source renders an interval or a bare name is unaffected.
        telemetry_url: Browser-reachable base URL of the telemetry store (e.g.
            ``http://localhost:5080``), or ``""`` when the store is not deployed
            or agent telemetry is off. Must be the address the OPERATOR's browser
            can reach — not the compose DNS name the containers use — because the
            dashboard renders it as a link the browser follows. Empty disables the
            dashboard's per-run telemetry link entirely.

    Returns:
        The dashboard HTML with the runtime config injected where the sentinel is present.
    """
    html = _DASHBOARD_HTML.read_text()
    config = json.dumps(
        {
            "facility_name": facility_name,
            "channel_strip_prefix": channel_strip_prefix,
            "telemetry_url": telemetry_url,
        }
    )
    return html.replace("/* OSPREY_CONFIG_PLACEHOLDER */", f"window.__OSPREY_CONFIG__ = {config};")
