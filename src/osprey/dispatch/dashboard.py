"""Dashboard HTML rendering with runtime config injection."""

from __future__ import annotations

import json
from pathlib import Path

_DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"


def render_dashboard_html(facility_name: str = "", pv_strip_prefix: str = "") -> str:
    """Read dashboard.html and inject runtime config via the OSPREY_CONFIG_PLACEHOLDER sentinel.

    The HTML carries a ``/* OSPREY_CONFIG_PLACEHOLDER */`` comment that this replaces
    with a ``window.__OSPREY_CONFIG__ = {...};`` literal. If the sentinel is absent,
    the HTML is returned unchanged — the replacement is a harmless no-op.

    Args:
        facility_name: Facility display name injected into the dashboard config.
        pv_strip_prefix: PV prefix the dashboard strips when rendering channel names.

    Returns:
        The dashboard HTML with the runtime config injected where the sentinel is present.
    """
    html = _DASHBOARD_HTML.read_text()
    config = json.dumps({"facility_name": facility_name, "pv_strip_prefix": pv_strip_prefix})
    return html.replace("/* OSPREY_CONFIG_PLACEHOLDER */", f"window.__OSPREY_CONFIG__ = {config};")
