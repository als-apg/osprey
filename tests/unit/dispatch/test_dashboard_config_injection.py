"""Tests for dashboard HTML runtime config injection (de-ALS)."""

from __future__ import annotations

from osprey.dispatch.dashboard import render_dashboard_html


def test_injects_config_and_consumes_sentinel():
    """render_dashboard_html replaces the sentinel with a window.__OSPREY_CONFIG__ literal."""
    html = render_dashboard_html(facility_name="ALS Demo", pv_strip_prefix="ALS:")

    # The injected config literal is present with the supplied values.
    assert "window.__OSPREY_CONFIG__ = {" in html
    assert '"facility_name": "ALS Demo"' in html
    assert '"pv_strip_prefix": "ALS:"' in html

    # The raw placeholder sentinel was consumed (replaced), not left behind.
    assert "/* OSPREY_CONFIG_PLACEHOLDER */" not in html


def test_default_injection_is_empty_config():
    """With no args, the injected config carries empty defaults."""
    html = render_dashboard_html()
    assert '"facility_name": ""' in html
    assert '"pv_strip_prefix": ""' in html
    assert '"telemetry_url": ""' in html
    assert "/* OSPREY_CONFIG_PLACEHOLDER */" not in html


def test_injects_the_browser_facing_telemetry_url():
    """The telemetry store's browser URL reaches the dashboard config verbatim."""
    html = render_dashboard_html(telemetry_url="http://localhost:5080")
    assert '"telemetry_url": "http://localhost:5080"' in html


def test_absent_telemetry_url_is_the_no_store_signal():
    """A deployment without the store injects an empty string, not a default.

    The dashboard reads empty as "hide the per-run telemetry link". Substituting
    a plausible default here would produce a link to a host that is not serving
    anything -- worse than no link, because it looks deliberate.
    """
    html = render_dashboard_html(facility_name="Test")
    assert '"telemetry_url": ""' in html
    assert "localhost:5080" not in html


def test_dashboard_is_de_alsed():
    """The hardcoded ALS PV-prefix strip is gone; the prefix is config-driven."""
    html = render_dashboard_html()
    # The old hardcoded ALS regex must not survive.
    assert "/^ALS:/" not in html
    # The PV display now strips a configurable prefix from window.__OSPREY_CONFIG__.
    assert "window.__OSPREY_CONFIG__.pv_strip_prefix" in html


def test_dashboard_token_handoff_uses_fragment_not_query():
    """The bearer-token URL handoff uses the fragment (#token), never the query string.

    Query-string tokens leak into server access logs and browser history; the
    fragment is not transmitted to the server.
    """
    html = render_dashboard_html()
    assert "searchParams.get('token')" not in html
    assert "window.location.hash" in html
