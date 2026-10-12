"""Tests for dashboard HTML runtime config injection."""

from __future__ import annotations

from osprey.dispatch.dashboard import render_dashboard_html


def test_injects_config_and_consumes_sentinel():
    """render_dashboard_html replaces the sentinel with a window.__OSPREY_CONFIG__ literal."""
    html = render_dashboard_html(facility_name="Example Demo", channel_strip_prefix="ERF:")

    # The injected config literal is present with the supplied values.
    assert "window.__OSPREY_CONFIG__ = {" in html
    assert '"facility_name": "Example Demo"' in html
    assert '"channel_strip_prefix": "ERF:"' in html

    # The raw placeholder sentinel was consumed (replaced), not left behind.
    assert "/* OSPREY_CONFIG_PLACEHOLDER */" not in html


def test_default_injection_is_empty_config():
    """With no args, the injected config carries empty defaults."""
    html = render_dashboard_html()
    assert '"facility_name": ""' in html
    assert '"channel_strip_prefix": ""' in html
    assert "/* OSPREY_CONFIG_PLACEHOLDER */" not in html


def test_config_names_no_telemetry_store():
    """The dashboard is handed no telemetry store address and builds no link to one."""
    html = render_dashboard_html(facility_name="Test")
    injected = html.split("window.__OSPREY_CONFIG__ = ", 1)[1].split(";", 1)[0]
    assert "telemetry" not in injected
    assert "telemetryUrlForRun" not in html


def test_injects_the_history_url():
    """The history page's URL reaches the dashboard config; empty by default."""
    url = "https://readers.example.org:10002/history/"
    assert f'"history_url": "{url}"' in render_dashboard_html(history_url=url)
    assert '"history_url": ""' in render_dashboard_html()
