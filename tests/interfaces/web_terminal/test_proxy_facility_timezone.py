"""Every panel page the hub relays carries the facility time zone on its root tag.

``facility-time.js`` renders every instant on the facility clock, and reads the
zone from the ``data-facility-timezone`` attribute on ``<html>``. A panel's own
server may not know that zone, so the hub's panel proxy stamps the zone it
resolves for its own pages on every relayed HTML document. A document that
already names a zone keeps its own: the zone is one deployment's configuration,
so the root tag carries the attribute exactly once.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from osprey.interfaces.common_middleware import (
    FACILITY_TIMEZONE_ATTRIBUTE,
    STORAGE_SCOPE_ATTRIBUTE,
)
from osprey.interfaces.web_terminal.routes.proxy import _stamp_facility_timezone
from tests.interfaces.web_terminal.test_proxy_storage_scope import _DOCUMENT, _relay, _root_tag

_RESOLVER = "osprey.interfaces.web_terminal.routes.proxy.get_facility_timezone"

_FACILITY_TIME_JS = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "osprey"
    / "interfaces"
    / "design_system"
    / "static"
    / "js"
    / "facility-time.js"
)


class TestStampFacilityTimezone:
    def test_the_zone_is_stamped_on_the_root_tag(self):
        out = _stamp_facility_timezone(_DOCUMENT, "text/html", "Asia/Tokyo")
        assert (
            '<html data-facility-timezone="Asia/Tokyo" lang="en" data-browse-orient="row">' in out
        )

    def test_a_document_that_names_its_own_zone_keeps_it(self):
        doc = '<html lang="en" data-facility-timezone="UTC"><head></head></html>'
        assert _stamp_facility_timezone(doc, "text/html", "Asia/Tokyo") == doc

    def test_only_the_root_tag_counts(self):
        doc = '<html lang="en"><body data-facility-timezone="x"></body></html>'
        out = _stamp_facility_timezone(doc, "text/html", "Asia/Tokyo")
        assert out.startswith('<html data-facility-timezone="Asia/Tokyo" lang="en">')

    @pytest.mark.parametrize("base_type", ["text/javascript", "text/css", "application/javascript"])
    def test_a_body_that_is_not_html_is_untouched(self, base_type):
        body = 'const s = "<html lang=x>";'
        assert _stamp_facility_timezone(body, base_type, "Asia/Tokyo") == body

    def test_a_fragment_with_no_root_tag_is_untouched(self):
        fragment = "<div><p>panel</p></div>"
        assert _stamp_facility_timezone(fragment, "text/html", "Asia/Tokyo") == fragment

    def test_the_value_is_escaped_for_the_attribute(self):
        out = _stamp_facility_timezone(_DOCUMENT, "text/html", 'a"><script>x</script>')
        assert "<script>x" not in out
        assert len(re.findall(r"<html\b", out)) == 1


class TestRelayedPanelDocument:
    def test_a_panel_page_carries_the_hubs_zone_once(self, tmp_path):
        with patch(_RESOLVER, return_value=ZoneInfo("Asia/Tokyo")):
            text, _ = _relay(
                tmp_path, None, "/panel/my-dash/", _DOCUMENT, "text/html; charset=utf-8"
            )
        assert 'data-facility-timezone="Asia/Tokyo"' in _root_tag(text)
        assert text.count("data-facility-timezone") == 1

    def test_on_a_mount_the_root_tag_carries_scope_and_zone(self, tmp_path):
        with patch(_RESOLVER, return_value=ZoneInfo("Asia/Tokyo")):
            text, _ = _relay(
                tmp_path, "alice", "/panel/my-dash/", _DOCUMENT, "text/html; charset=utf-8"
            )
        tag = _root_tag(text)
        assert f'{STORAGE_SCOPE_ATTRIBUTE}="alice"' in tag
        assert 'data-facility-timezone="Asia/Tokyo"' in tag

    def test_an_unconfigured_hub_stamps_utc(self, tmp_path):
        with patch(_RESOLVER, return_value=ZoneInfo("UTC")):
            text, _ = _relay(
                tmp_path, None, "/panel/my-dash/", _DOCUMENT, "text/html; charset=utf-8"
            )
        assert 'data-facility-timezone="UTC"' in _root_tag(text)

    def test_a_script_is_not_stamped(self, tmp_path):
        with patch(_RESOLVER, return_value=ZoneInfo("Asia/Tokyo")):
            text, _ = _relay(
                tmp_path,
                None,
                "/panel/my-dash/app.js",
                'const s = "<html lang=x>";',
                "text/javascript",
            )
        assert "data-facility-timezone" not in text


def test_the_stamp_names_the_attribute_the_browser_reads():
    source = _FACILITY_TIME_JS.read_text(encoding="utf-8")
    match = re.search(r"""ZONE_ATTRIBUTE\s*=\s*(['"])([^'"]+)\1""", source)
    assert match, "facility-time.js declares no ZONE_ATTRIBUTE literal"
    assert match.group(2) == FACILITY_TIMEZONE_ATTRIBUTE


def test_the_proxy_and_the_hub_pages_share_one_resolver():
    from osprey.interfaces.web_terminal import app
    from osprey.interfaces.web_terminal.routes import proxy

    assert proxy.get_facility_timezone is app.get_facility_timezone
