"""Every panel page relayed on a multi-user mount carries the mount's storage scope.

On a multi-user deployment every person's terminal is served from one origin,
so ``localStorage`` is shared across the roster. ``storage-scope.js`` keeps one
person's saved preferences out of another's by deriving every key from the
``data-osprey-storage-scope`` attribute on ``<html>``. The hub stamps its own
pages from Jinja; a panel page reaches the browser only through the hub's
panel proxy, and only the hub knows whose mount it is (a companion server may
be shared by the whole roster), so the proxy is the one producer of that stamp
for panel documents. Without a mount user no scope is stamped.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from osprey.interfaces.common_middleware import STORAGE_SCOPE_ATTRIBUTE
from osprey.interfaces.web_terminal.routes.proxy import _stamp_storage_scope
from tests.interfaces.web_terminal._proxy_fakes import (
    PANEL_DOCUMENT,
    relay_panel_response,
    root_tag,
)

#: The design-system scripts that read the attribute in the browser.
_READERS = ("storage-scope.js", "mode-boot.js", "rail-boot.js", "theme-boot.js")

_DESIGN_SYSTEM_JS = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "osprey"
    / "interfaces"
    / "design_system"
    / "static"
    / "js"
)


class TestStampStorageScope:
    def test_the_scope_is_the_first_attribute_of_the_root_tag(self):
        out = _stamp_storage_scope(PANEL_DOCUMENT, "text/html", "alice")
        assert '<html data-osprey-storage-scope="alice" lang="en" data-browse-orient="row">' in out
        assert out.count(STORAGE_SCOPE_ATTRIBUTE) == 1

    def test_an_empty_scope_relays_the_document_byte_for_byte(self):
        assert _stamp_storage_scope(PANEL_DOCUMENT, "text/html", "") == PANEL_DOCUMENT

    @pytest.mark.parametrize("base_type", ["text/javascript", "text/css", "application/javascript"])
    def test_a_body_that_is_not_html_is_untouched(self, base_type):
        body = 'const page = "<html lang=x><head></head></html>";'
        assert _stamp_storage_scope(body, base_type, "alice") == body

    def test_a_fragment_with_no_root_tag_is_untouched(self):
        assert _stamp_storage_scope("<p>x</p>", "text/html", "alice") == "<p>x</p>"

    def test_a_longer_tag_name_is_not_the_root_tag(self):
        out = _stamp_storage_scope("<htmlx><html>", "text/html", "alice")
        assert out == '<htmlx><html data-osprey-storage-scope="alice">'
        assert _stamp_storage_scope("<html-embed>", "text/html", "alice") == "<html-embed>"

    def test_the_tag_name_matches_in_any_case(self):
        out = _stamp_storage_scope("<HTML>", "text/html", "a")
        assert out == '<HTML data-osprey-storage-scope="a">'

    def test_the_value_is_escaped_for_the_attribute(self):
        out = _stamp_storage_scope(PANEL_DOCUMENT, "text/html", 'a"><script>x</script>')
        assert "<script>x" not in out
        assert len(re.findall(r"<html\b", out)) == 1

    def test_the_hubs_scope_precedes_one_the_backend_spelled(self):
        out = _stamp_storage_scope(
            '<html data-osprey-storage-scope="mallory">', "text/html", "alice"
        )
        assert out.startswith(
            '<html data-osprey-storage-scope="alice" data-osprey-storage-scope="mallory">'
        )


class TestRelayedPanelDocument:
    def test_a_panel_page_on_a_mount_carries_the_scope_once(self, tmp_path):
        text, _ = relay_panel_response(
            tmp_path, "alice", "/panel/my-dash/", PANEL_DOCUMENT, "text/html; charset=utf-8"
        )
        assert text.count(STORAGE_SCOPE_ATTRIBUTE) == 1
        assert f'{STORAGE_SCOPE_ATTRIBUTE}="alice"' in root_tag(text)

    def test_the_scope_names_the_user_the_forwarded_prefix_names(self, tmp_path):
        text, headers = relay_panel_response(
            tmp_path, "first.last", "/panel/my-dash/", PANEL_DOCUMENT, "text/html; charset=utf-8"
        )
        forwarded_user = headers["x-forwarded-prefix"].strip("/").split("/")[1]
        match = re.search(rf'{STORAGE_SCOPE_ATTRIBUTE}="([^"]*)"', root_tag(text))
        assert match
        assert match.group(1) == forwarded_user == "first.last"

    def test_without_a_mount_the_page_carries_no_scope(self, tmp_path):
        text, _ = relay_panel_response(
            tmp_path, None, "/panel/my-dash/", PANEL_DOCUMENT, "text/html; charset=utf-8"
        )
        assert STORAGE_SCOPE_ATTRIBUTE not in text
        assert re.sub(r' data-facility-timezone="[^"]*"', "", text) == PANEL_DOCUMENT

    def test_a_blank_user_is_no_mount(self, tmp_path):
        text, _ = relay_panel_response(
            tmp_path, "   ", "/panel/my-dash/", PANEL_DOCUMENT, "text/html; charset=utf-8"
        )
        assert STORAGE_SCOPE_ATTRIBUTE not in text

    def test_a_script_on_a_mount_is_not_stamped(self, tmp_path):
        text, _ = relay_panel_response(
            tmp_path,
            "alice",
            "/panel/my-dash/app.js",
            'const s = "<html lang=x>";',
            "text/javascript",
        )
        assert STORAGE_SCOPE_ATTRIBUTE not in text


def test_the_stamp_names_the_attribute_the_browser_reads():
    for name in _READERS:
        source = (_DESIGN_SYSTEM_JS / name).read_text(encoding="utf-8")
        match = re.search(r"""SCOPE_ATTRIBUTE\s*=\s*(['"])([^'"]+)\1""", source)
        assert match, f"{name} declares no SCOPE_ATTRIBUTE literal"
        assert match.group(2) == STORAGE_SCOPE_ATTRIBUTE, name
