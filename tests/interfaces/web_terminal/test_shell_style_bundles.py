"""The shell page links its own style sheets as two bundles, not one by one.

A cold load of the shell fetched every style sheet of the page as a request of
its own before the first paint. The sheets the shell owns are now served
concatenated, at fingerprinted addresses: ``base`` ahead of the vendored
sheets, ``shell`` behind them, each holding its sheets in the order the page
used to link them, so the cascade is the one the page had.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urljoin

import pytest

from osprey.interfaces._assets import CACHE_IMMUTABLE
from osprey.interfaces.web_terminal.app import SHELL_STYLE_BUNDLES, STATIC_DIR
from tests.interfaces.web_terminal._started_app import terminal_client

_STYLESHEET_LINK = re.compile(r'<link rel="stylesheet"[^>]*?href="([^"]+)"')
_BUNDLE = re.compile(r"/static/_v/[0-9a-f]+/_bundle/(\w+)\.css$")


@pytest.fixture
def workspace_dir(tmp_path: Path) -> Path:
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    return workspace


def _stylesheets(body: str) -> list[str]:
    return _STYLESHEET_LINK.findall(body)


def _own(links: list[str]) -> list[str]:
    """The links to sheets the shell itself owns: not vendored, not a theme."""
    return [
        link for link in links if not re.search(r"/vendor/|^https?:", link) and "hljs" not in link
    ]


def _preloaded_fonts(page: str) -> set[str]:
    return set(re.findall(r'<link rel="preload" as="font"[^>]*?href="([^"]+)"', page))


class TestTheShellLinksItsSheetsAsBundles:
    def test_every_sheet_of_the_shell_is_in_one_of_two_links(self, workspace_dir):
        with terminal_client(workspace_dir) as client:
            own = _own(_stylesheets(client.get("/").text))

        assert [_BUNDLE.search(link).group(1) for link in own] == ["base", "shell"]

    def test_no_sheet_of_a_bundle_is_also_linked_on_its_own(self, workspace_dir):
        with terminal_client(workspace_dir) as client:
            body = client.get("/").text

        linked = {re.sub(r"/_v/[0-9a-f]+/", "/", link) for link in _stylesheets(body)}
        bundled = {sheet for sheets in SHELL_STYLE_BUNDLES.values() for sheet in sheets}
        assert linked & bundled == set()

    def test_the_vendored_sheets_stay_between_the_two(self, workspace_dir):
        with terminal_client(workspace_dir) as client:
            links = _stylesheets(client.get("/").text)

        own = [index for index, link in enumerate(links) if _BUNDLE.search(link)]
        assert own == [0, len(links) - 1]
        assert len(links) > 2

    def test_the_bundle_links_carry_the_mount_prefix(self, workspace_dir):
        with terminal_client(
            workspace_dir, patch.dict("os.environ", {"OSPREY_TERMINAL_USER": "alice"})
        ) as client:
            own = _own(_stylesheets(client.get("/").text))

        assert len(own) == 2
        assert all(link.startswith("/u/alice/static/_v/") for link in own)

    def test_the_page_names_no_sheet_the_bundles_do_not_hold(self):
        # A sheet is added to the shell in ``SHELL_STYLE_BUNDLES``; a link
        # written into the page would be a request of its own again.
        page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        assert re.findall(r"prefixed\('(/(?:static/css|design-system/css)/[^']+)'\)", page) == []
        assert page.count("stylesheet_bundle(") == len(SHELL_STYLE_BUNDLES)


class TestABundleIsItsSheetsInOrder:
    @pytest.mark.parametrize("name", sorted(SHELL_STYLE_BUNDLES))
    def test_holds_every_source_sheet_whole_and_in_order(self, workspace_dir, name):
        with terminal_client(workspace_dir) as client:
            assets = client.app.state.assets
            body = client.get(assets.bundle_url(name)).text
            sources = {sheet: client.get(sheet).text for sheet in SHELL_STYLE_BUNDLES[name]}

        marks = [body.index(f"/* {sheet} */") for sheet in SHELL_STYLE_BUNDLES[name]]
        assert marks == sorted(marks)
        assert len(marks) == len(SHELL_STYLE_BUNDLES[name])
        for sheet, source in sources.items():
            if "url(" in source.replace("url(data:", "").replace('url("data:', ""):
                continue  # its references are rewritten; covered below
            assert source.rstrip() in body, sheet

    def test_every_sheet_named_exists(self, workspace_dir):
        with terminal_client(workspace_dir) as client:
            for sheets in SHELL_STYLE_BUNDLES.values():
                for sheet in sheets:
                    assert client.get(sheet).status_code == 200, sheet

    def test_no_sheet_is_in_two_bundles(self):
        every = [sheet for sheets in SHELL_STYLE_BUNDLES.values() for sheet in sheets]
        assert len(every) == len(set(every))

    def test_a_font_is_reached_at_the_address_the_page_preloads(self, workspace_dir):
        with terminal_client(workspace_dir) as client:
            page = client.get("/").text
            url = client.app.state.assets.bundle_url("base")
            body = client.get(url).text
            references = set(re.findall(r'url\("([^":]+\.(?:ttf|woff2?))"\)', body))
            reached = {urljoin(url, reference) for reference in references}
            answers = {address: client.get(address) for address in reached}

        assert reached
        for address, answer in answers.items():
            assert answer.status_code == 200, address
            assert answer.headers["cache-control"] == CACHE_IMMUTABLE, address
        assert _preloaded_fonts(page)
        assert _preloaded_fonts(page) <= reached

    def test_is_served_to_be_kept(self, workspace_dir):
        with terminal_client(workspace_dir) as client:
            answer = client.get(
                client.app.state.assets.bundle_url("shell"), headers={"Accept-Encoding": "gzip"}
            )

        assert answer.headers["cache-control"] == CACHE_IMMUTABLE
        assert answer.headers["content-encoding"] == "gzip"
        assert answer.headers["content-type"] == "text/css; charset=utf-8"


class TestAPageWithoutABundleIsStillStyled:
    def test_each_sheet_is_linked_in_order_when_the_bundle_cannot_be_built(self, workspace_dir):
        with terminal_client(workspace_dir) as client:
            assets = client.app.state.assets
            with patch.object(type(assets), "bundle_url", return_value=None):
                links = _stylesheets(client.get("/").text)
            version = assets.version()

        own = _own(links)
        expected = [
            re.sub(r"^(/static/fonts|/design-system|/static)/", rf"\1/_v/{version}/", sheet)
            for sheets in SHELL_STYLE_BUNDLES.values()
            for sheet in sheets
        ]
        assert own == expected
