"""The web terminal's proxy rewrite leaves the PV page's ARIEL search path alone.

The proxy rebases quoted root-absolute ``/api/`` literals in served JS onto the
serving panel's mount. The PV page reaches ARIEL's search through
``panelUrl('ariel', 'api/search')``, a rest with no leading slash, so the
rewrite must not turn it into a Channels route.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.interfaces.web_terminal.routes.proxy import _rewrite_content

_JS = Path(__file__).parents[3] / "src/osprey/interfaces/channel_finder/static/js"


def _ariel_lines(body: str) -> list[str]:
    return [line for line in body.splitlines() if "ariel" in line or "api/search" in line]


@pytest.mark.parametrize("module", ["directory-api.js", "pv-page.js"])
@pytest.mark.parametrize("outer", ["", "/u/a"])
def test_the_ariel_search_path_survives_the_rewrite(module: str, outer: str) -> None:
    body = (_JS / module).read_text(encoding="utf-8")
    rewritten = _rewrite_content(body, "channel-finder", outer)

    assert _ariel_lines(rewritten) == _ariel_lines(body)
    assert "/panel/channel-finder/api/search" not in rewritten


def test_the_page_names_the_search_as_a_rest_and_the_rewrite_still_works() -> None:
    page = (_JS / "pv-page.js").read_text(encoding="utf-8")
    assert "panelUrl('ariel', 'api/search')" in page
    assert "'/panel/" not in page and '"/panel/' not in page and "`/panel/" not in page

    api = (_JS / "directory-api.js").read_text(encoding="utf-8")
    assert "'/u/a/panel/channel-finder/api/directory/logbook/availability'" in _rewrite_content(
        api, "channel-finder", "/u/a"
    )
