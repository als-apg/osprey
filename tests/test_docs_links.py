"""Tests for the published documentation links runtime messages carry."""

from __future__ import annotations

import re
from pathlib import Path

import osprey.docs_links as docs_links
from osprey.docs_links import PERIMETER_LIMITS_URL
from osprey.port_layout import INDEX_MAX

_DOCS_SOURCE = Path(__file__).resolve().parents[1] / "docs" / "source"
_SITE = "https://als-apg.github.io/osprey/"


def _links() -> dict[str, str]:
    """Every public string constant of :mod:`osprey.docs_links`, by name."""
    return {
        name: value
        for name, value in vars(docs_links).items()
        if not name.startswith("_") and isinstance(value, str)
    }


def _page_source(url: str) -> Path:
    """The ``.rst`` file a published docs URL is built from."""
    path = url.removeprefix(_SITE).split("#", 1)[0].removesuffix(".html")
    return _DOCS_SOURCE / f"{path}.rst"


def test_every_docs_link_names_a_page_the_docs_tree_has() -> None:
    """A link that names a page the docs do not build sends the reader to a 404."""
    links = _links()
    assert len(links) >= 2
    for name, url in links.items():
        assert url.startswith(_SITE), f"{name} is not on the published docs site: {url}"
        assert _page_source(url).is_file(), f"{name} names no page in docs/source: {url}"


def test_every_docs_link_anchor_is_a_label_on_its_page() -> None:
    """A fragment only lands on its section when the page carries that label."""
    checked = []
    for name, url in _links().items():
        if "#" not in url:
            continue
        fragment = url.split("#", 1)[1]
        text = _page_source(url).read_text(encoding="utf-8")
        pattern = rf"^\.\. _{re.escape(fragment)}:\s*$"
        assert re.search(pattern, text, re.MULTILINE), (
            f"{name} links #{fragment}, which is not a label on {_page_source(url)}"
        )
        checked.append(url)
    assert PERIMETER_LIMITS_URL in checked


def test_the_perimeter_section_states_the_user_ceiling() -> None:
    """The documented ceiling is the one the allocator enforces."""
    text = (_DOCS_SOURCE / "how-to" / "deploy-a-facility.rst").read_text(encoding="utf-8")
    assert f"at most {INDEX_MAX + 1} users" in text
