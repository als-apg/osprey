"""The application-contract reference page against the number the code implements.

``reference/contracts/app-contract.rst`` is the one place the bump rule and the
history of contract numbers are written. These tests hold its history table to
:data:`osprey.apps.contract.APP_CONTRACT`, the refusal's link to the page, the
contracts index and the apps how-to to a link to it, and every other docs page
to not restating the bump rule.
"""

from __future__ import annotations

import re
from pathlib import Path

from osprey.apps.contract import APP_CONTRACT
from osprey.docs_links import APP_CONTRACT_URL

_DOCS = Path(__file__).resolve().parents[2] / "docs" / "source"
_CONTRACTS = _DOCS / "reference" / "contracts"
_HOW_TO = _DOCS / "how-to" / "apps"
_PAGE = _CONTRACTS / "app-contract.rst"
_SITE = "https://als-apg.github.io/osprey/"


def _history_numbers(text: str) -> list[int]:
    """The first-column values of the page's history ``list-table``, header row excluded."""
    match = re.search(r"^History\n=+\n(.*?)(?=^\S[^\n]*\n[=\-~]+\n|\Z)", text, re.M | re.S)
    assert match, "the page has no History section"
    section = match.group(1)
    assert ".. list-table::" in section, "the History section holds no list-table"
    rows = re.findall(r"^   \* - (.*)$", section, re.M)
    assert rows and rows[0].strip() == "Number", f"unexpected header row: {rows[:1]}"
    return [int(row.strip()) for row in rows[1:]]


def test_history_has_one_row_per_number_up_to_the_current_one() -> None:
    """Raising ``APP_CONTRACT`` without recording what it broke fails here."""
    numbers = _history_numbers(_PAGE.read_text(encoding="utf-8"))
    assert numbers == list(range(1, APP_CONTRACT + 1)), (
        f"history rows {numbers}; expected 1..{APP_CONTRACT}. "
        "Add the row for the new number to app-contract.rst."
    )


def test_the_refusal_links_this_page() -> None:
    """The refusal message's address is the page that lists the changes."""
    path = APP_CONTRACT_URL.removeprefix(_SITE).removesuffix(".html")
    assert path == "reference/contracts/app-contract"
    assert (_PAGE.parent.parents[1] / f"{path}.rst") == _PAGE
    assert _PAGE.is_file()


def test_the_contracts_index_links_the_page() -> None:
    """A card and a toctree entry make the page reachable from the contracts index."""
    index = (_CONTRACTS / "index.rst").read_text(encoding="utf-8")
    assert re.search(r"^      :link: app-contract$", index, re.M), "no card links app-contract"
    toctree = index.split(".. toctree::", 1)[1]
    assert re.search(r"^   app-contract$", toctree, re.M), "app-contract is not in the toctree"


def test_the_page_states_the_bump_rule() -> None:
    """The bump rule is written here, in its one wording."""
    text = " ".join(_PAGE.read_text(encoding="utf-8").split())
    assert (
        "Removing, renaming or retyping anything this page lists, or changing what it "
        "means, raises the number by one in the same change."
    ) in text


def test_the_how_to_links_the_page() -> None:
    """The apps how-to and the manifest-key reference both point at the page."""
    link = ":doc:`/reference/contracts/app-contract`"
    for name in ("index.rst", "build-your-own-app.rst"):
        text = (_HOW_TO / name).read_text(encoding="utf-8")
        assert link in text, f"how-to/apps/{name} does not link the app-contract page"


def test_the_bump_rule_is_on_this_page_only() -> None:
    """Other pages point at the page; none restates the bump rule."""
    phrase = "raises the number by one"
    others = [
        path.relative_to(_DOCS).as_posix()
        for path in sorted(_DOCS.rglob("*"))
        if path.is_file()
        and path != _PAGE
        and path.suffix in {".rst", ".md", ".txt", ".html", ".py"}
        and phrase in " ".join(path.read_text(encoding="utf-8", errors="replace").split())
    ]
    assert not others, f"the bump rule is restated in {others}; link app-contract instead"
