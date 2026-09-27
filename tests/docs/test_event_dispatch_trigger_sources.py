"""Every registered trigger source has an entry on the event-dispatch how-to.

``docs/source/how-to/agent-interfaces/event-dispatch.rst`` is where a trigger
author learns which ``source:`` values exist and what each one's
``source_config`` takes. A source registered without an entry there works, but
only for someone who has read its code: the ``triggers.yml`` a deployer writes
from the page can never name it, and the keys it reads are nowhere a deployer
looks.

The producer is the ``[project.entry-points."osprey.trigger_sources"]`` table of
``pyproject.toml``, the group the dispatcher loads its sources from. It is read
with :mod:`tomllib` rather than :func:`importlib.metadata.entry_points`, so the
check follows the declared source of truth and does not depend on how the
package was installed. An entry is the inline literal ``source: <name>``; the
page's definition list for each source is headed with exactly that.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The page under guard, relative to the repo root.
_PAGE = Path("docs") / "source" / "how-to" / "agent-interfaces" / "event-dispatch.rst"

#: The entry-point group the dispatcher discovers trigger sources in.
_GROUP = "osprey.trigger_sources"


def _registered_sources(root_dir: Path | None = None) -> list[str]:
    """The source names ``pyproject.toml`` registers, sorted."""
    root = root_dir if root_dir is not None else _REPO_ROOT
    with (root / "pyproject.toml").open("rb") as fh:
        project = tomllib.load(fh).get("project", {})
    return sorted(project.get("entry-points", {}).get(_GROUP, {}))


def _undocumented_sources(root_dir: Path | None = None) -> list[str]:
    """Every registered source whose ``source: <name>`` literal the page lacks."""
    root = root_dir if root_dir is not None else _REPO_ROOT
    page = root / _PAGE
    text = page.read_text(encoding="utf-8") if page.is_file() else ""
    return [name for name in _registered_sources(root) if f"``source: {name}``" not in text]


def test_every_registered_trigger_source_is_documented():
    assert _registered_sources(), f"no trigger sources registered under {_GROUP!r}"
    missing = _undocumented_sources()
    assert missing == [], (
        f"{_PAGE} has no entry for trigger source(s) {missing}; "
        "add a definition-list entry headed ``source: <name>`` for each"
    )


def test_a_registered_source_missing_from_the_page_is_reported(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        f'[project.entry-points."{_GROUP}"]\n'
        'alpha = "pkg.alpha:AlphaSource"\n'
        'beta = "pkg.beta:BetaSource"\n',
        encoding="utf-8",
    )
    page = tmp_path / _PAGE
    page.parent.mkdir(parents=True)
    page.write_text("``source: alpha``\n   Fires on alpha.\n", encoding="utf-8")

    assert _undocumented_sources(tmp_path) == ["beta"]
