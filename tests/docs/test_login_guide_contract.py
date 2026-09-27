"""The multi-user login guide against the CLI and the login service's own constants.

The multi-user guide is where an operator learns the commands that end access,
and a command that does not exist fails at the worst moment. This module holds
the guide to the CLI and to the login service's own constants: every
``osprey users <verb>`` the tree names is a registered verb.
"""

from __future__ import annotations

import re
from pathlib import Path

from osprey.cli.users_cmd import users

_REPO = Path(__file__).resolve().parents[2]
_ROOTS = ("docs/source", "src/osprey", "plugins")
_SUFFIXES = frozenset({".rst", ".md", ".py", ".j2", ".yml", ".yaml", ".html", ".js", ".txt"})

# An uppercase or templated verb (``VERB``, ``{verb}``) is not matched on purpose.
_VERB = re.compile(r"\bosprey users ([a-z][a-z-]*)")


def _unknown_verbs(text: str, verbs: set[str]) -> list[str]:
    """Return the ``osprey users`` verbs in ``text`` that are not in ``verbs``, in order."""
    return [match.group(1) for match in _VERB.finditer(text) if match.group(1) not in verbs]


def _tree_files() -> list[Path]:
    files: list[Path] = []
    for root in _ROOTS:
        base = _REPO / root
        if base.is_dir():
            files.extend(
                path
                for path in sorted(base.rglob("*"))
                if path.is_file() and path.suffix in _SUFFIXES
            )
    return files


def test_every_osprey_users_verb_the_tree_names_exists() -> None:
    """A documented ``osprey users`` verb is one the CLI registers."""
    # Arrange
    verbs = set(users.commands)

    # Act
    hits = 0
    unknown: list[str] = []
    for path in _tree_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            hits += len(_VERB.findall(line))
            unknown.extend(
                f"{path.relative_to(_REPO)}:{lineno}: {verb}"
                for verb in _unknown_verbs(line, verbs)
            )

    # Assert
    assert hits > 0, "no `osprey users <verb>` literal found; the scanned roots moved"
    assert unknown == [], "verbs `osprey users` does not register:\n" + "\n".join(unknown)


def test_the_verb_scan_reports_a_verb_that_does_not_exist() -> None:
    """The scan names a verb missing from the registered set and passes a known one."""
    # Arrange
    text = "run ``osprey users decommission alice`` and ``osprey users remove bob``"

    # Act
    unknown = _unknown_verbs(text, {"remove"})

    # Assert
    assert unknown == ["decommission"]
