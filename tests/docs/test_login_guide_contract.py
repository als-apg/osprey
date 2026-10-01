"""The multi-user login guide against the CLI and the login service's own constants.

The multi-user guide is where an operator learns the commands that end access,
and a command that does not exist fails at the worst moment. This module holds
the guide to the CLI and to the login service's own constants: every
``osprey users <verb>`` the tree names is a registered verb, and the guide
names every identity header and every path the login service answers on.
"""

from __future__ import annotations

import re
from pathlib import Path

from osprey.cli.users_cmd import users
from osprey.services.auth_sidecar import app as sidecar_app
from osprey.services.auth_sidecar.identity_headers import (
    ACCOUNT_HEADER,
    ROLE_HEADER,
    ROLE_SOURCE_HEADER,
    SUBJECT_HEADER,
)
from osprey.services.auth_sidecar.routes import entry, login, logout, oidc, verify

_REPO = Path(__file__).resolve().parents[2]
_ROOTS = ("docs/source", "src/osprey", "plugins")
_LOGIN_GUIDE = _REPO / "docs/source/how-to/web-terminal/multi-user/login.rst"
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


def test_the_login_guide_names_every_identity_header() -> None:
    """Each header nginx forwards from the login service's answer is on the page."""
    # Arrange
    text = _LOGIN_GUIDE.read_text(encoding="utf-8")

    # Act
    missing = [
        name
        for name in (ACCOUNT_HEADER, SUBJECT_HEADER, ROLE_HEADER, ROLE_SOURCE_HEADER)
        if f"``{name}``" not in text
    ]

    # Assert
    assert missing == []


def test_the_login_guide_names_the_login_service_paths() -> None:
    """Each path the login service answers on is on the page."""
    # Arrange
    text = _LOGIN_GUIDE.read_text(encoding="utf-8")
    paths = (
        verify.VERIFY_PATH,
        sidecar_app.HEALTH_PATH,
        login.LOGIN_PATH,
        oidc.LOGIN_PATH,
        oidc.CALLBACK_PATH,
        logout.LOGOUT_PATH,
        entry.ENTRY_PATH,
        oidc.ENTRY_PATH,
    )

    # Act
    missing = [path for path in paths if f"``{path}``" not in text]

    # Assert
    assert missing == []
