"""OSPREY's own applications pass the checks every deployment's applications face.

Each ``osprey.apps`` entry point the ``osprey-framework`` distribution declares is run
through :func:`osprey.cli.app_cmd.app_violations`, the per-application checks ``osprey app
check`` runs, so this repository holds its applications to exactly what a deployment is
told. Every application directory under ``osprey/apps/first_party/`` has an entry point and
every entry point a directory: the wheel ships whatever directory is present, and one with
no entry point would ship unreachable.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

import osprey.apps.first_party as first_party
from osprey.apps.origin import MANIFEST_FILE, AppOrigin, installed_apps
from osprey.cli.app_cmd import CheckedApp, app_violations
from tests.apps import _installed

#: The distribution OSPREY's own applications ship in.
FRAMEWORK_DIST = "osprey-framework"
FIRST_PARTY_DIR = Path(first_party.__file__).resolve().parent


def first_party_apps(entry_points: Iterable[Any] | None = None) -> dict[str, AppOrigin]:
    """The installed applications OSPREY's own distribution declares, by name."""
    return {
        name: origin
        for name, origin in installed_apps(entry_points).items()
        if origin.distribution == FRAMEWORK_DIST
    }


def _violations(apps: dict[str, AppOrigin]) -> list[str]:
    checked = [CheckedApp(name, origin) for name, origin in sorted(apps.items())]
    return [f"{violation} [{violation.rule}]" for violation in app_violations(checked)]


def test_every_first_party_app_passes_osprey_app_check() -> None:
    assert _violations(first_party_apps()) == []


def test_every_first_party_directory_has_an_entry_point_and_the_reverse() -> None:
    directories = {path.parent for path in FIRST_PARTY_DIR.glob(f"*/{MANIFEST_FILE}")}
    declared = {origin.directory for origin in first_party_apps().values()}

    assert directories == declared


# --- the check refuses a first-party application that breaks a rule -------------------

PACKAGE = "fixture_first_party"
MODULE = f"{first_party.__name__}.{PACKAGE}"
READY_PAGE = (
    '<html data-page-ready="true">\n<head>\n'
    + "".join(
        f'<link rel="stylesheet" href="/design-system/css/{sheet}">\n'
        for sheet in ("tokens.css", "base.css", "primitives.css")
    )
    + '</head>\n<body><main class="osprey-card"></main></body>\n</html>\n'
)


def _fake_first_party(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, str]
) -> dict[str, AppOrigin]:
    """The installed fixture application, placed at ``osprey.apps.first_party.<package>``."""
    package = tmp_path / PACKAGE
    shutil.copytree(_installed.DIRECTORY, package)
    for relative, text in files.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(first_party, "__path__", [*first_party.__path__, str(tmp_path)])
    return first_party_apps([_installed.entry_point(module=MODULE, dist=FRAMEWORK_DIST)])


@pytest.mark.parametrize(
    ("files", "refused"),
    [
        ({"own.py": f"from {MODULE}.tools import read_fixture\n"}, None),
        ({"panel/index.html": READY_PAGE}, None),
        (
            {"leak.py": "from osprey.registry import mcp\n"},
            f"{FRAMEWORK_DIST}:osprey/apps/first_party/{PACKAGE}/leak.py:1: osprey.registry.mcp: ",
        ),
        (
            {"panel/index.html": READY_PAGE.replace(' data-page-ready="true"', "")},
            f"{FRAMEWORK_DIST}:osprey/apps/first_party/{PACKAGE}/panel/index.html: "
            "never sets the page-ready mark",
        ),
    ],
    ids=["own-import", "ready-panel", "framework-import", "panel-never-ready"],
)
def test_a_first_party_app_faces_the_per_app_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, str],
    refused: str | None,
) -> None:
    apps = _fake_first_party(tmp_path, monkeypatch, files)
    assert [origin.module for origin in apps.values()] == [MODULE]

    lines = _violations(apps)

    if refused is None:
        assert lines == []
    else:
        [line] = lines
        assert line.startswith(refused)
