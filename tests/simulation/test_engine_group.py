"""The engines entry-point group is spelled once, by ``ENGINE_GROUP``, and nowhere else.

Two scans over the tracked files: no file uses the retired group name
``osprey.simulation.engines`` *as a group* (a module path of the same spelling
is not a group use), and every entry-point table that registers engines does so
under :data:`osprey_connectors.simulation.engines.ENGINE_GROUP`. A build whose
engine is registered only under the retired group is refused.
"""

from __future__ import annotations

import configparser
import re
import subprocess
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors.simulation.engines import ENGINE_GROUP
from tests.facility.test_wiring_slots import _files, _lines, _run

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The retired group name.
OLD_GROUP = "osprey.simulation.engines"

#: The files that may spell the retired name as a group: the tests asserting it
#: is no longer read, and the release history recording the rename.
ALLOWED = (
    "packages/osprey-connectors/tests/osprey_connectors_tests/test_engine_group_lookup.py",
    "tests/simulation/test_engine_group.py",
    "tests/connectors/test_composite.py",
    "changelog.d",
    "packages/osprey-connectors/changelog.d",
    "CHANGELOG.md",
    "packages/osprey-connectors/CHANGELOG.md",
    "RELEASE_NOTES.md",
)

_NAME = re.escape(OLD_GROUP)
_QUOTED = rf"[\"']{_NAME}[\"']"

#: The ways a file uses a name as an entry-point group.
GROUP_USES = (
    # An entry-point table header: pyproject.toml, setup.cfg or entry_points.txt.
    re.compile(
        rf"^\s*\[\s*(?:project\.entry-points\.|options\.entry_points\.)?(?:{_QUOTED}|{_NAME})\s*\]",
        re.M,
    ),
    # A lookup or an entry point: entry_points(group=...), EntryPoint(group=...).
    re.compile(rf"\bgroup\s*=\s*{_QUOTED}"),
    # A selection mapping: {"group": ...}.
    re.compile(rf"[\"']group[\"']\s*:\s*{_QUOTED}"),
    # A group constant: ENGINE_GROUP = ..., OLD_GROUP: str = ...
    re.compile(rf"^\s*\w*GROUP\w*\s*(?::[^=\n]*)?=\s*{_QUOTED}", re.M),
)

#: A pyproject entry-point table header, also inside a template string.
_TABLE = re.compile(r"^\s*\[project\.entry-points\.\"([^\"]+)\"\]", re.M)


def _tracked() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True
    ).stdout
    return [path for path in out.decode().split("\0") if path]


def _text(path: str) -> str | None:
    try:
        return (REPO_ROOT / path).read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _allowed(path: str) -> bool:
    return any(path == entry or path.startswith(entry + "/") for entry in ALLOWED)


def _is_engines(group: str) -> bool:
    return group.endswith(".engines")


@pytest.fixture(scope="module")
def tracked() -> list[str]:
    try:
        return _tracked()
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"not a git checkout: {error}")


@pytest.mark.parametrize("entry", ALLOWED)
def test_every_allowlisted_path_exists(entry: str) -> None:
    assert (REPO_ROOT / entry).exists(), f"allowlisted path {entry} does not exist"


@pytest.mark.parametrize(
    "line",
    [
        f'[project.entry-points."{OLD_GROUP}"]',
        f"[{OLD_GROUP}]",
        f'metadata.entry_points(group="{OLD_GROUP}")',
        f"EntryPoint(name='x', value='y', group='{OLD_GROUP}')",
        f'selection == {{"group": "{OLD_GROUP}", "name": "stub"}}',
        f'OLD_GROUP = "{OLD_GROUP}"',
        f'ENGINE_GROUP: str = "{OLD_GROUP}"',
    ],
)
def test_the_scan_sees_each_kind_of_group_use(line: str) -> None:
    assert any(pattern.search(line) for pattern in GROUP_USES), line


@pytest.mark.parametrize(
    "line",
    [
        f'"{OLD_GROUP}",',
        f"from {OLD_GROUP} import pyat",
        f'pyat = "{OLD_GROUP}.pyat"',
        f'"osprey.facility.validate -> {OLD_GROUP}.pyat",',
        f'[project.entry-points."{ENGINE_GROUP}"]',
        f'group="{ENGINE_GROUP}"',
    ],
)
def test_a_module_path_is_not_a_group_use(line: str) -> None:
    assert not any(pattern.search(line) for pattern in GROUP_USES), line


def test_no_tracked_file_uses_the_retired_group(tracked: list[str]) -> None:
    found = []
    for path in tracked:
        if _allowed(path):
            continue
        text = _text(path)
        if text is None or OLD_GROUP not in text:
            continue
        for pattern in GROUP_USES:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                found.append(f"{path}:{line}: {match.group(0).strip()}")
    assert not found, f"the retired group {OLD_GROUP} is used as a group:\n" + "\n".join(found)


def _tables(path: str, text: str) -> list[str]:
    """Every entry-point group the file's tables name."""
    name = Path(path).name
    if name == "pyproject.toml":
        project = tomllib.loads(text).get("project", {})
        return list(project.get("entry-points", {}))
    if name == "entry_points.txt":
        parser = configparser.ConfigParser(interpolation=None)
        parser.optionxform = str  # type: ignore[assignment,method-assign]
        parser.read_string(text)
        return parser.sections()
    if path.endswith(".py"):
        return _TABLE.findall(text)
    return []


def test_every_engines_table_registers_under_the_engine_group(tracked: list[str]) -> None:
    seen: list[str] = []
    wrong: list[str] = []
    for path in tracked:
        if not (path.endswith((".py", "pyproject.toml", "entry_points.txt"))):
            continue
        text = _text(path)
        if text is None:
            continue
        for group in _tables(path, text):
            if _is_engines(group):
                seen.append(path)
                if group != ENGINE_GROUP:
                    wrong.append(f"{path}: [{group}]")
    assert not wrong, f"engines registered under a group other than {ENGINE_GROUP}:\n" + "\n".join(
        wrong
    )
    # The scan reaches the root table, the fixture's dist-info and its template.
    assert {
        "pyproject.toml",
        "tests/facility/fixtures/fixture_kind/fixture_kind_deployment-0.1.dist-info/entry_points.txt",
        "tests/facility/_fixture_kind.py",
    } <= set(seen)


def test_a_build_whose_engine_is_under_the_retired_group_only_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R10: the build reads no other group, and its refusal names the one it reads."""
    real = metadata.entry_points
    retired = [
        metadata.EntryPoint(name=point.name, value=point.value, group=OLD_GROUP)
        for point in real(group=ENGINE_GROUP)
    ]
    others = [point for point in real() if point.group != ENGINE_GROUP]
    served = metadata.EntryPoints([*others, *retired])

    def entry_points(**selection: Any) -> metadata.EntryPoints:
        return served.select(**selection) if selection else served

    monkeypatch.setattr(metadata, "entry_points", entry_points)
    result = _run(tmp_path, _files())
    assert result.failed == "wiring", _lines(result)
    assert [(e.kind, e.record_id) for e in result.errors] == [
        ("engine-missing", "SR"),
        ("engine-missing", "plug"),
    ]
    for error in result.errors:
        assert f"engine pyat is not registered under {ENGINE_GROUP}" in error.format_message()
