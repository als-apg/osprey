"""Every directory that holds test modules is reached by recursion from the test paths.

A directory whose name matches a norecursedirs pattern is collected only when
named, so the suite would silently never run it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _pytest.pathlib import fnmatch_ex

REPO_ROOT = Path(__file__).resolve().parents[2]


def _test_roots(pytestconfig: pytest.Config) -> list[Path]:
    roots: list[Path] = []
    for pattern in pytestconfig.getini("testpaths") or ["tests"]:
        roots.extend(sorted(path for path in REPO_ROOT.glob(pattern) if path.is_dir()))
    return roots


def _unreached(
    roots: list[Path], python_files: list[str], norecursedirs: list[str]
) -> list[tuple[Path, str]]:
    """Each directory holding a test module below a root that a pattern stops, with that pattern."""
    stopped: dict[Path, str] = {}
    for root in roots:
        for module in sorted(root.rglob("*.py")):
            if not any(fnmatch_ex(pattern, module) for pattern in python_files):
                continue
            directory = root
            for part in module.relative_to(root).parent.parts:
                directory = directory / part
                pattern = next((p for p in norecursedirs if fnmatch_ex(p, directory)), None)
                if pattern is not None:
                    stopped.setdefault(directory, pattern)
                    break
    return sorted(stopped.items())


def test_no_test_directory_matches_a_norecursedirs_pattern(pytestconfig: pytest.Config) -> None:
    unreached = _unreached(
        _test_roots(pytestconfig),
        pytestconfig.getini("python_files"),
        pytestconfig.getini("norecursedirs"),
    )
    assert not unreached, "test directories recursion never reaches: " + ", ".join(
        f"{directory.relative_to(REPO_ROOT)} (norecursedirs {pattern!r})"
        for directory, pattern in unreached
    )


@pytest.mark.parametrize("name", ["build", "dist", "node_modules", ".hidden", "venv"])
def test_a_test_directory_named_like_a_norecursedirs_pattern_is_reported(
    tmp_path: Path, pytestconfig: pytest.Config, name: str
) -> None:
    norecursedirs = pytestconfig.getini("norecursedirs")
    nested = tmp_path / "tests" / name
    nested.mkdir(parents=True)
    (nested / "test_example.py").write_text("def test_example():\n    pass\n")
    (tmp_path / "tests" / "test_top.py").write_text("def test_top():\n    pass\n")

    unreached = _unreached([tmp_path / "tests"], pytestconfig.getini("python_files"), norecursedirs)

    assert [directory for directory, _ in unreached] == [nested]
    assert fnmatch_ex(unreached[0][1], nested)
