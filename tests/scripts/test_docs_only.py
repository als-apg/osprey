"""Tests for the docs-only classifier ci.yml's ``changes`` job runs.

A wrong ``true`` turns every pull-request-scoped e2e lane off on a change that
needed them, so each way the verdict can be ``true`` has a negative control.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests._by_path import load_module

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "docs_only.py"
docs_only = load_module("docs_only", _MODULE_PATH)


@pytest.mark.parametrize(
    "paths",
    [
        ["docs/source/index.rst"],
        ["docs/source/how-to/a.rst", "changelog.d/x.changed.md"],
        ["changelog.d/x.fixed.md"],
        ["docs/source/how-to/control-systems/examples/archiver-mock.yml"],
        # a docs page landing with its own guard test
        ["docs/source/index.rst", "tests/docs/test_redirects.py"],
        ["tests/docs/screenshots/test_video_probes_browser.py"],
    ],
)
def test_docs_and_changelog_paths_are_docs_only(paths: list[str]) -> None:
    assert docs_only.verdict(paths)[0] is True


@pytest.mark.parametrize(
    "paths",
    [
        ["docs/source/index.rst", "src/osprey/cli/main.py"],
        ["README.md"],
        # tests outside tests/docs/, even ones that read docs/
        ["docs/source/index.rst", "tests/test_docs_links.py"],
        ["docs/source/index.rst", "tests/e2e/test_dockerfile_e2e.py"],
        ["docs/source/index.rst", "tests/conftest.py"],
        ["docs/source/index.rst", ".github/workflows/ci.yml"],
        ["pyproject.toml"],
        # a lookalike prefix is not the docs tree
        ["docs-old/index.rst"],
        ["tests/docs-old/test_x.py"],
        ["src/osprey/docs/readme.rst"],
    ],
)
def test_any_path_outside_the_docs_tree_runs_every_lane(paths: list[str]) -> None:
    assert docs_only.verdict(paths)[0] is False


def test_a_rename_out_of_src_is_not_docs_only() -> None:
    """The changes job feeds both names of a rename, so moving a source file
    into docs/ is read as touching src/."""
    assert docs_only.verdict(["docs/source/moved.py", "src/osprey/moved.py"])[0] is False


@pytest.mark.parametrize("paths", [[], ["", "  ", "\n"]])
def test_an_empty_list_is_not_docs_only(paths: list[str]) -> None:
    """An unreadable diff must never turn lanes off."""
    assert docs_only.verdict(paths)[0] is False


def test_a_possibly_truncated_list_is_not_docs_only() -> None:
    paths = [f"docs/source/p{i}.rst" for i in range(docs_only.API_FILE_LIMIT)]
    assert docs_only.verdict(paths)[0] is False
    assert docs_only.verdict(paths[:-1])[0] is True


def test_the_script_prints_a_github_output_line() -> None:
    def run(stdin: str) -> str:
        return subprocess.run(
            [sys.executable, str(_MODULE_PATH)],
            input=stdin,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    assert run("docs/source/index.rst\nchangelog.d/a.md\n") == "docs_only=true\n"
    assert run("docs/source/index.rst\nsrc/osprey/x.py\n") == "docs_only=false\n"
    assert run("") == "docs_only=false\n"
