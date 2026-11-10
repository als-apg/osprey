"""The ``.env.variant`` example on the environment-variable page selects a variant.

``docs/source/reference/configuration/environment-variables.rst`` shows the one
line that picks a host's overlay profile, as ``<file>:   <KEY>=<name>``. The
build reads that setting through
:func:`osprey.cli.variant_selection.resolve_variant_selection`, so this test
writes the shown line into the shown file in a scratch repo and checks that it
selects ``profiles/<name>.yml``. A negative control shows that the same line
in the shell environment selects nothing, as the page says.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from osprey.cli.variant_selection import (
    VARIANT_SETTING_FILENAME,
    VARIANT_SETTING_KEY,
    resolve_variant_selection,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The page this test reads, relative to the repo root.
_PAGE = "docs/source/reference/configuration/environment-variables.rst"

#: The example line: a file name, a colon, then ``KEY=value``.
_LINE = re.compile(r"^\s+(?P<file>\.env\.\S+):\s+(?P<key>[A-Z_]+)=(?P<value>\S+)\s*$")


def _example() -> tuple[str, str, str]:
    """``(file, key, value)`` from the page's ``.env.variant`` text block."""
    lines = (_REPO_ROOT / _PAGE).read_text(encoding="utf-8").splitlines()
    for n, line in enumerate(lines):
        if line.strip() != ".. code-block:: text":
            continue
        for following in lines[n + 1 :]:
            if following.strip() and not following.startswith(" "):
                break
            match = _LINE.match(following)
            if match is not None and match["key"] == VARIANT_SETTING_KEY:
                return match["file"], match["key"], match["value"]
    pytest.fail(f"no `{VARIANT_SETTING_KEY}` example in a text block of {_PAGE}")


def _repo(tmp_path: Path, name: str) -> Path:
    """A scratch deployment repo that defines the variant *name*."""
    (tmp_path / "profile.yml").write_text("name: example\n", encoding="utf-8")
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / f"{name}.yml").write_text("name: overlay\n", encoding="utf-8")
    return tmp_path


def test_the_example_names_the_file_and_key_the_build_reads() -> None:
    file, key, _value = _example()
    assert (file, key) == (VARIANT_SETTING_FILENAME, VARIANT_SETTING_KEY)


def test_the_example_line_selects_its_overlay(tmp_path: Path) -> None:
    file, key, value = _example()
    repo = _repo(tmp_path, value)
    (repo / file).write_text(f"{key}={value}\n", encoding="utf-8")

    selection = resolve_variant_selection(repo)

    assert selection.name == value
    assert selection.path == repo / "profiles" / f"{value}.yml"


def test_the_shell_environment_selects_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: the page says the variable is not read from the shell."""
    _file, key, value = _example()
    repo = _repo(tmp_path, value)
    monkeypatch.setenv(key, value)

    selection = resolve_variant_selection(repo)

    assert (selection.name, selection.path) == (None, None)
