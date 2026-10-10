"""Ruff checks exactly the files the ruff pre-commit hook checks.

The hook hands ruff every tracked file that ``identify`` types as Python: a
``.py``, ``.pyi`` or ``.ipynb`` suffix, or an executable file whose shebang
names a Python interpreter. Plain ``ruff check`` discovers files by suffix
alone, so an extensionless script is invisible to it unless ``[tool.ruff]
extend-include`` names it, and a directory list narrower than the repository
misses whole trees the hook lints. Either gap lets the two gates disagree
about the same commit. These tests hold the configuration and every written
ruff invocation to the hook's selection.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Suffixes ``identify`` types as Python for the hook's ``types_or``.
PYTHON_SUFFIXES = (".py", ".pyi", ".ipynb")

#: A shebang ``identify`` maps to the ``python`` tag.
PYTHON_SHEBANG = re.compile(rb"^#![^\n]*\bpython[0-9.]*\b")

#: A ruff command line as written in a script, a workflow or a doc: the verb,
#: then its arguments up to the end of the shell word list.
RUFF_INVOCATION = re.compile(r"\bruff\s+(?:check|format)\b((?:[ \t]+[^\s`'\"|;&>)]+)*)")

#: Every place that runs ruff or tells a contributor how to.
INVOCATION_SURFACES = (
    ".github/workflows/ci.yml",
    "scripts/ci_check.sh",
    "scripts/quick_check.sh",
    "scripts/premerge_check.sh",
    "CONTRIBUTING.md",
    "docs/source/contributing/development-setup.rst",
    "plugins/osprey/skills/pre-commit/SKILL.md",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def _tracked_modes() -> dict[str, str]:
    """Map every tracked path to its git file mode."""
    modes: dict[str, str] = {}
    for line in _git("ls-files", "-s", "-z").split("\0"):
        if line:
            meta, path = line.split("\t", 1)
            modes[path] = meta.split()[0]
    return modes


def _is_python_script(path: str, mode: str) -> bool:
    if mode != "100755":
        return False
    with (REPO_ROOT / path).open("rb") as handle:
        return bool(PYTHON_SHEBANG.match(handle.readline()))


def _hook_selection() -> set[str]:
    """The tracked files the ruff pre-commit hook lints.

    The hook runs ``ruff check --force-exclude``, so a file under
    ``[tool.ruff] extend-exclude`` is skipped even when the hook hands it over.
    """
    excluded = _extend_exclude()
    return {
        path
        for path, mode in _tracked_modes().items()
        if (path.endswith(PYTHON_SUFFIXES) or _is_python_script(path, mode))
        and not any(path == entry or path.startswith(entry.rstrip("/") + "/") for entry in excluded)
    }


def _ruff_selection() -> set[str]:
    """The files ``ruff check .`` discovers from the repository root."""
    out = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--show-files", "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {Path(line).resolve().relative_to(REPO_ROOT).as_posix() for line in out.splitlines()}


def _extend_include() -> list[str]:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return config["tool"]["ruff"].get("extend-include", [])


def _extend_exclude() -> list[str]:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return config["tool"]["ruff"].get("extend-exclude", [])


def _paths_checked(text: str) -> list[tuple[int, list[str]]]:
    """Each ruff command line in *text*, as (line number, path arguments)."""
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        for match in RUFF_INVOCATION.finditer(line):
            paths = [arg for arg in match.group(1).split() if not arg.startswith("-")]
            found.append((number, paths))
    return found


def test_ruff_discovers_every_file_the_hook_lints() -> None:
    missing = _hook_selection() - _ruff_selection()
    assert not missing, (
        f"ruff check . does not see {sorted(missing)}; name each in [tool.ruff] extend-include"
    )


def test_ruff_discovers_no_file_git_ignores() -> None:
    candidates = sorted(_ruff_selection() - set(_tracked_modes()))
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=REPO_ROOT,
        input="\n".join(candidates),
        capture_output=True,
        text=True,
    ).stdout.split()
    assert not ignored, f"ruff check . lints files git ignores: {ignored}"


def test_extend_include_names_only_extensionless_python_scripts() -> None:
    modes = _tracked_modes()
    for entry in _extend_include():
        assert entry in modes, f"extend-include names an untracked path: {entry}"
        assert not entry.endswith(PYTHON_SUFFIXES), f"ruff already selects {entry} by suffix"
        assert _is_python_script(entry, modes[entry]), f"{entry} is not a Python script"


def test_extend_exclude_hides_only_the_generated_schema_modules() -> None:
    excluded = _extend_exclude()
    hidden = {
        path
        for path in _tracked_modes()
        if path.endswith(PYTHON_SUFFIXES)
        and any(path == entry or path.startswith(entry.rstrip("/") + "/") for entry in excluded)
    }
    assert hidden == {
        "src/osprey/facility/schema/_generated/__init__.py",
        "src/osprey/facility/schema/_generated/core.py",
    }, "extend-exclude hides tracked Python files ruff should lint, or the generated set moved"


def test_every_ruff_invocation_checks_the_whole_repository() -> None:
    narrowed = []
    for surface in INVOCATION_SURFACES:
        found = _paths_checked((REPO_ROOT / surface).read_text(encoding="utf-8"))
        assert found, f"{surface} no longer runs ruff; drop it from INVOCATION_SURFACES"
        narrowed += [f"{surface}:{number}: {paths}" for number, paths in found if paths != ["."]]
    assert not narrowed, "ruff must check `.`, as the pre-commit hook does:\n" + "\n".join(narrowed)


def test_a_directory_list_is_reported() -> None:
    text = (
        "uv run ruff check src/ packages/ tests/ --output-format=github\n"
        "echo \"Run 'ruff format src/ tests/' to fix\"\n"
        "   uv run ruff check --fix src/ tests/\n"
    )
    assert _paths_checked(text) == [
        (1, ["src/", "packages/", "tests/"]),
        (2, ["src/", "tests/"]),
        (3, ["src/", "tests/"]),
    ]


def test_the_whole_repository_passes() -> None:
    text = (
        "uv run ruff check . --output-format=github\n"
        "if uv run ruff format --check . >/dev/null 2>&1; then\n"
        "echo \"Run 'uv run ruff format .' to fix\"\n"
        "`ruff check .`\n"
    )
    assert [paths for _, paths in _paths_checked(text)] == [["."]] * 4
