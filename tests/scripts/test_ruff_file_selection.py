"""The files ruff checks, exactly those the pre-commit hook checks, and the rules it judges.

The hook hands ruff every tracked file that ``identify`` types as Python: a
``.py``, ``.pyi`` or ``.ipynb`` suffix, or an executable file whose shebang
names a Python interpreter. Plain ``ruff check`` discovers files by suffix
alone, so an extensionless script is invisible to it unless ``[tool.ruff]
extend-include`` names it, and a directory list narrower than the repository
misses whole trees the hook lints. Either gap lets the two gates disagree
about the same commit. These tests hold the configuration and every written
ruff invocation to the hook's selection.

The hygiene rules are pinned by piping text to ruff under a path and reading
the codes it reports: debug prints in the shipped trees, debugger calls,
commented-out code, quoted credentials outside the tests and work markers
without a link. A deliberate exception to one of them is a noqa that carries
its reason.
"""

from __future__ import annotations

import ast
import io
import json
import re
import subprocess
import sys
import tokenize
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Suffixes ``identify`` types as Python for the hook's ``types_or``.
PYTHON_SUFFIXES = (".py", ".pyi", ".ipynb")

#: A shebang ``identify`` maps to the ``python`` tag.
PYTHON_SHEBANG = re.compile(rb"^#![^\n]*\bpython[0-9.]*\b")

#: A ruff command line as written in a script, a workflow or a doc: the verb,
#: then its arguments up to the end of the shell word list.
RUFF_INVOCATION = re.compile(r"\bruff\s+(?:check|format)\b((?:[ \t]+[^\s`'\"|;&>)]+)*)")

#: The checks module that declares the ruff checks CI's ruff steps run.
CHECK_SCRIPT = "scripts/repo_checks.py"

#: Every place that runs ruff or tells a contributor how to.
INVOCATION_SURFACES = (
    CHECK_SCRIPT,
    "CONTRIBUTING.md",
    "docs/source/contributing/development-setup.rst",
    "plugins/osprey/skills/pre-commit/SKILL.md",
    "plugins/osprey/skills/test-audit/SKILL.md",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=60
    ).stdout


def _tracked_modes() -> dict[str, str]:
    """Map every tracked path present in the working tree to its git file mode.

    A tracked path deleted from the working tree is linted by nothing, so it is
    not part of either selection.
    """
    modes: dict[str, str] = {}
    for line in _git("ls-files", "-s", "-z").split("\0"):
        if line:
            meta, path = line.split("\t", 1)
            if (REPO_ROOT / path).exists():
                modes[path] = meta.split()[0]
    return modes


def _is_python_script(path: str, mode: str) -> bool:
    # A tracked file the working tree has deleted is handed to no hook.
    if mode != "100755" or not (REPO_ROOT / path).is_file():
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
        timeout=120,
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


def _declared_command_lines(source: str) -> str:
    """Every tuple of string literals in Python *source*, in order, each joined into a line;
    a starred prefix (``*UV_RUN``) is the command's launcher and is left out."""
    tuples = [
        node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Tuple) and node.elts
    ]
    lines = []
    for node in sorted(tuples, key=lambda node: (node.lineno, node.col_offset)):
        elts = [e for e in node.elts if not isinstance(e, ast.Starred)]
        words = [e.value for e in elts if isinstance(e, ast.Constant)]
        if words and len(words) == len(elts) and all(isinstance(w, str) for w in words):
            lines.append(" ".join(words))
    return "\n".join(lines)


def _surface_text(surface: str) -> str:
    """The text of *surface* in which its ruff command lines are written out."""
    text = (REPO_ROOT / surface).read_text(encoding="utf-8")
    return _declared_command_lines(text) if surface == CHECK_SCRIPT else text


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
        timeout=60,
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
        found = _paths_checked(_surface_text(surface))
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


def test_a_declared_directory_list_is_reported() -> None:
    source = (
        'CHECKS = (Check("ruff", ("uv", "run", "ruff", "check", "src/", "tests/")),)\n'
        'FIX = (("uv", "run", "ruff", "format", "."), ("scope/**", 1))\n'
    )
    assert [paths for _, paths in _paths_checked(_declared_command_lines(source))] == [
        ["src/", "tests/"],
        ["."],
    ]


def test_the_check_command_declares_both_ruff_checks() -> None:
    found = _paths_checked(_surface_text(CHECK_SCRIPT))
    assert len(found) >= 2, f"{CHECK_SCRIPT} declares {len(found)} ruff commands"


def test_the_whole_repository_passes() -> None:
    text = (
        "uv run ruff check . --output-format=github\n"
        "if uv run ruff format --check . >/dev/null 2>&1; then\n"
        "echo \"Run 'uv run ruff format .' to fix\"\n"
        "`ruff check .`\n"
    )
    assert [paths for _, paths in _paths_checked(text)] == [["."]] * 4


#: The codes of the hygiene rules; a noqa naming one carries its reason.
HYGIENE_CODES = re.compile(r"^(?:T20\d|T100|ERA001|S10[5-7]|TD003)$")

#: A line or file-level noqa: its codes, then whatever follows them.
NOQA = re.compile(r"#\s*(?:ruff:\s*)?noqa:\s*(?P<codes>[A-Z]+\d+(?:\s*,\s*[A-Z]+\d+)*)(?P<rest>.*)")

#: The reason a noqa gives after its codes.
REASON = re.compile(r"^\s+--\s+\S")


def _reported(text: str, path: str) -> list[str]:
    """The codes ruff reports for *text* linted as if it were at *path*."""
    out = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--stdin-filename", path]
        + ["--output-format", "json", "-"],
        cwd=REPO_ROOT,
        input=text,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    return sorted(diagnostic["code"] for diagnostic in json.loads(out))


def _unreasoned_hygiene_noqa(source: str) -> list[int]:
    """The lines of *source* whose noqa names a hygiene rule and gives no reason."""
    lines = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        match = NOQA.search(token.string)
        if match is None:
            continue
        codes = re.split(r"\s*,\s*", match["codes"])
        if any(HYGIENE_CODES.match(code) for code in codes) and not REASON.match(match["rest"]):
            lines.append(token.start[0])
    return lines


PRINT = 'print("x")\n'


@pytest.mark.parametrize("path", ["src/osprey/x.py", "packages/osprey-connectors/src/x.py"])
def test_a_print_in_a_shipped_tree_is_reported(path: str) -> None:
    assert _reported(PRINT, path) == ["T201"]


@pytest.mark.parametrize("path", ["tests/x.py", "scripts/x.py", "docs/x.py"])
def test_a_print_outside_the_shipped_trees_is_not_reported(path: str) -> None:
    assert _reported(PRINT, path) == []


def test_a_print_with_a_reasoned_noqa_is_not_reported() -> None:
    assert _reported('print("x")  # noqa: T201 -- the reason\n', "src/osprey/x.py") == []


def test_a_file_level_noqa_covers_every_print_of_a_program() -> None:
    text = "# ruff: noqa: T201 -- stdout is this program's output\n" + PRINT * 2
    assert _reported(text, "src/osprey/x.py") == []


def test_a_stale_print_noqa_is_reported() -> None:
    assert _reported("x = 1  # noqa: T201 -- the reason\n", "src/osprey/x.py") == ["RUF100"]


@pytest.mark.parametrize("path", ["src/osprey/x.py", "tests/x.py", "scripts/x.py"])
def test_a_debugger_call_is_reported_anywhere(path: str) -> None:
    assert _reported("breakpoint()\n", path) == ["T100"]


def test_commented_out_code_is_reported() -> None:
    assert _reported("x = 1\n# return value\n", "src/osprey/x.py") == ["ERA001"]


def test_a_heading_worded_as_prose_is_not_reported() -> None:
    assert _reported("x = 1\n# Tests of connect()\n", "tests/x.py") == []


CREDENTIALS = {
    "S105": 'password = "x"\n',
    "S106": 'def f(**kwargs):\n    return kwargs\n\n\nf(password="x")\n',
    "S107": 'def f(password="x"):\n    return password\n',
}


@pytest.mark.parametrize("code", sorted(CREDENTIALS))
def test_a_quoted_credential_in_shipped_code_is_reported(code: str) -> None:
    assert _reported(CREDENTIALS[code], "src/osprey/x.py") == [code]


@pytest.mark.parametrize("path", ["tests/x.py", "packages/osprey-connectors/tests/x.py"])
@pytest.mark.parametrize("code", sorted(CREDENTIALS))
def test_a_quoted_credential_in_a_test_is_not_reported(code: str, path: str) -> None:
    assert _reported(CREDENTIALS[code], path) == []


def test_a_work_marker_without_a_link_is_reported() -> None:
    assert _reported("x = 1  # TODO: tidy\n", "src/osprey/x.py") == ["TD003"]


@pytest.mark.parametrize("link", ["issue #12", "https://example.org/issues/12"])
def test_a_work_marker_with_a_link_is_not_reported(link: str) -> None:
    assert _reported(f"x = 1  # TODO: tidy, {link}\n", "src/osprey/x.py") == []


def test_a_work_marker_inside_a_string_is_not_reported() -> None:
    assert _reported('x = "# TODO: tidy"\n', "src/osprey/x.py") == []


def test_every_hygiene_noqa_gives_its_reason() -> None:
    unreasoned = []
    for path in sorted(_hook_selection()):
        if path.endswith(".ipynb"):
            continue
        source = (REPO_ROOT / path).read_text(encoding="utf-8")
        if "noqa" not in source:
            continue
        unreasoned += [f"{path}:{line}" for line in _unreasoned_hygiene_noqa(source)]
    assert not unreasoned, "write the reason as `# noqa: CODE -- reason`:\n" + "\n".join(unreasoned)


def test_a_hygiene_noqa_without_a_reason_is_found() -> None:
    source = (
        'print("x")  # noqa: T201\n'
        "# ruff: noqa: ERA001\n"
        'token = "x"  # noqa: E501, S105\n'
        'print("x")  # noqa: T201 -- the reason\n'
        "import os  # noqa: F401\n"
        'x = "# noqa: T201"\n'
    )
    assert _unreasoned_hygiene_noqa(source) == [1, 2, 3]
