"""The names a docs command carries are names the repository knows.

``test_cli_page_commands`` holds every ``osprey ...`` line to the click tree.
The other commands on the pages (``uv sync --extra ...``, ``pytest <path>``,
``npm run ...``, ``python -m ...``, a ``NAME=value`` line for a ``.env``) carry
names too, and each of those names is defined somewhere a test can read: an
extra in ``pyproject.toml``, a path under the repository root, a script in
``package.json``, a module on the import path, a variable some shipped module
reads. A reader who copies a line with a dead name gets an error, or worse, a
variable nothing reads. This check reads every such line and resolves each
name against the thing that defines it.

What is read
------------
Lines inside ``bash``, ``console``, ``shell``, ``sh`` and ``text`` code blocks
and ``::`` literal blocks (one leading ``$`` prompt is dropped), plus every line
of a block whose ``:caption:`` ends in ``.env``. Inline literals are read for
commands only, never for variable assignments. Every pattern is anchored to the
start of the line, so prose and output lines do not match.

* **Variables.** ``NAME=value`` and ``export NAME=value`` at the start of a
  line, several per line → ``NAME`` occurs as a word somewhere under
  ``src/osprey`` or ``packages`` (the trees ``test_environment_variable_page``
  searches), comments and docstrings included. That catches a name nothing
  in the shipped tree mentions; it does not prove a module reads it, and a
  page-local shell variable whose name happens to occur there (``LOGIN``,
  ``HOME``) passes.
* **Extras.** ``uv sync --extra X`` → ``X`` is in
  ``[project.optional-dependencies]``. ``pip install``, ``uv pip install`` and
  ``uv tool install|upgrade`` of a distribution whose name starts with
  ``osprey`` (or of ``.``) → the name is one of the repository's
  distributions and each ``[extra]`` is one it declares.
* **Paths.** Path arguments of ``pytest`` and ``ruff``, the script of
  ``python <script>``, and a line that runs ``scripts/...`` → the path exists
  under the repository root. A path ``osprey`` scaffolding writes into a
  project (``scripts/verify.sh``) resolves to its template instead.
* **npm.** ``npm run S`` → ``S`` is in ``package.json`` ``scripts``.
  ``npm ci`` and ``npm install`` pass; ``npm install -g
  @anthropic-ai/claude-code@V`` → ``V`` is the pinned Claude Code version.
* **Modules.** ``python -m M`` → an ``osprey`` module the import system finds
  and can run (a package with ``__main__``, or a module with a ``__main__``
  guard); any other module must be in :data:`_THIRD_PARTY_MODULES`.
* **Pins and asset paths**, anywhere on a page: ``jlab-archiver-client>=V``
  → ``V`` is the connector's required client version; ``/design-system/<file>``
  → the file is in the design system's static directory.

Not checked
-----------
Arguments of ``docker``, ``podman``, ``az``, ``claude``, ``codex``, ``ollama``,
``brew``, ``apt-get``, ``systemctl`` and ``git``; ``curl`` URLs; ``git clone``
URLs; YAML ``run:`` strings. Placeholders (a word with ``<``, ``{``, ``*`` or
``…``) are skipped.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shlex
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pytest

from tests.docs.test_environment_variable_page import _SOURCE_ROOTS, _shipped_text

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _REPO_ROOT / "docs" / "source"

#: A code directive, or a paragraph ending in ``::``, that opens a literal block.
#: The shape of ``test_cli_page_commands._BLOCK_OPENER``, with the directive's
#: language captured. Other directives (``.. note::``) hold prose and open nothing.
_BLOCK_OPENER = re.compile(
    r"^(\s*)(\.\. (?:code-block|code|sourcecode)::(?P<lang>.*)|(?!\.\. ).*::)\s*$"
)

#: Block languages whose lines are shell commands or ``.env`` lines.
_SHELL_LANGUAGES = frozenset({"bash", "console", "shell", "sh", "text", "literal"})

#: An RST inline literal on one line.
_LITERAL = re.compile(r"``([^`\n]+)``")

#: One leading ``NAME=value`` assignment; the value may be quoted.
_ASSIGNMENT = re.compile(r"""([A-Z][A-Z0-9_]*)=("[^"]*"|'[^']*'|\S*)\s*""")

#: ``name[extra,extra]spec`` of a pip requirement; ``.`` is the project itself.
_REQUIREMENT = re.compile(r"^(?P<name>\.|[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[(?P<extras>[^\]]*)\])?")

_ARCHIVER_PIN = re.compile(r"jlab-archiver-client>=([0-9][0-9.]*)")
_DESIGN_SYSTEM_ASSET = re.compile(r"/design-system/([A-Za-z0-9_./-]+\.[a-z]+)\b")
_CLAUDE_PACKAGE = "@anthropic-ai/claude-code@"

#: Modules a page runs with ``python -m`` that are not OSPREY's. A name not
#: listed and not under ``osprey`` fails, so a typo cannot hide as "third party".
_THIRD_PARTY_MODULES: dict[str, str] = {
    "venv": "the standard library's virtual-environment creator",
    "playwright": "Playwright's browser installer, in the Canopy dev venv",
    "canopy_mcp": "the Canopy MCP server, from the Canopy wheel",
    "canopy_mcp.skill": "the Canopy skill installer, from the Canopy wheel",
    "canopy_mcp.report": "the Canopy library report, from the Canopy wheel",
}

#: ``(page, name)`` pairs a page may write although they do not resolve, each
#: with its reason. ``page`` is relative to ``docs/source``. An entry whose pair
#: no longer fails is stale and fails :func:`test_every_allowlist_entry_is_needed`.
_ALLOWLIST: dict[tuple[str, str], str] = {
    ("how-to/web-terminal/canopy-library.rst", "CANOPY_DEV_ACCESS"): (
        "a shell variable the page exports for the claude mcp add header above it; "
        "nothing in OSPREY reads it"
    ),
    ("how-to/agent-interfaces/chat-bridges/microsoft-teams.rst", "SITE_ID"): (
        "a shell variable the page fills from az rest and reads in the LIST_ID lookup"
    ),
    ("how-to/agent-interfaces/chat-bridges/microsoft-teams.rst", "LIST_ID"): (
        "a shell variable the page fills from az rest and reads in the drive lookup below"
    ),
    ("contributing/development-setup.rst", "OSPREY_E2E_PROVIDER"): (
        "a test-suite knob read by tests/e2e/provider.py and the CI workflow, not by the "
        "shipped tree"
    ),
    ("contributing/development-setup.rst", "tests/path/to/test_file.py"): (
        "a placeholder path in the 'run one test file' example"
    ),
}


@dataclass(frozen=True)
class Name:
    """One name a command carries, and where."""

    page: str
    line: int
    kind: str
    value: str

    @property
    def id(self) -> str:
        return f"{self.page}:{self.line}:{self.value}"


# --------------------------------------------------------------------------- definitions


@lru_cache(maxsize=1)
def _pyproject() -> dict:
    return tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _distributions() -> dict[str, frozenset[str]]:
    """Every distribution the repository builds, with the extras it declares."""
    found: dict[str, frozenset[str]] = {}
    for path in [
        _REPO_ROOT / "pyproject.toml",
        *sorted(_REPO_ROOT.glob("packages/*/pyproject.toml")),
    ]:
        project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
        found[project["name"]] = frozenset(project.get("optional-dependencies", {}))
    return found


def _root_extras() -> frozenset[str]:
    return _distributions()[_pyproject()["project"]["name"]]


@lru_cache(maxsize=1)
def _npm_scripts() -> frozenset[str]:
    return frozenset(
        json.loads((_REPO_ROOT / "package.json").read_text(encoding="utf-8"))["scripts"]
    )


@lru_cache(maxsize=1)
def _rendered_paths() -> dict[str, Path]:
    """Project paths OSPREY scaffolding writes, mapped to their template."""
    from osprey.cli import deploy_scaffold_templates as t

    return {t.VERIFY_PATH: t.TEMPLATE_DIR / t.VERIFY_TEMPLATE}


def _claude_pin() -> str:
    from osprey.cli.templates.scaffolding import _DEFAULT_CLAUDE_CLI_VERSION

    return _DEFAULT_CLAUDE_CLI_VERSION


def _archiver_pin() -> str:
    from osprey_connectors.archiver._mya_archiver_connector import _REQUIRED_CLIENT

    return _REQUIRED_CLIENT


def _design_system_static() -> Path:
    spec = importlib.util.find_spec("osprey.interfaces.design_system")
    assert spec is not None and spec.origin is not None
    return Path(spec.origin).parent / "static"


# --------------------------------------------------------------------------- scanning


def _block_lines(lines: list[str]) -> list[tuple[int, str, str, str]]:
    """``(line number, text, language, caption)`` for every line inside a literal block."""
    inside: list[tuple[int, str, str, str]] = []
    index = 0
    while index < len(lines):
        opener = _BLOCK_OPENER.match(lines[index])
        index += 1
        if opener is None:
            continue
        base = len(opener.group(1))
        lang = opener.group("lang")
        language = "literal" if lang is None else (lang.strip() or "text")
        caption = ""
        while index < len(lines) and (
            not lines[index].strip() or lines[index].strip().startswith(":")
        ):
            option = lines[index].strip()
            if option.startswith(":caption:"):
                caption = option[len(":caption:") :].strip()
            index += 1
        while index < len(lines):
            text = lines[index]
            if text.strip() and len(text) - len(text.lstrip()) <= base:
                break
            if text.strip():
                inside.append((index + 1, text.strip(), language, caption))
            index += 1
    return inside


def _words(command: str) -> list[str]:
    """Shell words of ``command`` with a trailing comment dropped."""
    try:
        return shlex.split(command, comments=True)
    except ValueError:  # an open quote continued on the next line
        return command.split("#", 1)[0].split()


def _is_placeholder(word: str) -> bool:
    return any(mark in word for mark in "<>{}*…")


def _env_names(text: str) -> list[str]:
    """Names a line assigns at its start (``export`` optional)."""
    rest = text[len("export ") :] if text.startswith("export ") else text
    names: list[str] = []
    while (match := _ASSIGNMENT.match(rest)) is not None:
        names.append(match.group(1))
        rest = rest[match.end() :]
    return names


def _strip_assignments(text: str) -> str:
    rest = text[len("export ") :] if text.startswith("export ") else text
    while (match := _ASSIGNMENT.match(rest)) is not None:
        rest = rest[match.end() :]
    return rest


def _command_names(text: str, *, inline: bool = False) -> list[tuple[str, str]]:
    """``(kind, value)`` for every name the command on ``text`` carries.

    In an inline literal a bare ``scripts/...`` is a path named in prose, not a
    command, so ``inline`` leaves it alone.
    """
    words = _words(_strip_assignments(text))
    if words[:2] == ["uv", "run"]:
        words = words[2:]
        while words and words[0].startswith("-"):
            words = words[1:]
    if not words:
        return []
    head, args = words[0], words[1:]
    found: list[tuple[str, str]] = []
    if head == "uv" and args[:1] == ["sync"]:
        for index, word in enumerate(args):
            if word == "--extra" and index + 1 < len(args):
                found.append(("extra", args[index + 1]))
            elif word.startswith("--extra="):
                found.append(("extra", word.split("=", 1)[1]))
    elif (head == "pip" or head.endswith("/pip")) and args[:1] == ["install"]:
        found += _install_names(args[1:])
    elif head == "uv" and args[:2] in (
        ["pip", "install"],
        ["tool", "install"],
        ["tool", "upgrade"],
    ):
        found += _install_names(args[2:])
    elif head in ("pytest", "ruff"):
        found += [
            ("path", word.split("::", 1)[0])
            for word in args
            if "/" in word and not word.startswith("-")
        ]
    elif head in ("python", "python3") or head.endswith(("/python", "/python3")):
        if args[:1] == ["-m"] and len(args) > 1:
            found.append(("module", args[1]))
        elif args and not args[0].startswith("-"):
            found.append(("path", args[0]))
    elif not inline and head.startswith(("scripts/", "./scripts/")):
        found.append(("path", head))
    elif head == "npm" and args[:1] == ["run"] and len(args) > 1:
        found.append(("npm", args[1]))
    elif head == "npm" and args[:1] == ["install"]:
        found += [
            ("claude-pin", w[len(_CLAUDE_PACKAGE) :]) for w in args if w.startswith(_CLAUDE_PACKAGE)
        ]
    return [(kind, value) for kind, value in found if not _is_placeholder(value)]


def _install_names(args: list[str]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for word in args:
        if word.startswith("-"):
            continue
        match = _REQUIREMENT.match(word)
        if match is None:
            continue
        name = match.group("name")
        if name != "." and not name.startswith("osprey"):
            continue
        if name != ".":
            found.append(("distribution", name))
        extras = [e.strip() for e in (match.group("extras") or "").split(",") if e.strip()]
        found += [("install-extra", f"{name}[{extra}]") for extra in extras]
    return found


def _names(page: str, content: str) -> list[Name]:
    """Every checkable name on one page."""
    found: list[Name] = []
    lines = content.splitlines()
    for number, text, language, caption in _block_lines(lines):
        if language not in _SHELL_LANGUAGES and not caption.endswith(".env"):
            continue
        if text.startswith("$ "):
            text = text[2:].lstrip()
        found += [Name(page, number, "env", name) for name in _env_names(text)]
        found += [Name(page, number, kind, value) for kind, value in _command_names(text)]
    for number, line in enumerate(lines, start=1):
        for match in _LITERAL.finditer(line):
            literal = match.group(1).strip()
            if literal.startswith("$ "):
                literal = literal[2:]
            found += [
                Name(page, number, kind, value)
                for kind, value in _command_names(literal, inline=True)
            ]
        found += [Name(page, number, "archiver-pin", v) for v in _ARCHIVER_PIN.findall(line)]
        found += [
            Name(page, number, "design-system", v) for v in _DESIGN_SYSTEM_ASSET.findall(line)
        ]
    return sorted(set(found), key=lambda n: (n.line, n.kind, n.value))


@lru_cache(maxsize=1)
def _all_names() -> list[Name]:
    found: list[Name] = []
    for path in sorted(_DOCS.rglob("*.rst")):
        found += _names(path.relative_to(_DOCS).as_posix(), path.read_text(encoding="utf-8"))
    return found


# --------------------------------------------------------------------------- resolving


@lru_cache(maxsize=1)
def _shipped_words() -> frozenset[str]:
    """Every word (``[A-Za-z0-9_]+``) in the shipped tree, built once."""
    return frozenset(re.findall(r"[A-Za-z0-9_]+", _shipped_text()))


def _occurs_in_shipped_tree(name: str) -> bool:
    return name in _shipped_words()


def _module_problem(module: str) -> str | None:
    if not module.split(".")[0].startswith("osprey"):
        if module in _THIRD_PARTY_MODULES:
            return None
        return f"module {module!r} is neither under osprey nor a listed third-party module"
    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        return f"module {module!r} is not importable"
    if spec.submodule_search_locations is not None:
        if importlib.util.find_spec(f"{module}.__main__") is None:
            return f"package {module!r} has no __main__, so python -m cannot run it"
        return None
    source = Path(spec.origin).read_text(encoding="utf-8") if spec.origin else ""
    if '__name__ == "__main__"' not in source and "__name__ == '__main__'" not in source:
        return f"module {module!r} has no __main__ guard, so python -m does nothing"
    return None


def _path_problem(path: str) -> str | None:
    relative = path.removeprefix("./")
    if relative in _rendered_paths():
        return None
    if not (_REPO_ROOT / relative).exists():
        return f"path {relative!r} does not exist under the repository root"
    return None


def _problem(name: Name) -> str | None:
    """Why ``name`` does not resolve, or ``None`` when it does."""
    kind, value = name.kind, name.value
    if kind == "env":
        if _occurs_in_shipped_tree(value):
            return None
        return (
            f"{value} occurs as a word nowhere under {' or '.join(_SOURCE_ROOTS)} "
            "(code, comments or docstrings), so no shipped module can read it"
        )
    if kind == "extra":
        if value in _root_extras():
            return None
        return (
            f"extra {value!r} is not in [project.optional-dependencies] ({sorted(_root_extras())})"
        )
    if kind == "distribution":
        if value in _distributions():
            return None
        return (
            f"distribution {value!r} is not one this repository builds ({sorted(_distributions())})"
        )
    if kind == "install-extra":
        dist, extra = value[:-1].split("[", 1)
        dist = _pyproject()["project"]["name"] if dist == "." else dist
        if extra in _distributions().get(dist, frozenset()):
            return None
        return f"{dist} declares no extra {extra!r}"
    if kind == "path":
        return _path_problem(value)
    if kind == "npm":
        if value in _npm_scripts():
            return None
        return f"npm script {value!r} is not in package.json ({sorted(_npm_scripts())})"
    if kind == "module":
        return _module_problem(value)
    if kind == "claude-pin":
        pin = _claude_pin()
        return None if value == pin else f"Claude Code {value} is not the pinned {pin}"
    if kind == "archiver-pin":
        pin = _archiver_pin()
        return None if value == pin else f"jlab-archiver-client>={value} is not the required {pin}"
    if kind == "design-system":
        if (_design_system_static() / value).is_file():
            return None
        return f"/design-system/{value} is not a file in {_design_system_static()}"
    raise AssertionError(f"unknown kind {kind!r}")


# --------------------------------------------------------------------------- tests


@pytest.mark.parametrize("name", _all_names(), ids=lambda n: n.id)
def test_name_resolves(name: Name) -> None:
    """Each name a docs command carries is defined where the reader's shell looks."""
    if (name.page, name.value) in _ALLOWLIST:
        pytest.skip(_ALLOWLIST[(name.page, name.value)])
    problem = _problem(name)
    assert problem is None, f"docs/source/{name.page}:{name.line}: {problem}"


def test_every_allowlist_entry_is_needed() -> None:
    """An allowlist entry must still name a page/name pair that fails."""
    failing = {(n.page, n.value) for n in _all_names() if _problem(n) is not None}
    stale = sorted(set(_ALLOWLIST) - failing)
    assert stale == [], f"allowlist entries that no longer fail (remove them): {stale}"


def test_every_third_party_module_is_still_named() -> None:
    """A listed third-party module no page runs any more leaves the list."""
    named = {n.value for n in _all_names() if n.kind == "module"}
    stale = sorted(set(_THIRD_PARTY_MODULES) - named)
    assert stale == [], f"third-party modules no page runs (remove them): {stale}"


def test_rendered_paths_have_templates() -> None:
    """A path that resolves through scaffolding has a shipped template behind it."""
    missing = sorted(p for p, template in _rendered_paths().items() if not template.is_file())
    assert missing == [], f"scaffolded paths whose template is missing: {missing}"


def test_every_kind_is_found_on_the_real_pages() -> None:
    """Each sweep reads something, so none of them passes on an empty parse."""
    kinds = {n.kind for n in _all_names()}
    expected = {
        "env",
        "extra",
        "distribution",
        "path",
        "npm",
        "module",
        "claude-pin",
        "archiver-pin",
        "design-system",
    }
    assert expected <= kinds, f"sweeps that found nothing: {sorted(expected - kinds)}"


_NEGATIVE_PAGE = """\
Page
====

.. code-block:: bash

   uv sync --extra dev --extra no-such-extra
   pip install "osprey-framework[no-such-extra]"
   pip install osprey-not-a-distribution
   uv run pytest tests/no/such/dir/test_x.py -v
   python scripts/no_such_script.py
   npm run no-such-script
   npm install -g @anthropic-ai/claude-code@0.0.1
   export OSPREY_NO_SUCH_KNOB_EVER=1
   python -m osprey.no_such_module
   python -m some_unlisted_module

.. code-block:: ini
   :caption: my-project/.env

   OSPREY_NO_SUCH_DOTENV_KNOB=1

.. code-block:: yaml

   dependencies:
     - jlab-archiver-client>=0.0.1

.. code-block:: html

   <link rel="stylesheet" href="/design-system/css/no-such.css">

Prose mentioning NOT_A_LINE=1 and pip install osprey-x is not read.
"""


def test_negative_controls_fail() -> None:
    """A dead name of each kind is reported, and prose is not read."""
    names = _names("synthetic.rst", _NEGATIVE_PAGE)
    failing = {(n.kind, n.value) for n in names if _problem(n) is not None}
    assert failing == {
        ("extra", "no-such-extra"),
        ("install-extra", "osprey-framework[no-such-extra]"),
        ("distribution", "osprey-not-a-distribution"),
        ("path", "tests/no/such/dir/test_x.py"),
        ("path", "scripts/no_such_script.py"),
        ("npm", "no-such-script"),
        ("claude-pin", "0.0.1"),
        ("env", "OSPREY_NO_SUCH_KNOB_EVER"),
        ("env", "OSPREY_NO_SUCH_DOTENV_KNOB"),
        ("module", "osprey.no_such_module"),
        ("module", "some_unlisted_module"),
        ("archiver-pin", "0.0.1"),
        ("design-system", "css/no-such.css"),
    }
    passing = {(n.kind, n.value) for n in names if _problem(n) is None}
    assert passing == {("extra", "dev"), ("distribution", "osprey-framework")}


def test_positive_controls_resolve() -> None:
    """Real names of each kind resolve, including the scaffolded path."""
    page = """\
.. code-block:: console

   $ uv run --frozen python -m osprey.dispatch
   $ ./scripts/verify.sh --strict
   $ npm run test:js
   $ pip install "osprey-connectors"
"""
    names = _names("synthetic.rst", page)
    assert {(n.kind, n.value) for n in names} == {
        ("module", "osprey.dispatch"),
        ("path", "./scripts/verify.sh"),
        ("npm", "test:js"),
        ("distribution", "osprey-connectors"),
    }
    assert [n for n in names if _problem(n) is not None] == []
