"""What a message tells an operator to run, set or expect is true of this tree.

A warning, a refusal, a ``--help`` page and a banner written into ``.env`` are
read at the moment something went wrong, and followed as written. The tests
below hold the commands they quote to the click tree, the keys they name to the
config-key manifest, and the release they promise to the sunset constants.
"""

from __future__ import annotations

import ast
import re
import shlex
from collections.abc import Iterator
from pathlib import Path

import click
import yaml

from tests.docs.test_cli_page_commands import _options, _registered, _resolve
from tests.docs.test_environment_variable_page import _REPO_ROOT

_SRC = _REPO_ROOT / "src" / "osprey"

#: An ``osprey`` invocation quoted in prose, or closing a ``.env`` block banner.
_QUOTED_COMMAND = re.compile(r"'(osprey [^'\n]+)'|`(osprey [^`\n]+)`|\((osprey [^()\n]+)\)$")

#: Tuples of every spelling a block header was ever written under. They are
#: lookup keys for files already on an operator's disk, retired commands included.
_HISTORY_TUPLES = re.compile(r"_(?:BANNERS|HEADERS)$")

#: A placeholder: the command is a shape to fill in, not a line to run.
_PLACEHOLDER = re.compile(r"[<{\[…]|\.\.\.")

_UNDATED_SUNSET = re.compile(
    r"\b(?:deprecated|refused|removed)\b[^.]*\b(?:next|future)\b[^.]*\brelease", re.I
)


def _rel(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _messages() -> Iterator[tuple[Path, int, str]]:
    """Every string literal under ``src/osprey`` that is not a docstring."""
    for path in sorted(_SRC.rglob("*.py")):
        if "templates" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
        }
        skipped = docstrings | {
            id(element)
            for node in tree.body
            if isinstance(node, ast.Assign | ast.AnnAssign)
            and isinstance(node.value, ast.Tuple)
            and _HISTORY_TUPLES.search(
                ast.unparse(node.targets[0] if isinstance(node, ast.Assign) else node.target)
            )
            for element in node.value.elts
        }
        for node in ast.walk(tree):
            if id(node) in skipped:
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                yield path, node.lineno, node.value
            elif isinstance(node, ast.JoinedStr):
                parts = [
                    part.value if isinstance(part, ast.Constant) else "{…}" for part in node.values
                ]
                yield path, node.lineno, "".join(map(str, parts))


def _root() -> click.Command:
    from osprey.cli.main import cli

    return cli


def _problems(text: str) -> list[str]:
    """Why a command quoted in ``text`` cannot be run as written."""
    registered = _registered()
    problems = []
    quotes = ["".join(groups) for groups in _QUOTED_COMMAND.findall(text)]
    for quoted in (step.strip() for quote in quotes for step in quote.split("&&")):
        if _PLACEHOLDER.search(quoted) or not quoted.startswith("osprey "):
            continue
        words = shlex.split(quoted)[1:]
        path, unknown = _resolve(words, registered)
        if unknown:
            problems.append(f"`{quoted}`: no such command `{unknown}`")
            continue
        options = _options(registered[path] if path else _root())
        given = {word.split("=")[0] for word in words[len(path) :] if word.startswith("-")}
        problems += [f"`{quoted}`: no option `{word}`" for word in sorted(given - set(options))]
        problems += [
            f"`{quoted}`: `{option.opts[0]}` is required"
            for option in {id(o): o for o in options.values()}.values()
            if option.required and not given & set(option.opts)
        ]
    return problems


def test_a_command_a_message_quotes_runs_as_written() -> None:
    offences = [
        f"{_rel(path)}:{lineno}: {problem}"
        for path, lineno, text in _messages()
        for problem in _problems(text)
    ]
    assert not offences, "\n".join(offences)


def test_a_command_a_help_page_quotes_runs_as_written() -> None:
    offences = [
        f"osprey {' '.join(path)} --help: {problem}"
        for path, command in _registered().items()
        for problem in _problems(command.help or "")
    ]
    assert not offences, "\n".join(offences)


def test_no_help_page_names_a_deleted_config_key() -> None:
    manifest = _SRC / "profiles" / "config_key_manifest.yml"
    deleted = yaml.safe_load(manifest.read_text(encoding="utf-8"))["deleted"]
    offences = []
    for path, command in _registered().items():
        texts = [command.help or ""]
        texts += [p.help or "" for p in command.params if isinstance(p, click.Option)]
        named = set(re.findall(r"\bconfig\.([a-z_][\w.-]*\w)", " ".join(texts)))
        offences += [
            f"osprey {' '.join(path)}: config.{key}" for key in sorted(named) if key in deleted
        ]
    assert not offences, "help text names keys the manifest lists as deleted:\n" + "\n".join(
        offences
    )


def test_a_sunset_names_its_release() -> None:
    """A retirement takes the release that refuses it from the window's constant."""
    offences = [
        f"{_rel(path)}:{lineno}: {match.group(0)}"
        for path, lineno, text in _messages()
        if (match := _UNDATED_SUNSET.search(" ".join(text.split())))
    ]
    assert not offences, (
        "take the release from the window's *_REFUSED_FROM constant:\n" + "\n".join(offences)
    )
