"""Shell scripts under ``scripts/`` call the osprey CLI only with what it declares.

Every ``-m osprey.cli.main`` call in a tracked ``scripts/**/*.sh`` resolves, through
the real ``cli`` group, to a command that declares every subcommand, option and
positional it passes, and receives every option and argument that command requires.

The check is static: it resolves each call through the real ``cli`` group and parses
its tokens with each command's own parser, without invoking anything. The parser
converts no types and runs no callbacks, so a shell reference such as
``$PROV_DIR/prov`` reaches it as a literal string.
"""

from __future__ import annotations

import bisect
import shlex
from pathlib import Path

import click
import pytest

from osprey.cli.main import cli

_REPO = Path(__file__).resolve().parents[2]
_MARK = "-m osprey.cli.main"
_SHELL_OPERATOR = frozenset("();<>|&")


def _logical_lines(text: str) -> list[tuple[int, str, tuple[int, ...]]]:
    """Join backslash-continued lines.

    Returns ``(first_line, joined, starts)`` triples: ``first_line`` is the 1-based
    physical line where the joined command begins, and ``starts`` holds the offset in
    ``joined`` at which each physical line's content begins.
    """
    result: list[tuple[int, str, tuple[int, ...]]] = []
    first = 0
    joined: str | None = None
    starts: list[int] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if joined is None:
            first, joined, starts = number, "", [0]
            piece = line
        else:
            starts.append(len(joined) + 1)
            joined += " "
            piece = line
        if piece.endswith("\\"):
            joined += piece[:-1]
            continue
        joined += piece
        result.append((first, joined, tuple(starts)))
        joined = None
    if joined is not None:
        result.append((first, joined, tuple(starts)))
    return result


def _invocations(text: str) -> list[tuple[int, list[str]]]:
    """Return ``(line, tokens)`` for every CLI call outside a comment.

    ``line`` is the physical line holding the call's ``-m osprey.cli.main`` marker;
    ``tokens`` run up to the first shell operator.
    """
    calls: list[tuple[int, list[str]]] = []
    for first, joined, starts in _logical_lines(text):
        if joined.lstrip().startswith("#"):
            continue
        off = joined.find(_MARK)
        while off != -1:
            lexer = shlex.shlex(joined[off + len(_MARK) :], posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            tokens: list[str] = []
            for token in lexer:
                if token and set(token) <= _SHELL_OPERATOR:
                    break
                tokens.append(token)
            calls.append((first + bisect.bisect_right(starts, off) - 1, tokens))
            off = joined.find(_MARK, off + len(_MARK))
    return calls


_SCRIPTS = sorted(
    p for p in (_REPO / "scripts").rglob("*.sh") if _MARK in p.read_text(encoding="utf-8")
)

_CALLS = [
    pytest.param(script, line, tokens, id=f"{script.relative_to(_REPO)}:{line}")
    for script in _SCRIPTS
    for line, tokens in _invocations(script.read_text(encoding="utf-8"))
]


def _given(value: object) -> bool:
    """A parsed value counts as present when it is a string or a non-empty sequence."""
    return isinstance(value, str) or (isinstance(value, list | tuple) and len(value) > 0)


def _check(tokens: list[str]) -> tuple[tuple[str, ...], str | None]:
    """Resolve ``tokens`` through ``cli``; return the command path and a reason or None."""
    cmd: click.Command = cli
    ctx = click.Context(cli, info_name="osprey")
    args = list(tokens)
    path: list[str] = []
    while True:
        try:
            opts, rest, _ = cmd.make_parser(ctx).parse_args(args=args)
        except click.UsageError as e:
            return tuple(path), f"{type(e).__name__}: {e}"
        if isinstance(cmd, click.Group):
            if not rest:
                return tuple(path), "no subcommand given"
            sub = cmd.get_command(ctx, rest[0])
            if sub is None:
                return tuple(path), f"no subcommand {rest[0]!r}"
            path.append(rest[0])
            ctx = click.Context(sub, info_name=rest[0], parent=ctx)
            cmd, args = sub, rest[1:]
            continue
        if rest:
            return tuple(path), f"unexpected argument(s) {rest}"
        names = [p.name for p in cmd.params if p.required and not _given(opts.get(p.name))]
        if names:
            return tuple(path), f"missing required {names}"
        return tuple(path), None


def test_the_benchmark_worker_is_scanned():
    worker = _REPO / "scripts/benchmark/run_e2e_for_model.sh"
    assert worker in _SCRIPTS
    paths = {_check(tokens)[0] for _, tokens in _invocations(worker.read_text(encoding="utf-8"))}
    assert paths >= {
        ("init",),
        ("build",),
        ("ariel", "migrate"),
        ("sim", "apply"),
        ("ariel", "reembed"),
    }


@pytest.mark.parametrize(("script", "line", "tokens"), _CALLS)
def test_script_call_matches_the_declared_cli(script, line, tokens):
    reason = _check(tokens)[1]
    assert reason is None, (
        f"{script.relative_to(_REPO)}:{line}: osprey {' '.join(tokens)} -> {reason}"
    )


def test_logical_lines_join_continuations_and_keep_the_start_line():
    snippet = "a one \\\n  b two \\\n  c three\nd four\n"
    lines = [(n, s) for n, s, _ in _logical_lines(snippet)]
    assert len(lines) == 2
    assert lines[0][0] == 1
    assert all(part in lines[0][1] for part in ("a one", "b two", "c three"))
    assert "\\" not in lines[0][1]
    assert lines[1] == (4, "d four")


def test_invocations_stop_at_shell_operators_and_skip_comments():
    snippet = (
        '# "$PY" -m osprey.cli.main build --gone\n'
        '( cd x && "$PY" -m osprey.cli.main ariel migrate && "$PY" -m osprey.cli.main sim apply a'
        " --yes ) >&2\n"
    )
    assert [tokens for _, tokens in _invocations(snippet)] == [
        ["ariel", "migrate"],
        ["sim", "apply", "a", "--yes"],
    ]


@pytest.mark.parametrize(
    ("tokens", "fragment"),
    [
        pytest.param(
            [
                "build",
                "prov",
                "--preset",
                "control-assistant",
                "--skip-deps",
                "--skip-lifecycle",
                "--output-dir",
                "$PROV_DIR",
                "--set",
                "provider=als-apg",
            ],
            "--preset",
            id="build-dropped-options",
        ),
        pytest.param(["ariel", "nope"], "no subcommand 'nope'", id="unknown-subcommand"),
        pytest.param(["init", "dir", "extra"], "unexpected argument", id="extra-positional"),
        pytest.param(["ariel", "reembed", "--model", "m"], "dimension", id="missing-option"),
        pytest.param(
            ["ariel", "reembed", "--model", "m", "--dimension"],
            "--dimension",
            id="option-without-value",
        ),
        pytest.param(["sim", "apply", "--yes"], "names", id="missing-argument"),
        pytest.param(["--nope", "build"], "--nope", id="unknown-group-option"),
        pytest.param(["sim"], "no subcommand given", id="group-without-subcommand"),
    ],
)
def test_check_rejects_what_the_cli_does_not_declare(tokens, fragment):
    reason = _check(tokens)[1]
    assert reason is not None
    assert fragment in reason


def test_check_accepts_group_options_before_the_subcommand():
    assert _check(["-v", "build", "--skip-deps"]) == (("build",), None)
