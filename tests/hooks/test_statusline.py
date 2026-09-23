"""Tests for the deployed-agent statusline script.

``statusline.py`` reads Claude Code state JSON from stdin and emits one colored
status line to stdout. It is invoked as a script (``python3 statusline.py``),
so these tests drive it the same way — feeding JSON on stdin and asserting on
the (ANSI-stripped) line it prints, plus its graceful handling of malformed
input. Pure helpers (model shortening, context math) are also exercised by
direct import for exact-value assertions the rendered line would obscure.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

import osprey.templates.claude_code.claude.statusline as statusline

STATUSLINE_SCRIPT = (
    Path(__file__).parents[2]
    / "src"
    / "osprey"
    / "templates"
    / "claude_code"
    / "claude"
    / "statusline.py"
)

_ANSI = re.compile(r"\033\[[0-9;]*m")


def _strip(text: str) -> str:
    return _ANSI.sub("", text)


def _run(payload, cwd=None):
    """Invoke the statusline script with ``payload`` (dict or raw str) on stdin.

    Returns ``(returncode, ansi_stripped_stdout)``.
    """
    stdin_data = payload if isinstance(payload, str) else json.dumps(payload)
    result = subprocess.run(
        [sys.executable, str(STATUSLINE_SCRIPT)],
        input=stdin_data,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
    )
    return result.returncode, _strip(result.stdout)


#: The branch :func:`answering_git` reports.
_STUB_BRANCH = "osprey-test-branch"


@pytest.fixture
def answering_git(monkeypatch):
    """Make the statusline's ``git`` call answer with a branch, at once.

    The statusline holds that call to a one-second budget so that no repository
    can delay a prompt, and drops the branch when the budget runs out. A row
    that shells out to a real git therefore asserts a wall-clock race alongside
    the rendering, and loses that race on a machine running the suite in
    parallel.

    Answering in-process takes the budget, the ``PATH`` lookup and the state of
    any real repository out of the row without softening what it asserts: the
    stub records the argv it was handed, so the row pins the command the
    statusline sends as well as what it renders from the answer.
    """
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout=f"{_STUB_BRANCH}\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _render(payload) -> str:
    """Run the statusline in-process over ``payload``; return its stripped line."""
    stdout = io.StringIO()
    with (
        mock.patch.object(statusline.sys, "stdin", io.StringIO(json.dumps(payload))),
        mock.patch.object(statusline.sys, "stdout", stdout),
    ):
        statusline.main()
    return _strip(stdout.getvalue())


# ---------------------------------------------------------------------------
# Pure helpers (direct import)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("display_name", "expected"),
    [
        ("Claude 3.5 Sonnet", "Sonnet"),
        ("Claude Opus 4.8", "Opus"),
        ("Claude 3.5 Haiku", "Haiku"),
        ("claude sonnet 5", "Sonnet"),  # case-insensitive match
    ],
)
def test_model_short_maps_known_families(display_name, expected):
    assert statusline._model_short({"model": {"display_name": display_name}}) == expected


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("claude-sonnet-5", "Sonnet 5"),
        ("claude-haiku-4-5-20251001", "Haiku 4.5"),
        ("claude-fable-5-1", "Fable 5.1"),
        ("claude-opus-5[1m]", "Opus 5"),
        ("gpt-6-sol", "gpt-6-sol"),
    ],
)
def test_model_short_names_the_id_without_the_vendor_prefix(model_id, expected):
    data = {"model": {"id": model_id, "display_name": "Claude Something"}}
    assert statusline._model_short(data) == expected


def test_model_short_unknown_returns_raw():
    assert statusline._model_short({"model": {"display_name": "Gemini Pro"}}) == "Gemini Pro"


def test_model_short_missing_returns_question_mark():
    assert statusline._model_short({}) == "?"
    assert statusline._model_short({"model": {}}) == "?"


def test_context_math():
    """45% of a 200k window is 90k used; sizes are reported in k."""
    data = {"context_window": {"used_percentage": 45, "context_window_size": 200000}}
    pct, used_k, max_k = statusline._context(data)
    assert (pct, used_k, max_k) == (45, 90, 200)


def test_context_defaults_to_zero_when_absent():
    assert statusline._context({}) == (0, 0, 0)


# ---------------------------------------------------------------------------
# End-to-end stdin -> stdout contract
# ---------------------------------------------------------------------------


def test_line_contains_model_context_and_versions(tmp_path):
    """A fully-populated payload renders every segment; branch is absent because
    tmp_path is not a git repo."""
    payload = {
        "model": {"display_name": "Claude 3.5 Sonnet"},
        "context_window": {"used_percentage": 45, "context_window_size": 200000},
        "workspace": {"current_dir": str(tmp_path)},
        "version": "2.0.0",
    }
    rc, out = _run(payload, cwd=tmp_path)
    assert rc == 0
    assert "Sonnet" in out
    assert "45% 90k/200K" in out
    assert tmp_path.name in out
    assert "v2.0.0" in out
    assert "(" not in out  # no git branch parens for a non-repo dir


def test_branch_rendered_for_git_repo(tmp_path, answering_git):
    """When current_dir is a git repo, the abbreviated branch appears in parens."""
    out = _render(
        {
            "model": {"display_name": "Opus"},
            "workspace": {"current_dir": str(tmp_path)},
        }
    )

    assert f"({_STUB_BRANCH})" in out
    assert answering_git == [["git", "-C", str(tmp_path), "rev-parse", "--abbrev-ref", "HEAD"]]


def test_optional_segments_omitted_when_absent(tmp_path):
    """No version key -> no ``v...`` segment; the line still renders."""
    payload = {
        "model": {"display_name": "Haiku"},
        "workspace": {"current_dir": str(tmp_path)},
    }
    rc, out = _run(payload, cwd=tmp_path)
    assert rc == 0
    assert "Haiku" in out
    # A bare Claude Code version segment (" v<digits>") must not appear.
    assert not re.search(r"\bv\d", out.replace("osprey-v", ""))


def test_empty_stdin_is_handled_gracefully():
    """Empty stdin -> {} -> a line with the '?' model placeholder, exit 0."""
    rc, out = _run("")
    assert rc == 0
    assert "?" in out


def test_malformed_json_is_handled_gracefully():
    """Non-JSON stdin must not crash the statusline (exit 0, '?' model)."""
    rc, out = _run("this is not json{{{")
    assert rc == 0
    assert "?" in out


def test_output_is_single_line(tmp_path):
    """The statusline must emit exactly one line (no embedded newlines)."""
    payload = {
        "model": {"display_name": "Sonnet"},
        "workspace": {"current_dir": str(tmp_path)},
        "version": "1.2.3",
    }
    rc, out = _run(payload, cwd=tmp_path)
    assert rc == 0
    assert "\n" not in out.strip()
