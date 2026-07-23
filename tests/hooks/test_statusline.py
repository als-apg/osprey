"""Tests for the deployed-agent statusline script.

``statusline.py`` reads Claude Code state JSON from stdin and emits one colored
status line to stdout. It is invoked as a script (``python3 statusline.py``),
so these tests drive it the same way — feeding JSON on stdin and asserting on
the (ANSI-stripped) line it prints, plus its graceful handling of malformed
input. Pure helpers (model shortening, context math) are also exercised by
direct import for exact-value assertions the rendered line would obscure.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

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


def test_branch_rendered_for_git_repo(tmp_path):
    """When current_dir is a git repo, the abbreviated branch appears in parens."""
    init = subprocess.run(
        ["git", "init", "-b", "osprey-test-branch", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    if init.returncode != 0:
        pytest.skip("git unavailable or too old for `git init -b`")
    # `rev-parse --abbrev-ref HEAD` (what the statusline runs) needs a commit to
    # resolve — an unborn branch errors. Make an empty one with a local identity.
    commit = subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=test",
            "-C",
            str(tmp_path),
            "commit",
            "--allow-empty",
            "-m",
            "init",
        ],
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0:
        pytest.skip(f"git commit unavailable: {commit.stderr}")

    payload = {
        "model": {"display_name": "Opus"},
        "workspace": {"current_dir": str(tmp_path)},
    }
    rc, out = _run(payload, cwd=tmp_path)
    assert rc == 0
    assert "(osprey-test-branch)" in out


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
