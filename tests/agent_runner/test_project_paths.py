"""Tests for the Claude Code config-dir / project-dir resolution helpers.

Claude Code keeps per-project state under ``$CLAUDE_CONFIG_DIR/projects/
<encoded>/`` and falls back to ``~/.claude`` only when the variable is unset.
The per-user web-terminal container sets ``CLAUDE_CONFIG_DIR`` (and ``HOME``)
to the same mounted volume, so a reader that spells ``~/.claude`` looks one
``.claude`` too deep and finds nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.agent_runner.project_paths import (
    CLAUDE_CONFIG_DIR_ENV,
    claude_config_dir,
    claude_project_dir,
    encode_claude_project_path,
)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    monkeypatch.delenv(CLAUDE_CONFIG_DIR_ENV, raising=False)
    return tmp_path / "home"


def test_config_dir_defaults_to_dot_claude_under_home(fake_home):
    assert claude_config_dir() == fake_home / ".claude"


def test_config_dir_honours_the_environment(fake_home, tmp_path, monkeypatch):
    """``CLAUDE_CONFIG_DIR`` is used verbatim — no ``.claude`` appended."""
    configured = tmp_path / "data" / "claude-config"
    monkeypatch.setenv(CLAUDE_CONFIG_DIR_ENV, str(configured))

    assert claude_config_dir() == configured


def test_blank_environment_value_falls_back_to_home(fake_home, monkeypatch):
    monkeypatch.setenv(CLAUDE_CONFIG_DIR_ENV, "   ")

    assert claude_config_dir() == fake_home / ".claude"


def test_config_dir_expands_a_tilde(fake_home, monkeypatch):
    # ``expanduser`` reads $HOME, not the patched ``Path.home``.
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv(CLAUDE_CONFIG_DIR_ENV, "~/elsewhere")

    assert claude_config_dir() == fake_home / "elsewhere"


def test_project_dir_sits_under_projects_with_the_encoded_name(fake_home, tmp_path, monkeypatch):
    configured = tmp_path / "data" / "claude-config"
    monkeypatch.setenv(CLAUDE_CONFIG_DIR_ENV, str(configured))
    project = tmp_path / "app" / "my_project" / "build"

    assert claude_project_dir(project) == (
        configured / "projects" / encode_claude_project_path(project)
    )
