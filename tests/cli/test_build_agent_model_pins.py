"""``osprey build`` applies every ``claude_code.agent_models`` pin or refuses it.

Each test runs a real ``osprey init --preset hello-world`` and ``osprey build``
through ``CliRunner``, with the pin written into the profile's ``config:`` block.
The build logs its refusal through ``logger.error`` rather than printing it to
the command's output, so the refusal tests read ``caplog.text``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.init_cmd import init


def _init_repo(tmp_path: Path, *config_lines: str) -> Path:
    """A hello-world repo whose ``config:`` block starts with *config_lines*."""
    repo = tmp_path / "pins-repo"
    result = CliRunner().invoke(init, [str(repo), "--preset", "hello-world", "--no-git"])
    assert result.exit_code == 0, result.output
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    assert text.count("\nconfig:\n") == 1
    inserted = "".join(f"  {line}\n" for line in config_lines)
    profile.write_text(text.replace("\nconfig:\n", "\nconfig:\n" + inserted), encoding="utf-8")
    return repo


def _site_helper(repo: Path, model: str) -> None:
    """Ship the deployment's own agent file ``agents/site-helper.md``."""
    agents = repo / "agents"
    agents.mkdir(exist_ok=True)
    (agents / "site-helper.md").write_text(
        "---\n"
        "name: site-helper\n"
        "description: Answers questions about the site.\n"
        f"model: {model}\n"
        "---\n\n"
        "Answer questions about the site.\n",
        encoding="utf-8",
    )


def _build(repo: Path):
    return CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])


def test_a_pin_naming_no_agent_is_refused_with_the_agent_names(tmp_path, caplog):
    repo = _init_repo(tmp_path, "claude_code.agent_models.chanel-finder: claude-sonnet-5")

    with caplog.at_level(logging.ERROR):
        result = _build(repo)

    assert result.exit_code != 0
    assert "claude_code.agent_models.chanel-finder: no agent is called" in caplog.text
    assert "Agents: channel-finder, data-visualizer," in caplog.text


def test_a_pin_on_a_framework_agent_this_deployment_does_not_ship_builds(tmp_path):
    repo = _init_repo(tmp_path, "claude_code.agent_models.channel-finder: claude-haiku-4-5")

    result = _build(repo)

    assert result.exit_code == 0, result.output


def test_a_pin_the_profile_agent_file_overrides_is_refused(tmp_path, caplog):
    repo = _init_repo(tmp_path, "claude_code.agent_models.site-helper: claude-haiku-4-5")
    _site_helper(repo, "claude-sonnet-5")

    with caplog.at_level(logging.ERROR):
        result = _build(repo)

    assert result.exit_code != 0
    assert "claude-haiku-4-5 is not what agents/site-helper.md runs (claude-sonnet-5)" in (
        caplog.text
    )
    assert "so set model: there or remove the pin" in caplog.text


def test_a_pin_the_profile_agent_file_already_runs_builds(tmp_path):
    repo = _init_repo(tmp_path, "claude_code.agent_models.site-helper: claude-haiku-4-5")
    _site_helper(repo, "claude-haiku-4-5")

    result = _build(repo)

    assert result.exit_code == 0, result.output
    shipped = (repo / "build" / ".claude" / "agents" / "site-helper.md").read_text()
    assert "\nmodel: claude-haiku-4-5\n" in shipped
