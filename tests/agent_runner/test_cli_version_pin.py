"""One check of ``claude_code.cli_version`` on every surface that consumes it.

The pin ends up after ``@anthropic-ai/claude-code@`` in an ``npx`` or ``npm``
command line: the interactive launch argv, the project image's build argument
and the persona image's. In that position npm also accepts an alias, a URL, a
git address, a folder, a tarball in the working directory, a range and a
dist-tag — none of which names one build. So a pin is an exact version and
nothing else, and the same table of bad pins is refused wherever the value is
read.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from osprey.agent_runner import launcher
from osprey.agent_runner.launcher import build_claude_launch_argv
from osprey.cli.templates.scaffolding import _DEFAULT_CLAUDE_CLI_VERSION, _default_cli_version
from osprey.deployment.container_lifecycle import _resolve_claude_cli_version
from osprey.deployment.web_terminals.persona_images import _resolve_persona_claude_cli_version
from osprey.health.core.claude_cli import claude_cli_pinned
from osprey.health.models import Status

BAD_PINS = [
    # another package source
    "npm:another-package@1.0.0",
    "npm:@scope/another-package",
    "https://example.invalid/package.tgz",
    "git" + "+ssh://user@example.invalid/someone/repo",
    "github:someone/repo",
    "someone/repo",
    "file:../package",
    # a local file or folder npm would install
    "claude-code.tgz",
    "claude-code-2.1.146.tgz",
    "package.tar.gz",
    "../../elsewhere",
    "..",
    ".",
    "/opt/package",
    "C:\\package",
    # a range or a wildcard, which does not pin
    "^2.1.0",
    "^2.1",
    "~2.1.0",
    ">=2.1.0",
    ">=2",
    "2.x",
    "2.1.x",
    "*",
    "2.1.146 || 3.0.0",
    "2.1.0 - 2.2.0",
    "2",
    "2.1",
    "2.1.146.1",
    # a dist-tag, which no installed build reports as its version
    "latest",
    "stable",
    "next",
    # anything beyond the version
    "2.1.146@extra",
    "@2.1.146",
    "v2.1.146",
    "=2.1.146",
    "2.1.146+build.5",
    "2.1.146-",
    "2.1.146-beta_1",
    " 2.1.146",
    "2.1.146 ",
    "2.1.146\n",
    "2.1.146\n--registry=https://example.invalid",
    "2.1.146 --registry=https://example.invalid",
    "2.1 .146",
    "-2.1.146",
    "--version",
    "2.1.146; echo pwned",
    "$(id)",
    "",
    "   ",
]

GOOD_PINS = ["2.1.146", "2.1.267", "0.0.1", "10.20.300", "2.2.0-beta.1", "2.2.0-rc-1.2"]

#: A configured value that is not text at all.
NON_STRING_PINS = [2, 2.1, True, ["2.1.146"], {"version": "2.1.146"}]


@pytest.mark.parametrize("pin", GOOD_PINS)
def test_an_exact_version_is_a_valid_pin_and_comes_back_unchanged(pin):
    assert launcher.validate_cli_version(pin) == pin
    assert launcher.pinned_cli_version({"cli_version": pin}) == pin


@pytest.mark.parametrize("pin", BAD_PINS, ids=repr)
def test_anything_but_an_exact_version_is_refused_with_a_clear_message(pin):
    with pytest.raises(ValueError) as refused:
        launcher.validate_cli_version(pin)

    message = str(refused.value)
    assert "claude_code.cli_version" in message
    assert repr(pin) in message, "the offending value is shown"
    assert '"2.1.146"' in message, "one valid example is shown"
    assert "exact version" in message


@pytest.mark.parametrize("pin", NON_STRING_PINS, ids=repr)
def test_a_non_string_pin_is_refused(pin):
    with pytest.raises(ValueError, match="claude_code.cli_version") as refused:
        launcher.validate_cli_version(pin)
    with pytest.raises(ValueError, match="claude_code.cli_version"):
        launcher.pinned_cli_version({"cli_version": pin})

    assert repr(pin) in str(refused.value)


@pytest.mark.parametrize("cc_config", [{}, {"cli_version": None}, {"provider": "x"}], ids=repr)
def test_no_pin_is_no_version(cc_config):
    assert launcher.pinned_cli_version(cc_config) is None


@pytest.mark.parametrize("pin", GOOD_PINS)
def test_the_pin_reaches_npx_as_one_argv_element_after_the_package_name(pin):
    argv = build_claude_launch_argv({"cli_version": pin})

    assert argv[:3] == ["npx", "-y", f"@anthropic-ai/claude-code@{pin}"]


# -- every surface refuses the same table ------------------------------------

EVERY_BAD_PIN = [*BAD_PINS, *NON_STRING_PINS]


@pytest.mark.parametrize("pin", EVERY_BAD_PIN, ids=repr)
def test_the_interactive_launch_argv_refuses_a_bad_pin(pin):
    with pytest.raises(ValueError, match="claude_code.cli_version"):
        build_claude_launch_argv({"cli_version": pin})


@pytest.mark.parametrize("pin", EVERY_BAD_PIN, ids=repr)
def test_the_interactive_launch_argv_ignores_the_pin_entirely_under_no_pin(pin):
    assert build_claude_launch_argv({"cli_version": pin}, no_pin=True)[0] == launcher.CLI_NAME


@pytest.mark.parametrize("pin", EVERY_BAD_PIN, ids=repr)
def test_the_rendered_project_image_refuses_a_bad_pin(pin):
    with pytest.raises(ValueError, match="claude_code.cli_version"):
        _default_cli_version({"claude_code": {"cli_version": pin}})
    with pytest.raises(ValueError, match="claude_code.cli_version"):
        _default_cli_version({"claude_code_cli_version": pin})


def test_the_rendered_project_image_keeps_a_good_pin_and_its_default():
    nested: dict = {"claude_code": {"cli_version": "2.1.146"}}
    flat: dict = {"claude_code_cli_version": "2.1.147"}
    unpinned: dict = {"claude_code": {"cli_version": None}}
    bare: dict = {}

    for ctx in (nested, flat, unpinned, bare):
        _default_cli_version(ctx)

    assert nested["claude_code_cli_version"] == "2.1.146"
    assert flat["claude_code_cli_version"] == "2.1.147"
    assert unpinned["claude_code_cli_version"] == _DEFAULT_CLAUDE_CLI_VERSION
    assert bare["claude_code_cli_version"] == _DEFAULT_CLAUDE_CLI_VERSION


def test_the_framework_default_is_itself_a_valid_pin():
    assert launcher.validate_cli_version(_DEFAULT_CLAUDE_CLI_VERSION) == _DEFAULT_CLAUDE_CLI_VERSION


@pytest.mark.parametrize("pin", EVERY_BAD_PIN, ids=repr)
def test_the_project_image_build_argument_refuses_a_bad_pin(pin):
    with pytest.raises(ValueError, match="claude_code.cli_version"):
        _resolve_claude_cli_version({"claude_code": {"cli_version": pin}})


def test_the_project_image_build_argument_keeps_a_good_pin_and_its_default():
    assert _resolve_claude_cli_version({"claude_code": {"cli_version": "2.1.146"}}) == "2.1.146"
    assert _resolve_claude_cli_version({}) == _DEFAULT_CLAUDE_CLI_VERSION
    assert _resolve_claude_cli_version({"claude_code": {}}) == _DEFAULT_CLAUDE_CLI_VERSION


@pytest.mark.parametrize("pin", EVERY_BAD_PIN, ids=repr)
def test_the_persona_image_build_argument_refuses_a_bad_pin(tmp_path: Path, pin):
    import yaml

    (tmp_path / "config.yml").write_text(
        yaml.safe_dump({"claude_code": {"cli_version": pin}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="claude_code.cli_version"):
        _resolve_persona_claude_cli_version(str(tmp_path))


def test_the_persona_image_build_argument_keeps_a_good_pin(tmp_path: Path):
    (tmp_path / "config.yml").write_text(
        "claude_code:\n  cli_version: '2.1.146'\n", encoding="utf-8"
    )

    assert _resolve_persona_claude_cli_version(str(tmp_path)) == "2.1.146"


@pytest.mark.parametrize("pin", EVERY_BAD_PIN, ids=repr)
def test_the_health_check_reports_a_bad_pin_and_runs_nothing(monkeypatch, pin):
    def refuse(*_args, **_kwargs):
        raise AssertionError("a bad pin must reach no command line")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", refuse)
    monkeypatch.setattr(launcher.subprocess, "run", refuse)

    (row,) = asyncio.run(claude_cli_pinned({"claude_code": {"cli_version": pin}})())

    assert row.name == "claude_cli_pinned"
    assert row.status == Status.ERROR
    assert "claude_code.cli_version" in row.message
