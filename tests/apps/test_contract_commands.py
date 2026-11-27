"""Every refused application is reported, by every command, in one run.

``loader.load`` collects each per-manifest refusal and raises once, so a
deployment with several broken ``app.yml`` files shows all of them at the first
``osprey app check``, ``osprey build`` or ``osprey validate``; ``osprey set``
warns with the same lines instead of swallowing them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.apps.errors import AppManifestError
from osprey.apps.loader import load
from osprey.cli.app_cmd import app
from osprey.cli.set_cmd import set as set_command
from osprey.cli.validate_cmd import validate
from tests.apps._contract import CONTRACT
from tests.cli._lifecycle_build import invoke_build
from tests.fixtures.lifecycle_repo import (  # noqa: F401
    lifecycle_repo,
    lifecycle_repo_factory,
)

#: A refusal line, as opposed to click's ``Usage: ... [OPTIONS]`` line.
_FINDING = re.compile(r"^apps/\S+/app\.yml: .* \[\w+\]$")

#: A contract this OSPREY does not implement.
OTHER = CONTRACT + 1


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_app(repo: Path, directory: str, *, name: str | None = None, contract: int) -> None:
    """``apps/<directory>/app.yml``: one container, otherwise valid, declaring *contract*."""
    app_dir = repo / "apps" / directory
    app_dir.mkdir(parents=True, exist_ok=True)
    name = name or directory
    (app_dir / "app.yml").write_text(
        f"contract: {contract}\nname: {name}\n"
        f"containers:\n  {name}-service:\n    runtime: osprey\n"
        "    command: [python, -m, server]\n    health: /health\n",
        encoding="utf-8",
    )


def _profile(repo: Path, apps: list[str], **keys: object) -> None:
    path = repo / "profile.yml"
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    profile["apps"] = {name: {} for name in apps}
    profile.update(keys)
    path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")


def _check(runner: CliRunner, repo: Path) -> Result:
    return runner.invoke(app, ["check", "--repo", str(repo)])


def _findings(output: str) -> list[str]:
    """The refusal lines of a command's output: ``<where>: <message> [rule]``.

    Click prints a usage error's first line after ``Error: ``; that prefix is dropped.
    """
    lines = (line.strip().removeprefix("Error: ") for line in output.splitlines())
    return [line for line in lines if _FINDING.match(line)]


def test_two_mismatched_apps_give_two_findings_in_one_run_and_nothing_else(
    runner: CliRunner,
    lifecycle_repo: Path,  # noqa: F811
) -> None:
    _write_app(lifecycle_repo, "alpha", contract=OTHER)
    _write_app(lifecycle_repo, "beta", contract=OTHER)
    _profile(lifecycle_repo, ["alpha", "beta"])

    with pytest.raises(AppManifestError) as info:
        load(lifecycle_repo, ["alpha", "beta"])
    result = _check(runner, lifecycle_repo)

    assert [(v.app, v.rule) for v in info.value.violations] == [
        ("alpha", "contract"),
        ("beta", "contract"),
    ]
    assert result.exit_code == 1, result.output
    findings = _findings(result.output)
    assert len(findings) == 2, result.output
    assert findings[0].startswith("apps/alpha/app.yml: contract: ")
    assert findings[1].startswith("apps/beta/app.yml: contract: ")
    assert all(line.endswith("[contract]") for line in findings)
    assert "2 violation(s)." in result.output
    assert "names no application" not in result.output


def test_a_contract_mismatch_and_a_directory_name_mismatch_are_both_reported(
    runner: CliRunner,
    lifecycle_repo: Path,  # noqa: F811
) -> None:
    _write_app(lifecycle_repo, "alpha", contract=OTHER)
    _write_app(lifecycle_repo, "beta", name="gamma", contract=CONTRACT)
    _write_app(lifecycle_repo, "delta", contract=CONTRACT)
    _profile(lifecycle_repo, ["alpha", "beta", "delta"])

    with pytest.raises(AppManifestError) as info:
        load(lifecycle_repo, ["alpha", "beta", "delta"])
    result = _check(runner, lifecycle_repo)

    assert [(v.app, v.rule) for v in info.value.violations] == [
        ("alpha", "contract"),
        ("beta", "name_matches_directory"),
    ]
    assert [m.name for m in info.value.loaded] == ["delta"]
    assert result.exit_code == 1, result.output
    findings = _findings(result.output)
    assert [line.rpartition(" [")[2] for line in findings] == [
        "contract]",
        "name_matches_directory]",
    ]


def test_build_app_check_and_validate_print_the_same_refusal_line(
    runner: CliRunner,
    lifecycle_repo: Path,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    _write_app(lifecycle_repo, "knob", contract=OTHER)
    _profile(lifecycle_repo, ["knob"])

    checked = _check(runner, lifecycle_repo)
    validated = runner.invoke(validate, [str(lifecycle_repo / "profile.yml")])
    built = invoke_build(lifecycle_repo)

    [line] = _findings(checked.output)
    assert line.startswith("apps/knob/app.yml: contract: ")
    assert line.endswith("[contract]")
    assert checked.exit_code == 1, checked.output
    assert validated.exit_code == 2, validated.output
    assert line in validated.output
    assert built.exit_code != 0
    assert line in built.output + caplog.text


def test_a_missing_app_and_a_mismatched_app_in_the_apps_block_are_both_reported(
    runner: CliRunner,
    lifecycle_repo: Path,  # noqa: F811
) -> None:
    _write_app(lifecycle_repo, "knob", contract=OTHER)
    _profile(lifecycle_repo, ["knob", "ghost"])

    result = runner.invoke(validate, [str(lifecycle_repo / "profile.yml")])

    assert result.exit_code == 2, result.output
    assert "apps.ghost names no application" in result.output
    assert "apps.knob names no application" not in result.output
    [line] = _findings(result.output)
    assert line.startswith("apps/knob/app.yml: contract: ")
    assert "1 violation(s)." in result.output


def test_a_default_panel_naming_a_mismatched_apps_panel_reports_only_the_refusal(
    runner: CliRunner,
    lifecycle_repo: Path,  # noqa: F811
) -> None:
    _write_app(lifecycle_repo, "knob", contract=OTHER)
    _profile(lifecycle_repo, ["knob"], default_panel="knob")

    result = runner.invoke(validate, [str(lifecycle_repo / "profile.yml")])

    assert result.exit_code == 2, result.output
    assert "default_panel" not in result.output
    assert "Build profile validation failed" not in result.output
    [line] = _findings(result.output)
    assert line.startswith("apps/knob/app.yml: contract: ")
    assert "1 violation(s)." in result.output


def test_osprey_set_warns_with_the_refusal_lines(
    lifecycle_repo: Path,  # noqa: F811
) -> None:
    _write_app(lifecycle_repo, "knob", contract=OTHER)
    _profile(lifecycle_repo, ["knob"])

    result = CliRunner().invoke(
        set_command,
        [
            "--repo",
            str(lifecycle_repo),
            "config.claude_code.agent_models.knob-helper=claude-haiku-4-5",
        ],
    )

    assert "apps/knob/app.yml: contract: " in result.output
    assert "[contract]" in result.output
    assert "1 violation(s)." in result.output
